"""df collection and filtering for Δw in neighborhood of t_start"""
from collections.abc import Callable
from typing import Any, Literal

PresynapseFilter = Literal["all", "active", "inactive"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from models.unit_node_id import parse_unit_node_id
from visualize_webapp.common import _opt_label
from visualize_webapp.constants import _NETWORK_N_DIGITS, _NETWORK_SAMPLES_PER_DIGIT
from visualize_webapp.notebook.scoring_constants import N_SAMPLES_PER_TRIAL
from visualize_webapp.notebook.scoring_helpers import _unit_key, sample_ics_for_digits

PERIOD_KEYS = [
	"experiment_id",
	"model_id",
	"optimizer_id",
	"run_id",
	"neuron_id",
	"period_id",
]

RUN_GROUP_COLS = ["experiment_id", "model_id", "optimizer_id", "run_id"]

FILTER_STRATEGY_TO_COLUMN = {
	"angle": "angle_score",
	"length": "length_score",
	"robustness": "robustness_score",
	"total score": "total_score",
}


def robustness_score_column(df: pd.DataFrame) -> str:
	"""Prefer `minmax_normalized_robustness_scores` when present in `df`."""
	if "minmax_normalized_robustness_scores" in df.columns:
		return "minmax_normalized_robustness_scores"
	return "robustness_score"


def filter_score_column(strategy: str, df: pd.DataFrame | None = None) -> str:
	key = str(strategy).strip().lower()
	if key not in FILTER_STRATEGY_TO_COLUMN:
		raise ValueError(
			f"Unknown filter_strategy={strategy!r}; expected one of {sorted(FILTER_STRATEGY_TO_COLUMN)}"
		)
	if key == "robustness" and df is not None:
		return robustness_score_column(df)
	return FILTER_STRATEGY_TO_COLUMN[key]


def format_opt_display(optimizer_id: str) -> str:
	"""Human-readable optimizer name (e.g. `Pure Shampoo Lr0.001`)."""
	return " ".join(
		part.lower().capitalize() for part in _opt_label(optimizer_id).split("_")
	)


def _apply_iteration_time_xticks(ax, *, fontsize: float) -> None:
	ax.set_xlabel("iteration (time)", fontsize=fontsize)
	ticks = ax.get_xticks()
	if len(ticks) == 0:
		return
	tick_labels = [
		r"$t_\text{start}$" if np.isclose(t, 0) else (f"{int(t) if float(t).is_integer() else t:g}")
		for t in ticks
	]
	ax.set_xticks(ticks)
	ax.set_xticklabels(tick_labels, fontsize=fontsize)


def _title_period_point_counts(df: pd.DataFrame) -> tuple[int, int]:
	"""Return (n_periods, n_points) for plot titles.

	Each **Δw point** is one presynaptic weight update at a trial timestep where that
	input neuron was active on an assigned-digit sample: one (t_rel, w_init, Δw) scatter
	cell, not a separate BTSP period.
	"""
	n_periods = len(df)
	n_points = int(df["n_events"].sum()) if "n_events" in df.columns and n_periods else 0
	return n_periods, n_points


def _unit_node_ids_from_nts(nts: dict) -> list[str]:
	raw = nts.get("unit_node_ids")
	if raw is not None and len(raw):
		return [str(x) for x in raw]
	ids: set[str] = set()
	for key in nts:
		if key.startswith("act__"):
			ids.add(key[5:])
		elif key.startswith("weights__"):
			ids.add(key[9:])
	return sorted(ids)


def _hidden_layer_index(layer_name: str) -> int:
	return int(layer_name.split(".")[-1]) // 2


def _prev_hidden_layer_from_name(layer_name: str) -> str | None:
	if not str(layer_name).startswith("hidden."):
		return None
	k = int(str(layer_name).split(".")[-1])
	if k <= 0:
		return None
	return f"hidden.{k - 2}"


def _prev_hidden_layer_name(nts, layer_name: str) -> str | None:
	prev = _prev_hidden_layer_from_name(layer_name)
	if prev is None:
		return None
	for nid in _unit_node_ids_from_nts(nts):
		parsed = parse_unit_node_id(nid)
		if parsed and parsed["layer_name"] == prev and parsed["unit_type"] == "neuron":
			return prev
	return None


def _neurons_in_layer(nts, layer_name: str) -> list[tuple[int, str]]:
	out: list[tuple[int, str]] = []
	for uid in _unit_node_ids_from_nts(nts):
		parsed = parse_unit_node_id(uid)
		if parsed and parsed["layer_name"] == layer_name and parsed["unit_type"] == "neuron":
			out.append((parsed["unit_index"], uid))
	return sorted(out)


def _assigned_digits_from_cached_trial(
	nts,
	neuron_id: str,
	*,
	trial_start_cp: int,
	assignment_row: int,
) -> np.ndarray:
	act = np.asarray(
		nts[_unit_key("act", neuron_id)][trial_start_cp : trial_start_cp + N_SAMPLES_PER_TRIAL],
		dtype=np.float64,
	)
	row = int(assignment_row)
	active = (act[row] > 0).reshape((_NETWORK_N_DIGITS, _NETWORK_SAMPLES_PER_DIGIT))
	assigned = np.argwhere(np.all(active, axis=1)).flatten()
	if assigned.size < 1:
		raise ValueError(f"No assigned digits at trial row {row} for {neuron_id}")
	return assigned


def collect_tstart_vicinity_dw_from_row(
	nts,
	row,
	*,
	presynapse_filter: PresynapseFilter = "active",
	require_active_at_t: bool | None = None,
) -> dict[str, Any] | None:
	"""Δw events using cached periods_df trial/PMA columns (no recovery_periods rebuild).

	`presynapse_filter`:
	- `"active"`: presynapse active on an assigned-digit sample at that timestep
	- `"all"`: all presynaptic weights in the time window
	- `"inactive"`: presynapses not active at that timestep (full \\ active)
	"""
	if require_active_at_t is not None:
		presynapse_filter = "active" if require_active_at_t else "all"
	parsed = parse_unit_node_id(str(row.neuron_id))
	if parsed is None or not parsed["layer_name"].startswith("hidden."):
		return None
	if _hidden_layer_index(parsed["layer_name"]) <= 0:
		return None

	prev_layer_name = _prev_hidden_layer_name(nts, parsed["layer_name"])
	if prev_layer_name is None:
		return None

	trial_start_cp = int(row.trial_start_cp)
	t_start = int(row.point_of_max_acceleration)
	t_assign = int(row.assignment_within_trial)
	# delta = int(min(t_start, t_assign - t_start)) <- other potential definition
	delta = int(t_start)
	if delta < 0:
		return None

	t_rel_min, t_rel_max = t_start - delta, t_start + delta
	if t_rel_min < 0 or t_rel_max >= N_SAMPLES_PER_TRIAL:
		return None

	w_key = _unit_key("weights", str(row.neuron_id))
	if w_key not in nts:
		return None

	try:
		assigned_digits = _assigned_digits_from_cached_trial(
			nts,
			str(row.neuron_id),
			trial_start_cp=trial_start_cp,
			assignment_row=t_assign,
		)
	except (KeyError, ValueError, IndexError):
		return None

	sample_cols = sample_ics_for_digits(assigned_digits)
	prev_neurons = _neurons_in_layer(nts, prev_layer_name)
	w_trial = np.asarray(
		nts[w_key][trial_start_cp : trial_start_cp + N_SAMPLES_PER_TRIAL],
		dtype=np.float64,
	)
	act_trials = {
		j_idx: np.asarray(
			nts[_unit_key("act", prev_nid)][trial_start_cp : trial_start_cp + N_SAMPLES_PER_TRIAL],
			dtype=np.float64,
		)
		for j_idx, prev_nid in prev_neurons
	}
	w_init_row = w_trial[t_start]

	t_rel_list: list[int] = []
	j_idx_list: list[int] = []
	w_init_list: list[float] = []
	delta_w_list: list[float] = []
	for t in range(t_rel_min, t_rel_max + 1):
		for j_idx, _prev_nid in prev_neurons:
			if j_idx >= w_trial.shape[1]:
				continue
			active_at_t = (act_trials[j_idx][t, sample_cols] > 0).any()
			if presynapse_filter == "active" and not active_at_t:
				continue
			if presynapse_filter == "inactive" and active_at_t:
				continue
			t_rel_list.append(t - t_start)
			j_idx_list.append(j_idx)
			w_init_list.append(float(w_init_row[j_idx]))
			delta_w_list.append(float(w_trial[t, j_idx] - w_init_row[j_idx]))

	if not t_rel_list:
		return None

	return _vicinity_event_dict(
		t_start, t_assign, delta, t_rel_list, j_idx_list, w_init_list, delta_w_list
	)


def _vicinity_event_dict(
	t_start: int,
	t_assign: int,
	delta: int,
	t_rel_list: list[int],
	j_idx_list: list[int],
	w_init_list: list[float],
	delta_w_list: list[float],
) -> dict[str, Any]:
	return {
		"t_start": t_start,
		"t_assign": t_assign,
		"delta": delta,
		"t_rel_lo": int(-delta),
		"t_rel_hi": int(delta),
		"t_rel": np.asarray(t_rel_list, dtype=np.int32),
		"j_idx": np.asarray(j_idx_list, dtype=np.int32),
		"w_init": np.asarray(w_init_list, dtype=np.float64),
		"delta_w": np.asarray(delta_w_list, dtype=np.float64),
		"n_events": len(t_rel_list),
	}


def collect_tstart_vicinity_dw_full_and_inactive_from_row(
	nts,
	row,
) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
	"""Full-window events plus inactive-only (full \\ active); one NTS load per run."""
	full_vicinity = collect_tstart_vicinity_dw_from_row(nts, row, presynapse_filter="all")
	if full_vicinity is None:
		return None
	inactive_vic = collect_tstart_vicinity_dw_from_row(nts, row, presynapse_filter="inactive")
	return full_vicinity, inactive_vic


def build_dw_vicinity_df(
	periods: pd.DataFrame,
	*,
	load_nts: Callable[..., Any],
	desc: str = "Δw vicinity",
	presynapse_filter: PresynapseFilter = "active",
	require_active_at_t: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
	"""Load neuron timeseries once per run; return one row per plottable period."""
	if require_active_at_t is not None:
		presynapse_filter = "active" if require_active_at_t else "all"

	rows_out: list[dict[str, Any]] = []
	skipped: dict[str, int] = {}
	if periods.empty:
		return pd.DataFrame(columns=PERIOD_KEYS), skipped

	grouped = periods.groupby(RUN_GROUP_COLS, sort=False)
	for run_key, group in tqdm(grouped, desc=desc, leave=False):
		eid, mid, oid, rid = run_key
		try:
			nts = load_nts(eid, mid, oid, rid=rid, w_cache=False)
			if not nts or not _unit_node_ids_from_nts(nts):
				raise FileNotFoundError(
					f"neuron_timeseries.npz missing or empty for {(eid, mid, oid, rid)}"
				)
		except Exception:
			skipped["nts_load"] = skipped.get("nts_load", 0) + len(group)
			continue

		for row in group.itertuples():
			vic = collect_tstart_vicinity_dw_from_row(
				nts, row, presynapse_filter=presynapse_filter
			)
			if vic is None:
				skipped["not_plottable"] = skipped.get("not_plottable", 0) + 1
				continue
			rows_out.append(
				{
					"experiment_id": eid,
					"model_id": mid,
					"optimizer_id": oid,
					"run_id": rid,
					"neuron_id": str(row.neuron_id),
					"period_id": int(row.period_id),
					**vic,
				}
			)

		del nts

	return pd.DataFrame(rows_out), skipped


def build_dw_vicinity_df_full_and_inactive(
	periods: pd.DataFrame,
	*,
	load_nts: Callable[..., Any],
	desc: str = "Δw vicinity (full + inactive)",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
	"""One NTS pass per run: full-window and inactive-only rows per plottable period."""
	rows_full: list[dict[str, Any]] = []
	rows_inactive: list[dict[str, Any]] = []
	skipped: dict[str, int] = {}
	if periods.empty:
		return pd.DataFrame(columns=PERIOD_KEYS), pd.DataFrame(columns=PERIOD_KEYS), skipped

	grouped = periods.groupby(RUN_GROUP_COLS, sort=False)
	for run_key, group in tqdm(grouped, desc=desc, leave=False):
		eid, mid, oid, rid = run_key
		try:
			nts = load_nts(eid, mid, oid, rid=rid, w_cache=False)
			if not nts or not _unit_node_ids_from_nts(nts):
				raise FileNotFoundError(
					f"neuron_timeseries.npz missing or empty for {(eid, mid, oid, rid)}"
				)
		except Exception:
			skipped["nts_load"] = skipped.get("nts_load", 0) + len(group)
			continue

		for row in group.itertuples():
			out = collect_tstart_vicinity_dw_full_and_inactive_from_row(nts, row)
			if out is None:
				skipped["not_plottable_full"] = skipped.get("not_plottable_full", 0) + 1
				continue
			full_vic, inactive_vic = out
			base = {
				"experiment_id": eid,
				"model_id": mid,
				"optimizer_id": oid,
				"run_id": rid,
				"neuron_id": str(row.neuron_id),
				"period_id": int(row.period_id),
			}
			rows_full.append({**base, **full_vic})
			if inactive_vic is not None:
				rows_inactive.append({**base, **inactive_vic})
			else:
				skipped["no_inactive_events"] = skipped.get("no_inactive_events", 0) + 1

		del nts

	return pd.DataFrame(rows_full), pd.DataFrame(rows_inactive), skipped


def merge_dw_with_periods(
	df_dw_vicinity: pd.DataFrame,
	periods: pd.DataFrame,
) -> pd.DataFrame:
	"""Inner join: vicinity arrays + score/experiment columns from `periods`."""
	if df_dw_vicinity.empty:
		return df_dw_vicinity.copy()
	extra_cols = [c for c in periods.columns if c not in df_dw_vicinity.columns]
	return df_dw_vicinity.merge(periods[PERIOD_KEYS + extra_cols], on=PERIOD_KEYS, how="inner")


def build_merged_df_dw(
	all_periods_by_experiment: dict[str, pd.DataFrame],
	load_nts: Callable[..., Any],
	*,
	desc: str = "Δw vicinity (all periods)",
	require_active_at_t: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
	"""Concat period pools, load vicinity arrays once per run, merge scores."""
	if not all_periods_by_experiment:
		return pd.DataFrame(), {}

	periods_all = pd.concat(all_periods_by_experiment.values(), ignore_index=True)
	periods_all = periods_all.drop_duplicates(subset=PERIOD_KEYS)
	df_dw_vicinity, skipped = build_dw_vicinity_df(
		periods_all,
		load_nts=load_nts,
		desc=desc,
		require_active_at_t=require_active_at_t,
	)
	df_dw = merge_dw_with_periods(df_dw_vicinity, periods_all)
	return df_dw, skipped


def build_merged_df_dw_full_and_inactive(
	all_periods_by_experiment: dict[str, pd.DataFrame],
	load_nts: Callable[..., Any],
	*,
	desc: str = "Δw vicinity (full + inactive)",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
	"""Build full-window and inactive-only (full \\ active) merged DataFrames in one NTS pass."""
	if not all_periods_by_experiment:
		return pd.DataFrame(), pd.DataFrame(), {}

	periods_all = pd.concat(all_periods_by_experiment.values(), ignore_index=True)
	periods_all = periods_all.drop_duplicates(subset=PERIOD_KEYS)
	df_full_vic, df_inactive_vic, skipped = build_dw_vicinity_df_full_and_inactive(
		periods_all,
		load_nts=load_nts,
		desc=desc,
	)
	df_dw_full = merge_dw_with_periods(df_full_vic, periods_all)
	df_dw_inactive = merge_dw_with_periods(df_inactive_vic, periods_all)
	return df_dw_full, df_dw_inactive, skipped


def select_dw_periods(
	df_dw: pd.DataFrame,
	*,
	experiment: str | None = None,
	experiments: list[str] | None = None,
	optimizer_id: str | None = None,
	top_n: int | None = None,
	score_column: str = "total_score",
) -> pd.DataFrame:
	out = df_dw
	if experiment is not None:
		out = out[out["experiment"] == experiment]
	if experiments is not None:
		out = out[out["experiment"].isin(experiments)]
	if optimizer_id is not None:
		out = out[out["optimizer_id"] == optimizer_id]
	if top_n is not None:
		out = out.dropna(subset=[score_column]).sort_values(score_column, ascending=False).head(top_n)
	return out.reset_index(drop=True)


def stack_dw_events(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	if df.empty:
		return (
			np.array([], dtype=np.float64),
			np.array([], dtype=np.float64),
			np.array([], dtype=np.float64),
		)
	t_rel = np.concatenate([np.asarray(r, dtype=np.float64) for r in df["t_rel"]])
	w_init = np.concatenate([np.asarray(r, dtype=np.float64) for r in df["w_init"]])
	delta_w = np.concatenate([np.asarray(r, dtype=np.float64) for r in df["delta_w"]])
	return t_rel, w_init, delta_w


def vlim_from_df(df: pd.DataFrame, q: float = 98) -> float:
	_, _, delta_w = stack_dw_events(df)
	if delta_w.size == 0:
		return 1e-6
	return max(float(np.percentile(np.abs(delta_w), q)), 1e-12)


def shared_t_rel_xlim_from_dfs(
	dfs: list[pd.DataFrame],
	*,
	pad: float = 0.5,
) -> tuple[float, float] | None:
	"""Union of per-period `[t_rel_lo, t_rel_hi]` (or stacked events), with scatter-style padding."""
	lo_vals: list[float] = []
	hi_vals: list[float] = []
	for df in dfs:
		if df.empty:
			continue
		if "t_rel_lo" in df.columns and "t_rel_hi" in df.columns:
			lo_vals.append(float(df["t_rel_lo"].min()))
			hi_vals.append(float(df["t_rel_hi"].max()))
			continue
		t_rel, _, _ = stack_dw_events(df)
		if t_rel.size:
			lo_vals.append(float(t_rel.min()))
			hi_vals.append(float(t_rel.max()))
	if not lo_vals:
		return None
	return (min(lo_vals) - pad, max(hi_vals) + pad)


def _w_pad_from_range(w_min: float, w_max: float) -> float:
	return max(0.05 * (w_max - w_min), 1e-4) if w_max > w_min else 1e-3


def shared_w_edges_from_dfs(
	dfs: list[pd.DataFrame],
	*,
	n_w_bins: int = 51,
) -> np.ndarray | None:
	"""Union of stacked `w_init` ranges across panels, with scatter-style padding."""
	w_mins: list[float] = []
	w_maxs: list[float] = []
	for df in dfs:
		if df.empty:
			continue
		_, w_arr, _ = stack_dw_events(df)
		if w_arr.size:
			w_mins.append(float(w_arr.min()))
			w_maxs.append(float(w_arr.max()))
	if not w_mins:
		return None
	w_min, w_max = min(w_mins), max(w_maxs)
	w_pad = _w_pad_from_range(w_min, w_max)
	return np.linspace(w_min - w_pad, w_max + w_pad, n_w_bins + 1)


def _integer_t_rel_xticks(xlim: tuple[float, float]) -> np.ndarray:
	lo, hi = xlim
	ticks = {0} if lo <= 0 <= hi else set()
	ticks.update(range(int(np.ceil(lo)), int(np.floor(hi)) + 1))
	return np.array(sorted(ticks), dtype=np.float64)


def shared_t_rel_xticks_from_xlim(
	xlim: tuple[float, float],
	*,
	n_per_side: int = 4,
) -> np.ndarray:
	"""Integer ticks: 0 plus *n_per_side* values from `linspace(0, max_xtick)` on each side."""
	lo, hi = xlim
	lo_i = int(np.ceil(lo))
	hi_i = int(np.floor(hi))
	max_xtick = max(-lo_i if lo_i < 0 else 0, hi_i if hi_i > 0 else 0)
	if max_xtick == 0:
		return np.array([0.0], dtype=np.float64) if lo <= 0 <= hi else np.array([], dtype=np.float64)

	side = np.unique(np.round(np.linspace(0, max_xtick, n_per_side + 1)[1:]).astype(int))
	side = side[side > 0]
	ticks: list[float] = []
	if lo <= 0 <= hi:
		ticks.append(0.0)
	for t in side:
		if t <= hi_i:
			ticks.append(float(t))
	for t in side:
		if -t >= lo_i:
			ticks.append(float(-t))
	return np.array(sorted(set(ticks)), dtype=np.float64)


def _apply_shared_t_rel_xaxis(
	ax,
	xlim: tuple[float, float],
	*,
	fontsize: float,
	xticks: np.ndarray | None = None,
) -> None:
	ax.set_xlim(xlim)
	ax.set_xticks(_integer_t_rel_xticks(xlim) if xticks is None else xticks)
	_apply_iteration_time_xticks(ax, fontsize=fontsize)


def _t_bounds_for_grid(
	df: pd.DataFrame,
	*,
	t_xlim: tuple[float, float] | None = None,
) -> tuple[float, float]:
	"""Time axis extent: explicit xlim, else per-period window, else stacked events."""
	if t_xlim is not None:
		return t_xlim
	if not df.empty and "t_rel_lo" in df.columns and "t_rel_hi" in df.columns:
		return float(df["t_rel_lo"].min()), float(df["t_rel_hi"].max())
	t_arr, _, _ = stack_dw_events(df)
	return float(t_arr.min()), float(t_arr.max())


def _integer_t_edges(t_lo: float, t_hi: float) -> np.ndarray:
	"""One bin per integer t_rel; avoids empty columns from uniform linspace binning."""
	lo_i = int(np.floor(t_lo))
	hi_i = int(np.ceil(t_hi))
	return (np.arange(hi_i - lo_i + 2, dtype=np.float64) + lo_i) - 0.5


def aggregate_dw_events_grid(
	df: pd.DataFrame,
	*,
	n_t_bins: int = 51,
	n_w_bins: int = 51,
	w_edges: np.ndarray | None = None,
	t_xlim: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
	"""Bin (t_rel, w_init) and average Δw across selected periods."""
	t_arr, w_arr, dw_arr = stack_dw_events(df)
	if t_arr.size == 0:
		return None

	t_lo, t_hi = _t_bounds_for_grid(df, t_xlim=t_xlim)
	t_edges = _integer_t_edges(t_lo, t_hi)
	if w_edges is None:
		w_min, w_max = float(w_arr.min()), float(w_arr.max())
		w_pad = _w_pad_from_range(w_min, w_max)
		w_edges = np.linspace(w_min - w_pad, w_max + w_pad, n_w_bins + 1)

	sum_grid, _, _ = np.histogram2d(t_arr, w_arr, bins=[t_edges, w_edges], weights=dw_arr)
	cnt_grid, _, _ = np.histogram2d(t_arr, w_arr, bins=[t_edges, w_edges])
	with np.errstate(invalid="ignore", divide="ignore"):
		mean_grid = sum_grid / cnt_grid
	mean_grid[cnt_grid == 0] = np.nan
	return mean_grid, t_edges, w_edges


def aggregate_dw_events_grid_diff(
	df: pd.DataFrame,
	df_baseline: pd.DataFrame,
	*,
	n_t_bins: int = 51,
	n_w_bins: int = 51,
	w_edges: np.ndarray | None = None,
	t_xlim: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
	"""Bin (t_rel, w_init) and average Δw; return `mean(df) - mean(df_baseline)`.

	For full-vs-inactive heatmaps, pass full as `df` and inactive-only as `df_baseline`.
	"""
	grid_a = aggregate_dw_events_grid(
		df, n_t_bins=n_t_bins, n_w_bins=n_w_bins, w_edges=w_edges, t_xlim=t_xlim
	)
	grid_b = aggregate_dw_events_grid(
		df_baseline,
		n_t_bins=n_t_bins,
		n_w_bins=n_w_bins,
		w_edges=w_edges,
		t_xlim=t_xlim,
	)
	if grid_a is None and grid_b is None:
		return None
	if grid_a is None:
		mean_b, t_edges, w_edges = grid_b
		return -mean_b, t_edges, w_edges
	if grid_b is None:
		mean_a, t_edges, w_edges = grid_a
		return mean_a, t_edges, w_edges
	mean_a, t_edges, w_edges = grid_a
	mean_b, _, _ = grid_b
	diff = mean_a - mean_b
	both_nan = np.isnan(mean_a) & np.isnan(mean_b)
	diff[both_nan] = np.nan
	return diff, t_edges, w_edges


def vlim_from_diff_grid_pairs(
	pairs: list[tuple[pd.DataFrame, pd.DataFrame]],
	*,
	q: float = 98,
	n_t_bins: int = 51,
	n_w_bins: int = 51,
	w_edges: np.ndarray | None = None,
	t_xlim: tuple[float, float] | None = None,
) -> float:
	abs_vals: list[float] = []
	for df, df_baseline in pairs:
		grid_data = aggregate_dw_events_grid_diff(
			df,
			df_baseline,
			n_t_bins=n_t_bins,
			n_w_bins=n_w_bins,
			w_edges=w_edges,
			t_xlim=t_xlim,
		)
		if grid_data is None:
			continue
		diff_grid, _, _ = grid_data
		finite = diff_grid[np.isfinite(diff_grid)]
		if finite.size:
			abs_vals.extend(np.abs(finite).tolist())
	if not abs_vals:
		return 1e-6
	return max(float(np.percentile(abs_vals, q)), 1e-12)


def aggregate_dw_events_grid_ratio(
	df_active: pd.DataFrame,
	df_full: pd.DataFrame,
	df_inactive: pd.DataFrame,
	*,
	n_t_bins: int = 51,
	n_w_bins: int = 51,
	w_edges: np.ndarray | None = None,
	t_xlim: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
	"""Bin (t_rel, w_init); return `(mean_active - mean_inactive) / mean_full` per cell."""
	grid_active = aggregate_dw_events_grid(
		df_active,
		n_t_bins=n_t_bins,
		n_w_bins=n_w_bins,
		w_edges=w_edges,
		t_xlim=t_xlim,
	)
	grid_full = aggregate_dw_events_grid(
		df_full,
		n_t_bins=n_t_bins,
		n_w_bins=n_w_bins,
		w_edges=w_edges,
		t_xlim=t_xlim,
	)
	grid_inactive = aggregate_dw_events_grid(
		df_inactive,
		n_t_bins=n_t_bins,
		n_w_bins=n_w_bins,
		w_edges=w_edges,
		t_xlim=t_xlim,
	)
	ref = grid_full or grid_active or grid_inactive
	if ref is None:
		return None
	_, t_edges, w_edges_out = ref
	shape = ref[0].shape

	def _mean_or_nan(grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> np.ndarray:
		if grid is None:
			return np.full(shape, np.nan)
		return grid[0]

	mean_active = _mean_or_nan(grid_active)
	mean_full = _mean_or_nan(grid_full)
	mean_inactive = _mean_or_nan(grid_inactive)
	with np.errstate(divide="ignore", invalid="ignore"):
		ratio = (mean_active - mean_inactive) / mean_full
	ratio[~np.isfinite(ratio)] = np.nan
	all_nan = np.isnan(mean_active) & np.isnan(mean_full) & np.isnan(mean_inactive)
	ratio[all_nan] = np.nan
	return ratio, t_edges, w_edges_out


def vlim_from_ratio_grid_triples(
	triples: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
	*,
	q: float = 98,
	n_t_bins: int = 51,
	n_w_bins: int = 51,
	w_edges: np.ndarray | None = None,
	t_xlim: tuple[float, float] | None = None,
) -> float:
	abs_vals: list[float] = []
	for df_active, df_full, df_inactive in triples:
		grid_data = aggregate_dw_events_grid_ratio(
			df_active,
			df_full,
			df_inactive,
			n_t_bins=n_t_bins,
			n_w_bins=n_w_bins,
			w_edges=w_edges,
			t_xlim=t_xlim,
		)
		if grid_data is None:
			continue
		ratio_grid, _, _ = grid_data
		finite = ratio_grid[np.isfinite(ratio_grid)]
		if finite.size:
			abs_vals.extend(np.abs(finite).tolist())
	if not abs_vals:
		return 1e-6
	return max(float(np.percentile(abs_vals, q)), 1e-12)


def plot_period_tstart_vicinity_dw_scatter_grid(
	df: pd.DataFrame,
	*,
	experiment_label: str,
	ncols: int = 10,
	cmap: str = "RdBu_r",
	parse_neuron_id: Callable[[str], str] | None = None,
):
	if df.empty:
		print(f"No plottable Δw periods for {experiment_label!r}.")
		return None

	if parse_neuron_id is None:

		def parse_neuron_id(neuron_id: str) -> str:
			parsed = parse_unit_node_id(neuron_id)
			if parsed is None:
				return str(neuron_id)
			layer_idx = int(parsed["layer_name"].split(".")[-1]) // 2
			return r"$n^{(" + str(layer_idx) + r")}_{" + str(parsed["unit_index"]) + r"}$"

	n_periods = len(df)
	nrows = int(np.ceil(n_periods / ncols))
	fig, axes = plt.subplots(
		nrows,
		ncols,
		figsize=(2.5 * ncols, 2.3 * nrows),
		squeeze=False,
		sharey=True,
		constrained_layout=True,
	)
	axes_flat = axes.ravel()

	_, _, all_dw = stack_dw_events(df)
	_, w_init_stack, _ = stack_dw_events(df)
	vlim = vlim_from_df(df)
	w_min, w_max = float(w_init_stack.min()), float(w_init_stack.max())
	w_pad = max(0.05 * (w_max - w_min), 1e-4) if w_max > w_min else 1e-3
	ylim_shared = (w_min - w_pad, w_max + w_pad)

	for ax in axes_flat[n_periods:]:
		ax.axis("off")

	sc_last = None
	for ax, row in zip(axes_flat, df.itertuples(index=False)):
		t_rel = np.asarray(row.t_rel)
		w_init = np.asarray(row.w_init)
		delta_w = np.asarray(row.delta_w)
		sc_last = ax.scatter(
			t_rel,
			w_init,
			c=delta_w,
			cmap=cmap,
			vmin=-vlim,
			vmax=vlim,
			s=12,
			marker="s",
			linewidths=0,
			edgecolors="none",
		)
		ax.set_xlim(int(row.t_rel_lo) - 0.5, int(row.t_rel_hi) + 0.5)
		_apply_iteration_time_xticks(ax, fontsize=6)
		score = getattr(row, "total_score", np.nan)
		ax.set_title(
			f"{format_opt_display(row.optimizer_id)}, {parse_neuron_id(row.neuron_id)}\n"
			r"$t_\text{start}=" + f"{row.t_start}$, " + r"$\delta=" + f"{row.delta}$, score={score:.3f}",
			fontsize=7,
		)

	axes_flat[0].set_ylim(ylim_shared)
	for i, ax in enumerate(axes_flat[:n_periods]):
		if i % ncols == 0:
			ax.set_ylabel("initial w_ij", fontsize=7)

	cbar = fig.colorbar(sc_last, ax=axes_flat[:n_periods], fraction=0.02, pad=0.02)
	cbar.set_label(r"$\Delta W_{i,j}$")
	fig.suptitle(
		f"{experiment_label} — all {n_periods} BTSP periods "
		+ r"($\Delta W_{ij}$ around $t_\text{start}$"
		+ ", layer $l-1$)",
		fontsize=12,
		y=1.02,
	)
	return fig


def plot_mean_tstart_vicinity_dw_heatmap(
	df: pd.DataFrame,
	*,
	experiment_label: str,
	cmap: str = "RdBu_r",
	n_t_bins: int = 51,
	n_w_bins: int = 51,
	ytick_fontsize: float = 8,
):
	grid_data = aggregate_dw_events_grid(df, n_t_bins=n_t_bins, n_w_bins=n_w_bins)
	if grid_data is None:
		print(f"No aggregate Δw grid for {experiment_label!r}.")
		return None

	mean_grid, t_edges, w_edges = grid_data
	vlim = vlim_from_df(df)

	fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
	extent = (t_edges[0], t_edges[-1], w_edges[0], w_edges[-1])
	im = ax.imshow(
		mean_grid.T,
		origin="lower",
		aspect="auto",
		extent=extent,
		cmap=cmap,
		vmin=-vlim,
		vmax=vlim,
		interpolation="nearest",
	)
	_apply_iteration_time_xticks(ax, fontsize=10)
	ax.set_ylabel("initial $w_{ij}$", fontsize=10)
	ax.tick_params(axis="y", labelsize=ytick_fontsize)
	n_periods, n_points = _title_period_point_counts(df)
	ax.set_title(
		f"{experiment_label} — mean Δw around t_start"
		f" (n={n_periods} periods, {n_points} Δw points)",
		fontsize=11,
	)
	cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_label("mean Δw$_{ij}$")
	return fig


def plot_mean_tstart_vicinity_dw_heatmap_on_ax(
	ax,
	df: pd.DataFrame,
	*,
	cmap: str = "RdBu_r",
	n_t_bins: int = 51,
	n_w_bins: int = 51,
	vlim: float | None = None,
	xlim: tuple[float, float] | None = None,
	xticks: np.ndarray | None = None,
	show_ylabel: bool = True,
	ylabel_fontsize: float = 10,
	ytick_fontsize: float = 8,
	w_edges: np.ndarray | None = None,
):
	grid_data = aggregate_dw_events_grid(
		df, n_t_bins=n_t_bins, n_w_bins=n_w_bins, w_edges=w_edges, t_xlim=xlim
	)
	if grid_data is None:
		ax.axis("off")
		return None

	mean_grid, t_edges, w_edges = grid_data
	if vlim is None:
		vlim = vlim_from_df(df)

	extent = (t_edges[0], t_edges[-1], w_edges[0], w_edges[-1])
	im = ax.imshow(
		mean_grid.T,
		origin="lower",
		aspect="auto",
		extent=extent,
		cmap=cmap,
		vmin=-vlim,
		vmax=vlim,
		interpolation="nearest",
	)
	if xlim is not None:
		_apply_shared_t_rel_xaxis(ax, xlim, fontsize=10, xticks=xticks)
	else:
		_apply_iteration_time_xticks(ax, fontsize=10)
	if show_ylabel:
		ax.set_ylabel(r"$w_{ij}^{init}$", fontsize=ylabel_fontsize)
	ax.tick_params(axis="y", labelsize=ytick_fontsize)
	return im


def plot_mean_tstart_vicinity_dw_heatmap_diff_on_ax(
	ax,
	df: pd.DataFrame,
	df_baseline: pd.DataFrame,
	*,
	cmap: str = "PRGn",
	n_t_bins: int = 51,
	n_w_bins: int = 51,
	vlim: float | None = None,
	xlim: tuple[float, float] | None = None,
	xticks: np.ndarray | None = None,
	show_ylabel: bool = True,
	ylabel_fontsize: float = 10,
	ytick_fontsize: float = 8,
	w_edges: np.ndarray | None = None,
):
	grid_data = aggregate_dw_events_grid_diff(
		df,
		df_baseline,
		n_t_bins=n_t_bins,
		n_w_bins=n_w_bins,
		w_edges=w_edges,
		t_xlim=xlim,
	)
	if grid_data is None:
		ax.axis("off")
		return None

	diff_grid, t_edges, w_edges = grid_data
	if vlim is None:
		finite = diff_grid[np.isfinite(diff_grid)]
		vlim = (
			max(float(np.percentile(np.abs(finite), 98)), 1e-12)
			if finite.size
			else 1e-6
		)

	extent = (t_edges[0], t_edges[-1], w_edges[0], w_edges[-1])
	im = ax.imshow(
		diff_grid.T,
		origin="lower",
		aspect="auto",
		extent=extent,
		cmap=cmap,
		vmin=-vlim,
		vmax=vlim,
		interpolation="nearest",
	)
	if xlim is not None:
		_apply_shared_t_rel_xaxis(ax, xlim, fontsize=10, xticks=xticks)
	else:
		_apply_iteration_time_xticks(ax, fontsize=10)
	if show_ylabel:
		ax.set_ylabel(r"$w_{ij}^{init}$", fontsize=ylabel_fontsize)
	ax.tick_params(axis="y", labelsize=ytick_fontsize)
	return im


def plot_mean_tstart_vicinity_dw_heatmap_ratio_on_ax(
	ax,
	df_active: pd.DataFrame,
	df_full: pd.DataFrame,
	df_inactive: pd.DataFrame,
	*,
	cmap: str = "PRGn",
	n_t_bins: int = 51,
	n_w_bins: int = 51,
	vlim: float | None = None,
	xlim: tuple[float, float] | None = None,
	xticks: np.ndarray | None = None,
	show_ylabel: bool = True,
	ylabel_fontsize: float = 10,
	ytick_fontsize: float = 8,
	w_edges: np.ndarray | None = None,
):
	grid_data = aggregate_dw_events_grid_ratio(
		df_active,
		df_full,
		df_inactive,
		n_t_bins=n_t_bins,
		n_w_bins=n_w_bins,
		w_edges=w_edges,
		t_xlim=xlim,
	)
	if grid_data is None:
		ax.axis("off")
		return None

	ratio_grid, t_edges, w_edges = grid_data
	if vlim is None:
		finite = ratio_grid[np.isfinite(ratio_grid)]
		vlim = (
			max(float(np.percentile(np.abs(finite), 98)), 1e-12)
			if finite.size
			else 1e-6
		)

	extent = (t_edges[0], t_edges[-1], w_edges[0], w_edges[-1])
	im = ax.imshow(
		ratio_grid.T,
		origin="lower",
		aspect="auto",
		extent=extent,
		cmap=cmap,
		vmin=-vlim,
		vmax=vlim,
		interpolation="nearest",
	)
	if xlim is not None:
		_apply_shared_t_rel_xaxis(ax, xlim, fontsize=10, xticks=xticks)
	else:
		_apply_iteration_time_xticks(ax, fontsize=10)
	if show_ylabel:
		ax.set_ylabel(r"$w_{ij}^{init}$", fontsize=ylabel_fontsize)
	ax.tick_params(axis="y", labelsize=ytick_fontsize)
	return im
