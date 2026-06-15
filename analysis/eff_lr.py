"""Computation and plots for effective learning rate analysis"""
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import ticker as mpl_ticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from compute_results.defaults import DEFAULT_ADAM_EPS
from models.unit_node_id import parse_unit_node_id
from analysis.common import _opt_color, _opt_type
from analysis.dw_vicinity import (
	PERIOD_KEYS,
	RUN_GROUP_COLS,
	filter_score_column,
	format_opt_display,
	select_dw_periods,
)
from analysis.scoring_constants import N_SAMPLES_PER_TRIAL
from analysis.scoring_helpers import is_pretrain_shuffle_mislabel_experiment

# Re-use period selection from Δw vicinity (same keys / score columns).
select_efflr_periods = select_dw_periods


def is_cat1_sample_shuffle_experiment(experiment_id: str) -> bool:
	"""Category-1 sample-shuffle experiments (`cat1_sample_shuffle_*`)."""
	return is_pretrain_shuffle_mislabel_experiment(experiment_id)


def is_cat2_sequence_experiment(experiment_id: str) -> bool:
	return str(experiment_id).startswith("cat2_sequence_")


def _opt_order_without_sgd(opt_order: list[str]) -> list[str]:
	"""SGD has constant η_eff — omit from comparative eff-LR bar plots."""
	return [oid for oid in opt_order if "sgd" not in str(oid).lower()]


def _parse_lr_from_optimizer_id(oid: str) -> float:
	return float(oid.rsplit("_lr", 1)[1])


def _hidden_layer_index(layer_name: str) -> int:
	return int(layer_name.split(".")[-1]) // 2


def _signal_row_for_checkpoint(sigs: dict, metrics: dict, cp_idx: int) -> int | None:
	cp_iters = metrics.get("checkpoint_iterations")
	if cp_iters is None or cp_idx < 0 or cp_idx >= len(cp_iters):
		return None
	target = int(cp_iters[cp_idx])
	sig_iters = sigs.get("iteration")
	if sig_iters is None:
		return cp_idx
	matches = np.where(np.asarray(sig_iters) == target)[0]
	if len(matches):
		return int(matches[0])
	if 0 <= target < len(sig_iters):
		return target
	return None


def _unit_col_idx(sigs: dict, neuron_id: str) -> int:
	uid_list = [str(n) for n in sigs.get("unit_node_ids", [])]
	return uid_list.index(str(neuron_id)) if str(neuron_id) in uid_list else -1


def _scalar_from_signal_row(sigs: dict, oid: str, neuron_id: str, row_idx: int) -> float:
	if row_idx is None or row_idx < 0:
		return float("nan")
	otype = _opt_type(oid)
	safe = str(neuron_id).replace(":", "__")
	lr = _parse_lr_from_optimizer_id(oid)

	if otype == "sgd":
		return lr

	if otype in ("adagrad", "grafted_shampoo"):
		key = f"effective_lr__{safe}"
		if key in sigs:
			arr = sigs[key]
			if row_idx >= len(arr):
				return float("nan")
			val = arr[row_idx]
			if isinstance(val, np.ndarray):
				return float(np.nanmean(val))
			return float(val)
		return float("nan")

	if otype == "pure_shampoo":
		if "h_inv_norm" not in sigs:
			return float("nan")
		arr = sigs["h_inv_norm"]
		if row_idx >= len(arr):
			return float("nan")
		col = _unit_col_idx(sigs, neuron_id)
		if arr.ndim == 2 and col >= 0:
			return float(arr[row_idx, col])
		return float(arr[row_idx])

	if otype == "adam":
		key = f"exp_avg_sq__{safe}"
		if key not in sigs:
			return float("nan")
		arr = sigs[key]
		if row_idx >= len(arr):
			return float("nan")
		val = arr[row_idx]
		# TODO: make per parameter analysis
		if isinstance(val, np.ndarray):
			v = np.sqrt(np.nanmean(val))
			return lr / (float(v) + DEFAULT_ADAM_EPS)
		return lr / (float(val) + DEFAULT_ADAM_EPS)

	return float("nan")


def _effective_lr_at_trial_iter(
	sigs: dict,
	metrics: dict,
	oid: str,
	neuron_id: str,
	trial_start_cp: int,
	t: int,
) -> float:
	cp_idx = int(trial_start_cp) + int(t)
	row_idx = _signal_row_for_checkpoint(sigs, metrics, cp_idx)
	return _scalar_from_signal_row(sigs, oid, neuron_id, row_idx if row_idx is not None else -1)


def collect_efflr_from_row(sigs, metrics, row) -> dict[str, Any] | None:
	parsed = parse_unit_node_id(str(row.neuron_id))
	if parsed is None or not parsed["layer_name"].startswith("hidden."):
		return None
	if _hidden_layer_index(parsed["layer_name"]) <= 0:
		return None

	t_novelty = 0
	t_start = int(row.point_of_max_acceleration)
	t_assign = int(row.assignment_within_trial)
	if not (0 <= t_novelty < t_assign < N_SAMPLES_PER_TRIAL):
		return None

	trial_start_cp = int(row.trial_start_cp)
	x = np.arange(N_SAMPLES_PER_TRIAL, dtype=np.int64)
	y = np.array(
		[
			_effective_lr_at_trial_iter(
				sigs,
				metrics,
				str(row.optimizer_id),
				str(row.neuron_id),
				trial_start_cp,
				int(t),
			)
			for t in x
		],
		dtype=np.float64,
	)
	if not np.any(np.isfinite(y)):
		return None

	return {
		"t_start": t_start,
		"t_assign": t_assign,
		"t_novelty": t_novelty,
		"x": x,
		"y": y,
	}


def build_efflr_df(
	df_efflr_raw: pd.DataFrame | None,
	periods: pd.DataFrame,
	*,
	load_sigs: Callable[..., Any],
	load_metrics: Callable[..., Any],
	desc: str = "effective LR",
) -> tuple[pd.DataFrame, dict[str, int]]:
	"""Load signals/metrics once per run; return one row per plottable period."""
	if df_efflr_raw is not None:
		return df_efflr_raw, {}

	rows_out: list[dict[str, Any]] = []
	skipped: dict[str, int] = {}
	if periods.empty:
		return pd.DataFrame(columns=PERIOD_KEYS), skipped

	for run_key, group in tqdm(periods.groupby(RUN_GROUP_COLS, sort=False), desc=desc, leave=False):
		eid, mid, oid, rid = run_key
		try:
			sigs = load_sigs(eid, mid, oid, rid=rid, w_cache=False)
			metrics = load_metrics(eid, mid, oid, rid=rid, w_cache=False)
			if not sigs or not metrics or "unit_node_ids" not in sigs:
				raise FileNotFoundError("signals.npz missing or empty")
		except Exception:
			skipped["sigs_load"] = skipped.get("sigs_load", 0) + len(group)
			continue

		for row in group.itertuples():
			rec = collect_efflr_from_row(sigs, metrics, row)
			if rec is None:
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
					**rec,
				}
			)

	return pd.DataFrame(rows_out), skipped


def merge_efflr_with_periods(
	df_efflr: pd.DataFrame,
	periods: pd.DataFrame,
) -> pd.DataFrame:
	if df_efflr.empty:
		return df_efflr.copy()
	extra_cols = [c for c in periods.columns if c not in df_efflr.columns]
	return df_efflr.merge(periods[PERIOD_KEYS + extra_cols], on=PERIOD_KEYS, how="inner")


def filter_exp_order_for_efflr_plot(
	exp_order: list[str],
	all_periods_by_experiment: dict[str, pd.DataFrame],
	experiment_predicate: Callable[[str], bool],
) -> list[str]:
	"""Display labels whose underlying `experiment_id` passes `experiment_predicate`."""
	out: list[str] = []
	for exp_label in exp_order:
		df = all_periods_by_experiment.get(exp_label)
		if df is None or df.empty:
			continue
		eid = str(df["experiment_id"].iloc[0])
		if experiment_predicate(eid):
			out.append(exp_label)
	return out


_WELCH_ALPHA = 0.05
_SEM_Z = 1.96
_EFFLR_BRACKET_STEP_FRAC = 0.086
_EFFLR_BRACKET_YLIM_PAD_FRAC = 0.175


def gamma_rel_novelty_samples_by_panel(
	df_efflr: pd.DataFrame,
	*,
	exp_order: list[str],
	opt_order: list[str],
	score_column: str,
	top_n: int,
) -> dict[tuple[str, str], list[float]]:
	"""Per-period `(γ(t)−γ(0))/γ(0)` around `t_start` (top-N periods per optimizer)."""
	out: dict[tuple[str, str], list[float]] = {}
	for exp_label in exp_order:
		for oid in opt_order:
			sel = select_efflr_periods(
				df_efflr,
				experiment=exp_label,
				optimizer_id=oid,
				top_n=top_n,
				score_column=score_column,
			)
			per_period: list[float] = []
			for row in sel.itertuples():
				val = _mean_gamma_rel_novelty_tstart_vicinity(
					np.asarray(row.y, dtype=np.float64),
					int(row.t_start),
					int(row.t_assign),
				)
				if np.isfinite(val):
					per_period.append(val)
			if per_period:
				out[(exp_label, oid)] = per_period
	return out


def gamma_rel_novelty_samples_all_strategies(
	df_efflr: pd.DataFrame,
	*,
	exp_order: list[str],
	opt_order: list[str],
	strategy_order: list[str],
	top_n: int,
) -> dict[str, dict[tuple[str, str], list[float]]]:
	"""`gamma_rel_novelty_samples_by_panel` for each period-selection strategy."""
	opts = _opt_order_without_sgd(opt_order)
	return {
		strat: gamma_rel_novelty_samples_by_panel(
			df_efflr,
			exp_order=exp_order,
			opt_order=opts,
			score_column=filter_score_column(strat, df_efflr),
			top_n=top_n,
		)
		for strat in strategy_order
	}


def mean_gamma_rel_novelty_by_panel(
	df_efflr: pd.DataFrame,
	*,
	exp_order: list[str],
	opt_order: list[str],
	score_column: str,
	top_n: int,
) -> dict[tuple[str, str], float]:
	"""Mean `(γ(t)−γ(0))/γ(0)` around `t_start`, then mean over top-N periods."""
	samples = gamma_rel_novelty_samples_by_panel(
		df_efflr,
		exp_order=exp_order,
		opt_order=opt_order,
		score_column=score_column,
		top_n=top_n,
	)
	return {k: float(np.mean(v)) for k, v in samples.items()}


def mean_gamma_rel_novelty_all_strategies(
	df_efflr: pd.DataFrame,
	*,
	exp_order: list[str],
	opt_order: list[str],
	strategy_order: list[str],
	top_n: int,
) -> dict[str, dict[tuple[str, str], float]]:
	"""`mean_gamma_rel_novelty_by_panel` for each period-selection strategy."""
	opts = _opt_order_without_sgd(opt_order)
	return {
		strat: mean_gamma_rel_novelty_by_panel(
			df_efflr,
			exp_order=exp_order,
			opt_order=opts,
			score_column=filter_score_column(strat, df_efflr),
			top_n=top_n,
		)
		for strat in strategy_order
	}


def _sem_stderr(x: np.ndarray) -> float:
	x = np.asarray(x, dtype=np.float64)
	x = x[np.isfinite(x)]
	if x.size < 2:
		return 0.0
	return float(np.std(x, ddof=1) / np.sqrt(x.size))


def _holm_stepdown_reject(pvals: list[float], alpha: float = _WELCH_ALPHA) -> list[bool]:
	"""Holm step-down on `pvals`; return reject flags in original list order."""
	m = len(pvals)
	rejected = [False] * m
	if m == 0:
		return rejected
	order = sorted(range(m), key=lambda i: pvals[i])
	for rank, idx in enumerate(order):
		if pvals[idx] <= alpha / (m - rank):
			rejected[idx] = True
		else:
			break
	return rejected


def _welch_t_pvalue_arrays(a: np.ndarray, b: np.ndarray) -> float:
	"""Welch two-sample p-value for period-level samples."""
	a = np.asarray(a, dtype=np.float64)
	b = np.asarray(b, dtype=np.float64)
	a = a[np.isfinite(a)]
	b = b[np.isfinite(b)]
	if a.size < 2 or b.size < 2:
		return float("nan")
	sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
	ma, mb = float(np.mean(a)), float(np.mean(b))
	if sa == 0.0 and sb == 0.0 and ma == mb:
		return 1.0
	res = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
	p = float(res.pvalue)
	return p if np.isfinite(p) else float("nan")


def _welch_pairwise_holm_sig_matrix(
	samples_by_group: dict[str, np.ndarray],
	order: list[str],
) -> np.ndarray:
	"""Symmetric Holm-significant pairwise Welch flags over `order` (k×k)."""
	k = len(order)
	sig_mat = np.zeros((k, k), dtype=bool)
	pairs: list[tuple[int, int, float]] = []
	for i, gi in enumerate(order):
		for j, gj in enumerate(order):
			if i >= j:
				continue
			ai = samples_by_group.get(gi)
			aj = samples_by_group.get(gj)
			if ai is None or aj is None:
				continue
			p = _welch_t_pvalue_arrays(ai, aj)
			if np.isfinite(p):
				pairs.append((i, j, p))
	if not pairs:
		return sig_mat
	rej_flags = _holm_stepdown_reject([t[2] for t in pairs])
	for (i, j, _), rej in zip(pairs, rej_flags, strict=True):
		if rej:
			sig_mat[i, j] = sig_mat[j, i] = True
	return sig_mat


def _draw_optimizer_sig_brackets(
	ax: plt.Axes,
	order: list[str],
	sig_mat: np.ndarray,
	*,
	y_base: float,
	y_step: float,
) -> None:
	"""Draw brackets for Holm-significant Welch optimizer pairs at stacked heights."""
	plotted: set[tuple[int, int]] = set()
	stack = 0
	pitch = max(y_step * 1.32, y_step)
	riser = y_step * 0.30
	for i in range(len(order)):
		for j in range(i + 1, len(order)):
			if not sig_mat[i, j]:
				continue
			key = (i, j)
			if key in plotted:
				continue
			plotted.add(key)
			y = y_base + stack * pitch
			stack += 1
			x1, x2 = float(i), float(j)
			ax.plot(
				[x1, x1, x2, x2],
				[y, y + riser, y + riser, y],
				lw=1.4,
				color="0.15",
				clip_on=False,
				zorder=5,
			)
			ax.text(
				(x1 + x2) / 2.0,
				y + riser,
				"*",
				ha="center",
				va="bottom",
				fontsize=12,
				color="0.1",
				clip_on=False,
				zorder=6,
			)


def _efflr_gamma_panel_y_extent(
	heights: list[float],
	sem_errs: list[float],
	sig_mat: np.ndarray,
) -> float:
	"""Half-span for symmetric y-limits (data + 1.96 SEM + Holm bracket headroom)."""
	hi = 0.0
	lo = 0.0
	for h, err in zip(heights, sem_errs):
		if not np.isfinite(h):
			continue
		err = err if np.isfinite(err) else 0.0
		hi = max(hi, float(h + err))
		lo = min(lo, float(h - err))
	span = max(hi - lo, 1e-12)
	n_sig = int(np.sum(np.triu(sig_mat, k=1)))
	if n_sig:
		bracket_top = hi + 0.02 * span + (_EFFLR_BRACKET_YLIM_PAD_FRAC * span) * n_sig
		hi = max(hi, bracket_top)
	return max(abs(hi), abs(lo), 1e-6)


def _delta_efflr_over_lr(y: np.ndarray) -> np.ndarray:
	"""`(γ_eff(t) - γ_eff(t-1)) / γ_eff(t-1)` over the trial."""
	return (y[1:] - y[:-1]) / y[:-1]


def _efflr_vicinity_delta(t_start: int, t_assign: int) -> int:
	"""Half-width around `t_start` (same as commented definition in dw_vicinity)."""
	return int(min(int(t_start), int(t_assign) - int(t_start)))


def _mean_gamma_rel_novelty_tstart_vicinity(
	y: np.ndarray, t_start: int, t_assign: int
) -> float:
	"""Mean `(γ(t)−γ(0))/γ(0)` for trial points in `[t_start−δ, t_start+δ]`."""
	t_start = int(t_start)
	t_assign = int(t_assign)
	delta = _efflr_vicinity_delta(t_start, t_assign)
	if delta < 0:
		return float("nan")
	t_rel_min = t_start - delta
	t_rel_max = t_start + delta
	if t_rel_min < 0 or t_rel_max >= N_SAMPLES_PER_TRIAL:
		return float("nan")
	y = np.asarray(y, dtype=np.float64)
	y0 = float(y[0])
	if not np.isfinite(y0) or abs(y0) <= 1e-15:
		return float("nan")
	slice_y = y[t_rel_min : t_rel_max + 1]
	rel = (slice_y - y0) / y0
	finite = rel[np.isfinite(rel)]
	if not finite.size:
		return float("nan")
	return float(np.mean(finite))


def _symmetric_y_half_span(y: np.ndarray, *, floor: float = 1e-12) -> float:
	finite = y[np.isfinite(y)]
	if not finite.size:
		return floor
	return max(float(np.max(np.abs(finite))), floor)


_EFFLR_DELTA_SCI_THRESHOLD = 1e-3


def _apply_efflr_delta_yaxis_format(yaxis, dlim: float, tick_size: float) -> None:
	"""Use scientific ticks + top offset when |Δγ/γ| is tiny (avoids wide decimals)."""
	if dlim > _EFFLR_DELTA_SCI_THRESHOLD:
		return
	fmt = mpl_ticker.ScalarFormatter(useMathText=True)
	fmt.set_scientific(True)
	fmt.set_powerlimits((0, 0))
	fmt.set_useOffset(True)
	yaxis.set_major_formatter(fmt)
	yaxis.get_offset_text().set_fontsize(tick_size)


def _plot_efflr_segment(
	ax: plt.Axes,
	x: np.ndarray,
	y: np.ndarray,
	*,
	t_assign: int,
	color: str,
	lw: float,
	ls: str,
	alpha_full: float,
	alpha_post: float,
) -> None:
	pre_end = min(int(t_assign) + 1, len(x))
	if pre_end > 0:
		ax.plot(x[:pre_end], y[:pre_end], color=color, lw=lw, ls=ls, alpha=alpha_full)
	if int(t_assign) < len(x):
		ax.plot(
			x[int(t_assign) :],
			y[int(t_assign) :],
			color=color,
			lw=lw,
			ls=ls,
			alpha=alpha_post,
		)


def _plot_efflr_trial_trajectory(
	ax: plt.Axes,
	x: np.ndarray,
	y: np.ndarray,
	*,
	color: str,
	t_start: int,
	t_assign: int,
	lw: float = 1.2,
	tick_size: float,
	show_delta_ylabel: bool = False,
) -> None:
	"""Full trial trace: solid before assignment, faded from assignment through trial end."""
	x = np.asarray(x, dtype=np.int64)
	
	t_assign = int(t_assign)
	t_start = int(t_start)

	_plot_efflr_segment(
		ax,
		x,
		y,
		t_assign=t_assign,
		color=color,
		lw=lw,
		ls="-",
		alpha_full=1.0,
		alpha_post=0.3,
	)

	ax2 = ax.twinx()
	y_delta = _delta_efflr_over_lr(y)
	_plot_efflr_segment(
		ax2,
		x[1:],
		y_delta,
		t_assign=t_assign,
		color="#5c5552",
		lw=lw/2,
		ls="--",
		alpha_full=1.0,
		alpha_post=0.3,
	)

	ax.axvline(t_start, color="0.35", ls="--", lw=1.0)
	ax.axvline(t_assign, color="blue", lw=1.0)
	ax.set_xlim(0, N_SAMPLES_PER_TRIAL - 1)
	dlim = _symmetric_y_half_span(y_delta)
	ax2.set_ylim(-dlim, dlim)
	ax2.axhline(0, color="0.4", lw=0.6, zorder=0)
	_apply_efflr_delta_yaxis_format(ax2.yaxis, dlim, tick_size)

	ax.tick_params(axis="both", labelsize=tick_size)
	ax2.tick_params(axis="y", labelsize=tick_size)
	ax2.tick_params(axis="x", labelbottom=False)
	# Offset multiplier (e.g. "1e6") does not follow tick labelsize.
	for axis in (ax.yaxis, ax2.yaxis):
		axis.get_offset_text().set_fontsize(tick_size)
	if show_delta_ylabel:
		ax2.set_ylabel(r"$\frac{\Delta\gamma_{\mathrm{eff}}}{\gamma_{\mathrm{eff}}}$", fontsize=tick_size*2, rotation=0, labelpad=10, va="center",color="#5c5552")

def plot_efflr_trajectories_for_optimizer(
	oid: str,
	panel_by_exp: dict[str, pd.DataFrame],
	*,
	exp_order: list[str],
	top_n: int,
	filter_strategy: str,
	score_column: str,
	tick_size: float,
) -> plt.Figure | None:
	if not any(not panel_by_exp.get(exp_label, pd.DataFrame()).empty for exp_label in exp_order):
		return None

	n_exp = len(exp_order)
	n_cols = top_n
	fig, axes = plt.subplots(
		n_exp,
		n_cols,
		figsize=(2.2 * n_cols, 2.0 * n_exp),
		squeeze=False,
		sharex=True,
		constrained_layout=True,
	)
	color = _opt_color(oid)

	for i, exp_label in enumerate(exp_order):
		sel = panel_by_exp.get(exp_label, pd.DataFrame())
		for j in range(n_cols):
			ax = axes[i, j]
			if j >= len(sel):
				ax.axis("off")
				continue
			row = sel.iloc[j]
			_plot_efflr_trial_trajectory(
				ax,
				np.asarray(row.x),
				np.asarray(row.y),
				color=color,
				t_start=int(row.t_start),
				t_assign=int(row.t_assign),
				tick_size=tick_size,
				show_delta_ylabel=(j == len(sel) - 1),
			)
			if i == n_exp - 1:
				ax.set_xlabel("trial iter (from novelty)", fontsize=7)
			if j == 0:
				ax.set_ylabel(r"$\gamma_{\mathrm{eff}}$", fontsize=tick_size*2, rotation=0, labelpad=10, va="center",color=color)
				ax.text(
					-0.8,
					0.5,
					exp_label,
					transform=ax.transAxes,
					rotation=90,
					va="center",
					ha="center",
					fontsize=10,
					clip_on=False,
				)
			if i == 0:
				ax.set_title(f"#{j + 1}", fontsize=10)
			score = float(row[score_column]) if score_column in sel.columns else np.nan
			ax.text(
				0.45,
				0.95,
				f"score={score:.2f}",
				transform=ax.transAxes,
				va="top",
				ha="left",
				fontsize=6,
				color="0.2",
				zorder=10,
			)

	fig.suptitle(
		f"Effective LR — {format_opt_display(oid)} (top {top_n} per experiment by {filter_strategy})",
		fontsize=11,
		y=1.02,
	)
	return fig


def plot_efflr_gamma_rel_novelty_grid(
	samples_by_strategy: dict[str, dict[tuple[str, str], list[float]]],
	*,
	exp_order: list[str],
	opt_order: list[str],
	strategy_order: list[str],
	top_n: int,
	tick_size: float,
	category_label: str,
) -> plt.Figure | None:
	"""Rows = experiments, cols = selection strategies, bars = optimizers."""
	opt_order = _opt_order_without_sgd(opt_order)
	if not exp_order or not opt_order or not strategy_order:
		return None

	n_exp = len(exp_order)
	n_strat = len(strategy_order)
	n_opt = len(opt_order)
	fig, axes = plt.subplots(
		n_exp,
		n_strat,
		figsize=(1.5 * n_opt * n_strat, 1.5 * n_exp),
		squeeze=False,
		sharey=True,
	)

	x = np.arange(n_opt)
	opt_labels = [format_opt_display(oid) for oid in opt_order]
	bar_colors = [_opt_color(oid) for oid in opt_order]

	panel_stats: list[dict] = []
	for exp_label in exp_order:
		for strat_label in strategy_order:
			panel = samples_by_strategy.get(strat_label, {})
			heights: list[float] = []
			sem_errs: list[float] = []
			samples_by_oid: dict[str, np.ndarray] = {}
			for oid in opt_order:
				samples = panel.get((exp_label, oid), [])
				if samples:
					arr = np.asarray(samples, dtype=np.float64)
					heights.append(float(np.mean(arr)))
					sem_errs.append(_SEM_Z * _sem_stderr(arr))
					samples_by_oid[oid] = arr
				else:
					heights.append(float("nan"))
					sem_errs.append(0.0)
			sig_mat = _welch_pairwise_holm_sig_matrix(samples_by_oid, opt_order)
			panel_stats.append(
				{
					"exp_label": exp_label,
					"strat_label": strat_label,
					"heights": heights,
					"sem_errs": sem_errs,
					"sig_mat": sig_mat,
					"y_half": _efflr_gamma_panel_y_extent(heights, sem_errs, sig_mat),
				}
			)

	vlim = max((st["y_half"] for st in panel_stats), default=1.0)

	for i, exp_label in enumerate(exp_order):
		for j, strat_label in enumerate(strategy_order):
			ax = axes[i, j]
			st = panel_stats[i * n_strat + j]
			heights = st["heights"]
			sem_errs = st["sem_errs"]
			plot_heights = [h if np.isfinite(h) else 0.0 for h in heights]
			bars = ax.bar(
				x,
				plot_heights,
				color=bar_colors,
				width=0.5,
				edgecolor="none",
				zorder=2,
			)
			for idx, (h, err) in enumerate(zip(heights, sem_errs)):
				if not np.isfinite(h) or err <= 0:
					continue
				ax.errorbar(
					x[idx],
					h,
					yerr=err,
					fmt="none",
					ecolor="0.15",
					capsize=2,
					linewidth=0.9,
					zorder=3,
				)
			ax.axhline(0, color="red", alpha=0.15, lw=0.8, zorder=1)
			ax.set_ylim(-vlim, vlim)
			ax.tick_params(axis="y", labelsize=tick_size)

			hi = max(
				(
					float(h + err)
					for h, err in zip(heights, sem_errs)
					if np.isfinite(h)
				),
				default=0.0,
			)
			lo = min(
				(
					float(h - err)
					for h, err in zip(heights, sem_errs)
					if np.isfinite(h)
				),
				default=0.0,
			)
			span = max(hi - lo, 2 * vlim, 1e-12)
			y_base = hi + 0.1 * span
			y_step = _EFFLR_BRACKET_STEP_FRAC * span
			_draw_optimizer_sig_brackets(
				ax,
				opt_order,
				st["sig_mat"],
				y_base=y_base,
				y_step=y_step,
			)

			for bar, h, err in zip(bars, heights, sem_errs):
				if not np.isfinite(h):
					continue
				label_y = h + err if h >= 0 else h - err
				ax.text(
					bar.get_x() + bar.get_width() / 2,
					label_y,
					f"{h:+.2f}",
					ha="center",
					va="bottom" if h >= 0 else "top",
					fontsize=tick_size,
				)
			if i == 0:
				ax.set_title(strat_label)
			if j == 0:
				ax.set_ylabel(exp_label.replace(" ", "\n"))
			if i < n_exp - 1:
				ax.tick_params(axis="x", labelbottom=False)
			else:
				ax.set_xticks(x)
				ax.set_xticklabels(opt_labels, fontsize=tick_size * 1.5)

	legend_handles = [
		Patch(facecolor=_opt_color(oid), label=format_opt_display(oid)) for oid in opt_order
	]
	fig.legend(
		handles=legend_handles,
		loc="upper center",
		bbox_to_anchor=(0.5, 0.99),
		ncol=min(n_opt, 6),
		framealpha=0.92,
		fontsize=12,
	)

	fig.suptitle(
		f"{category_label} mean "
		r"$(\gamma_{\mathrm{eff}}(t)-\gamma_{\mathrm{eff}}(0))/\gamma_{\mathrm{eff}}(0)$ "
		r"around $t_{\mathrm{start}}$ "
		r"($\delta=\min(t_{\mathrm{start}},\,t_{\mathrm{assign}}-t_{\mathrm{start}})$; "
		f"mean over $t$, then over top {top_n} periods per optimizer; "
		f"bars: mean $\\pm$ {_SEM_Z:.2f} SEM; Holm-corrected Welch pairs: *)",
		y=1.04,
	)
	fig.supylabel(
		r"$\frac{\gamma_{\mathrm{eff}} - \gamma_{\mathrm{eff}}^{(0)}}{\gamma_{\mathrm{eff}}^{(0)}}$",
		x=0.01,
		va="center",
		rotation=0,
		y=0.5,
	)
	fig.tight_layout()
	return fig


def plot_efflr_gamma_rel_novelty_for_category(
	df_efflr: pd.DataFrame,
	exp_order: list[str],
	opt_order: list[str],
	all_periods_by_experiment: dict[str, pd.DataFrame],
	*,
	experiment_predicate: Callable[[str], bool],
	category_label: str,
	category_slug: str,
	strategy_order: list[str],
	top_n: int,
	tick_size: float,
) -> plt.Figure | None:
	"""Build and return the all-strategies γ/γ⁽⁰⁾ bar grid for one experiment category."""
	cat_exp_order = filter_exp_order_for_efflr_plot(
		exp_order, all_periods_by_experiment, experiment_predicate
	)
	if not cat_exp_order:
		print(f"No {category_label} experiments — skip effective-LR γ/γ⁽⁰⁾ bar plot")
		return None

	samples_by_strategy = gamma_rel_novelty_samples_all_strategies(
		df_efflr,
		exp_order=cat_exp_order,
		opt_order=opt_order,
		strategy_order=strategy_order,
		top_n=top_n,
	)
	if not any(samples_by_strategy.values()):
		print(f"No {category_label} eff-LR γ/γ⁽⁰⁾ data for any strategy")
		return None

	return plot_efflr_gamma_rel_novelty_grid(
		samples_by_strategy,
		exp_order=cat_exp_order,
		opt_order=opt_order,
		strategy_order=strategy_order,
		top_n=top_n,
		tick_size=tick_size,
		category_label=category_label,
	)
