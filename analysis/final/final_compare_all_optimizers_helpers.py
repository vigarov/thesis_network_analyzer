"""Helpers for computations in FINAL_compare_all_optimizers.ipynb """
import gzip
import hashlib
import json
import pickle
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch
from matplotlib.colors import to_rgba
from tqdm.auto import tqdm

from experiments.mnist import MNISTWrapper
from models import get_model
from models.unit_node_id import parse_unit_node_id

from analysis.common import (
	_opt_color,
	_opt_display,
	_opt_label,
	_sort_optimizer_ids,
)
from analysis.constants import RESULTS, _NETWORK_SAMPLES_PER_DIGIT
from analysis.io import _metrics, _nts, _pp_dead, _pp_neuron_digit
from analysis.final.final_experiment_display import (
	ExperimentDisplayLabels,
	two_row_cat12_from_sorted,
)
from analysis.plot_helpers import add_trial_boundaries_mpl
from analysis.scoring_helpers import (
	angle_score_function,
	filter_periods,
	find_impacted_periods,
	final_compute_period_repr_trajectory_summary,
	final_stream_repr_distance_through_time,
	get_all_model_saliency_maps,
	get_all_points_of_max_acceleration,
	get_assignments,
	get_iteration,
	get_neuron_saliency_maps_from_all,
	get_n_assigned_digits,
	get_period_angles,
	get_period_checkpoint_indices,
	get_recovery_periods,
	get_trial_starts,
	init_cw_ssim_pyramid,
	is_pretrain_shuffle_mislabel_experiment,
	length_score_function,
	n_active_decay,
	n_partial_decay,
	remove_inelligible_periods,
	index_grp,
)

# --- checkpoint I/O + rescore ---
def _run_checkpoint_dir(save_dir: str, eid: str, mid: str, oid: str, rid: str) -> Path:
	sha = hashlib.sha256(f"{eid}|{mid}|{oid}|{rid}".encode("utf-8")).hexdigest()[:10]
	return Path(save_dir) / sha


def _save_run_checkpoint(save_dir: str, eid: str, mid: str, oid: str, rid: str, out: dict[str, Any]) -> None:
	d = _run_checkpoint_dir(save_dir, eid, mid, oid, rid)
	d.mkdir(parents=True, exist_ok=True)
	pd.DataFrame([{"eid": eid, "mid": mid, "oid": oid, "rid": rid}]).to_csv(d / "key.csv", index=False)
	with gzip.open(d / "results.pkl.gz", "wb", compresslevel=9) as f:
		pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_run_checkpoint(save_dir: str, eid: str, mid: str, oid: str, rid: str) -> dict[str, Any] | None:
	d = _run_checkpoint_dir(save_dir, eid, mid, oid, rid)
	key_csv = d / "key.csv"
	res_gz = d / "results.pkl.gz"
	res_pkl = d / "results.pkl"
	if not key_csv.is_file() or (not res_gz.is_file() and not res_pkl.is_file()):
		return None
	key_df = pd.read_csv(key_csv)
	if len(key_df) != 1:
		return None
	if (
		str(key_df["eid"].iloc[0]) != str(eid)
		or str(key_df["mid"].iloc[0]) != str(mid)
		or str(key_df["oid"].iloc[0]) != str(oid)
		or str(key_df["rid"].iloc[0]) != str(rid)
	):
		return None
	if res_gz.is_file():
		with gzip.open(res_gz, "rb") as f:
			return pickle.load(f)
	with open(res_pkl, "rb") as f:
		return pickle.load(f)



def _is_cat2_recover_reinforce_experiment(eid: str) -> bool:
	s = str(eid)
	return "cat2_sequence_recover" in s or "cat2_sequence_reinforce" in s


def _parse_digit_a_from_config(eid: str, mid: str, rid: str) -> int | None:
	path = RESULTS / eid / rid / mid / "config.json"
	if not path.is_file():
		return None
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return None
	exp_cfg = data.get("experiment_config") or {}
	if "digitA" not in exp_cfg:
		return None
	return int(exp_cfg["digitA"])


def reparse_digit_a_final_run(
	out: dict[str, Any], eid: str, mid: str, rid: str
) -> tuple[dict[str, Any], bool]:
	"""Add `digitA` to `periods_df` for recover/reinforce runs; return `(out, changed)`."""
	if out.get("error") or not _is_cat2_recover_reinforce_experiment(eid):
		return out, False
	digit_a = _parse_digit_a_from_config(eid, mid, rid)
	if digit_a is None:
		return out, False
	df = out.get("periods_df")
	if df is None:
		return out, False
	out = {**out}
	if df.empty:
		out["periods_df"] = df.assign(digitA=digit_a)
		return out, True
	df = df.copy()
	changed = "digitA" not in df.columns or not (df["digitA"] == digit_a).all()
	df["digitA"] = digit_a
	out["periods_df"] = df
	return out, changed



def _robustness_score_column(
	df: pd.DataFrame,
	*,
	use_minmax_normalized_robustness: bool,
) -> str:
	"""Column used for the robustness term in weighted `total_score`."""
	if use_minmax_normalized_robustness and "minmax_normalized_robustness_scores" in df.columns:
		return "minmax_normalized_robustness_scores"
	return "robustness_score"


def apply_score_factors_to_periods_df(
	df: pd.DataFrame,
	score_factors: dict[str, float],
	*,
	use_minmax_normalized_robustness: bool = False,
) -> pd.DataFrame:
	"""Weighted sum of cached score components (no repr / PMA recomputation)."""
	df = df.copy()
	rob_col = _robustness_score_column(
		df,
		use_minmax_normalized_robustness=use_minmax_normalized_robustness,
	)
	for col in score_factors:
		if col == "robustness_score":
			if rob_col not in df.columns:
				df[rob_col] = np.nan
		elif col not in df.columns:
			df[col] = np.nan
	df["total_score"] = sum(
		df[rob_col if c == "robustness_score" else c].astype(float) * w
		for c, w in score_factors.items()
	)
	return df


def rescore_final_run(
	out: dict[str, Any],
	score_factors: dict[str, float],
	*,
	use_minmax_normalized_robustness: bool = False,
) -> dict[str, Any]:
	"""Recompute `total_score` and `mean_total_score` from logged component columns."""
	if out.get("error"):
		return out
	df = out.get("periods_df")
	if df is None:
		return out
	out = {**out}
	if df.empty:
		out["mean_total_score"] = float(np.nan)
		return out
	df = apply_score_factors_to_periods_df(
		df,
		score_factors,
		use_minmax_normalized_robustness=use_minmax_normalized_robustness,
	)
	out["periods_df"] = df
	out["mean_total_score"] = float(pd.to_numeric(df["total_score"], errors="coerce").mean())
	return out


def minmax_normalize_robustness_all_results(
	all_results: dict[tuple[str, str, str, str], dict],
	score_factors: dict[str, float],
	*,
	save_dir: str,
) -> int:
	"""Min-max normalize `robustness_score` across all runs, rescore, and save."""
	pending_keys: list[tuple[str, str, str, str]] = []
	raw_values: list[float] = []
	for key, out in all_results.items():
		if out.get("error"):
			continue
		df = out.get("periods_df")
		if df is None or df.empty or "robustness_score" not in df.columns:
			continue
		if "minmax_normalized_robustness_scores" in df.columns:
			continue
		pending_keys.append(key)
		raw_values.extend(
			pd.to_numeric(df["robustness_score"], errors="coerce").dropna().astype(float).tolist()
		)
	if not pending_keys or not raw_values:
		return 0

	r_min = float(np.min(raw_values))
	r_max = float(np.max(raw_values))
	denom = r_max - r_min

	def _normalize_value(value: float) -> float:
		if denom == 0.0:
			return 0.0
		return (value - r_min) / denom

	n_saved = 0
	for key in pending_keys:
		eid, mid, oid, rid = key
		out = {**all_results[key]}
		df = out["periods_df"].copy()
		r = pd.to_numeric(df["robustness_score"], errors="coerce")
		df["minmax_normalized_robustness_scores"] = r.map(
			lambda x: _normalize_value(float(x)) if pd.notna(x) else np.nan
		)
		df = apply_score_factors_to_periods_df(
			df,
			score_factors,
			use_minmax_normalized_robustness=True,
		)
		out["periods_df"] = df
		out["mean_total_score"] = float(pd.to_numeric(df["total_score"], errors="coerce").mean())
		all_results[key] = out
		_save_run_checkpoint(save_dir, eid, mid, oid, rid, out)
		n_saved += 1
	return n_saved



class DummyExp(MNISTWrapper):
	def experiment_id(self):
		return ""

	def _build_trials(self, batch_size: int, seed: int, *, num_experiment_runs: int):
		return []

# --- core compute ---
def collect_final_run_data(
	eid: str,
	mid: str,
	oid: str,
	rid: str,
	prebuilt_pyramids: dict,
	*,
	project_root: Path,
	expert_saliency_by_oid: dict[str, dict] | None = None,
	n_checkpoint_samples: int,
	strict: bool,
	btsp_start_strategy: str,
	use_real_progression: bool,
	expert_model_dir: str,
	score_factors: dict[str, float],
) -> dict[str, Any]:
	"""Load post-processed data, sample `n_checkpoint_samples` checkpoints uniformly, stream `repr_distance`, build per-period table."""
	root = project_root
	is_shuffle_mislabel = is_pretrain_shuffle_mislabel_experiment(eid)
	peak_agg = "max" if is_shuffle_mislabel else "min"
	metrics = _metrics(eid, mid, oid, rid=rid, w_cache=False)
	df_nd = cast(pd.DataFrame, _pp_neuron_digit(eid, mid, oid, rid=rid, w_cache=False))
	if df_nd is None or df_nd.empty:
		return {"error": "empty neuron_digit", "eid": eid, "mid": mid, "oid": oid, "rid": rid}

	df_nd = df_nd.copy()
	df_nd["n_active"] = df_nd["status"].apply(
		lambda s: 0
		if s == "inactive"
		else (_NETWORK_SAMPLES_PER_DIGIT if s == "assigned" else int(s.split("_")[-1]))
	)
	df_nd["iteration"] = df_nd["checkpoint_idx"].map(lambda c: get_iteration(metrics, c))

	df_dead = cast(pd.DataFrame, _pp_dead(eid, mid, oid, rid=rid, w_cache=False))
	if df_dead is not None and not df_dead.empty:
		df_dead = df_dead.copy()
		df_dead["iteration"] = df_dead["checkpoint_idx"].map(lambda c: get_iteration(metrics, c))

	nts = _nts(eid, mid, oid, rid=rid, w_cache=False)

	n_assigned_digits = get_n_assigned_digits(df_nd)
	recovery_periods_any = get_recovery_periods(metrics, df_nd, n_assigned_digits, df_dead, mode="assigned_any")
	recovery_periods_same = get_recovery_periods(metrics, df_nd, n_assigned_digits, df_dead, mode="assigned_same")
	impacted = find_impacted_periods(recovery_periods_any, recovery_periods_same)
	n_any = len(recovery_periods_any)
	impacted_pct = float(len(impacted) / n_any) if n_any else 0.0

	recovery_periods = recovery_periods_same
	recovery_periods = remove_inelligible_periods(recovery_periods, require_until_trial_end=not is_shuffle_mislabel)
	if df_dead is not None and not df_dead.empty:
		recovery_periods = filter_periods(recovery_periods, df_dead, strict=strict)

	if recovery_periods.empty:
		empty = pd.DataFrame(
			columns=[
				*index_grp,
				"layer_name",
				"n_checkpoints",
				"d_bar",
				"repr_trajectory_dispersion",
				"robustness_score",
				"d_tk",
				"sampled_checkpoint_idx",
				"sampled_iteration",
				"period_start_cp",
				"period_end_cp",
				"period_start_iter",
				"period_end_iter",
				"total_length_iter",
				"trial_start_iter",
				"trial_start_cp",
				"assignment_within_trial",
				"point_of_max_acceleration",
				"activation_angle",
				"angle_score",
				"length_score",
				"n_partial_decay_score",
				"n_active_decay_score",
				"total_score",
			]
		)
		return {
			"eid": eid,
			"mid": mid,
			"oid": oid,
			"rid": rid,
			"impacted_pct": impacted_pct,
			"periods_df": empty,
			"n_periods": 0,
			"mean_total_score": float(np.nan),
		}

	period_angles = get_period_angles(
		nts,
		recovery_periods,
		from_point=btsp_start_strategy,
		use_real_progression=use_real_progression,
		peak_time_aggregate=peak_agg,
	)
	angles_scores = period_angles.assign(
		angle_score=lambda df: df["activation_angle"].map(angle_score_function)
	)

	length_scores = (
		recovery_periods[index_grp + ["total_length_iter"]]
		.assign(length_score=lambda df: df["total_length_iter"].map(length_score_function))
		.drop(columns={"total_length_iter"})
		.set_index(index_grp)
	)

	assignments = get_assignments(df_nd, recovery_periods)
	assignment_scores = assignments.groupby(index_grp, sort=False).apply(
		lambda df: pd.Series(
			{
				"n_partial_decay_score": n_partial_decay(int(df["n_partial_at_start"].iloc[0])),
				"n_active_decay_score": n_active_decay(int(df["n_assigned_at_start"].iloc[0])),
			}
		)
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	eval_digits = DummyExp().evaluation_inputs(device)[0]

	_, prebuilt_pyramids = init_cw_ssim_pyramid(device, prebuilt_pyramids)

	required_nids = list(recovery_periods["neuron_id"].unique())
	expert_dir = root / expert_model_dir.replace("!OPT", oid)
	expert_saliency: dict
	if expert_saliency_by_oid is not None and oid in expert_saliency_by_oid:
		expert_saliency = expert_saliency_by_oid[oid]
		missing = [nid for nid in required_nids if nid not in expert_saliency]
		if missing:
			report = json.loads((expert_dir / "report.json").read_text())
			expert_model = get_model(report["model_class"], **report["model_config"]).to(device)
			expert_model.load_state_dict(torch.load(expert_dir / "model.pt", map_location="cpu", weights_only=True))
			expert_model.eval()
			all_expert_saliency_maps = get_all_model_saliency_maps(expert_model, eval_digits, device)
			for nid in tqdm(missing, desc=f"Expert maps (extra) {oid[:16]}…", leave=False):
				expert_saliency[nid] = get_neuron_saliency_maps_from_all(all_expert_saliency_maps, nid)
			expert_model.cpu()
			del expert_model, all_expert_saliency_maps
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
	else:
		report = json.loads((expert_dir / "report.json").read_text())
		expert_model = get_model(report["model_class"], **report["model_config"]).to(device)
		expert_model.load_state_dict(torch.load(expert_dir / "model.pt", map_location="cpu", weights_only=True))
		expert_model.eval()
		all_expert_saliency_maps = get_all_model_saliency_maps(expert_model, eval_digits, device)
		expert_saliency = {
			nid: get_neuron_saliency_maps_from_all(all_expert_saliency_maps, nid)
			for nid in tqdm(required_nids, desc=f"Expert maps {oid[:16]}…", leave=False)
		}
		expert_model.cpu()
		del expert_model, all_expert_saliency_maps
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		if expert_saliency_by_oid is not None:
			expert_saliency_by_oid[oid] = expert_saliency

	checkpoint_df = get_period_checkpoint_indices(
		recovery_periods,
		n_points=n_checkpoint_samples,
		nts=nts,
		metrics=metrics,
		btsp_start_strategy=btsp_start_strategy,
		peak_time_aggregate=peak_agg,
	)

	df_repr_tt = final_stream_repr_distance_through_time(
		nts,
		recovery_periods,
		expert_saliency,
		checkpoint_df,
		device=device,
		prebuilt_pyramids=prebuilt_pyramids,
	)

	summary = final_compute_period_repr_trajectory_summary(df_repr_tt, recovery_periods)
	if summary.empty:
		empty_cols = [
			*index_grp,
			"layer_name",
			"n_checkpoints",
			"d_bar",
			"repr_trajectory_dispersion",
			"robustness_score",
			"d_tk",
			"sampled_checkpoint_idx",
			"sampled_iteration",
			"period_start_cp",
			"period_end_cp",
			"period_start_iter",
			"period_end_iter",
			"total_length_iter",
			"trial_start_iter",
			"trial_start_cp",
			"assignment_within_trial",
			"point_of_max_acceleration",
			"activation_angle",
			"angle_score",
			"length_score",
			"n_partial_decay_score",
			"n_active_decay_score",
			"total_score",
		]
		return {
			"eid": eid,
			"mid": mid,
			"oid": oid,
			"rid": rid,
			"impacted_pct": impacted_pct,
			"periods_df": pd.DataFrame(columns=empty_cols),
			"n_periods": int(len(recovery_periods)),
			"mean_total_score": float(np.nan),
		}

	meta_cols = index_grp + ["start", "end", "start_iter", "end_iter", "total_length_iter"]
	summary = summary.merge(
		recovery_periods[meta_cols].drop_duplicates(subset=index_grp),
		on=index_grp,
		how="left",
	).rename(
		columns={
			"start": "period_start_cp",
			"end": "period_end_cp",
			"start_iter": "period_start_iter",
			"end_iter": "period_end_iter",
		}
	)

	pma = get_all_points_of_max_acceleration(nts, recovery_periods, peak_time_aggregate=peak_agg).reset_index()
	summary = summary.merge(
		pma[["neuron_id", "period_id", "point_of_max_acceleration"]],
		on=index_grp,
		how="left",
	)

	trial_rows: list[dict[str, Any]] = []
	for _, row in summary.iterrows():
		ts, delta, scp = get_trial_starts(recovery_periods, row["neuron_id"], row["period_id"])
		trial_rows.append(
			{"trial_start_iter": int(ts), "trial_start_cp": int(scp), "assignment_within_trial": int(delta)}
		)
	summary = pd.concat([summary.reset_index(drop=True), pd.DataFrame(trial_rows)], axis=1)

	summary = summary.merge(angles_scores.reset_index(), on=index_grp, how="left")
	summary = summary.merge(length_scores.reset_index(), on=index_grp, how="left")
	summary = summary.merge(assignment_scores.reset_index(), on=index_grp, how="left")

	for col in score_factors:
		if col not in summary.columns:
			summary[col] = np.nan
	summary["total_score"] = sum(summary[c].astype(float) * w for c, w in score_factors.items())

	return {
		"eid": eid,
		"mid": mid,
		"oid": oid,
		"rid": rid,
		"impacted_pct": impacted_pct,
		"periods_df": summary,
		"n_periods": int(len(recovery_periods)),
		"mean_total_score": float(pd.to_numeric(summary["total_score"], errors="coerce").mean()),
	}


def final_run_one(
	eid: str,
	mid: str,
	oid: str,
	rid: str,
	prebuilt_pyramids: dict,
	expert_saliency_by_oid: dict[str, dict] | None,
	*,
	project_root: Path,
	save_dir: str,
	n_checkpoint_samples: int,
	strict: bool,
	btsp_start_strategy: str,
	use_real_progression: bool,
	expert_model_dir: str,
	score_factors: dict[str, float],
) -> dict[str, Any]:
	collected = collect_final_run_data(
		eid,
		mid,
		oid,
		rid,
		prebuilt_pyramids,
		project_root=project_root,
		expert_saliency_by_oid=expert_saliency_by_oid,
		n_checkpoint_samples=n_checkpoint_samples,
		strict=strict,
		btsp_start_strategy=btsp_start_strategy,
		use_real_progression=use_real_progression,
		expert_model_dir=expert_model_dir,
		score_factors=score_factors,
	)
	return collected

# --- significance ---
# --- statistics: Holm-adjusted Welch pairs + bracket drawing ---


def _holm_stepdown_reject(pvals: list[float], alpha: float = 0.05) -> list[bool]:
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


def _welch_t_pvalue(vi: pd.Series, vj: pd.Series) -> float:
	"""Welch `p`-value. If both groups are constant with the same mean, return `1.0`; else `ttest_ind` (warnings not suppressed)."""
	a = pd.to_numeric(vi, errors="coerce").astype(np.float64).dropna().to_numpy()
	b = pd.to_numeric(vj, errors="coerce").astype(np.float64).dropna().to_numpy()
	if len(a) < 2 or len(b) < 2:
		return float("nan")
	sa, sb = float(np.nanstd(a, ddof=1)), float(np.nanstd(b, ddof=1))
	ma, mb = float(np.nanmean(a)), float(np.nanmean(b))
	if sa == 0.0 and sb == 0.0 and ma == mb:
		return 1.0
	res = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
	p = float(res.pvalue)
	return p if np.isfinite(p) else float("nan")




def _ks_2samp_pvalue(vi: pd.Series, vj: pd.Series) -> float:
	"""Two-sample KS `p`-value for distribution difference."""
	a = pd.to_numeric(vi, errors="coerce").astype(np.float64).dropna().to_numpy()
	b = pd.to_numeric(vj, errors="coerce").astype(np.float64).dropna().to_numpy()
	if len(a) < 2 or len(b) < 2:
		return float("nan")
	if len(np.unique(a)) == 1 and len(np.unique(b)) == 1 and a[0] == b[0]:
		return 1.0
	res = stats.ks_2samp(a, b, method="auto")
	p = float(res.pvalue)
	return p if np.isfinite(p) else float("nan")

def welch_pairwise_matrix(
	df: pd.DataFrame,
	value_col: str,
	group_col: str,
	order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
	"""Return symmetric p-value and significance (Holm) matrices over `order`."""
	k = len(order)
	p_mat = np.full((k, k), np.nan)
	sig_mat = np.zeros((k, k), dtype=bool)
	pairs: list[tuple[int, int, float]] = []
	for i, gi in enumerate(order):
		for j, gj in enumerate(order):
			if i >= j:
				continue
			vi = df.loc[df[group_col] == gi, value_col]
			vj = df.loc[df[group_col] == gj, value_col]
			p = _welch_t_pvalue(vi, vj)
			if not np.isfinite(p):
				continue
			p_mat[i, j] = p
			p_mat[j, i] = p
			pairs.append((i, j, p))
	if not pairs:
		return p_mat, sig_mat
	p_list = [t[2] for t in pairs]
	rej_flags = _holm_stepdown_reject(p_list, alpha=0.05)
	for (i, j, _), r in zip(pairs, rej_flags, strict=True):
		if r:
			sig_mat[i, j] = sig_mat[j, i] = True
	return p_mat, sig_mat

def ks_pairwise_matrix(
	df: pd.DataFrame,
	value_col: str,
	group_col: str,
	order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
	"""Return symmetric p-value and significance (Holm) matrices over `order` (KS two-sample)."""
	k = len(order)
	p_mat = np.full((k, k), np.nan)
	sig_mat = np.zeros((k, k), dtype=bool)
	pairs: list[tuple[int, int, float]] = []
	for i, gi in enumerate(order):
		for j, gj in enumerate(order):
			if i >= j:
				continue
			vi = df.loc[df[group_col] == gi, value_col]
			vj = df.loc[df[group_col] == gj, value_col]
			p = _ks_2samp_pvalue(vi, vj)
			if not np.isfinite(p):
				continue
			p_mat[i, j] = p
			p_mat[j, i] = p
			pairs.append((i, j, p))
	if not pairs:
		return p_mat, sig_mat
	p_list = [t[2] for t in pairs]
	rej_flags = _holm_stepdown_reject(p_list, alpha=0.05)
	for (i, j, _), r in zip(pairs, rej_flags, strict=True):
		if r:
			sig_mat[i, j] = sig_mat[j, i] = True
	return p_mat, sig_mat


def draw_sig_brackets(
	ax,
	order: list[str],
	sig_mat: np.ndarray,
	y_base: float,
	y_step: float,
) -> None:
	"""Draw brackets for Holm-significant Welch pairs at stacked heights."""
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
			ax.plot([x1, x1, x2, x2], [y, y + riser, y + riser, y], lw=1.4, color="0.15")
			ax.text((x1 + x2) / 2.0, y + riser, "*", ha="center", va="bottom", fontsize=14, color="0.1")

# --- facet / distribution plots ---


def _two_row_cat12_for_labels(
	exps: list[str],
	labels: ExperimentDisplayLabels,
) -> tuple[list[str], list[str], list[str]]:
	"""Split sorted display labels: Cat1 (sort ranks 100-102), Cat2 sequence (200-205); remainder is `other`."""
	return two_row_cat12_from_sorted(exps, labels)



def _whisker_low_high(x: np.ndarray) -> tuple[float, float]:
	"""Tukey whisker ends (1.5 IQR): extreme data inside the fences."""
	x = np.asarray(x, dtype=np.float64)
	x = x[np.isfinite(x)]
	if x.size == 0:
		return 0.0, 1.0
	q1, q3 = np.percentile(x, [25, 75])
	iqr = float(q3 - q1)
	if iqr == 0.0:
		return float(np.min(x)), float(np.max(x))
	lo_f, hi_f = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
	inside = x[(x >= lo_f) & (x <= hi_f)]
	if inside.size == 0:
		return float(np.min(x)), float(np.max(x))
	return float(np.min(inside)), float(np.max(inside))


def _outlier_mask(x: np.ndarray, lo_w: float, hi_w: float) -> np.ndarray:
	x = np.asarray(x, dtype=np.float64)
	return np.isfinite(x) & ((x < lo_w) | (x > hi_w))


def _score_facet_need_hi(
	sub: pd.DataFrame,
	value_col: str,
	order: list[str],
	*,
	sig_bracket_ylim_pad_frac: float,
) -> float:
	"""Upper y (including Holm/Welch bracket headroom) matching the old `facet_box_welch` scaling."""
	if len(order) < 2:
		return 1.0
	natural_hi = 0.0
	for oid in order:
		v = sub.loc[sub["optimizer_id"] == oid, value_col].dropna().to_numpy(dtype=np.float64)
		if v.size == 0:
			continue
		_, wh = _whisker_low_high(v)
		natural_hi = max(natural_hi, wh, float(np.nanmax(v)))
	span = natural_hi if natural_hi > 0 else 1.0
	_, sig = welch_pairwise_matrix(sub, value_col, "optimizer_id", order)
	n_sig = max(1, int(np.triu(sig, 1).sum()))
	return float(natural_hi + 0.02 * span + (sig_bracket_ylim_pad_frac * span) * n_sig)


def _fixed_ylim_for_score_metric(
	subm: pd.DataFrame,
	*,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	sig_bracket_ylim_pad_frac: float,
) -> tuple[float, float]:
	"""One shared (0, ymax) for all experiment facets of a single score metric."""
	score_hi = 0.0
	exps_m = sort_experiment_names(subm["experiment"].unique())
	for exp in exps_m:
		sub = subm[subm["experiment"] == exp]
		order = _sort_optimizer_ids(sub["optimizer_id"].unique().tolist())
		score_hi = max(score_hi, _score_facet_need_hi(sub, "value", order, sig_bracket_ylim_pad_frac=sig_bracket_ylim_pad_frac))
	return (0.0, float(score_hi)) if score_hi > 0 else (0.0, 1.0)


def _annotate_optimizer_means(
	ax,
	sub: pd.DataFrame,
	order: list[str],
	value_col: str,
	*,
	decimals: int = 2,
) -> None:
	for xi, oid in enumerate(order):
		vals = pd.to_numeric(sub.loc[sub["optimizer_id"] == oid, value_col], errors="coerce").dropna()
		if vals.empty:
			continue
		mean = float(vals.mean())
		min_loc = float(vals.min())-0.5
		ax.text(
			float(xi),
			min_loc,
			f"{mean:.{decimals}f}",
			ha="center",
			va="bottom",
			fontsize=9,
			color="0.15",
			clip_on=False,
			zorder=5,
		)
		ax.scatter([float(xi)], [mean], color="0.15",marker="D", s=10, alpha=0.5, zorder=10)
	vals_all = pd.to_numeric(sub[value_col], errors="coerce").dropna()
	if vals_all.empty:
		return
	exp_mean = float(vals_all.mean())
	ax.text(
		0.95,
		0.95,
		r"$\mu_{\text{exp}}="+f"{exp_mean:.{decimals}f}$",
		transform=ax.transAxes,
		ha="right",
		va="top",
		fontsize=12,
		color="0.15",
		clip_on=False,
		zorder=11,
	)


def facet_box_welch(
	df: pd.DataFrame,
	value_col: str,
	experiment_col: str,
	title: str,
	ylabel: str,
	filename: str,
	plot_dir: str,
	sig_bracket_step_frac: float,
	sig_bracket_ylim_pad_frac: float,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	labels: ExperimentDisplayLabels,
	figsize_per: tuple[float, float] = (5.2, 4.2),
	*,
	fixed_ylim: tuple[float, float] | None = None,
	showfliers: bool = True,
	annotate_means: bool = False,
	mean_decimals: int = 2,
) -> None:
	"""One subplot per experiment: box by optimizer + within-experiment Holm/Welch."""
	exps = sort_experiment_names(df[experiment_col].unique())
	if not exps:
		return

	def _one(ax, exp: str, *, set_ylabel: bool = False, set_xlabel: bool = False) -> None:
		sub = df[df[experiment_col] == exp]
		order = _sort_optimizer_ids(sub["optimizer_id"].unique().tolist())
		if len(order) < 2:
			ax.set_visible(False)
			return
		palette = [_opt_color(o) for o in order]
		sns.boxplot(
			data=sub,
			x="optimizer_id",
			y=value_col,
			hue="optimizer_id",
			order=order,
			hue_order=order,
			palette=palette,
			legend=False,
			ax=ax,
			linewidth=0.9,
			showfliers=showfliers,
			flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
		)
		ax.set_xticks(np.arange(len(order)))
		if set_xlabel:
			ax.set_xticklabels([_opt_display(o) for o in order], rotation=18, ha="right")
			ax.set_xlabel("optimizer")
		else:
			ax.set_xticklabels([])
			ax.set_xlabel("")
		if set_ylabel:
			ax.set_ylabel(ylabel)
		else:
			ax.set_ylabel("")
		ax.set_title(str(exp))
		lo_m, hi_m = ax.get_ylim()
		span_m = hi_m - lo_m if hi_m > lo_m else 1.0
		_, sig = welch_pairwise_matrix(sub, value_col, "optimizer_id", order)
		draw_sig_brackets(ax, order, sig, y_base=hi_m + 0.02 * span_m, y_step=sig_bracket_step_frac * span_m)
		if fixed_ylim is not None:
			ax.set_ylim(fixed_ylim)
		else:
			ax.set_ylim(lo_m, hi_m + (sig_bracket_ylim_pad_frac * span_m) * max(1, int(np.triu(sig, 1).sum())))
		if annotate_means:
			_annotate_optimizer_means(ax, sub, order, value_col, decimals=mean_decimals)

	row1, row2, other = _two_row_cat12_for_labels(exps, labels)
	use_two_rows = bool(row1 and row2) and not other
	if use_two_rows:
		ncols = max(len(row1), len(row2))
		fig, axes = plt.subplots(2, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * 2), squeeze=False)
		for j, exp in enumerate(row1):
			_one(axes[0, j], exp, set_ylabel=(j == 0), set_xlabel=False)
		for j in range(len(row1), ncols):
			axes[0, j].set_visible(False)
		for j, exp in enumerate(row2):
			_one(axes[1, j], exp, set_ylabel=(j == 0), set_xlabel=True)
		for j in range(len(row2), ncols):
			axes[1, j].set_visible(False)
		legend_opts = _sort_optimizer_ids(df["optimizer_id"].unique().tolist())
		legend_handles = [
			mpl.patches.Patch(
				facecolor=_opt_color(o),
				edgecolor="0.25",
				linewidth=0.55,
				label=_opt_display(o),
			)
			for o in legend_opts
		]
		fig.legend(
			handles=legend_handles,
			title="optimizer",
			loc="upper center",
			bbox_to_anchor=(0.70, 0.8),
			fontsize=12,
		)
	else:
		ncols = min(3, len(exps))
		nrows = int(np.ceil(len(exps) / ncols))
		fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False)
		ax_array = np.ravel(axes)
		for j, (ax, exp) in enumerate(zip(ax_array, exps)):
			row_idx = j // ncols
			col_idx = j % ncols
			_one(
				ax,
				exp,
				set_ylabel=(col_idx == 0),
				set_xlabel=(row_idx == nrows - 1),
			)
		for ax in ax_array[len(exps) :]:
			ax.set_visible(False)
	fig.suptitle(title, y=1.02, fontsize=14)
	plt.tight_layout()
	plt.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	plt.show()


def _length_whisker_glob(df: pd.DataFrame, value_col: str) -> float:
	whisk_glob = 0.0
	for (_e, oid), g in df.groupby(["experiment", "optimizer_id"], sort=False):
		v = g[value_col].dropna().to_numpy(dtype=np.float64)
		if v.size:
			_, wh = _whisker_low_high(v)
			whisk_glob = max(whisk_glob, wh)
	return float(whisk_glob)


def _length_final_hi_for_subplot(
	y_main_top: float,
	span: float,
	sub: pd.DataFrame,
	value_col: str,
	order: list[str],
	*,
	sig_bracket_ylim_pad_frac: float,
) -> float:
	if len(order) < 2:
		return float(y_main_top + 0.02 * span + sig_bracket_ylim_pad_frac * span)
	_, sig = welch_pairwise_matrix(sub, value_col, "optimizer_id", order)
	n_sig = max(1, int(np.triu(sig, 1).sum()))
	return float(y_main_top + 0.02 * span + (sig_bracket_ylim_pad_frac * span) * n_sig)


def _draw_length_box_manual_fliers(
	ax,
	sub: pd.DataFrame,
	value_col: str,
	order: list[str],
	*,
	y_main_top: float,
	flier_kw: dict,
) -> int:
	"""`showfliers=False` boxplot + manual low/high fliers; return high-fliers with y > `y_main_top`."""
	palette = [_opt_color(o) for o in order]
	sns.boxplot(
		data=sub,
		x="optimizer_id",
		y=value_col,
		hue="optimizer_id",
		order=order,
		hue_order=order,
		palette=palette,
		legend=False,
		ax=ax,
		linewidth=0.9,
		showfliers=False,
	)
	n_hidden_total = 0
	dy = max(0.012 * y_main_top, 1.0) if y_main_top > 0 else 1.0
	low_strip = 0.02 * y_main_top if y_main_top > 0 else 0.02
	for xi, oid in enumerate(order):
		v = sub.loc[sub["optimizer_id"] == oid, value_col].dropna().to_numpy(dtype=np.float64)
		if v.size == 0:
			continue
		lo_w, hi_w = _whisker_low_high(v)
		m = _outlier_mask(v, lo_w, hi_w)
		if not np.any(m):
			continue
		out = v[m]
		lows = out[out < lo_w]
		highs = out[out > hi_w]
		if lows.size:
			for j in range(int(lows.size)):
				ax.scatter([float(xi)], [float(low_strip + j * dy)], **flier_kw)
		if highs.size:
			in_band = highs[highs <= y_main_top]
			n_hidden_total += int(np.sum(highs > y_main_top))
			if in_band.size:
				n = int(in_band.size)
				for j in range(n):
					y_draw = float(y_main_top - (j + 1) * dy)
					if y_draw <= hi_w:
						y_draw = float(hi_w + (j + 1) * max(dy * 0.35, 1.0))
					ax.scatter([float(xi)], [y_draw], **flier_kw)
	return n_hidden_total


def _length_shared_final_hi(
	len_df: pd.DataFrame,
	value_col: str,
	experiment_col: str,
	exps: list[str],
	pool_df: pd.DataFrame | None,
	*,
	sig_bracket_ylim_pad_frac: float,
) -> tuple[float, float, float]:
	"""`FINAL_HI`, `y_main_top`, `span` shared by all period-length panels."""
	whisk_glob = _length_whisker_glob(len_df, value_col)
	if pool_df is not None and not pool_df.empty:
		whisk_glob = max(whisk_glob, _length_whisker_glob(pool_df, value_col))
	span = whisk_glob if whisk_glob > 0 else 1.0
	y_main_top = whisk_glob + 0.08 * span
	final_hi = y_main_top + 0.02 * span + sig_bracket_ylim_pad_frac * span
	for exp in exps:
		sub = len_df[len_df[experiment_col] == exp]
		order = _sort_optimizer_ids(sub["optimizer_id"].unique().tolist())
		final_hi = max(
			final_hi,
			_length_final_hi_for_subplot(
				y_main_top, span, sub, value_col, order, sig_bracket_ylim_pad_frac=sig_bracket_ylim_pad_frac
			),
		)
	if pool_df is not None and not pool_df.empty and pool_df["optimizer_id"].nunique() > 1:
		orderp = _sort_optimizer_ids(pool_df["optimizer_id"].unique().tolist())
		final_hi = max(
			final_hi,
			_length_final_hi_for_subplot(
				y_main_top, span, pool_df, value_col, orderp, sig_bracket_ylim_pad_frac=sig_bracket_ylim_pad_frac
			),
		)
	return float(final_hi), float(y_main_top), float(span)


def facet_box_welch_length_iter(
	len_df: pd.DataFrame,
	pool_df: pd.DataFrame | None,
	experiment_col: str,
	title: str,
	ylabel: str,
	filename: str,
	plot_dir: str,
	sig_bracket_step_frac: float,
	sig_bracket_ylim_pad_frac: float,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	labels: ExperimentDisplayLabels,
	figsize_per: tuple[float, float] = (5.2, 4.2),
) -> None:
	"""Period length facets: shared y, manual fliers, hidden-high count annotation."""
	exps = sort_experiment_names(len_df[experiment_col].unique())
	if not exps:
		return
	FINAL_HI, y_main_top, span = _length_shared_final_hi(
		len_df, "value", experiment_col, exps, pool_df, sig_bracket_ylim_pad_frac=sig_bracket_ylim_pad_frac
	)
	flier_kw = {"marker": "o", "s": 9, "alpha": 0.35, "color": "0.35", "linewidths": 0, "zorder": 3}

	def _one_len(ax, exp: str) -> None:
		sub = len_df[len_df[experiment_col] == exp]
		order = _sort_optimizer_ids(sub["optimizer_id"].unique().tolist())
		if len(order) < 2:
			ax.set_visible(False)
			return
		n_hid = _draw_length_box_manual_fliers(ax, sub, "value", order, y_main_top=y_main_top, flier_kw=flier_kw)
		ax.set_xticks(np.arange(len(order)))
		ax.set_xticklabels([_opt_display(o) for o in order], rotation=18, ha="right")
		ax.set_xlabel("")
		ax.set_ylabel(ylabel)
		ax.set_title(str(exp))
		_, sig = welch_pairwise_matrix(sub, "value", "optimizer_id", order)
		draw_sig_brackets(ax, order, sig, y_base=y_main_top + 0.02 * span, y_step=sig_bracket_step_frac * span)
		ax.set_ylim(0.0, FINAL_HI)
		if n_hid > 0:
			xc = 0.5 * float(len(order) - 1)
			_y_text = FINAL_HI - 0.2 * span
			ax.annotate(
				str(n_hid),
				xy=(xc, FINAL_HI),
				xytext=(xc, _y_text),
				textcoords="data",
				ha="center",
				va="bottom",
				fontsize=11,
				color="0.2",
				clip_on=False,
				arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.0, shrinkA=0, shrinkB=2),
			)

	row1, row2, other = _two_row_cat12_for_labels(exps, labels)
	use_two_rows = bool(row1 and row2) and not other
	if use_two_rows:
		ncols = max(len(row1), len(row2))
		fig, axes = plt.subplots(2, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * 2), squeeze=False)
		for j, exp in enumerate(row1):
			_one_len(axes[0, j], exp)
		for j in range(len(row1), ncols):
			axes[0, j].set_visible(False)
		for j, exp in enumerate(row2):
			_one_len(axes[1, j], exp)
		for j in range(len(row2), ncols):
			axes[1, j].set_visible(False)
	else:
		ncols = min(3, len(exps))
		nrows = int(np.ceil(len(exps) / ncols))
		fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False)
		for ax, exp in zip(np.ravel(axes), exps):
			_one_len(ax, exp)
		for ax in np.ravel(axes)[len(exps) :]:
			ax.set_visible(False)
	fig.suptitle(title, y=1.02, fontsize=14)
	plt.tight_layout()
	plt.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	plt.show()


def facet_box_welch_t_start_by_experiment(
	pma_df: pd.DataFrame,
	title: str,
	filename: str,
	plot_dir: str,
	sig_bracket_step_frac: float,
	sig_bracket_ylim_pad_frac: float,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	labels: ExperimentDisplayLabels,
	experiment_col: str = "experiment",
	figsize_per: tuple[float, float] = (5.2, 4.2),
) -> None:
	"""One figure: subplots per experiment for t_start; Holm-corrected KS pairwise brackets."""
	exps = sort_experiment_names(pma_df[experiment_col].unique())
	if not exps:
		return

	y_tops: list[float] = []

	def _one_pma(ax, exp: str) -> None:
		sub = pma_df[pma_df[experiment_col] == exp]
		order = _sort_optimizer_ids(sub["optimizer_id"].unique().tolist())
		if len(order) < 2:
			ax.text(
				0.5,
				0.5,
				"Not enough optimizers for pairwise tests",
				ha="center",
				va="center",
				transform=ax.transAxes,
			)
			ax.axis("off")
			return
		sns.boxplot(
			data=sub,
			x="optimizer_id",
			y="value",
			hue="optimizer_id",
			order=order,
			hue_order=order,
			palette=[_opt_color(o) for o in order],
			legend=False,
			ax=ax,
			linewidth=0.9,
			flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
		)
		ax.set_xticks(np.arange(len(order)))
		ax.set_xticklabels([_opt_display(o) for o in order], rotation=18, ha="right")
		ax.set_xlabel("")
		ax.set_ylabel(r"$t_{\mathrm{start}}$ (samples before assignment)")
		ax.set_title(str(exp))
		lo, hi = ax.get_ylim()
		span = hi - lo if hi > lo else 1.0
		_, sig = ks_pairwise_matrix(sub, "value", "optimizer_id", order)
		draw_sig_brackets(ax, order, sig, y_base=hi + 0.02 * span, y_step=sig_bracket_step_frac * span)
		y_top = hi + (sig_bracket_ylim_pad_frac * span) * max(1, int(np.triu(sig, 1).sum()))
		y_tops.append(float(y_top))

	row1, row2, other = _two_row_cat12_for_labels(exps, labels)
	use_two_rows = bool(row1 and row2) and not other
	if use_two_rows:
		ncols = max(len(row1), len(row2))
		fig, axes = plt.subplots(2, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * 2), squeeze=False)
		for j, exp in enumerate(row1):
			_one_pma(axes[0, j], exp)
		for j in range(len(row1), ncols):
			axes[0, j].set_visible(False)
		for j, exp in enumerate(row2):
			_one_pma(axes[1, j], exp)
		for j in range(len(row2), ncols):
			axes[1, j].set_visible(False)
	else:
		ncols = min(3, len(exps))
		nrows = int(np.ceil(len(exps) / ncols))
		fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False)
		for ax, exp in zip(np.ravel(axes), exps):
			_one_pma(ax, exp)
		for ax in np.ravel(axes)[len(exps) :]:
			ax.set_visible(False)
	y_max = max(y_tops) if y_tops else 1.0
	y_max = max(y_max, 1e-9)
	for ax in np.ravel(axes):
		if not ax.get_visible() or not ax.patches:
			continue
		ax.set_ylim(0.0, y_max)
	fig.suptitle(title, y=1.02, fontsize=14)
	plt.tight_layout()
	plt.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	plt.show()


# --- offset-from-novelty bincounts (t_start / PMA timing) ---


def _novelty_offset_from_period_row(row: pd.Series, *, cat1: bool) -> float:
	"""Offset of max-acceleration time from the category novelty reference.

	Cat1: novelty at experiment start (iter 0) → global iter = trial_start + t_start.
	Cat2: novelty at that period's trial start → trial-relative t_start.
	"""
	pma = pd.to_numeric(row.get("point_of_max_acceleration"), errors="coerce")
	if not np.isfinite(pma):
		return np.nan
	if cat1:
		ts = pd.to_numeric(row.get("trial_start_iter"), errors="coerce")
		if not np.isfinite(ts):
			return np.nan
		return float(ts + pma)
	return float(pma)


def build_period_novelty_offset_df(
	all_results: dict,
	rename_fn: Callable[[str], str],
) -> pd.DataFrame:
	"""One row per period with category-specific offset from novelty."""
	rows: list[dict[str, Any]] = []
	for (eid, _mid, oid, _rid), payload in all_results.items():
		if payload.get("error") or "periods_df" not in payload:
			continue
		pdf = payload["periods_df"]
		if pdf is None or pdf.empty:
			continue
		eid_s = str(eid)
		cat1 = bool(is_pretrain_shuffle_mislabel_experiment(eid_s))
		cat2 = eid_s.startswith("cat2_sequence_")
		if not cat1 and not cat2:
			continue
		for _, row in pdf.iterrows():
			offset = _novelty_offset_from_period_row(row, cat1=cat1)
			if not np.isfinite(offset):
				continue
			rows.append(
				{
					"experiment_id": eid_s,
					"experiment": rename_fn(eid_s),
					"optimizer_id": str(oid),
					"offset_from_novelty": offset,
					"category": "cat1" if cat1 else "cat2",
				}
			)
	return pd.DataFrame(rows)


def _offset_bin_edges(
	values: np.ndarray,
	bin_width: float,
	*,
	pad_bins: int = 1,
) -> np.ndarray:
	v = np.asarray(values, dtype=np.float64)
	v = v[np.isfinite(v)]
	if v.size == 0:
		return np.arange(0.0, bin_width, bin_width)
	lo = float(np.floor(np.min(v) / bin_width) * bin_width) - pad_bins * bin_width
	hi = float(np.ceil(np.max(v) / bin_width) * bin_width) + pad_bins * bin_width
	if hi <= lo:
		hi = lo + bin_width
	return np.arange(lo, hi + bin_width, bin_width)


def _bincount_proportions(
	offsets: np.ndarray,
	bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	v = np.asarray(offsets, dtype=np.float64)
	v = v[np.isfinite(v)]
	if v.size == 0:
		centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2.0
		return centers, np.zeros_like(centers, dtype=np.float64)
	counts, _ = np.histogram(v, bins=bin_edges)
	centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2.0
	return centers, counts.astype(np.float64) / float(v.size)


def load_neuro_experimental_bars(path: str | Path) -> pd.DataFrame:
	"""Load neuro ground-truth histogram: `index` = time, `*_val` = counts."""
	df = pd.read_csv(path)
	rename = {
		"teal_val": "btsp_val",
		"magenta_val": "other_val",
	}
	df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
	if "index" not in df.columns:
		raise ValueError("%s: missing column 'index' (time)" % (path,))
	for col in ("index", "btsp_val", "other_val"):
		if col not in df.columns:
			raise ValueError("%s: missing column %r" % (path, col))
		df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def _histogram_bars_to_ecdf_steps(
	x: np.ndarray,
	counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""ECDF (post-step) from histogram bar centers and counts."""
	x = np.asarray(x, dtype=np.float64)
	w = np.asarray(counts, dtype=np.float64)
	mask = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
	x, w = x[mask], w[mask]
	if x.size == 0:
		return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
	order = np.argsort(x)
	x, w = x[order], w[order]
	return x, np.cumsum(w) / float(w.sum())


def _draw_neuro_ground_truth_offset_cdf(
	ax: plt.Axes,
	neuro_bars: pd.DataFrame | None,
) -> None:
	"""Overlay neuro experimental reference ECDFs (BTSP thick black, other thin gray)."""
	if neuro_bars is None or neuro_bars.empty:
		return
	time = neuro_bars["index"].to_numpy(dtype=np.float64)
	x_btsp, y_btsp = _histogram_bars_to_ecdf_steps(
		time,
		neuro_bars["btsp_val"].to_numpy(dtype=np.float64),
	)
	if x_btsp.size:
		ax.step(
			x_btsp,
			y_btsp,
			where="post",
			color="black",
			linewidth=1.5,
			label="BTSP (neuro)",
			zorder=10,
		)
	x_other, y_other = _histogram_bars_to_ecdf_steps(
		time,
		neuro_bars["other_val"].to_numpy(dtype=np.float64),
	)
	if x_other.size:
		ax.step(
			x_other,
			y_other,
			where="post",
			color="0.6",
			linewidth=1.0,
			label="other (neuro)",
			zorder=11,
			ls="--"
		)


def _neuro_ground_truth_legend_handles() -> tuple[list[Any], list[str]]:
	from matplotlib.lines import Line2D

	handles = [
		Line2D([0], [0], color="black", linewidth=2.8, label="BTSP (neuro)"),
		Line2D([0], [0], color="0.6", linewidth=1.0, label="other (neuro)", ls="--"),
	]
	labels = ["BTSP (neuro)", "other (neuro)"]
	return handles, labels


def _novelty_offset_mode(mode: str) -> str:
	m = str(mode).strip().lower()
	if m == "bar":
		m = "stack_bar"
	if m not in ("cdf", "stack_bar", "neighbor_bar"):
		raise ValueError(
			"mode must be 'cdf', 'stack_bar', or 'neighbor_bar', got %r" % (mode,),
		)
	return m


def _novelty_offset_uses_bins(mode: str) -> bool:
	return mode in ("stack_bar", "neighbor_bar")


def _offset_xlim_from_values(values: np.ndarray, *, pad_frac: float = 0.02) -> tuple[float, float]:
	v = np.asarray(values, dtype=np.float64)
	v = v[np.isfinite(v)]
	if v.size == 0:
		return 0.0, 1.0
	lo, hi = float(np.min(v)), float(np.max(v))
	pad = pad_frac * (hi - lo) if hi > lo else 1.0
	return lo - pad, hi + pad


def _annotate_optimizer_n_counts(ax, sub: pd.DataFrame, order: list[str]) -> None:
	for i, oid in enumerate(order):
		n = int((sub["optimizer_id"] == oid).sum())
		ax.text(
			0.9,
			0.9 - (i * 0.1),
			f"n={n}",
			transform=ax.transAxes,
			ha="center",
			va="bottom",
			fontsize=9,
			color=_opt_color(oid),
			clip_on=False,
		)


def _draw_optimizer_bincount_stack_bars(
	ax,
	sub: pd.DataFrame,
	order: list[str],
	bin_edges: np.ndarray,
	bin_width: float,
) -> None:
	"""Stacked bars at each bin: one segment per optimizer (bottom = first in order), y = proportion within optimizer."""
	if not order:
		return
	bin_step = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else float(bin_width)
	bar_w = bin_step * 0.92
	_annotate_optimizer_n_counts(ax, sub, order)
	bottom: np.ndarray | None = None
	for oid in order:
		vals = sub.loc[sub["optimizer_id"] == oid, "offset_from_novelty"].to_numpy(dtype=np.float64)
		centers, props = _bincount_proportions(vals, bin_edges)
		if bottom is None:
			bottom = np.zeros_like(props, dtype=np.float64)
		ax.bar(
			centers,
			props,
			width=bar_w * 0.95,
			bottom=bottom,
			color=_opt_color(oid),
			edgecolor="0.25",
			linewidth=0.45,
			align="center",
		)
		bottom = bottom + props


def _draw_optimizer_bincount_neighbor_bars(
	ax,
	sub: pd.DataFrame,
	order: list[str],
	bin_edges: np.ndarray,
	bin_width: float,
) -> None:
	"""Grouped bars at each bin: one bar per optimizer side-by-side, y = proportion within optimizer."""
	if not order:
		return
	bin_step = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else float(bin_width)
	n_opt = len(order)
	group_w = bin_step * 0.92
	bar_w = group_w / n_opt * 0.95
	_annotate_optimizer_n_counts(ax, sub, order)
	for i, oid in enumerate(order):
		vals = sub.loc[sub["optimizer_id"] == oid, "offset_from_novelty"].to_numpy(dtype=np.float64)
		centers, props = _bincount_proportions(vals, bin_edges)
		x = centers + (i - 0.5 * (n_opt - 1)) * bar_w
		ax.bar(
			x,
			props,
			width=bar_w,
			color=_opt_color(oid),
			edgecolor="0.25",
			linewidth=0.45,
			align="center",
		)


def offset_at_cdf_gt_than(vals: np.ndarray, *, cdf_threshold: float = 0.5) -> float:
    """Smallest offset where ECDF (post-steps, y=i/n) exceeds 0.5."""
    v = np.sort(np.asarray(vals, dtype=np.float64))
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return np.nan
    idx = int(np.searchsorted(np.arange(1, n + 1, dtype=np.float64) / n, cdf_threshold, side="right"))
    if idx >= n:
        return float(v[-1])
    return float(v[idx])

def auc_ecdf(
    vals: np.ndarray,
    *,
    xlo: float,
    xhi: float,
) -> float:
    """Mean post-step ECDF height on a shared [xlo, xhi] window (facet-wide).

    Matches overlaid ECDF plots that use the same xlim for all optimizers.
    Equivalent to integrating F_n on [xlo, xhi] with F=0 left of data and F=1
    right of max(vals), then dividing by (xhi - xlo).
    """
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    span = float(xhi - xlo)
    if v.size == 0 or span <= 0.0:
        return np.nan
    return float(1.0 - (v.mean() - xlo) / span)


def novelty_offset_xlim(
    sub: pd.DataFrame,
    *,
    pad_frac: float = 0.02,
) -> tuple[float, float]:
    """Shared xlim for one experiment facet (all optimizers pooled)."""
    return _offset_xlim_from_values(
        sub["offset_from_novelty"].to_numpy(dtype=np.float64),
        pad_frac=pad_frac,
    )

def _draw_optimizer_offset_cdf(
	ax,
	sub: pd.DataFrame,
	order: list[str],
	*,
	cdf_threshold: float = 0.6,
	print_auc: bool = True,
) -> pd.DataFrame:
	"""One ECDF line per optimizer. Returns per-optimizer AUC (shared facet xlim)."""
	cols = ["optimizer_id", "optimizer", "n", "offset_cdf_gt", "auc"]
	if not order:
		return pd.DataFrame(columns=cols)

	xlo, xhi = novelty_offset_xlim(sub)
	rows: list[dict[str, Any]] = []

	for i, oid in enumerate(order):
		vals = sub.loc[
			sub["optimizer_id"] == oid, "offset_from_novelty"
		].to_numpy(dtype=np.float64)
		offset = offset_at_cdf_gt_than(vals, cdf_threshold=cdf_threshold)
		auc = auc_ecdf(vals, xlo=xlo, xhi=xhi)
		vals_f = vals[np.isfinite(vals)]
		n = int(vals_f.size)
		rows.append(
			{
				"optimizer_id": oid,
				"optimizer": _opt_display(oid),
				"n": n,
				"offset_cdf_gt": offset,
				"auc": auc,
			}
		)
		if print_auc:
			print(
				f"{_opt_display(oid)} CDF > {cdf_threshold} at {offset}, "
				f"AUC={auc:.3f}"
			)
		ax.text(
			0.9,
			0.9 - (i * 0.1),
			f"n={n}",
			transform=ax.transAxes,
			ha="center",
			va="bottom",
			fontsize=9,
			color=_opt_color(oid),
			clip_on=False,
		)
		if n == 0:
			continue
		x = np.sort(vals_f)
		y = np.arange(1, n + 1, dtype=np.float64) / float(n)
		ax.step(
			x,
			y,
			where="post",
			color=_opt_color(oid),
			linewidth=1.5,
			label=_opt_display(oid),
			alpha=0.6,
		)

	return pd.DataFrame(rows, columns=cols)

def summarize_mean_optimizer_auc(
	auc_df: pd.DataFrame,
	*,
	title: str = "",
) -> pd.DataFrame:
	if auc_df.empty:
		return auc_df
	summary = (
		auc_df.groupby(["optimizer_id", "optimizer"], as_index=False)
		.agg(
			mean_auc=("auc", "mean"),
			std_auc=("auc", "std"),
			n_experiments=("experiment", "nunique"),
		)
		.sort_values("mean_auc", ascending=False)
	)
	if title:
		print(title)
	print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
	return summary

def _novelty_offset_legend_handles(
	offset_df: pd.DataFrame,
	*,
	mode: str = "cdf",
	include_neuro_ground_truth: bool = False,
) -> tuple[list[Any], list[str]]:
	mode = _novelty_offset_mode(mode)
	order = _sort_optimizer_ids(offset_df["optimizer_id"].unique().tolist())
	if _novelty_offset_uses_bins(mode):
		from matplotlib.patches import Patch

		handles = [
			Patch(facecolor=_opt_color(oid), edgecolor="0.25", linewidth=0.45, label=_opt_display(oid))
			for oid in order
		]
	else:
		from matplotlib.lines import Line2D

		handles = [
			Line2D([0], [0], color=_opt_color(oid), linewidth=1.5, label=_opt_display(oid))
			for oid in order
		]
	labels = [_opt_display(oid) for oid in order]
	if include_neuro_ground_truth:
		neuro_handles, neuro_labels = _neuro_ground_truth_legend_handles()
		handles = handles + neuro_handles
		labels = labels + neuro_labels
	return handles, labels


def facet_bincount_novelty_offset_by_experiment(
	offset_df: pd.DataFrame,
	title: str,
	filename: str,
	plot_dir: str,
	sig_bracket_step_frac: float,
	sig_bracket_ylim_pad_frac: float,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	labels: ExperimentDisplayLabels,
	experiment_col: str = "experiment",
	*,
	mode: str = "cdf",
	bin_width: float,
	figsize_per: tuple[float, float] = (7.5, 2.8),
	print_mean_auc: bool = True,
	neuro_experimental_bars: pd.DataFrame | None = None,
) -> None:
	"""Facets: offset from novelty by optimizer (`mode='cdf'`, `stack_bar`, or `neighbor_bar`).

	Cat1 experiments on the top row and Cat2 on the bottom when both are present;
	the optimizer legend sits beside the top row.
	"""
	mode = _novelty_offset_mode(mode)
	if offset_df.empty:
		return
	exps = sort_experiment_names(offset_df[experiment_col].unique())
	if not exps:
		return
	bin_edges: np.ndarray | None = None
	if _novelty_offset_uses_bins(mode):
		bin_edges = _offset_bin_edges(offset_df["offset_from_novelty"].to_numpy(), bin_width)
	show_neuro_gt = mode == "cdf" and neuro_experimental_bars is not None
	legend_handles, legend_labels = _novelty_offset_legend_handles(
		offset_df,
		mode=mode,
		include_neuro_ground_truth=show_neuro_gt,
	)
	auc_parts: list[pd.DataFrame] = []
	def _draw_one(ax: plt.Axes, exp: str) -> None:
		sub = offset_df[offset_df[experiment_col] == exp]
		order = _sort_optimizer_ids(sub["optimizer_id"].unique().tolist())
		if _novelty_offset_uses_bins(mode):
			assert bin_edges is not None
			if mode == "stack_bar":
				_draw_optimizer_bincount_stack_bars(ax, sub, order, bin_edges, bin_width)
			else:
				_draw_optimizer_bincount_neighbor_bars(ax, sub, order, bin_edges, bin_width)
			ax.set_xlim(bin_edges[0], bin_edges[-1])
			ax.set_ylim(0.0, None)
		else:
			auc_sub = _draw_optimizer_offset_cdf(ax, sub, order)
			_draw_neuro_ground_truth_offset_cdf(ax, neuro_experimental_bars)
			auc_sub["experiment"] = exp
			auc_parts.append(auc_sub)
			ax.set_xlim(*novelty_offset_xlim(sub))
			ax.set_ylim(0.0, 1.0)
		ax.set_ylabel("proportion" if _novelty_offset_uses_bins(mode) else "eCDF")
		ax.set_title(str(exp))
		ax.tick_params(axis="y", labelsize=10)
		ax.tick_params(axis="x", labelsize=10)

	row1, row2, other = _two_row_cat12_for_labels(exps, labels)
	use_two_rows = bool(row1 and row2) and not other
	if use_two_rows:
		ncols = max(len(row1), len(row2))
		fig, axes = plt.subplots(
			2,
			ncols,
			figsize=(figsize_per[0] * ncols, figsize_per[1] * 2),
			sharex=True,
			squeeze=False,
		)
		for j, exp in enumerate(row1):
			_draw_one(axes[0, j], exp)
		for j in range(len(row1), ncols):
			axes[0, j].set_visible(False)
		for j, exp in enumerate(row2):
			_draw_one(axes[1, j], exp)
		for j in range(len(row2), ncols):
			axes[1, j].set_visible(False)
		x_ax = axes[1, len(row2) - 1]
	else:
		n_cols = len(exps)
		fig, axes = plt.subplots(
			1,
			n_cols,
			figsize=(figsize_per[0] * n_cols, figsize_per[1]),
			sharex=True,
			squeeze=False,
		)
		for ax, exp in zip(np.ravel(axes), exps):
			_draw_one(ax, exp)
		x_ax = np.ravel(axes)[-1]

	if _novelty_offset_uses_bins(mode):
		x_ax.set_xlabel(r"offset from novelty (iterations, bin width = %g)" % bin_width)
	else:
		x_ax.set_xlabel(r"offset from novelty (iterations)")
	fig.suptitle(title, fontsize=14)
	plt.tight_layout()
	if legend_handles:
		if use_two_rows:
			anchor_ax = axes[0, len(row1) - 1] if len(row1) < len(row2) else axes[1, len(row2) - 1]
			bbox = anchor_ax.get_position()
			legend_y = 0.5 * (bbox.y0 + bbox.y1)
			fig.legend(
				legend_handles,
				legend_labels,
				title="optimizer",
				loc="center left",
				bbox_to_anchor=(bbox.x1 + 0.02, legend_y),
				frameon=True,
				fontsize=10,
			)
		else:
			fig.legend(
				legend_handles,
				legend_labels,
				title="optimizer",
				loc="center left",
				bbox_to_anchor=(1.01, 0.5),
				frameon=True,
				fontsize=10,
			)
	plt.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	plt.show()


	auc_df = pd.concat(auc_parts, ignore_index=True) if auc_parts else pd.DataFrame()
	if print_mean_auc and not auc_df.empty:
		summarize_mean_optimizer_auc(auc_df, title=f"Mean AUC — {title}")
	return auc_df


def plot_bincount_novelty_offset_pooled(
	offset_df: pd.DataFrame,
	title: str,
	filename: str,
	plot_dir: str,
	sig_bracket_step_frac: float,
	sig_bracket_ylim_pad_frac: float,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	labels: ExperimentDisplayLabels,
	*,
	mode: str = "cdf",
	bin_width: float,
	figsize: tuple[float, float] = (9.5, 4.8),
	neuro_experimental_bars: pd.DataFrame | None = None,
) -> None:
	"""Single panel: pooled offset from novelty (`mode='cdf'`, `stack_bar`, or `neighbor_bar`)."""
	mode = _novelty_offset_mode(mode)
	if offset_df.empty:
		return
	fig, ax = plt.subplots(figsize=figsize)
	order = _sort_optimizer_ids(offset_df["optimizer_id"].unique().tolist())
	if _novelty_offset_uses_bins(mode):
		bin_edges = _offset_bin_edges(offset_df["offset_from_novelty"].to_numpy(), bin_width)
		if mode == "stack_bar":
			_draw_optimizer_bincount_stack_bars(ax, offset_df, order, bin_edges, bin_width)
		else:
			_draw_optimizer_bincount_neighbor_bars(ax, offset_df, order, bin_edges, bin_width)
		ax.set_xlabel(r"offset from novelty (iterations, bin width = %g)" % bin_width)
		ax.set_ylabel("proportion")
		ax.set_xlim(bin_edges[0], bin_edges[-1])
		ax.set_ylim(0.0, None)
	else:
		auc_df = _draw_optimizer_offset_cdf(ax, offset_df, order)
		_draw_neuro_ground_truth_offset_cdf(ax, neuro_experimental_bars)
		ax.set_xlabel(r"offset from novelty (iterations)")
		ax.set_ylabel("eCDF")
		ax.set_xlim(*novelty_offset_xlim(offset_df))
		ax.set_ylim(0.0, 1.0)
	ax.tick_params(axis="y", labelsize=10)
	ax.tick_params(axis="x", labelsize=10)
	ax.set_title(title,y=1.02)
	show_neuro_gt = mode == "cdf" and neuro_experimental_bars is not None
	legend_handles, legend_labels = _novelty_offset_legend_handles(
		offset_df,
		mode=mode,
		include_neuro_ground_truth=show_neuro_gt,
	)
	ax.legend(
		legend_handles,
		legend_labels,
		title="optimizer",
		# bbox_to_anchor=(1.02, 1),
		loc=("lower" if mode == "cdf" else "upper") +" center",
		fontsize=10,
	)
	plt.tight_layout()
	plt.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	plt.show()


def _sort_layer_display_names(layer_names) -> list[str]:
	"""Stable layer order: DNN `hidden.{k}` by numeric k, then other names lexicographically."""
	names = [str(x) for x in layer_names if x is not None and str(x) not in ("nan", "")]

	def sort_key(ln: str) -> tuple[int, int | str]:
		if ln.startswith("hidden."):
			try:
				return (0, int(ln.rsplit(".", 1)[-1]))
			except ValueError:
				return (1, ln)
		return (2, ln)

	return sorted(set(names), key=sort_key)


def parse_ln(ln: str) -> str:
	return "Hidden\nLayer " + str(int(str(ln).split(".")[-1]) // 2)


def _fill_btsp_layer_grid(
	sub: pd.DataFrame,
	order_layers: list[str],
	order_opts: list[str],
) -> pd.DataFrame:
	"""Ensure every (layer, optimizer) pair exists; missing counts default to 0."""
	from analysis.final.final_additional_plots_helpers import (
		fill_layer_optimizer_grid,
	)

	return fill_layer_optimizer_grid(sub, order_layers, order_opts, "n_btsp")


def facet_bar_btsp_periods_per_layer(
	agg: pd.DataFrame,
	experiment_col: str,
	title: str,
	filename: str,
	plot_dir: str,
	sig_bracket_step_frac: float,
	sig_bracket_ylim_pad_frac: float,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	labels: ExperimentDisplayLabels,
	figsize_per: tuple[float, float] = (5.2, 4.2),
	*,
	show: bool = True,
	save: bool = True,
) -> tuple[plt.Figure | None, np.ndarray | None, list[tuple[plt.Axes, str]]]:
	"""One subplot per experiment: grouped bars — layer (x) vs BTSP period count (y), hue = optimizer."""
	if agg.empty or "n_btsp" not in agg.columns:
		return None, None, []
	exps = sort_experiment_names(agg[experiment_col].unique())
	if not exps:
		return None, None, []
	canonical_layers = _sort_layer_display_names(agg["layer_name"].unique())
	plotted_axes: list[tuple[plt.Axes, str]] = []

	def _one(ax, exp: str) -> None:
		sub = agg[agg[experiment_col] == exp].copy()
		if sub.empty:
			ax.set_visible(False)
			return
		order_layers = canonical_layers or _sort_layer_display_names(sub["layer_name"].unique())
		order_opts = _sort_optimizer_ids(sub["optimizer_id"].unique().tolist())
		if not order_layers or not order_opts:
			ax.set_visible(False)
			return
		sub = _fill_btsp_layer_grid(sub, order_layers, order_opts)
		palette = [_opt_color(o) for o in order_opts]
		sns.barplot(
			data=sub,
			x="layer_name",
			y="n_btsp",
			hue="optimizer_id",
			order=order_layers,
			hue_order=order_opts,
			palette=palette,
			ax=ax,
			edgecolor="0.25",
			linewidth=0.55,
			legend=False,
		)
		ax.set_xlabel("layer")
		ax.set_ylabel("BTSP period count")
		ax.set_title(str(exp))
		ax.set_xticks(np.arange(len(order_layers)))
		ax.set_xticklabels([parse_ln(ln) for ln in order_layers], fontsize=10, rotation=0)
		plotted_axes.append((ax, str(exp)))

	row1, row2, other = _two_row_cat12_for_labels(exps, labels)
	# Always put Cat1 on row 0 and Cat2 on row 1 when both are present (do not wrap with cat2 on row 0).
	use_cat12_rows = bool(row1 and row2)
	if use_cat12_rows:
		ncols = max(len(row1), len(row2), 1)
		n_extra_rows = int(np.ceil(len(other) / ncols)) if other else 0
		nrows = 2 + n_extra_rows
		fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False)
		for j, exp in enumerate(row1):
			_one(axes[0, j], exp)
		for j in range(len(row1), ncols):
			axes[0, j].set_visible(False)
		for j, exp in enumerate(row2):
			_one(axes[1, j], exp)
		for j in range(len(row2), ncols):
			axes[1, j].set_visible(False)
		for k, exp in enumerate(other):
			r, c = divmod(k, ncols)
			_one(axes[2 + r, c], exp)
		for k in range(len(other), n_extra_rows * ncols):
			r, c = divmod(k, ncols)
			axes[2 + r, c].set_visible(False)
	else:
		ncols = min(3, len(exps))
		nrows = int(np.ceil(len(exps) / ncols))
		fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False)
		for ax, exp in zip(np.ravel(axes), exps):
			_one(ax, exp)
		for ax in np.ravel(axes)[len(exps) :]:
			ax.set_visible(False)
	legend_opts = _sort_optimizer_ids(agg["optimizer_id"].unique().tolist())
	legend_handles = [
		mpl.patches.Patch(
			facecolor=_opt_color(o),
			edgecolor="0.25",
			linewidth=0.55,
			label=_opt_label(o),
		)
		for o in legend_opts
	]
	fig.legend(
		handles=legend_handles,
		title="optimizer",
		loc="upper center",
		bbox_to_anchor=(0.70, 0.8),
		fontsize=12,
	)
	fig.suptitle(title, fontsize=14)
	plt.tight_layout()
	if save:
		fig.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	if show:
		plt.show()
	return fig, axes, plotted_axes


def _is_grafted_shampoo_optimizer(oid: str) -> bool:
	return str(oid).lower().startswith("grafted_shampoo")


def facet_lines_exp_d_tk(
	df_traj: pd.DataFrame,
	experiment_col: str,
	title: str,
	filename: str,
	plot_dir: str,
	sig_bracket_step_frac: float,
	sig_bracket_ylim_pad_frac: float,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	labels: ExperimentDisplayLabels,
	figsize_per: tuple[float, float] = (8, 5),
) -> None:
	"""Per experiment: d(t_k) vs iteration; color by neuron_id; one line per period_id."""
	if df_traj.empty:
		return
	exps = sort_experiment_names(df_traj[experiment_col].unique())
	if not exps:
		return

	def _one(ax, exp: str) -> None:
		sub = df_traj[df_traj[experiment_col] == exp].copy()
		if sub.empty:
			ax.set_visible(False)
			return

		rows = []
		for _, row in sub.iterrows():
			if pd.isna(row.get("period_id")):
				continue
			nid = str(row["neuron_id"])
			pid = str(row["period_id"])
			d_tk = np.asarray(row["d_tk"], dtype=np.float64).ravel()
			it = np.asarray(row["sampled_iteration"], dtype=np.float64).ravel()
			m = int(min(d_tk.size, it.size))
			if m < 2:
				continue
			d_tk, it = d_tk[:m], it[:m]
			mask = np.isfinite(d_tk) & np.isfinite(it)
			if not np.any(mask):
				continue
			rows.append(
				pd.DataFrame(
					{
						"iteration": it[mask],
						"d_tk": d_tk[mask],
						"neuron_id": nid,
						"period_id": pid,
					}
				)
			)

		if not rows:
			ax.set_visible(False)
			return

		flat = pd.concat(rows, ignore_index=True)

		sns.lineplot(
			data=flat,
			x="iteration",
			y="d_tk",
			hue="neuron_id",
			units="period_id",
			estimator=None,
			marker=".",
			markers=True,
			dashes=False,
			markersize=7,
			linewidth=0.9,
			alpha=0.7,
			legend=False,
			ax=ax,
		)
		ax.set_xlabel("iteration")
		ax.set_ylabel(r"$d(t_k)$")
		ax.set_xlim(0, None)
		ax.set_title(str(exp))

	row1, row2, other = _two_row_cat12_for_labels(exps, labels)
	use_two_rows = bool(row1 and row2) and not other
	if use_two_rows:
		ncols = max(len(row1), len(row2))
		fig, axes = plt.subplots(2, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * 2), squeeze=False)
		for j, exp in enumerate(row1):
			_one(axes[0, j], exp)
		for j in range(len(row1), ncols):
			axes[0, j].set_visible(False)
		for j, exp in enumerate(row2):
			_one(axes[1, j], exp)
		for j in range(len(row2), ncols):
			axes[1, j].set_visible(False)
	else:
		ncols = min(3, len(exps))
		nrows = int(np.ceil(len(exps) / ncols))
		fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False)
		for ax, exp in zip(np.ravel(axes), exps):
			_one(ax, exp)
		for ax in np.ravel(axes)[len(exps) :]:
			ax.set_visible(False)
	fig.suptitle(title, y=1.02, fontsize=14)
	plt.tight_layout()
	plt.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	plt.show()


def _norm_nid_pid_key(neuron_id: Any, period_id: Any) -> tuple[str, Any]:
	"""Stable (neuron_id, period_id) key for matching user highlights to rows."""
	nid = str(neuron_id)
	if pd.isna(period_id):
		return nid, None
	try:
		x = np.asarray(period_id, dtype=np.float64).reshape(-1)[0]
	except (TypeError, ValueError):
		return nid, str(period_id)
	if np.isfinite(x) and float(int(x)) == float(x):
		return nid, int(x)
	return nid, float(x)


def plot_d_tk_picked_optimizer_experiment_highlights(
	df_traj: pd.DataFrame,
	optimizer_id: str,
	experiment: str,
	highlights_nid_pid: Iterable[tuple[Any, Any]],
	title: str,
	filename: str,
	plot_dir: str,
	sig_bracket_step_frac: float,
	sig_bracket_ylim_pad_frac: float,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	*,
	trace_alpha_faint: float = 0.2,
	trace_alpha_highlight: float = 1.0,
	figsize: tuple[float, float] = (14, 6),
) -> None:
	"""All $d(t_k)$ traces for one (optimizer, experiment); faint vs highlighted by (neuron_id, period_id)."""
	if df_traj.empty:
		return
	sub = df_traj[
		(df_traj["optimizer_id"] == optimizer_id) & (df_traj["experiment"] == experiment)
	].copy()
	if sub.empty:
		print(f"plot_d_tk_picked...: no rows for optimizer_id={optimizer_id!r} experiment={experiment!r}")
		return

	hl_set = {_norm_nid_pid_key(n, p) for n, p in highlights_nid_pid}

	nids = sorted({str(x) for x in sub["neuron_id"].unique()})
	pal = sns.color_palette("tab10", n_colors=max(10, len(nids)))
	color_by_nid = {nid: pal[(i+3) % len(pal)] for i, nid in enumerate(nids)}

	fig, ax = plt.subplots(figsize=figsize)

	for _, row in sub.iterrows():
		if pd.isna(row.get("period_id")):
			continue
		nid = str(row["neuron_id"])
		pid_raw = row["period_id"]
		key = _norm_nid_pid_key(nid, pid_raw)
		d_tk = np.asarray(row["d_tk"], dtype=np.float64).ravel()
		it = np.asarray(row["sampled_iteration"], dtype=np.float64).ravel()
		m = int(min(d_tk.size, it.size))
		if m < 2:
			continue
		d_tk, it = d_tk[:m], it[:m]
		mask = np.isfinite(d_tk) & np.isfinite(it)
		if not np.any(mask):
			continue
		it, d_tk = it[mask], d_tk[mask]
		is_highlighted = key in hl_set
		alpha = trace_alpha_highlight if is_highlighted else trace_alpha_faint
		z = 3 if is_highlighted else 1
		lw = 1.35 if is_highlighted else 0.9
		ax.plot(
			it,
			d_tk,
			marker=".",
			markersize=7 if is_highlighted else 5,
			linewidth=lw,
			alpha=alpha,
			color=color_by_nid.get(nid, (0.4, 0.4, 0.4)),
			zorder=z,
		)

	ax.set_xlabel("iteration")
	ax.set_ylabel(r"$d(t_k)$")
	ax.set_xlim(0, None)
	ax.set_title(title)
	# fig.suptitle(title, y=1.02, fontsize=14)
	plt.tight_layout()
	plt.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	plt.show()

# --- long-form builders ---

def build_df_long(
	all_results: dict[tuple[str, str, str, str], dict],
	rename_fn: Callable[[str], str],
) -> pd.DataFrame:
	"""Long-form table for plotting (one row per period-level metric sample)."""
	rows_plot: list[dict] = []
	for (eid, mid, oid, rid), v in all_results.items():
		if v.get("error") or "periods_df" not in v:
			print("Error for", eid, mid, oid, rid)
			continue
		periods_df = v["periods_df"]
		if periods_df is None or periods_df.empty:
			continue
		periods_df = periods_df.copy()
		periods_df["experiment_id"] = eid
		periods_df["optimizer_id"] = oid
		periods_df["experiment"] = periods_df["experiment_id"].map(rename_fn)
		for col in [
			"total_score",
			"angle_score",
			"length_score",
			"n_partial_decay_score",
			"n_active_decay_score",
			"robustness_score",
			"repr_trajectory_dispersion",
			"total_length_iter",
			"activation_angle",
			"point_of_max_acceleration",
		]:
			if col not in periods_df.columns:
				continue
			sub = periods_df[["experiment_id", "experiment", "optimizer_id", col]].dropna(subset=[col])
			sub = sub.rename(columns={col: "value"})
			sub["metric"] = col
			rows_plot.extend(sub.to_dict("records"))
	return pd.DataFrame(rows_plot)


def build_df_counts(
	all_results: dict[tuple[str, str, str, str], dict],
	rename_fn: Callable[[str], str],
) -> pd.DataFrame:
	"""Run-level period counts (one sample per run for testing)."""
	count_rows: list[dict] = []
	for (eid, mid, oid, rid), v in all_results.items():
		if v.get("error"):
			continue
		count_rows.append(
			{
				"experiment_id": eid,
				"experiment": rename_fn(eid),
				"optimizer_id": oid,
				"n_periods": int(v.get("n_periods", 0)),
				"run_id": rid,
				"model_id": mid,
			}
		)
	return pd.DataFrame(count_rows)

def _experiment_id_str(eid) -> str:
	return str(eid)

def _is_cat2_sequence_experiment(eid) -> bool:
	return _experiment_id_str(eid).startswith("cat2_sequence_")

def _length_iter_pool_tag(
	df: pd.DataFrame,
	labels: ExperimentDisplayLabels,
) -> pd.Series:
	"""Tags for pooled period-length rows: recover_reinforce, shuffle, nopt_control (mutually exclusive)."""
	e = df["experiment_id"].map(_experiment_id_str)
	disp = df["experiment"].astype(str)
	rec_re = (
		e.str.contains("cat2_sequence_recover")
		| e.str.contains("cat2_sequence_reinforce")
		| disp.isin(["Recover", "Reinforce"])
	)
	shuffle = (
		(
			e.map(is_pretrain_shuffle_mislabel_experiment)
			& ~e.str.startswith("cat1_sample_shuffle_control_tr")
		)
		| e.str.contains("cat2_sequence_labelperm")
		| e.str.startswith("cat2_sequence_pretrain_control_")
	) & ~rec_re
	nopt_control = (
		e.str.startswith("cat1_sample_shuffle_control_tr")
		| e.str.startswith("cat2_sequence_control_tr")
		| e.str.startswith("digit_")
		| disp.isin(
			[
				labels.no_pretrain,
				labels.shuffle_nopre,
				labels.seq_nopre,
				"Control",
				"Control (no pretrain)",
				"Control (✗PT)",  # legacy display name
				"Control Shuffle (✗PT)",  # legacy display name
			]
		)
	) & ~rec_re & ~shuffle
	return pd.Series(
		np.select(
			[rec_re.to_numpy(), shuffle.to_numpy(), nopt_control.to_numpy()],
			["recover_reinforce", "shuffle", "nopt_control"],
			default="",
		),
		index=df.index,
		dtype=object,
	)

def _length_pooled_limits(
	sub: pd.DataFrame,
	value_col: str,
	order: list[str],
	*,
	sig_bracket_ylim_pad_frac: float,
) -> tuple[float, float, float]:
	whisk_glob = _length_whisker_glob(sub, value_col)
	span = whisk_glob if whisk_glob > 0 else 1.0
	y_main_top = whisk_glob + 0.08 * span
	final_hi = _length_final_hi_for_subplot(
		y_main_top, span, sub, value_col, order, sig_bracket_ylim_pad_frac=sig_bracket_ylim_pad_frac
	)
	return float(final_hi), float(y_main_top), float(span)


def length_pooled_one(
	pool_part: pd.DataFrame,
	title: str,
	filename: str,
	*,
	plot_dir: str,
	length_ylabel: str,
	sig_bracket_step_frac: float,
	sig_bracket_ylim_pad_frac: float,
	figsize: tuple[float, float] = (9.5, 5.2),
) -> None:
	if pool_part.empty or pool_part["optimizer_id"].nunique() < 1:
		return
	order = _sort_optimizer_ids(pool_part["optimizer_id"].unique().tolist())
	if len(order) < 2:
		fig, ax = plt.subplots(figsize=(7.5, 4.5))
		ax.text(0.5, 0.5, "Not enough optimizers for pairwise tests", ha="center", va="center")
		ax.axis("off")
		ax.set_title(title)
		plt.tight_layout()
		plt.savefig(Path(plot_dir) / filename, bbox_inches="tight")
		plt.show()
		return
	fig, ax = plt.subplots(figsize=figsize)
	FINAL_HI, y_main_top, span = _length_pooled_limits(
		pool_part, "value", order, sig_bracket_ylim_pad_frac=sig_bracket_ylim_pad_frac
	)
	flier_kw = {"marker": "o", "s": 9, "alpha": 0.35, "color": "0.35", "linewidths": 0, "zorder": 3}
	n_hid = _draw_length_box_manual_fliers(ax, pool_part, "value", order, y_main_top=y_main_top, flier_kw=flier_kw)
	ax.set_xticks(np.arange(len(order)))
	ax.set_xticklabels([_opt_display(o) for o in order], rotation=18, ha="right")
	ax.set_xlabel("")
	ax.set_ylabel(length_ylabel)
	ax.set_title(title)
	_, sig = welch_pairwise_matrix(pool_part, "value", "optimizer_id", order)
	draw_sig_brackets(ax, order, sig, y_base=y_main_top + 0.02 * span, y_step=sig_bracket_step_frac * span)
	ax.set_ylim(0.0, FINAL_HI)
	if n_hid > 0:
		xc = 0.5 * float(len(order) - 1)
		_y_text = FINAL_HI - 0.15 * span
		ax.annotate(
			str(n_hid),
			xy=(xc, FINAL_HI),
			xytext=(xc, _y_text),
			textcoords="data",
			ha="center",
			va="bottom",
			fontsize=11,
			color="0.2",
			clip_on=False,
			arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.0, shrinkA=0, shrinkB=2),
		)
	plt.tight_layout()
	plt.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	plt.show()

# --- BTSP overlay ---
def _overlay_btsp_layer_group_mean_regression(
	ax: plt.Axes,
	sub: pd.DataFrame,
	order_layers: list[str],
	group_col: str = "optimizer_id",
) -> None:
	"""Linear fit with x/y mean-centered within group_col; intercept absorbs group means."""
	if sub.empty or len(order_layers) < 2:
		return
	fit = sub.copy()
	layer_pos = {ln: i for i, ln in enumerate(order_layers)}
	fit["layer_idx"] = fit["layer_name"].map(layer_pos)
	fit = fit.dropna(subset=["layer_idx"])
	if len(fit) < 2 or fit[group_col].nunique() < 1:
		return
	fit["y_dm"] = fit["n_btsp"] - fit.groupby(group_col, observed=False)["n_btsp"].transform("mean")
	fit["x_dm"] = fit["layer_idx"] - fit.groupby(group_col, observed=False)["layer_idx"].transform("mean")
	if fit["x_dm"].nunique() < 2:
		return
	slope, _, r_value, _, _ = stats.linregress(fit["x_dm"], fit["y_dm"])
	intercept = float(fit["n_btsp"].mean() - slope * fit["layer_idx"].mean())
	x_range = np.arange(0,len(order_layers),0.01, dtype=float)
	def lr(x):
		return slope * x + intercept
	regression_x_to_plot = x_range[lr(x_range) > 0]
	ax.plot(
		regression_x_to_plot,
		lr(regression_x_to_plot),
		color="0.1",
		linestyle="--",
		linewidth=2.0,
		zorder=10,
		clip_on=False,
	)
	r2 = float(r_value**2)
	ax.text(
		0.05,
		0.95,
		f"slope = {slope:+.3f} / layer\n$R^2$ = {r2:.3f}",
		transform=ax.transAxes,
		ha="left",
		va="top",
		fontsize=10,
		color="0.1",
		zorder=11,
		clip_on=False,
	)
	ax.set_ylim(-0.01, None)
def build_btsp_layer_df(
	all_results: dict[tuple[str, str, str, str], dict],
	rename_fn: Callable[[str], str],
) -> pd.DataFrame:
	"""BTSP period counts per layer (all optimizers)."""
	rows: list[dict] = []
	for (eid, mid, oid, rid), v in all_results.items():
		if v.get("error") or "periods_df" not in v:
			continue
		pdf = v["periods_df"]
		if pdf is None or pdf.empty or "layer_name" not in pdf.columns:
			continue
		pdf = pdf.copy()
		for _, row in pdf.iterrows():
			ln = row.get("layer_name")
			if pd.isna(ln) or str(ln) == "":
				continue
			rows.append(
				{
					"experiment_id": eid,
					"experiment": rename_fn(eid),
					"optimizer_id": oid,
					"layer_name": str(ln),
				}
			)
	return pd.DataFrame(rows)


def build_d_tk_trajectories_df(
	all_results: dict[tuple[str, str, str, str], dict],
	rename_fn: Callable[[str], str],
) -> pd.DataFrame:
	"""Grafted Shampoo: d(t_k) vs iteration trajectories."""
	rows: list[dict] = []
	for (eid, mid, oid, rid), v in all_results.items():
		if v.get("error") or "periods_df" not in v:
			continue
		pdf = v["periods_df"]
		if pdf is None or pdf.empty:
			continue
		if "d_tk" not in pdf.columns or "sampled_iteration" not in pdf.columns:
			continue
		if "neuron_id" not in pdf.columns:
			continue
		pdf = pdf.copy()
		pdf["experiment"] = rename_fn(eid)
		for _, row in pdf.iterrows():
			rows.append(
				{
					"optimizer_id": oid,
					"experiment": row["experiment"],
					"neuron_id": row["neuron_id"],
					"period_id": row.get("period_id"),
					"d_tk": row["d_tk"],
					"sampled_iteration": row["sampled_iteration"],
				}
			)
	return pd.DataFrame(rows)


# --- training curves ---
def _metrics_x_axis(metrics: dict) -> tuple[np.ndarray, str]:
	tags = [str(t) for t in metrics.get("checkpoint_tags", [])]
	cp_iters = metrics.get("checkpoint_iterations")
	if cp_iters is not None and len(cp_iters) == len(tags):
		return np.asarray(cp_iters, dtype=float), "iteration"
	return np.arange(len(tags), dtype=float), "checkpoint"


def _safe_slug(s: str) -> str:
	out = []
	for ch in str(s):
		if ch.isalnum():
			out.append(ch)
		elif ch in "-_":
			out.append(ch)
		else:
			out.append("_")
	return "".join(out).strip("_") or "experiment"




def _early_trials_x_max(
	metrics: dict,
	x_mode: str,
	*,
	n_trials: int,
	default_max: float,
) -> float:
	iters = metrics.get("trial_end_iterations", metrics.get("stage_end_iterations"))
	if x_mode == "iteration" and iters is not None and len(iters) >= n_trials:
		return float(iters[n_trials - 1])
	cp_idxs = metrics.get(
		"trial_end_checkpoint_idxs", metrics.get("stage_end_checkpoint_idxs")
	)
	if x_mode == "checkpoint" and cp_idxs is not None and len(cp_idxs) >= n_trials:
		return float(int(cp_idxs[n_trials - 1]))
	return default_max


def _first_digit_for_training_plot(
	eid: str,
	mid: str,
	rid: str,
	all_results: dict[tuple[str, str, str, str], dict],
) -> int | None:
	"""Base `digitA` for recover/reinforce trial labels (from cache or config.json)."""
	if not _is_cat2_recover_reinforce_experiment(eid):
		return None
	for (e, m, _oid, r), payload in all_results.items():
		if e != eid or m != mid or r != rid:
			continue
		if payload.get("error"):
			continue
		df = payload.get("periods_df")
		if df is not None and not df.empty and "digitA" in df.columns:
			return int(df["digitA"].iloc[0])
	return _parse_digit_a_from_config(eid, mid, rid)


def _plot_train_loss_on_ax(
	ax,
	eid: str,
	mid: str,
	rid: str,
	oids: list[str],
	*,
	x_max: float | None = None,
) -> tuple[dict | None, str]:
	"""Plot training loss (+ SE) on `ax`; optional `x_max` truncates the x range."""
	first_m: dict | None = None
	x_mode = "iteration"
	for oid in oids:
		m = _metrics(eid, mid, oid, rid=rid, w_cache=False)
		if not m or "loss_all_train_mean" not in m:
			continue
		if first_m is None:
			first_m = m
		xs_full, x_mode = _metrics_x_axis(m)
		mask = np.ones(xs_full.shape, dtype=bool)
		if x_max is not None:
			mask = xs_full <= x_max + 1e-9
		if not np.any(mask):
			continue
		xs = xs_full[mask]
		col = _opt_color(oid)
		lab = _opt_display(oid)
		fill = to_rgba(col, 0.22)
		y = np.asarray(m["loss_all_train_mean"], dtype=float)[mask]
		se_key = "loss_all_train_se"
		if se_key in m:
			se = np.asarray(m[se_key], dtype=float).copy()[mask]
			if se.size:
				se[0] = 0.0
			ax.fill_between(xs, y - 1.96 * se, y + 1.96 * se, color=fill, linewidth=0)
		ax.plot(xs, y, color=col, label=lab, marker=".", ms=4, lw=1.6, alpha=0.75)
	return first_m, x_mode


def plot_experiment_training_loss(
	eid: str,
	mid: str,
	rid: str,
	oids: list[str],
	*,
	rename_fn: Callable[[str], str],
	plot_dir: str,
	n_trials: int,
	default_max_iter: float,
	show: bool = True,
	all_results: dict[tuple[str, str, str, str], dict] | None = None,
) -> None:
	"""Full training loss; cat2 experiments also get an early-trials zoom panel."""
	oids = _sort_optimizer_ids(oids)
	with_zoom = _is_cat2_sequence_experiment(eid)
	n_rows = 2 if with_zoom else 1
	fig, axes = plt.subplots(n_rows, 1, figsize=(9.5, 4.2 * n_rows), squeeze=False)

	first_m, x_mode = _plot_train_loss_on_ax(axes[0, 0], eid, mid, rid, oids)
	if first_m is None:
		plt.close(fig)
		return

	first_digit = None
	if _is_cat2_sequence_experiment(eid):
		first_digit = _first_digit_for_training_plot(eid, mid, rid, all_results or {})
		add_trial_boundaries_mpl(
			axes[0, 0], first_m, x_mode, report=True, first_digit=first_digit
		)

	x_label = "Iteration" if x_mode == "iteration" else "Checkpoint"
	axes[0, 0].set_ylabel("Training loss")
	axes[0, 0].legend(loc="best", fontsize=9, framealpha=0.92)

	if with_zoom:
		x_max = _early_trials_x_max(
			first_m, x_mode, n_trials=n_trials, default_max=default_max_iter
		)
		ax_zoom = axes[1, 0]
		zoom_m, zoom_x_mode = _plot_train_loss_on_ax(
			ax_zoom, eid, mid, rid, oids, x_max=x_max
		)
		if zoom_m is not None:
			plot_x_full, _ = _metrics_x_axis(zoom_m)
			zoom_end_idx = int(np.searchsorted(plot_x_full, x_max, side="right"))
			add_trial_boundaries_mpl(
				ax_zoom,
				zoom_m,
				zoom_x_mode,
				plot_x_per_segment=plot_x_full,
				max_idx=zoom_end_idx,
				report=True,
				first_digit=first_digit,
			)
			ax_zoom.set_xlim(
				float(plot_x_full[0]) if plot_x_full.size else 0.0, x_max
			)
			ax_zoom.set_ylabel("Training loss")
			ax_zoom.set_xlabel(x_label)
			ax_zoom.legend(loc="best", fontsize=9, framealpha=0.92)
	else:
		axes[0, 0].set_xlabel(x_label)

	exp_label = rename_fn(eid)
	title = fr"{exp_label} ‒ training loss"
	if with_zoom:
		title += f" (first {n_trials} trials below)"
	fig.suptitle(title, y=1.01 if with_zoom else 1.02, fontsize=14)
	fig.tight_layout()

	fname = f"final_training_loss_{_safe_slug(eid)}.pdf"
	plt.savefig(Path(plot_dir) / fname, bbox_inches="tight")
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_experiment_test_accuracy(
	eid: str,
	mid: str,
	rid: str,
	oids: list[str],
	*,
	rename_fn: Callable[[str], str],
	plot_dir: str,
	show: bool = True,
	all_results: dict[tuple[str, str, str, str], dict] | None = None,
) -> None:
	oids = _sort_optimizer_ids(oids)
	first_m: dict | None = None
	x_mode = "iteration"

	fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.2))
	for oid in oids:
		m = _metrics(eid, mid, oid, rid=rid, w_cache=False)
		if not m or "acc_all_test" not in m:
			continue
		if first_m is None:
			first_m = m
		xs, x_mode = _metrics_x_axis(m)
		col = _opt_color(oid)
		lab = _opt_display(oid)
		y = np.asarray(m["acc_all_test"], dtype=float)
		ax.plot(xs, y, color=col, label=lab, marker=".", ms=4, lw=1.6)

	if first_m is None:
		plt.close(fig)
		return

	if _is_cat2_sequence_experiment(eid):
		first_digit = _first_digit_for_training_plot(eid, mid, rid, all_results or {})
		add_trial_boundaries_mpl(ax, first_m, x_mode, report=True, first_digit=first_digit)

	x_label = "Iteration" if x_mode == "iteration" else "Checkpoint"
	ax.set_ylabel("all_test accuracy")
	ax.set_xlabel(x_label)
	ax.set_ylim(-0.02, 1.05)
	ax.legend(loc="best", fontsize=9, framealpha=0.92)

	exp_label = rename_fn(eid)
	fig.suptitle(fr"{exp_label} ‒ all_test accuracy", y=1.02, fontsize=14)
	fig.tight_layout()

	fname = f"final_training_accuracy_{_safe_slug(eid)}.pdf"
	plt.savefig(Path(plot_dir) / fname, bbox_inches="tight")
	if show:
		plt.show()
	else:
		plt.close(fig)


# --- period pool ---
# plot selected periods distribution per optimizer here
from models.unit_node_id import parse_unit_node_id


def _is_category_experiment(eid) -> bool:
	return is_pretrain_shuffle_mislabel_experiment(eid) or _is_cat2_sequence_experiment(eid)


def _hidden_layer_index_from_row(row) -> int:
	if hasattr(row, "layer_name") and pd.notna(getattr(row, "layer_name", None)):
		return int(str(row.layer_name).split(".")[-1]) // 2
	parsed = parse_unit_node_id(str(row.neuron_id))
	if parsed is None:
		return -1
	return int(parsed["layer_name"].split(".")[-1]) // 2


def _deep_layer_period_pool(df: pd.DataFrame) -> pd.DataFrame:
	pool = df.copy()
	pool["layer_idx"] = pool.apply(_hidden_layer_index_from_row, axis=1)
	deep = pool[pool["layer_idx"] > 0]
	return deep if not deep.empty else pool

def build_category_period_tables(
	all_results: dict[tuple[str, str, str, str], dict],
	rename_fn: Callable[[str], str],
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
	"""Pooled period table (category experiments only) and per-experiment deep-layer pools."""
	rows: list[dict] = []
	for (eid, mid, oid, rid), v in all_results.items():
		if not _is_category_experiment(eid):
			continue
		if v.get("error") or "periods_df" not in v:
			continue
		pdf = v["periods_df"]
		if pdf is None or pdf.empty:
			continue
		exp_label = rename_fn(eid)
		for _, row in pdf.iterrows():
			if pd.isna(row.get("total_score")):
				continue
			if pd.isna(row.get("trial_start_cp")) or pd.isna(row.get("assignment_within_trial")):
				continue
			if pd.isna(row.get("point_of_max_acceleration")):
				continue
			rows.append(
				{
					"experiment_id": eid,
					"experiment": exp_label,
					"model_id": mid,
					"run_id": rid,
					"optimizer_id": oid,
					"neuron_id": row["neuron_id"],
					"period_id": int(row["period_id"]),
					"layer_name": row.get("layer_name"),
					"total_score": float(row["total_score"]),
					"angle_score": float(row["angle_score"]) if pd.notna(row.get("angle_score")) else np.nan,
					"length_score": float(row["length_score"]) if pd.notna(row.get("length_score")) else np.nan,
					"robustness_score": float(row["robustness_score"])
					if pd.notna(row.get("robustness_score"))
					else np.nan,
					"minmax_normalized_robustness_scores": float(row["minmax_normalized_robustness_scores"])
					if pd.notna(row.get("minmax_normalized_robustness_scores"))
					else np.nan,
					"trial_start_cp": int(row["trial_start_cp"]),
					"assignment_within_trial": int(row["assignment_within_trial"]),
					"point_of_max_acceleration": int(row["point_of_max_acceleration"]),
				}
			)
	df_cat_periods = pd.DataFrame(rows)
	all_periods_by_experiment: dict[str, pd.DataFrame] = {}
	for exp_label in sort_experiment_names(df_cat_periods["experiment"].unique()):
		exp_sub = df_cat_periods[df_cat_periods["experiment"] == exp_label]
		pool = _deep_layer_period_pool(exp_sub).sort_values("total_score", ascending=False)
		all_periods_by_experiment[exp_label] = pool.reset_index(drop=True)
		print(
			f"{exp_label}: {len(pool)} / {len(exp_sub)} periods"
			f" ({int((pool['layer_idx'] > 0).sum())} with layer_idx>0)"
		)
	return df_cat_periods, all_periods_by_experiment


def plot_all_periods_optimizer_distribution(
	all_periods_by_experiment: dict[str, pd.DataFrame],
	*,
	plot_dir: str,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
) -> None:
	if not all_periods_by_experiment:
		print("No category-experiment periods — skip optimizer distribution plot")
		return
	exp_order = sort_experiment_names(all_periods_by_experiment.keys())
	n_exp = len(exp_order)
	fig_dist, axes_dist = plt.subplots(
		1,
		n_exp,
		figsize=(max(4.2 * n_exp, 8), 4.8),
		squeeze=False,
		sharey=True,
	)
	axes_dist = axes_dist.ravel()
	opt_order = _sort_optimizer_ids(
		pd.concat(all_periods_by_experiment.values(), ignore_index=True)["optimizer_id"].unique().tolist()
	)
	palette = [_opt_color(o) for o in opt_order]
	for ax, exp_label in zip(axes_dist, exp_order):
		sel = all_periods_by_experiment[exp_label]
		counts = sel["optimizer_id"].value_counts().reindex(opt_order, fill_value=0)
		ax.bar(
			range(len(opt_order)),
			counts.to_numpy(),
			color=palette,
			edgecolor="0.25",
			linewidth=0.55,
		)
		ax.set_xticks(range(len(opt_order)))
		ax.set_xticklabels([_opt_label(o) for o in opt_order], rotation=35, ha="right", fontsize=9)
		ax.set_title(exp_label, fontsize=11)
		ax.set_ylabel("n periods" if ax is axes_dist[0] else "")
	fig_dist.suptitle(
		"All BTSP periods per experiment (deep layers) ‒ optimizer mix",
		y=1.04,
		fontsize=13,
	)
	fig_dist.tight_layout()
	plt.savefig(Path(plot_dir) / "final_all_periods_optimizer_distribution.pdf", bbox_inches="tight")
	plt.show()


def parse_neuron_id_for_display(neuron_id: str) -> str:
	parsed = parse_unit_node_id(neuron_id)
	if parsed is None:
		return str(neuron_id)
	layer_idx = int(parsed["layer_name"].split(".")[-1]) // 2
	return r"$n^{(" + str(layer_idx) + r")}_{" + str(parsed["unit_index"]) + r"}$"


def save_efflr_gamma_rel_category_figure(
	fig,
	*,
	plot_dir: str,
	top_n_per_optimizer: int,
	category_slug: str,
) -> None:
	if fig is None:
		return
	plt.savefig(
		Path(plot_dir)
		/ f"final_top{top_n_per_optimizer}peropt_allstrategies_efflr_relgamma_{category_slug}.pdf",
		bbox_inches="tight",
	)
	plt.show()

