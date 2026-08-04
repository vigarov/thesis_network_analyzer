"""Helpers for FINAL_additional_plots.ipynb — peak ratio, BTSP neuron %, correlations."""
import gzip
import hashlib
import json
import pickle
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.signal import find_peaks

from analysis.common import _opt_color, _opt_display, _sort_optimizer_ids
from analysis.constants import RESULTS
from analysis.io import _metrics, _pp_neuron_digit
from analysis.dw_vicinity import robustness_score_column
from analysis.final.final_experiment_display import (
	ExperimentDisplayLabels,
	two_row_cat12_from_sorted,
)
from analysis.plot_helpers import add_trial_boundaries_mpl
from analysis.final.plot_dataset_context import insert_filename_suffix
from analysis.scoring_helpers import is_pretrain_shuffle_mislabel_experiment


def apply_presentation_defaults(plot_dir: str) -> None:
	sns.set_theme(context="talk", style="whitegrid", font_scale=0.95)
	mpl.rcParams["figure.dpi"] = 120
	mpl.rcParams["savefig.dpi"] = 200
	mpl.rcParams["axes.titlesize"] = 13
	Path(plot_dir).mkdir(parents=True, exist_ok=True)


def _run_checkpoint_dir(save_dir: str, eid: str, mid: str, oid: str, rid: str) -> Path:
	sha = hashlib.sha256(f"{eid}|{mid}|{oid}|{rid}".encode("utf-8")).hexdigest()[:10]
	return Path(save_dir) / sha


def load_run_checkpoint(
	save_dir: str, eid: str, mid: str, oid: str, rid: str
) -> dict[str, Any] | None:
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


def load_all_results_from_cache(
	tree: dict, save_dir: str
) -> dict[tuple[str, str, str, str], dict]:
	all_results: dict[tuple[str, str, str, str], dict] = {}
	for eid, models in sorted(tree.items()):
		for mid, runs in sorted(models.items()):
			for rid, oids in sorted(runs.items()):
				for oid in oids:
					loaded = load_run_checkpoint(save_dir, eid, mid, oid, rid)
					if loaded is not None:
						all_results[(eid, mid, oid, rid)] = loaded
	return all_results


def _experiment_id_str(eid) -> str:
	return str(eid)


def is_cat2_sequence_experiment(eid) -> bool:
	return _experiment_id_str(eid).startswith("cat2_sequence_")


def is_cat2_recover_reinforce_experiment(eid: str) -> bool:
	s = str(eid)
	return "cat2_sequence_recover" in s or "cat2_sequence_reinforce" in s


def is_category_experiment(eid) -> bool:
	return is_pretrain_shuffle_mislabel_experiment(eid) or is_cat2_sequence_experiment(eid)


def parse_digit_a_from_config(eid: str, mid: str, rid: str) -> int | None:
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


def is_control_experiment_id(
	eid: str, experiment_label: str, labels: ExperimentDisplayLabels
) -> bool:
	e = _experiment_id_str(eid)
	disp = str(experiment_label)
	if e.startswith("cat1_sample_shuffle_control_tr"):
		return True
	if e.startswith("cat2_sequence_control_tr"):
		return True
	if e.startswith("digit_"):
		return True
	return disp in (
		labels.no_pretrain,
		labels.shuffle_nopre,
		labels.seq_nopre,
		"Control",
		"Control (no pretrain)",
		"Control (✗PT)",
		"Control Shuffle (✗PT)",
	)


def safe_slug(s: str) -> str:
	out = []
	for ch in str(s):
		if ch.isalnum():
			out.append(ch)
		elif ch in "-_":
			out.append(ch)
		elif ch == "✗":
			out.append("no")
		else:
			out.append("_")
	return "".join(out).strip("_") or "experiment"


def _two_row_cat12_for_labels(
	exps: list[str], labels: ExperimentDisplayLabels
) -> tuple[list[str], list[str], list[str]]:
	return two_row_cat12_from_sorted(exps, labels)


def _sort_layer_display_names(layer_names) -> list[str]:
	names = [str(x) for x in layer_names if x is not None and str(x) not in ("nan", "")]

	def sort_key(ln: str) -> tuple[int, int | str]:
		if ln.startswith("hidden."):
			try:
				return (0, int(ln.rsplit(".", 1)[-1]))
			except ValueError:
				return (1, ln)
		return (2, ln)

	return sorted(set(names), key=sort_key)


# --- statistics: Holm-adjusted Welch pairs + bracket drawing ---


def _holm_stepdown_reject(pvals: list[float], alpha: float = 0.05) -> list[bool]:
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


def welch_pairwise_matrix(
	df: pd.DataFrame,
	value_col: str,
	group_col: str,
	order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
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


def draw_sig_brackets(
	ax,
	order: list[str],
	sig_mat: np.ndarray,
	y_base: float,
	y_step: float,
) -> None:
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
			ax.text(
				(x1 + x2) / 2.0, y + riser, "*", ha="center", va="bottom", fontsize=14, color="0.1"
			)


def _whisker_low_high(x: np.ndarray) -> tuple[float, float]:
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


def _facet_need_hi(
	sub: pd.DataFrame,
	value_col: str,
	order: list[str],
	*,
	sig_bracket_ylim_pad_frac: float,
) -> float:
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


def _fixed_ylim_for_metric(
	df: pd.DataFrame,
	value_col: str,
	experiment_col: str,
	*,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	sig_bracket_ylim_pad_frac: float,
) -> tuple[float, float]:
	score_hi = 0.0
	for exp in sort_experiment_names(df[experiment_col].unique()):
		sub = df[df[experiment_col] == exp]
		order = _sort_optimizer_ids(sub["optimizer_id"].unique().tolist())
		score_hi = max(score_hi, _facet_need_hi(sub, value_col, order, sig_bracket_ylim_pad_frac=sig_bracket_ylim_pad_frac))
	return (0.0, float(score_hi)) if score_hi > 0 else (0.0, 1.0)


def facet_box_welch_cat12(
	df: pd.DataFrame,
	value_col: str,
	experiment_col: str,
	title: str,
	ylabel: str,
	filename: str,
	*,
	plot_dir: str,
	sig_bracket_step_frac: float,
	sig_bracket_ylim_pad_frac: float,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	labels: ExperimentDisplayLabels,
	figsize_per: tuple[float, float] = (5.2, 4.2),
	fixed_ylim: tuple[float, float] | None = None,
	show: bool = True,
) -> None:
	exps = sort_experiment_names(df[experiment_col].unique())
	if not exps:
		return

	def _one(ax, exp: str, set_ylabel: bool = False) -> None:
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
			showfliers=True,
			flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
		)
		ax.set_xticks(np.arange(len(order)))
		ax.set_xticklabels([_opt_display(o) for o in order], rotation=18, ha="right")
		ax.set_xlabel("")
		if set_ylabel:
			ax.set_ylabel(ylabel)
		else:
			ax.set_ylabel("")
		ax.set_title(str(exp))
		lo_m, hi_m = ax.get_ylim()
		span_m = hi_m - lo_m if hi_m > lo_m else 1.0
		_, sig = welch_pairwise_matrix(sub, value_col, "optimizer_id", order)
		draw_sig_brackets(
			ax, order, sig, y_base=hi_m + 10 + 0.02 * span_m, y_step=sig_bracket_step_frac * span_m + 10.0
		)
		if fixed_ylim is not None:
			ax.set_ylim(fixed_ylim)
		else:
			ax.set_ylim(
				lo_m, hi_m + (sig_bracket_ylim_pad_frac * span_m) * max(1, int(np.triu(sig, 1).sum()))
			)

	row1, row2, other = _two_row_cat12_for_labels(exps, labels)
	use_two_rows = bool(row1 and row2) and not other
	if use_two_rows:
		ncols = max(len(row1), len(row2))
		fig, axes = plt.subplots(
			2, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * 2), squeeze=False
		)
		for j, exp in enumerate(row1):
			# Set ylabel only for leftmost axes in each row
			_one(axes[0, j], exp, set_ylabel=(j == 0))
		for j in range(len(row1), ncols):
			axes[0, j].set_visible(False)
		for j, exp in enumerate(row2):
			_one(axes[1, j], exp, set_ylabel=(j == 0))
		for j in range(len(row2), ncols):
			axes[1, j].set_visible(False)
		order = _sort_optimizer_ids(df["optimizer_id"].unique().tolist())
		legend_handles = [
			mpl.patches.Patch(
				facecolor=_opt_color(o),
				edgecolor="0.25",
				linewidth=0.55,
				label=_opt_display(o),
			)
			for o in order
		]
		fig.legend(
			handles=legend_handles,
			title="optimizer",
			loc="upper center",
			bbox_to_anchor=(0.70, 0.8),
			frameon=True,
			fontsize=10,
		)
	else:
		ncols = min(3, len(exps))
		nrows = int(np.ceil(len(exps) / ncols))
		fig, axes = plt.subplots(
			nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False
		)
		ax_array = np.ravel(axes)
		for j, (ax, exp) in enumerate(zip(ax_array, exps)):
			# Set ylabel only for leftmost column in each row
			row_idx = j // ncols
			col_idx = j % ncols
			_one(ax, exp, set_ylabel=(col_idx == 0))
		for ax in ax_array[len(exps) :]:
			ax.set_visible(False)
	fig.suptitle(title, y=1.02, fontsize=14)
	plt.tight_layout()
	plt.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	if show:
		plt.show()
	else:
		plt.close(fig)


def _annotate_grouped_bar_values(
	ax: plt.Axes,
	xs: Iterable[float],
	heights: Iterable[float],
	yerr: Iterable[float] | None = None,
	*,
	optimizer_indices: Iterable[int] | None = None,
	fontsize: float = 4.0,
	fmt: str = ".2f",
	position: str = "above",
	white_threshold: float = 0.3,
	optimizer_base_y_step_frac: float = 0.04,
) -> None:
	"""Bar value labels: `above` error caps, or `inside_bottom` stacked by optimizer index."""
	yerr_list = list(yerr) if yerr is not None else None
	opt_idx_list = list(optimizer_indices) if optimizer_indices is not None else None
	ymin, ymax = ax.get_ylim()
	span = max(float(ymax) - float(ymin), 1e-9)
	for i, (xi, h) in enumerate(zip(xs, heights)):
		if not np.isfinite(h):
			continue
		hv = float(h)
		if position == "inside_bottom":
			if hv <= 0.0:
				continue
			opt_idx = int(opt_idx_list[i]) if opt_idx_list is not None and i < len(opt_idx_list) else 0
			y = (opt_idx%3) * optimizer_base_y_step_frac * ymax
			# color = "white" if hv > white_threshold else "0.15"
			color = "0.05"
			va = "bottom"
			bbox = dict(facecolor='white', alpha=0.4, pad=0)
		else:
			err = float(yerr_list[i]) if yerr_list is not None and i < len(yerr_list) else 0.0
			pad = 0.012 * span
			y = hv + err + pad
			color = "0.15"
			va = "bottom"
			bbox = None
		ax.text(
			xi,
			y,
			format(hv, fmt),
			ha="center",
			va=va,
			fontsize=fontsize,
			color=color,
			clip_on=False,
			zorder=12,
			bbox=bbox,
		)


def _draw_layer_optimizer_grouped_bars(
	ax: plt.Axes,
	sub: pd.DataFrame,
	order_layers: list[str],
	order_opts: list[str],
	y_col: str,
	*,
	sem_col: str | None = None,
	ci_mult: float = 1.96,
	annotate_bars: bool = False,
	bar_label_fontsize: float = 6.0,
	bar_label_fmt: str = ".2f",
	annotate_bars_inside: bool = False,
) -> None:
	"""Grouped bars: x = layer, hue = optimizer; optional `sem_col` → ±ci_mult·SEM error bars."""
	idx = sub.set_index(["layer_name", "optimizer_id"])
	lookup = idx[y_col]
	lookup_sem = idx[sem_col] if sem_col and sem_col in sub.columns else None
	n_layer = len(order_layers)
	n_opt = len(order_opts)
	x = np.arange(n_layer, dtype=float)
	group_w = 0.8
	bar_w = group_w / max(n_opt, 1) * 0.92
	all_xs: list[float] = []
	all_heights: list[float] = []
	all_yerr: list[float] = []
	all_optimizer_indices: list[int] = []
	for oi, opt in enumerate(order_opts):
		heights: list[float] = []
		yerr: list[float] = []
		xs: list[float] = []
		for li, layer in enumerate(order_layers):
			try:
				h = float(lookup.loc[(layer, opt)])
			except KeyError:
				h = 0.0
			heights.append(h)
			if lookup_sem is not None:
				try:
					sem = float(lookup_sem.loc[(layer, opt)])
				except KeyError:
					sem = 0.0
				yerr.append(ci_mult * sem)
			xs.append(x[li] + (oi - 0.5 * (n_opt - 1)) * bar_w)
			all_optimizer_indices.append(oi)
		bar_kw: dict = dict(
			width=bar_w,
			color=_opt_color(opt),
			edgecolor="0.25",
			linewidth=0.55,
		)
		if yerr:
			bar_kw["yerr"] = yerr
			bar_kw["capsize"] = 2.5
			bar_kw["error_kw"] = {"elinewidth": 0.9, "ecolor": "0.2", "capthick": 0.9}
		ax.bar(xs, heights, **bar_kw)
		all_xs.extend(xs)
		all_heights.extend(heights)
		all_yerr.extend(yerr)
	if annotate_bars:
		_annotate_grouped_bar_values(
			ax,
			all_xs,
			all_heights,
			all_yerr if all_yerr else None,
			optimizer_indices=all_optimizer_indices,
			fontsize=bar_label_fontsize,
			fmt=bar_label_fmt,
			position="inside_bottom" if annotate_bars_inside else "above",
		)
	ax.set_xticks(x)
	ax.set_xticklabels(order_layers)


def _plot_layer_optimizer_bars_on_ax(
	ax: plt.Axes,
	sub: pd.DataFrame,
	order_layers: list[str],
	order_opts: list[str],
	y_col: str,
	*,
	sem_col: str | None = None,
	ci_mult: float = 1.96,
	annotate_bars: bool = False,
	bar_label_fontsize: float = 6.0,
	bar_label_fmt: str = ".2f",
	annotate_bars_inside: bool = False,
) -> None:
	if sem_col and sem_col in sub.columns:
		_draw_layer_optimizer_grouped_bars(
			ax,
			sub,
			order_layers,
			order_opts,
			y_col,
			sem_col=sem_col,
			ci_mult=ci_mult,
			annotate_bars=annotate_bars,
			bar_label_fontsize=bar_label_fontsize,
			bar_label_fmt=bar_label_fmt,
			annotate_bars_inside=annotate_bars_inside,
		)
	else:
		palette = [_opt_color(o) for o in order_opts]
		sns.barplot(
			data=sub,
			x="layer_name",
			y=y_col,
			hue="optimizer_id",
			order=order_layers,
			hue_order=order_opts,
			palette=palette,
			ax=ax,
			edgecolor="0.25",
			linewidth=0.55,
			legend=False,
		)


def facet_bar_by_experiment_cat12(
	agg: pd.DataFrame,
	experiment_col: str,
	y_col: str,
	title: str,
	ylabel: str,
	filename: str,
	*,
	plot_dir: str,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	labels: ExperimentDisplayLabels,
	figsize_per: tuple[float, float] = (5.2, 4.2),
	show: bool = True,
	save: bool = True,
	sem_col: str | None = None,
	ci_mult: float = 1.96,
	annotate_bars: bool = False,
	bar_label_fontsize: float = 6.0,
	bar_label_fmt: str = ".2f",
	annotate_bars_inside: bool = False,
) -> tuple[plt.Figure | None, np.ndarray | None, list[tuple[plt.Axes, str]]]:
	if agg.empty or y_col not in agg.columns:
		return None, None, []
	exps = sort_experiment_names(agg[experiment_col].unique())
	if not exps:
		return None, None, []
	plotted_axes: list[tuple[plt.Axes, str]] = []

	def _one(ax, exp: str, set_ylabel: bool = False) -> None:
		sub = agg[agg[experiment_col] == exp].copy()
		if sub.empty:
			ax.set_visible(False)
			return
		order_layers = _sort_layer_display_names(sub["layer_name"].unique())
		order_opts = _sort_optimizer_ids(sub["optimizer_id"].unique().tolist())
		if not order_layers:
			ax.set_visible(False)
			return
		_plot_layer_optimizer_bars_on_ax(
			ax,
			sub,
			order_layers,
			order_opts,
			y_col,
			sem_col=sem_col,
			ci_mult=ci_mult,
			annotate_bars=annotate_bars,
			bar_label_fontsize=bar_label_fontsize,
			bar_label_fmt=bar_label_fmt,
			annotate_bars_inside=annotate_bars_inside,
		)
		ax.set_xlabel("layer")
		if set_ylabel:
			ax.set_ylabel(ylabel, fontsize=10)
		else:
			ax.set_ylabel("")
		ax.set_title(str(exp))
		if annotate_bars and not annotate_bars_inside:
			_bot, top = ax.get_ylim()
			ax.set_ylim(_bot, top * 1.08 if top > 0 else top + 0.5)
		# ax.tick_params(axis="x", rotation=22)
		plotted_axes.append((ax, str(exp)))

	row1, row2, other = _two_row_cat12_for_labels(exps, labels)
	use_cat12_rows = bool(row1 and row2)
	if use_cat12_rows:
		ncols = max(len(row1), len(row2), 1)
		n_extra_rows = int(np.ceil(len(other) / ncols)) if other else 0
		nrows = 2 + n_extra_rows
		fig, axes = plt.subplots(
			nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False
		)
		for j, exp in enumerate(row1):
			_one(axes[0, j], exp, set_ylabel=(j == 0))
		for j in range(len(row1), ncols):
			axes[0, j].set_visible(False)
		for j, exp in enumerate(row2):
			_one(axes[1, j], exp, set_ylabel=(j == 0))
		for j in range(len(row2), ncols):
			axes[1, j].set_visible(False)
		for k, exp in enumerate(other):
			r, c = divmod(k, ncols)
			_one(axes[2 + r, c], exp, set_ylabel=(c == 0))
		for k in range(len(other), n_extra_rows * ncols):
			r, c = divmod(k, ncols)
			axes[2 + r, c].set_visible(False)
	else:
		ncols = min(3, len(exps))
		nrows = int(np.ceil(len(exps) / ncols))
		fig, axes = plt.subplots(
			nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False
		)
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
	fig.suptitle(title, fontsize=14)
	plt.tight_layout()
	if save:
		fig.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	if show:
		plt.show()
	return fig, axes, plotted_axes


def facet_bar_by_experiment_row(
	agg: pd.DataFrame,
	experiment_col: str,
	experiments: Iterable[str],
	y_col: str,
	title: str,
	ylabel: str,
	filename: str,
	*,
	plot_dir: str,
	figsize_per: tuple[float, float] = (5.2, 4.2),
	show: bool = True,
	save: bool = True,
	sem_col: str | None = None,
	ci_mult: float = 1.96,
	annotate_bars: bool = False,
	bar_label_fontsize: float = 6.0,
	bar_label_fmt: str = ".2f",
	annotate_bars_inside: bool = False,
) -> tuple[plt.Figure | None, np.ndarray | None, list[tuple[plt.Axes, str]]]:
	"""One row of subplots, fixed experiment order (e.g. Shuffle_seq, Recover, Reinforce)."""
	if agg.empty or y_col not in agg.columns:
		return None, None, []
	exps = [str(e) for e in experiments]
	exps = [e for e in exps if e in set(agg[experiment_col].astype(str))]
	if not exps:
		return None, None, []
	plotted_axes: list[tuple[plt.Axes, str]] = []
	ncols = len(exps)
	fig, axes = plt.subplots(
		1, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1]), squeeze=False
	)

	def _one(ax, exp: str, set_ylabel: bool = False) -> None:
		sub = agg[agg[experiment_col].astype(str) == exp].copy()
		if sub.empty:
			ax.set_visible(False)
			return
		order_layers = _sort_layer_display_names(sub["layer_name"].unique())
		order_opts = _sort_optimizer_ids(sub["optimizer_id"].unique().tolist())
		if not order_layers:
			ax.set_visible(False)
			return
		_plot_layer_optimizer_bars_on_ax(
			ax,
			sub,
			order_layers,
			order_opts,
			y_col,
			sem_col=sem_col,
			ci_mult=ci_mult,
			annotate_bars=annotate_bars,
			bar_label_fontsize=bar_label_fontsize,
			bar_label_fmt=bar_label_fmt,
			annotate_bars_inside=annotate_bars_inside,
		)
		ax.set_xlabel("layer")
		if set_ylabel:
			ax.set_ylabel(ylabel, fontsize=10)
		else:
			ax.set_ylabel("")
		ax.set_title(str(exp))
		if annotate_bars and not annotate_bars_inside:
			_bot, top = ax.get_ylim()
			ax.set_ylim(_bot, top * 1.08 if top > 0 else top + 0.5)
		plotted_axes.append((ax, str(exp)))

	for j, exp in enumerate(exps):
		_one(axes[0, j], exp, set_ylabel=(j == 0))

	legend_opts = _sort_optimizer_ids(agg["optimizer_id"].unique().tolist())
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
		ncols=len(legend_opts),
		handles=legend_handles,
		title="optimizer",
		loc="center right",
		bbox_to_anchor=(1.0, 0.90),
		frameon=False,
		fontsize=9,
		title_fontsize=10,
	)
	fig.suptitle(title, fontsize=14)
	plt.tight_layout()
	if save:
		fig.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	if show:
		plt.show()
	return fig, axes, plotted_axes


def fill_layer_optimizer_grid(
	sub: pd.DataFrame,
	order_layers: list[str],
	order_opts: list[str],
	value_col: str,
	*,
	fill_value: float = 0.0,
) -> pd.DataFrame:
	"""Every (layer, optimizer) pair exists; missing `value_col` defaults to `fill_value`."""
	if not order_layers or not order_opts:
		return sub
	work = sub.groupby(["layer_name", "optimizer_id"], as_index=False, observed=False)[
		value_col
	].sum()
	idx = pd.MultiIndex.from_product(
		[order_layers, order_opts], names=["layer_name", "optimizer_id"]
	)
	filled = (
		work.set_index(["layer_name", "optimizer_id"])[value_col]
		.reindex(idx, fill_value=fill_value)
		.reset_index()
	)
	for col in sub.columns:
		if col in filled.columns or col in ("layer_name", "optimizer_id", value_col):
			continue
		vals = sub[col].dropna()
		filled[col] = vals.iloc[0] if len(vals) else None
	return filled


def fill_btsp_agg_per_experiment(
	agg: pd.DataFrame,
	*,
	experiment_col: str = "experiment",
	value_col: str = "n_btsp",
) -> pd.DataFrame:
	"""Zero-fill missing (experiment, layer, optimizer) rows before BTSP bar/regression plots."""
	if agg.empty or value_col not in agg.columns:
		return agg
	canonical_layers = _sort_layer_display_names(agg["layer_name"].unique())
	canonical_opts = _sort_optimizer_ids(agg["optimizer_id"].unique().tolist())
	parts: list[pd.DataFrame] = []
	for exp in agg[experiment_col].astype(str).unique():
		sub = agg[agg[experiment_col].astype(str) == exp]
		parts.append(
			fill_layer_optimizer_grid(sub, canonical_layers, canonical_opts, value_col)
		)
	return pd.concat(parts, ignore_index=True)


def _layer_optimizer_series_for_regression(
	sub: pd.DataFrame,
	order_layers: list[str],
	oid: str,
	value_col: str,
	group_col: str = "optimizer_id",
	*,
	fill_missing_layers: bool = True,
) -> pd.DataFrame:
	"""One row per layer; missing layers use 0 only when `fill_missing_layers`."""
	layer_pos = {ln: i for i, ln in enumerate(order_layers)}
	g = sub[sub[group_col] == oid]
	rows = []
	for ln in order_layers:
		match = g[g["layer_name"] == ln]
		if len(match):
			y = float(match[value_col].iloc[0])
		elif fill_missing_layers:
			y = 0.0
		else:
			continue
		rows.append({"layer_name": ln, "layer_idx": layer_pos[ln], value_col: y})
	return pd.DataFrame(rows)


def _linregress_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
	slope, intercept, r_value, _, _ = stats.linregress(x, y)
	if np.isfinite(r_value):
		r2 = float(r_value**2)
	elif len(y) > 0 and np.allclose(y, y[0]):
		r2 = 1.0
	else:
		r2 = float("nan")
	return float(slope), float(intercept), r2, float(r_value) if np.isfinite(r_value) else float("nan")


def overlay_layer_group_mean_regression(
	ax: plt.Axes,
	sub: pd.DataFrame,
	order_layers: list[str],
	value_col: str,
	group_col: str = "optimizer_id",
	*,
	fill_missing_layers: bool = True,
) -> None:
	if sub.empty or len(order_layers) < 2:
		return
	if fill_missing_layers:
		fit = fill_layer_optimizer_grid(
			sub,
			order_layers,
			_sort_optimizer_ids(sub[group_col].unique().tolist()),
			value_col,
		)
	else:
		fit = (
			sub.groupby(["layer_name", group_col], as_index=False, observed=False)[value_col]
			.mean()
			.copy()
		)
	layer_pos = {ln: i for i, ln in enumerate(order_layers)}
	fit["layer_idx"] = fit["layer_name"].map(layer_pos)
	fit = fit.dropna(subset=["layer_idx"])
	if len(fit) < 2 or fit[group_col].nunique() < 1:
		return
	fit["y_dm"] = fit[value_col] - fit.groupby(group_col, observed=False)[value_col].transform(
		"mean"
	)
	fit["x_dm"] = fit["layer_idx"] - fit.groupby(group_col, observed=False)["layer_idx"].transform(
		"mean"
	)
	if fit["x_dm"].nunique() < 2:
		return
	slope, _, r_value, _, _ = stats.linregress(fit["x_dm"], fit["y_dm"])
	intercept = float(fit[value_col].mean() - slope * fit["layer_idx"].mean())
	x_range = np.arange(0, len(order_layers), 0.01, dtype=float)

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
	def parse_ln(i):
		return f"Hidden {i}"
	ax.set_xticks(range(5))
	ax.set_xticklabels([parse_ln(i) for i in range(5)],fontsize=10, rotation = 0)
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


def overlay_layer_per_optimizer_regression(
	ax: plt.Axes,
	sub: pd.DataFrame,
	order_layers: list[str],
	value_col: str,
	group_col: str = "optimizer_id",
	order_opts: list[str] | None = None,
	*,
	fill_missing_layers: bool = True,
) -> dict[str, tuple[float, float]]:
	"""One linear fit per optimizer (layer index vs value); returns oid -> (slope, R²).

	When `fill_missing_layers` is True, missing layers count as 0 (BTSP plots).
	When False, only layers present in `sub` are used (e.g. score-per-layer plots).
	"""
	if sub.empty or len(order_layers) < 2:
		return {}
	opts = order_opts or _sort_optimizer_ids(sub[group_col].unique().tolist())
	x_range = np.arange(0, len(order_layers), 0.01, dtype=float)
	out: dict[str, tuple[float, float]] = {}
	for oid in _sort_optimizer_ids(opts):
		g = _layer_optimizer_series_for_regression(
			sub,
			order_layers,
			oid,
			value_col,
			group_col,
			fill_missing_layers=fill_missing_layers,
		)
		if len(g) < 2 or g["layer_idx"].nunique() < 2:
			continue
		slope, intercept, r2, _r_value = _linregress_r2(
			g["layer_idx"].to_numpy(dtype=float), g[value_col].to_numpy(dtype=float)
		)

		def lr(x: np.ndarray) -> np.ndarray:
			return slope * x + intercept

		_, top_y = ax.get_ylim()
		regression_x_to_plot = x_range[(lr(x_range) > 0) & (lr(x_range) < top_y)]
		if len(regression_x_to_plot) > 0:
			ax.plot(
				regression_x_to_plot,
				lr(regression_x_to_plot),
				color=_opt_color(oid),
				linestyle="-",
				linewidth=1.6,
				zorder=9,
				clip_on=False,
				alpha=0.4,
			)
		out[oid] = (slope, r2)
	ax.set_ylim(-0.01, None)
	return out


def layer_abbrev_number(layer_name: str) -> int | None:
	"""Map `hidden.{k}` to display index `k // 2` (0, 1, …); `None` if not a hidden layer."""
	ln = str(layer_name)
	if not ln.startswith("hidden."):
		return None
	try:
		return int(ln.rsplit(".", 1)[-1]) // 2
	except ValueError:
		return None


def cat2_shuffle_recover_reinforce_labels(labels: ExperimentDisplayLabels) -> list[str]:
	return [labels.shuffle_seq, "Recover", "Reinforce"]


def add_btsp_period_pct_within_optimizer(agg: pd.DataFrame) -> pd.DataFrame:
	"""Share of BTSP periods per layer within each (experiment, optimizer), in percent."""
	out = agg.copy()
	totals = out.groupby(["experiment", "optimizer_id"], observed=False)["n_btsp"].transform("sum")
	out["pct_btsp_periods"] = np.where(totals > 0, 100.0 * out["n_btsp"] / totals, 0.0)
	return out


def _sem_across_values(s: pd.Series) -> float:
	v = pd.to_numeric(s, errors="coerce").dropna()
	n = len(v)
	if n <= 1:
		return 0.0
	return float(v.std(ddof=1) / np.sqrt(n))


def aggregate_layer_metric_mean_sem(
	df: pd.DataFrame,
	group_cols: list[str],
	value_col: str,
) -> pd.DataFrame:
	"""Mean and SEM of `value_col` per `group_cols` (e.g. experiment × layer × optimizer)."""
	if df.empty or value_col not in df.columns:
		return pd.DataFrame(columns=[*group_cols, value_col, "sem"])
	return (
		df.groupby(group_cols, observed=False)[value_col]
		.agg(mean="mean", sem=_sem_across_values)
		.reset_index()
		.rename(columns={"mean": value_col})
	)


def pool_btsp_mean_over_experiments(
	agg: pd.DataFrame,
	experiment_col: str,
	pool_experiments: Iterable[str],
	value_col: str,
) -> pd.DataFrame:
	"""Mean and SEM of `value_col` across listed experiments, per (layer, optimizer)."""
	pool = [str(e) for e in pool_experiments]
	sub = agg[agg[experiment_col].astype(str).isin(pool)]
	if sub.empty:
		return pd.DataFrame(columns=["layer_name", "optimizer_id", value_col, "sem"])
	grouped = sub.groupby(["layer_name", "optimizer_id"], observed=False)[value_col]
	return grouped.agg(mean="mean", sem=_sem_across_values).reset_index().rename(
		columns={"mean": value_col}
	)


def plot_btsp_pooled_optimizer_layer_bars(
	agg: pd.DataFrame,
	value_col: str,
	ylabel: str,
	title: str,
	filename: str,
	*,
	plot_dir: str,
	show: bool = True,
	save: bool = True,
	figsize: tuple[float, float] = (11.0, 5.0),
	annotate_bars: bool = False,
	bar_label_fontsize: float = 6.0,
	bar_label_fmt: str = ".2f",
	annotate_bars_inside: bool = False,
) -> plt.Figure | None:
	"""Single panel: major x = optimizer; grouped bars (left→right) = layers 0…4."""
	if agg.empty or value_col not in agg.columns:
		return None
	order_opts = _sort_optimizer_ids(agg["optimizer_id"].unique().tolist())
	order_layers = _sort_layer_display_names(agg["layer_name"].unique())
	if not order_opts or not order_layers:
		return None

	idx = agg.set_index(["optimizer_id", "layer_name"])
	lookup = idx[value_col]
	has_sem = "sem" in agg.columns
	lookup_sem = idx["sem"] if has_sem else None
	n_opt = len(order_opts)
	n_layer = len(order_layers)
	group_centers = np.arange(n_opt, dtype=float)
	group_w = 0.82
	bar_w = group_w / max(n_layer, 1) * 0.92
	ci_mult = 1.96

	fig, ax = plt.subplots(figsize=figsize)
	layer_colors = [plt.cm.plasma(i / max(n_layer - 1, 1)) for i in range(n_layer)]
	all_xs: list[float] = []
	all_heights: list[float] = []
	all_yerr: list[float] = []
	all_optimizer_indices: list[int] = []
	for li, layer in enumerate(order_layers):
		heights: list[float] = []
		yerr: list[float] = []
		xs: list[float] = []
		lab = layer_abbrev_number(layer)
		for oi, opt in enumerate(order_opts):
			try:
				h = float(lookup.loc[(opt, layer)])
			except KeyError:
				h = 0.0
			heights.append(h)
			if lookup_sem is not None:
				try:
					sem = float(lookup_sem.loc[(opt, layer)])
				except KeyError:
					sem = 0.0
				yerr.append(ci_mult * sem)
			xs.append(group_centers[oi] + (li - 0.5 * (n_layer - 1)) * bar_w)
			all_optimizer_indices.append(oi)
		bar_kw: dict = dict(
			width=bar_w,
			color=layer_colors[li],
			edgecolor="0.25",
			linewidth=0.55,
			label=str(lab if lab is not None else layer),
		)
		if yerr:
			bar_kw["yerr"] = yerr
			bar_kw["capsize"] = 2.5
			bar_kw["error_kw"] = {"elinewidth": 0.9, "ecolor": "0.2", "capthick": 0.9}
		ax.bar(xs, heights, **bar_kw)
		all_xs.extend(xs)
		all_heights.extend(heights)
		all_yerr.extend(yerr)
	if annotate_bars:
		_annotate_grouped_bar_values(
			ax,
			all_xs,
			all_heights,
			all_yerr if all_yerr else None,
			optimizer_indices=all_optimizer_indices,
			fontsize=bar_label_fontsize,
			fmt=bar_label_fmt,
			position="inside_bottom" if annotate_bars_inside else "above",
		)
	ax.set_xticks(group_centers)
	ax.set_xticklabels([_opt_display(o) for o in order_opts], rotation=22, ha="right")
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.legend(title="layer", loc="best", fontsize=9, ncol=min(n_layer, 5))
	if annotate_bars and not annotate_bars_inside:
		_bot, top = ax.get_ylim()
		ax.set_ylim(_bot, top * 1.08 if top > 0 else top + 0.5)
	plt.tight_layout()
	if save:
		fig.savefig(Path(plot_dir) / filename, bbox_inches="tight")
	if show:
		plt.show(fig)
	return fig


def _is_cat2_shuffle_recover_reinforce_exp(exp: str) -> bool:
	"""True for Shuffle_seq, Recover, and Reinforce display labels."""
	if exp in ("Recover", "Reinforce"):
		return True
	s = str(exp)
	return "Shuffle" in s and "seq" in s


def _slope_at_best_r2_among_exps(
	results: dict,
	opt: str,
	exp_names: list[str],
) -> float:
	import numpy as np

	best_r2 = -np.inf
	best_slope = np.nan
	for exp in exp_names:
		if exp not in results or opt not in results[exp]:
			continue
		slope, r2 = results[exp][opt]
		if np.isnan(r2):
			continue
		if r2 > best_r2:
			best_r2 = r2
			best_slope = slope
	return float(best_slope)


def build_latex_table(results, ref_name="slopes_r2", *, best_slope_ref_name: str | None = None):
	import numpy as np

	exps = list(results.keys())
	opts = _sort_optimizer_ids(list({opt for exp in results for opt in results[exp].keys()}))

	slopes = {opt: [] for opt in opts}
	r2s = {opt: [] for opt in opts}

	for opt in opts:
		for exp in exps:
			if exp in results and opt in results[exp]:
				s, r = results[exp][opt]
			else:
				s, r = np.nan, np.nan
			slopes[opt].append(s)
			r2s[opt].append(r)

	slope_cols = np.array([slopes[opt] for opt in opts], dtype=float)
	r2_cols = np.array([r2s[opt] for opt in opts], dtype=float)

	slope_max = np.nanmax(slope_cols, axis=0)
	r2_max = np.nanmax(r2_cols, axis=0)

	def fmt(val, is_col_max, is_row_max, color):
		if np.isnan(val):
			return r"--"

		s = f"{val:+.4f}" if abs(val) > 1 else f"{val:.4f}"

		# row emphasis (colored underline)
		if is_row_max:
			s = rf"{{\setul{{-.1pt}}{{2pt}} \setulcolor{{{color}}} \ul{{{s}}} }}"

		# column emphasis (bold)
		if is_col_max:
			s = rf"\textbf{{{s}}}"

		return s

	latex = []
	latex.append(r"\begin{table}[ht]")
	latex.append(r"\centering")
	latex.append(r"\ra{1.3}")
	latex.append(r"\resizebox{0.95\textwidth}{!}{%")

	col_spec = "l " + " ".join(["cc"] * len(exps))
	latex.append(rf"\begin{{tabular}}{{{col_spec}}}")
	latex.append(r"\toprule")

	# header row 1
	header1 = [r"\textbf{Optimizer}"]
	for exp in exps:
		exp_clean = (
			exp.replace("✗", r"\xmark ")
			   .replace("✔", r"\cmark ")
			   .replace("Control", "Ctrl.")
			   .replace("sequential", "seq.")
		)
		header1.append(rf"\multicolumn{{2}}{{c}}{{\textbf{{{exp_clean}}}}}")

	latex.append(" & ".join(header1) + r" \\")
	latex.append(r"\midrule")

	# header row 2
	header2 = [""]
	for _ in exps:
		header2.append(r"\textbf{Slope} & \textbf{$R^2$}")

	latex.append(" & ".join(header2) + r" \\")
	latex.append(r"\midrule")

	# body
	for opt in opts:
		row = [rf"\textbf{{{_opt_display(opt)}}}"]

		for j, exp in enumerate(exps):
			s = slopes[opt][j]
			r = r2s[opt][j]

			s_col_max = (not np.isnan(s)) and np.isclose(s, slope_max[j])
			r_col_max = (not np.isnan(r)) and np.isclose(r, r2_max[j])

			s_row_max = (not np.isnan(s)) and np.nanmax(slopes[opt]) == s
			r_row_max = (not np.isnan(r)) and np.nanmax(r2s[opt]) == r

			s_fmt = fmt(s, s_col_max, s_row_max, "ExpPink")
			r_fmt = fmt(r, r_col_max, r_row_max, "ExpTeal")

			row.append(rf"{s_fmt} & {r_fmt}")

		latex.append(" & ".join(row) + r" \\")

	latex.append(r"\bottomrule")
	latex.append(r"\end{tabular}%")
	latex.append(r"}")

	latex.append(
		r"\caption{Bold = bigger across optimizers per column; "
		r"colored underline = bigger within optimizer row. "
		r"One emphasis independently for {\setul{-.1pt}{2pt} \setulcolor{ExpPink} \ul{slope}} and {\setul{-.1pt}{2pt} \setulcolor{ExpTeal} \ul{$R^2$}}}."
	)
	latex.append(fr"\label{{tab:{ref_name}}}")
	latex.append(r"\end{table}")

	cat2_exps = [exp for exp in exps if _is_cat2_shuffle_recover_reinforce_exp(exp)]
	best_slopes = [_slope_at_best_r2_among_exps(results, opt, cat2_exps) for opt in opts]
	best_max = np.nanmax(best_slopes) if best_slopes else np.nan

	def fmt_best_slope(val: float) -> str:
		if np.isnan(val):
			return r"--"
		s = f"{val:+.4f}" if abs(val) > 1 else f"{val:.4f}"
		if (not np.isnan(best_max)) and np.isclose(val, best_max):
			s = rf"\textbf{{{s}}}"
		return s

	latex2 = []
	latex2.append(r"\begin{table}[ht]")
	latex2.append(r"\centering")
	latex2.append(r"\ra{1.3}")
	col_spec = "c" * len(opts)
	latex2.append(rf"\begin{{tabular}}{{{col_spec}}}")
	latex2.append(r"\toprule")
	latex2.append(" & ".join(rf"\textbf{{{_opt_display(opt)}}}" for opt in opts) + r" \\")
	latex2.append(r"\midrule")
	latex2.append(
		" & ".join(fmt_best_slope(s) for s in best_slopes) + r" \\"
	)
	latex2.append(r"\bottomrule")
	latex2.append(r"\end{tabular}")
	latex2.append(
		r"\caption{Slope at the experiment with highest $R^2$ among "
		r"$\text{Shuffle}_{seq}$, Recover, and Reinforce (per optimizer). "
		r"Bold = largest across optimizers.}"
	)
	_best_label = best_slope_ref_name or f"{ref_name}_best_r2_slope"
	latex2.append(fr"\label{{tab:{_best_label}}}")
	latex2.append(r"\end{table}")

	return "\n".join(latex), "\n".join(latex2)

# --- metrics / training loss ---


def metrics_x_axis(metrics: dict) -> tuple[np.ndarray, str]:
	tags = [str(t) for t in metrics.get("checkpoint_tags", [])]
	cp_iters = metrics.get("checkpoint_iterations")
	if cp_iters is not None and len(cp_iters) == len(tags):
		return np.asarray(cp_iters, dtype=float), "iteration"
	return np.arange(len(tags), dtype=float), "checkpoint"


def early_trials_x_max(
	metrics: dict,
	x_mode: str,
	*,
	n_trials: int,
	default_max: float,
) -> float:
	iters = metrics.get("trial_end_iterations", metrics.get("stage_end_iterations"))
	if x_mode == "iteration" and iters is not None and len(iters) >= n_trials:
		return float(iters[n_trials - 1])
	cp_idxs = metrics.get("trial_end_checkpoint_idxs", metrics.get("stage_end_checkpoint_idxs"))
	if x_mode == "checkpoint" and cp_idxs is not None and len(cp_idxs) >= n_trials:
		return float(int(cp_idxs[n_trials - 1]))
	return default_max


def is_control_global_or_shuffle_all_experiment(eid: str) -> bool:
	"""True for Cat1 control-global and shuffle-all experiment ids."""
	e = _experiment_id_str(eid)
	if e.startswith("cat1_sample_shuffle_control_tr"):
		return True
	if e.startswith("cat1_sample_shuffle_finetune_"):
		return True
	if "pretrain_shuffle_mislabel" in e and "_digits" not in e:
		return True
	return False


def find_peaks_kwargs_for_experiment(
	eid: str,
	*,
	find_peaks_kwargs: dict | None = None,
	find_peaks_kwargs_control_global_shuffle_all: dict | None = None,
) -> dict:
	"""Pick find_peaks kwargs: alt set for control-global / shuffle-all, else default."""
	if is_control_global_or_shuffle_all_experiment(eid):
		return dict(find_peaks_kwargs_control_global_shuffle_all or find_peaks_kwargs or {})
	return dict(find_peaks_kwargs or {})


def find_peaks_params_label(find_peaks_kwargs: dict | None = None) -> str:
	"""Short label for plot titles from `FIND_PEAKS_KWARGS`."""
	kw = find_peaks_kwargs or {}
	parts: list[str] = []
	if "prominence" in kw and kw["prominence"] is not None:
		parts.append(f"prominence={kw['prominence']}")
	for key in ("distance", "height", "width", "threshold"):
		if key in kw and kw[key] is not None:
			parts.append(f"{key}={kw[key]}")
	return ", ".join(parts) if parts else "default"


def peak_ratios_per_peak(y: np.ndarray, find_peaks_kwargs: dict | None = None) -> np.ndarray:
	"""Per detected peak: `loss_at_peak / mean(training loss)` (one value per peak)."""
	y = np.asarray(y, dtype=float)
	y = y[np.isfinite(y)]
	if y.size == 0:
		return np.array([], dtype=float)
	mean_y = float(np.mean(y))
	if mean_y == 0.0 or not np.isfinite(mean_y):
		return np.array([], dtype=float)
	kw = dict(find_peaks_kwargs or {})
	peak_idx, _ = find_peaks(y, **kw)
	if peak_idx.size == 0:
		return np.array([], dtype=float)
	return np.asarray(y[peak_idx], dtype=float) / mean_y


def build_peak_ratio_df(
	tree: dict,
	*,
	rename_fn: Callable[[str], str],
	category_only: bool = True,
	find_peaks_kwargs: dict | None = None,
	find_peaks_kwargs_control_global_shuffle_all: dict | None = None,
) -> pd.DataFrame:
	"""One row per detected peak (not one mean per run)."""
	rows: list[dict] = []
	for eid, models in sorted(tree.items()):
		if category_only and not is_category_experiment(eid):
			continue
		fp_kw = find_peaks_kwargs_for_experiment(
			eid,
			find_peaks_kwargs=find_peaks_kwargs,
			find_peaks_kwargs_control_global_shuffle_all=find_peaks_kwargs_control_global_shuffle_all,
		)
		for mid, runs in sorted(models.items()):
			for rid, oids in sorted(runs.items()):
				for oid in oids:
					m = _metrics(eid, mid, oid, rid=rid, w_cache=False)
					if not m or "loss_all_train_mean" not in m:
						continue
					y = np.asarray(m["loss_all_train_mean"], dtype=float)
					ratios = peak_ratios_per_peak(y, fp_kw)
					for peak_i, ratio in enumerate(ratios):
						if not np.isfinite(ratio):
							continue
						rows.append(
							{
								"experiment_id": eid,
								"experiment": rename_fn(eid),
								"model_id": mid,
								"run_id": rid,
								"optimizer_id": oid,
								"peak_index": int(peak_i),
								"peak_ratio": float(ratio),
							}
						)
	return pd.DataFrame(rows)


def _peak_ratio_per_optimizer_means(
	df_peak: pd.DataFrame,
	experiment_labels: list[str],
	*,
	value_col: str = "peak_ratio",
	experiment_col: str = "experiment",
) -> pd.Series:
	"""One mean per (experiment, optimizer) when multiple experiments, else per optimizer."""
	sub = df_peak.loc[df_peak[experiment_col].isin(experiment_labels)].copy()
	sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
	sub = sub.dropna(subset=[value_col, "optimizer_id"])
	if sub.empty:
		return pd.Series(dtype=float)
	grouper = (
		["experiment", "optimizer_id"]
		if len(experiment_labels) > 1
		else ["optimizer_id"]
	)
	return sub.groupby(grouper, observed=False)[value_col].mean()


def print_peak_ratio_pooled_summaries(
	df_peak: pd.DataFrame,
	*,
	labels: ExperimentDisplayLabels,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	value_col: str = "peak_ratio",
	experiment_col: str = "experiment",
) -> None:
	"""Print pooled peak-ratio means and Welch tests vs cat2 PT and shuffle-seq controls."""
	if df_peak.empty or value_col not in df_peak.columns:
		print("peak_ratio summary: empty dataframe — skip")
		return

	def _pool_mean(mask: pd.Series) -> tuple[float, int]:
		v = pd.to_numeric(df_peak.loc[mask, value_col], errors="coerce").dropna()
		return (float(v.mean()) if len(v) else float("nan"), int(len(v)))

	exps = sort_experiment_names(df_peak[experiment_col].unique())
	row1, row2, _other = _two_row_cat12_for_labels(exps, labels)

	def _mean_for_experiments(exp_labels: list[str]) -> tuple[float, int]:
		if not exp_labels:
			return float("nan"), 0
		return _pool_mean(df_peak[experiment_col].isin(exp_labels))

	m_cat1, n_cat1 = _mean_for_experiments(row1)
	m_cat2, n_cat2 = _mean_for_experiments(row2)
	m_rec, n_rec = _pool_mean(df_peak[experiment_col] == "Recover")
	m_reinf, n_reinf = _pool_mean(df_peak[experiment_col] == "Reinforce")
	m_rr, n_rr = _pool_mean(df_peak[experiment_col].isin(["Recover", "Reinforce"]))
	m_seq_pre, n_seq_pre = _pool_mean(df_peak[experiment_col] == labels.seq_pre)
	m_shuffle_seq, n_shuffle_seq = _pool_mean(df_peak[experiment_col] == labels.shuffle_seq)

	print("--- Peak ratio pooled means (one value per detected peak) ---")
	print(
		f"Category 1 / row 1 ({len(row1)} experiments, n={n_cat1} peaks): "
		f"mean={m_cat1:.6g}"
	)
	if row1:
		print(f"  experiments: {', '.join(row1)}")
	print(
		f"Category 2 / row 2 ({len(row2)} experiments, n={n_cat2} peaks): "
		f"mean={m_cat2:.6g}"
	)
	if row2:
		print(f"  experiments: {', '.join(row2)}")
	print(f"Recover (n={n_rec} peaks): mean={m_rec:.6g}")
	print(f"Reinforce (n={n_reinf} peaks): mean={m_reinf:.6g}")
	print(f"Recover + Reinforce (n={n_rr} peaks): mean={m_rr:.6g}")
	print(f"{labels.seq_pre} (n={n_seq_pre} peaks): mean={m_seq_pre:.6g}")
	print(f"{labels.shuffle_seq} (n={n_shuffle_seq} peaks): mean={m_shuffle_seq:.6g}")

	rr_opt = _peak_ratio_per_optimizer_means(
		df_peak, ["Recover", "Reinforce"], value_col=value_col, experiment_col=experiment_col
	)
	seq_pre_opt = _peak_ratio_per_optimizer_means(
		df_peak, [labels.seq_pre], value_col=value_col, experiment_col=experiment_col
	)
	shuffle_seq_opt = _peak_ratio_per_optimizer_means(
		df_peak, [labels.shuffle_seq], value_col=value_col, experiment_col=experiment_col
	)
	print(
		"--- Welch t-test on per-optimizer means "
		"(one mean per experiment×optimizer across peaks; unequal group sizes) ---"
	)
	print(
		f"Recover+Reinforce vs {labels.seq_pre}: "
		f"p={_welch_t_pvalue(rr_opt, seq_pre_opt):.6g} "
		f"(n1={len(rr_opt)}, n2={len(seq_pre_opt)})"
	)
	print(
		f"Recover+Reinforce vs {labels.shuffle_seq}: "
		f"p={_welch_t_pvalue(rr_opt, shuffle_seq_opt):.6g} "
		f"(n1={len(rr_opt)}, n2={len(shuffle_seq_opt)})"
	)


def build_btsp_neuron_pct_rows(
	all_results: dict[tuple[str, str, str, str], dict],
	rename_fn: Callable[[str], str],
) -> pd.DataFrame:
	rows: list[dict] = []
	for (eid, mid, oid, rid), v in all_results.items():
		if v.get("error") or "periods_df" not in v:
			continue
		pdf = v["periods_df"]
		if pdf is None or pdf.empty:
			continue
		df_nd = _pp_neuron_digit(eid, mid, oid, rid=rid, w_cache=False)
		if df_nd is None or df_nd.empty or "layer_name" not in df_nd.columns:
			continue
		total_by_layer = (
			df_nd[["neuron_id", "layer_name"]]
			.drop_duplicates("neuron_id")
			.groupby("layer_name", observed=False)["neuron_id"]
			.nunique()
		)
		btsp_by_layer = (
			pdf.dropna(subset=["layer_name"])
			.groupby("layer_name", observed=False)["neuron_id"]
			.nunique()
		)
		for layer_name, n_total in total_by_layer.items():
			if pd.isna(layer_name) or str(layer_name) == "" or n_total == 0:
				continue
			n_btsp = int(btsp_by_layer.get(layer_name, 0))
			rows.append(
				{
					"experiment_id": eid,
					"experiment": rename_fn(eid),
					"optimizer_id": oid,
					"layer_name": str(layer_name),
					"pct_btsp_neurons": 100.0 * n_btsp / int(n_total),
				}
			)
	return pd.DataFrame(rows)


def build_layer_period_score_rows(
	all_results: dict[tuple[str, str, str, str], dict],
	rename_fn: Callable[[str], str],
	*,
	score_col: str = "total_score",
) -> pd.DataFrame:
	"""One row per BTSP period (for bar-plot mean ± SEM across periods in each layer)."""
	rows: list[dict] = []
	for (eid, mid, oid, rid), v in all_results.items():
		if v.get("error") or "periods_df" not in v:
			continue
		pdf = v["periods_df"]
		if pdf is None or pdf.empty or "layer_name" not in pdf.columns:
			continue
		if score_col not in pdf.columns:
			continue
		experiment = rename_fn(eid)
		for _, row in pdf.iterrows():
			layer_name = row.get("layer_name")
			if pd.isna(layer_name) or str(layer_name) in ("", "nan"):
				continue
			score = pd.to_numeric(row.get(score_col), errors="coerce")
			if pd.isna(score):
				continue
			rows.append(
				{
					"experiment_id": eid,
					"experiment": experiment,
					"model_id": mid,
					"run_id": rid,
					"optimizer_id": oid,
					"layer_name": str(layer_name),
					score_col: float(score),
				}
			)
	return pd.DataFrame(rows)


def build_layer_mean_score_rows(
	all_results: dict[tuple[str, str, str, str], dict],
	rename_fn: Callable[[str], str],
	*,
	score_col: str = "total_score",
) -> pd.DataFrame:
	"""Per-run mean `score_col` across BTSP periods in each layer (for across-layer summaries)."""
	rows: list[dict] = []
	for (eid, mid, oid, rid), v in all_results.items():
		if v.get("error") or "periods_df" not in v:
			continue
		pdf = v["periods_df"]
		if pdf is None or pdf.empty or "layer_name" not in pdf.columns:
			continue
		if score_col not in pdf.columns:
			continue
		for layer_name, grp in pdf.groupby("layer_name", observed=False):
			if pd.isna(layer_name) or str(layer_name) in ("", "nan"):
				continue
			scores = pd.to_numeric(grp[score_col], errors="coerce").dropna()
			if scores.empty:
				continue
			rows.append(
				{
					"experiment_id": eid,
					"experiment": rename_fn(eid),
					"model_id": mid,
					"run_id": rid,
					"optimizer_id": oid,
					"layer_name": str(layer_name),
					"mean_layer_score": float(scores.mean()),
				}
			)
	return pd.DataFrame(rows)


def print_mean_score_across_layers(
	df: pd.DataFrame,
	*,
	labels: ExperimentDisplayLabels,
	value_col: str = "mean_layer_score",
	period_df: pd.DataFrame | None = None,
	period_value_col: str = "total_score",
	experiment_col: str = "experiment",
	optimizer_col: str = "optimizer_id",
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	exclude_layers: Iterable[str] = ("head",),
) -> None:
	"""Stdout: per experiment × optimizer, mean score averaged across layers (then across runs).

	When `period_df` is given, the experiment mean matches plot `μ_exp`: flat mean over
	all period-level scores in that experiment. Also prints Cat1/Cat2 pooled period means
	and a Welch t-test between them (unequal variance, all individual period samples).
	"""
	if df.empty or value_col not in df.columns:
		print("(no score rows — skip across-layer means)")
		return
	work = df.copy()
	if exclude_layers:
		excl = {str(x) for x in exclude_layers}
		work = work[~work["layer_name"].astype(str).isin(excl)]
	if work.empty:
		print("(no score rows after layer filter)")
		return
	run_cols = [
		c
		for c in ("experiment_id", "model_id", "run_id", experiment_col, optimizer_col)
		if c in work.columns
	]
	if len(run_cols) < 3:
		run_cols = [experiment_col, optimizer_col]
	per_run = (
		work.groupby(run_cols, observed=False)[value_col]
		.mean()
		.reset_index(name="mean_across_layers")
	)
	summary = (
		per_run.groupby([experiment_col, optimizer_col], observed=False)["mean_across_layers"]
		.mean()
		.reset_index()
	)
	exps = sort_experiment_names(summary[experiment_col].unique())
	print("Mean total score across layers (per experiment × optimizer):")
	for exp in exps:
		print(f"\n{exp}")
		sub = summary[summary[experiment_col].astype(str) == str(exp)]
		for oid in _sort_optimizer_ids(sub[optimizer_col].unique().tolist()):
			val = float(sub.loc[sub[optimizer_col] == oid, "mean_across_layers"].iloc[0])
			print(f"  {_opt_display(oid)}: {val:.4f}")
		exp_mean: float | None = None
		if period_df is not None and period_value_col in period_df.columns:
			exp_periods = period_df[period_df[experiment_col].astype(str) == str(exp)]
			vals = pd.to_numeric(exp_periods[period_value_col], errors="coerce").dropna()
			if not vals.empty:
				exp_mean = float(vals.mean())
		if exp_mean is None and not sub.empty:
			exp_runs = per_run[per_run[experiment_col].astype(str) == str(exp)]
			if not exp_runs.empty:
				exp_mean = float(exp_runs["mean_across_layers"].mean())
		if exp_mean is not None:
			print(f"  experiment mean (all periods): {exp_mean:.4f}")

	if period_df is not None and period_value_col in period_df.columns:
		row1, row2, _other = _two_row_cat12_for_labels(exps, labels)

		def _pool_period_samples(exp_labels: list[str]) -> pd.Series:
			if not exp_labels:
				return pd.Series(dtype=float)
			mask = period_df[experiment_col].astype(str).isin([str(e) for e in exp_labels])
			return pd.to_numeric(period_df.loc[mask, period_value_col], errors="coerce").dropna()

		vals_cat1 = _pool_period_samples(row1)
		vals_cat2 = _pool_period_samples(row2)

		print("\n--- Category pooled means (all period samples) ---")
		if not vals_cat1.empty:
			print(
				f"Category 1 / row 1 ({len(row1)} experiments, n={len(vals_cat1)} periods): "
				f"mean={float(vals_cat1.mean()):.4f}"
			)
			if row1:
				print(f"  experiments: {', '.join(row1)}")
		else:
			print("Category 1 / row 1: (no period samples)")
		if not vals_cat2.empty:
			print(
				f"Category 2 / row 2 ({len(row2)} experiments, n={len(vals_cat2)} periods): "
				f"mean={float(vals_cat2.mean()):.4f}"
			)
			if row2:
				print(f"  experiments: {', '.join(row2)}")
		else:
			print("Category 2 / row 2: (no period samples)")

		if len(vals_cat1) >= 2 and len(vals_cat2) >= 2:
			p = _welch_t_pvalue(vals_cat1, vals_cat2)
			sig = bool(np.isfinite(p) and p < 0.05)
			sig_str = "significant" if sig else "not significant"
			print(
				f"Welch t-test (Cat1 vs Cat2, unequal variance): "
				f"p={p:.6g} ({sig_str} at α=0.05)"
			)
		else:
			print("Welch t-test (Cat1 vs Cat2): skipped (need ≥2 samples per group)")
		
		last_exps = ["Recover", "Reinforce"]
		vals_pool = _pool_period_samples(last_exps)
		print("Pooled (Recover + Reinforce):")
		print(f"  mean={float(vals_pool.mean()):.4f}")
		p = _welch_t_pvalue(vals_cat1, vals_pool)
		sig = bool(np.isfinite(p) and p < 0.05)
		sig_str = "significant" if sig else "not significant"
		print(
			f"Welch t-test (Cat1 vs [Rec.;Rei.], unequal variance): "
			f"p={p:.6g} ({sig_str} at α=0.05)"
		)


_BTSP_LAYER_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h")


def build_btsp_neuron_pct_segment_rows(
	all_results: dict[tuple[str, str, str, str], dict],
	rename_fn: Callable[[str], str],
	labels: "ExperimentDisplayLabels",
) -> pd.DataFrame:
	"""Cumulative BTSP neuron % per segment for shuffle / recover / reinforce.

	Segments are defined by `trial_start_iter` in each `periods_df`:

	* Shuffle (labelperm): 10 × 100-iter trials, skip=0  →
	  `segment = trial_start_iter // 100`
	* Recover / Reinforce: 10 × 300-iter groups of 3 trials, skip=1000  →
	  `segment = (trial_start_iter − 1000) // 300`
	  Periods with `trial_start_iter < 1000` (within-experiment pretrain) are
	  excluded.

	At segment k the value is the % of unique neurons with a BTSP period in
	segments 0..k (running union across segments — a neuron is never double-counted).
	Returns one row per (eid, mid, oid, rid, layer_name, segment_index).
	"""
	rows: list[dict] = []
	for (eid, mid, oid, rid), v in all_results.items():
		if v.get("error") or "periods_df" not in v:
			continue
		pdf = v["periods_df"]
		if pdf is None or pdf.empty:
			continue
		experiment = rename_fn(eid)
		spec = _btsp_segment_spec_for_experiment(experiment, eid, labels)
		if spec is None:
			continue

		skip = spec["skip_pretrain_iters"]
		seg_iters = spec["segment_iters"]
		n_segs = spec["n_segments"]

		if "trial_start_iter" not in pdf.columns:
			continue
		pdf_work = pdf.copy()
		pdf_work["_tsi"] = pd.to_numeric(pdf_work["trial_start_iter"], errors="coerce")
		pdf_work = pdf_work.dropna(subset=["_tsi", "neuron_id", "layer_name"])
		pdf_work["_seg"] = ((pdf_work["_tsi"] - skip) / seg_iters).astype(int)
		# keep only segments belonging to the post-skip window
		pdf_work = pdf_work[(pdf_work["_seg"] >= 0) & (pdf_work["_seg"] < n_segs)]

		df_nd = _pp_neuron_digit(eid, mid, oid, rid=rid, w_cache=False)
		if df_nd is None or df_nd.empty or "layer_name" not in df_nd.columns:
			continue
		total_by_layer = (
			df_nd[["neuron_id", "layer_name"]]
			.drop_duplicates("neuron_id")
			.groupby("layer_name", observed=False)["neuron_id"]
			.nunique()
		)

		# accumulate unique neurons segment-by-segment
		seen_by_layer: dict[str, set] = {str(ln): set() for ln in total_by_layer.index}
		for seg_idx in range(n_segs):
			seg_pdf = pdf_work[pdf_work["_seg"] == seg_idx]
			for layer_name, n_total in total_by_layer.items():
				if pd.isna(layer_name) or str(layer_name) in ("", "nan") or n_total == 0:
					continue
				ln = str(layer_name)
				new_neurons = set(
					seg_pdf.loc[seg_pdf["layer_name"] == layer_name, "neuron_id"]
					.astype(str)
					.tolist()
				)
				seen_by_layer[ln] |= new_neurons
				rows.append(
					{
						"experiment_id": eid,
						"experiment": experiment,
						"optimizer_id": oid,
						"layer_name": ln,
						"segment": seg_idx,
						"pct_btsp_neurons": 100.0 * len(seen_by_layer[ln]) / int(n_total),
					}
				)
	return pd.DataFrame(rows)


def plot_btsp_neuron_pct_segment_trajectories(
	df: pd.DataFrame,
	*,
	labels: "ExperimentDisplayLabels",
	plot_dir: str,
	show: bool = True,
	save: bool = True,
	figsize_per: tuple[float, float] = (5.5, 4.5),
	filename_suffix: str = "",
	title_suffix: str = "",
) -> "plt.Figure | None":
	"""1-row × 3-col figure: shuffle / recover / reinforce.

	x = run index (labeled "novelty introduction") — each x-tick is one experimental
	run (a different digit / digit-pair scenario).
	y = cumulative % of unique neurons that had at least one BTSP period across runs 0..k
	(running union across runs, no double-counting).
	One trace per (optimizer × layer): color = optimizer, marker = layer.
	"""
	experiments = cat2_shuffle_recover_reinforce_labels(labels)
	df = df[df["experiment"].isin(experiments) & (df["layer_name"].astype(str) != "head")].copy()
	if df.empty:
		return None

	order_layers = _sort_layer_display_names(df["layer_name"].unique())
	order_opts = _sort_optimizer_ids(df["optimizer_id"].unique().tolist())
	markers = _BTSP_LAYER_MARKERS

	# mean across runs for each (experiment, optimizer, layer, segment)
	agg = (
		df.groupby(["experiment", "optimizer_id", "layer_name", "segment"], observed=False)[
			"pct_btsp_neurons"
		]
		.mean()
		.reset_index()
	)

	fig, axes = plt.subplots(
		1, 3, figsize=(figsize_per[0] * 3, figsize_per[1]), squeeze=False
	)

	for col_idx, exp in enumerate(experiments):
		ax = axes[0, col_idx]
		sub = agg[agg["experiment"] == exp]
		if sub.empty:
			ax.set_visible(False)
			continue
		n_segs = int(sub["segment"].max()) + 1 if not sub.empty else 10
		for oid in order_opts:
			color = _opt_color(oid)
			opt_sub = sub[sub["optimizer_id"] == oid]
			for li, layer in enumerate(order_layers):
				marker = markers[li % len(markers)]
				layer_sub = opt_sub[opt_sub["layer_name"] == layer].sort_values("segment")
				if layer_sub.empty:
					continue
				ax.plot(
					layer_sub["segment"].to_numpy(),
					layer_sub["pct_btsp_neurons"].to_numpy(),
					color=color,
					marker=marker,
					linewidth=1.4,
					markersize=5,
				)
		ax.set_xlabel("novelty introduction")
		ax.set_ylabel("cumulative % neurons with BTSP period" if col_idx == 0 else "")
		ax.set_title(exp)
		ax.set_xticks(range(n_segs))
		ax.set_xticklabels([str(i) for i in range(n_segs)])

	# shared legend: optimizer (color) + layer (marker)
	opt_handles = [
		Line2D([0], [0], color=_opt_color(o), linewidth=2, label=_opt_display(o))
		for o in order_opts
	]
	layer_handles = [
		Line2D(
			[0], [0],
			color="0.35",
			marker=markers[li % len(markers)],
			linewidth=0,
			markersize=7,
			label=(
				f"L{layer_abbrev_number(layer)}"
				if layer_abbrev_number(layer) is not None
				else str(layer)
			),
		)
		for li, layer in enumerate(order_layers)
	]
	sep = [Line2D([0], [0], color="none", label="")]
	fig.legend(
		handles=opt_handles + sep + layer_handles,
		loc="center right",
		bbox_to_anchor=(1.18, 0.5),
		frameon=True,
		fontsize=9,
		title="optimizer / layer",
	)

	fig.suptitle(
		f"Cumulative % neurons with BTSP period per time segment{title_suffix}",
		fontsize=13,
		y=1.02,
	)
	plt.tight_layout()
	if save:
		fig.savefig(
			Path(plot_dir)
			/ insert_filename_suffix(
				"additional_btsp_neuron_pct_cumulative_segments.pdf", filename_suffix
			),
			bbox_inches="tight",
		)
	if show:
		plt.show()
	else:
		plt.close(fig)
	return fig


def _btsp_segment_spec_for_experiment(
	experiment: str,
	experiment_id: str,
	labels: ExperimentDisplayLabels,
) -> dict[str, int] | None:
	"""Post-pretrain windows: shuffle-seq 1000 it (10×100); recover/reinforce 3000 it (10×300)."""
	exp = str(experiment)
	eid = str(experiment_id)
	if exp == labels.shuffle_seq or "cat2_sequence_labelperm" in eid:
		return {
			"segment_iters": 100,
			"n_segments": 10,
			"total_iters": 1000,
			"skip_pretrain_iters": 0,  # no pretrain to skip: all trials start from iter 0
		}
	if exp == "Recover" or "cat2_sequence_recover" in eid:
		return {
			"segment_iters": 300,
			"n_segments": 10,
			"total_iters": 3000,
			"skip_pretrain_iters": 1000,
		}
	if exp == "Reinforce" or "cat2_sequence_reinforce" in eid:
		return {
			"segment_iters": 300,
			"n_segments": 10,
			"total_iters": 3000,
			"skip_pretrain_iters": 1000,
		}
	return None


def _period_reference_iter(row: pd.Series) -> float:
	for col in ("period_start_iter", "start_iter"):
		if col in row.index:
			val = pd.to_numeric(row.get(col), errors="coerce")
			if pd.notna(val):
				return float(val)
	return float("nan")


def _offset_from_novelty_for_period_row(row: pd.Series, eid: str) -> float:
	"""PMA timing relative to category novelty (matches `_novelty_offset_from_period_row`).

	Cat1: novelty at experiment start → `trial_start_iter + point_of_max_acceleration`.
	Cat2: novelty at period trial start → `point_of_max_acceleration` (trial-relative).
	"""
	pma = pd.to_numeric(row.get("point_of_max_acceleration"), errors="coerce")
	if not np.isfinite(pma):
		return float("nan")
	if is_pretrain_shuffle_mislabel_experiment(str(eid)):
		ts = pd.to_numeric(row.get("trial_start_iter"), errors="coerce")
		if not np.isfinite(ts):
			return float("nan")
		return float(ts + pma)
	return float(pma)


def build_period_metrics_df(
	all_results: dict[tuple[str, str, str, str], dict],
	rename_fn: Callable[[str], str],
) -> pd.DataFrame:
	"""Per-period metrics for correlation plots.

	`t_start` is PMA offset from category novelty (cat1: `trial_start_iter + PMA`;
	cat2: trial-relative PMA). Exposed as `t_start` for compatibility with `t_start_inv`.
	"""
	rows: list[dict] = []
	base_cols = ["activation_angle", "total_length_iter"]
	pma_col = "point_of_max_acceleration"
	for (eid, mid, oid, rid), v in all_results.items():
		if not is_category_experiment(eid):
			continue
		if v.get("error") or "periods_df" not in v:
			continue
		pdf = v["periods_df"]
		if pdf is None or pdf.empty:
			continue
		rob_col = robustness_score_column(pdf)
		required_cols = base_cols + [rob_col, pma_col]
		if is_pretrain_shuffle_mislabel_experiment(str(eid)):
			required_cols.append("trial_start_iter")
		if any(col not in pdf.columns for col in required_cols):
			continue
		for _, row in pdf.iterrows():
			if any(pd.isna(row.get(c)) for c in base_cols + [rob_col, pma_col]):
				continue
			offset = _offset_from_novelty_for_period_row(row, str(eid))
			if not np.isfinite(offset):
				continue
			rows.append(
				{
					"experiment_id": eid,
					"experiment": rename_fn(eid),
					"optimizer_id": oid,
					"activation_angle": float(row["activation_angle"]),
					"total_length_iter": float(row["total_length_iter"]),
					"robustness_score": float(row[rob_col]),
					"t_start": offset,
				}
			)
	return pd.DataFrame(rows)


def _spearman_annotation(x: np.ndarray, y: np.ndarray) -> str:
	mask = np.isfinite(x) & np.isfinite(y)
	x, y = x[mask], y[mask]
	if x.size < 3:
		return "n<3"
	r, p = stats.spearmanr(x, y)
	if not np.isfinite(r):
		return "n/a"
	p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
	return rf"$\rho$={r:+.2f}, p={p_str}"


def _corr_matrix_text_color(rgba: tuple[float, ...]) -> str:
	red = int(round(rgba[0] * 255))
	green = int(round(rgba[1] * 255))
	blue = int(round(rgba[2] * 255))
	if red * 0.299 + green * 0.587 + blue * 0.114 > 186:
		return "#000000"
	return "#ffffff"


def _plot_spearman_corr_matrix(
	ax,
	sub: pd.DataFrame,
	cols: list[str] | None = None,
	labels: list[str] | None = None,
) -> bool:
	if cols is None or labels is None:
		raise ValueError("cols and labels are required")
	cols = list(cols)
	labels = list(labels)
	vals = sub[cols].dropna()
	n = len(cols)
	if len(vals) < 3:
		ax.set_visible(False)
		return False
	corr_s = vals.corr(method="spearman").to_numpy()
	im = ax.imshow(corr_s, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
	ax.grid(False)
	ax.set_xticks(range(n))
	ax.set_yticks(range(n))
	ax.set_xticklabels(labels, fontsize=8)
	ax.set_yticklabels(labels, fontsize=8)
	cmap = im.get_cmap()
	norm = im.norm
	for i in range(n):
		for k in range(n):
			val = float(corr_s[i, k])
			text_color = _corr_matrix_text_color(cmap(norm(val)))
			ax.text(
				k,
				i,
				f"{val:+.2f}",
				ha="center",
				va="center",
				fontsize=7,
				color=text_color,
			)
	ax.set_title("Spearman", fontsize=8, pad=2)
	return True


def _corr_scatter_inner_hspace(n_pairs: int) -> float:
	"""Vertical gap between stacked scatter rows (room for each row's xlabel)."""
	return max(0.85, 0.35 + 0.10 * n_pairs)


def _set_three_ticks_from_limits(ax, axis: Literal["x", "y"]) -> None:
	lim = ax.get_xlim() if axis == "x" else ax.get_ylim()
	lo, hi = float(lim[0]), float(lim[1])
	if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
		return
	mid = lo + 0.5 * (hi - lo)
	ticks = [lo, mid, hi]
	if axis == "x":
		ax.set_xticks(ticks)
		ax.set_xlim(lo, hi)
	else:
		ax.set_yticks(ticks)
		ax.set_ylim(lo, hi)


def _scatter_corr_pair(
	ax,
	sub: pd.DataFrame,
	xc: str,
	yc: str,
	xlab: str,
	ylab: str,
	color: str,
	*,
	set_ylabel: bool = True,
) -> None:
	x = sub[xc].to_numpy(dtype=float)
	y = sub[yc].to_numpy(dtype=float)
	m = np.isfinite(x) & np.isfinite(y)
	if np.sum(m) < 2:
		ax.set_visible(False)
		return
	ax.scatter(x[m], y[m], s=16, alpha=0.35, color=color, edgecolors="none")
	ax.set_xlabel(xlab, fontsize=9, labelpad=3)
	ax.set_ylabel(ylab if set_ylabel else "", fontsize=9)
	ax.text(
		0.04,
		0.96,
		_spearman_annotation(x[m], y[m]),
		transform=ax.transAxes,
		ha="left",
		va="top",
		fontsize=7,
	)
	ax.autoscale()
	_set_three_ticks_from_limits(ax, "y")
	ax.tick_params(axis="both", which="major", labelsize=7, length=3)


def plot_correlations_per_experiment(
	df_periods: pd.DataFrame,
	*,
	plot_dir: str,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	corr_pairs: list[tuple[str, str, str, str]],
	corr_mat_cols: list[str],
	corr_mat_labels: list[str],
	show: bool = False,
	filename_suffix: str = "",
) -> None:
	if df_periods.empty:
		return
	n_pairs = len(corr_pairs)
	for exp in sort_experiment_names(df_periods["experiment"].unique()):
		exp_sub = df_periods[df_periods["experiment"] == exp]
		opts = _sort_optimizer_ids(exp_sub["optimizer_id"].unique().tolist())
		if not opts:
			continue
		ncols = len(opts)
		inner_hspace = _corr_scatter_inner_hspace(n_pairs)
		fig = plt.figure(figsize=(3.8 * ncols, 1.65 * n_pairs + 2.4))
		gs_outer = fig.add_gridspec(
			2, ncols, height_ratios=[n_pairs * 0.95, 1.0], hspace=0.45, wspace=0.32
		)
		for j, oid in enumerate(opts):
			sub = exp_sub[exp_sub["optimizer_id"] == oid]
			color = _opt_color(oid)
			inner = gs_outer[0, j].subgridspec(n_pairs, 1, hspace=inner_hspace)
			for row, (xc, yc, xlab, ylab) in enumerate(corr_pairs):
				ax = fig.add_subplot(inner[row, 0])
				if row == 0:
					ax.set_title(_opt_display(oid), fontsize=10)
				_scatter_corr_pair(
					ax, sub, xc, yc, xlab, ylab, color, set_ylabel=(j == 0)
				)
			ax_hm = fig.add_subplot(gs_outer[1, j])
			ax_hm.tick_params(axis="y", labelsize=10)
			_plot_spearman_corr_matrix(ax_hm, sub, cols=corr_mat_cols, labels=corr_mat_labels)
		fig.suptitle(f"{exp} — period metric correlations", fontsize=13, y=0.98)
		fig.subplots_adjust(top=0.93)
		fname = insert_filename_suffix(
			f"additional_corr_{safe_slug(exp)}.pdf", filename_suffix
		)
		plt.savefig(Path(plot_dir) / fname, bbox_inches="tight")
		if show:
			plt.show()
		else:
			plt.close(fig)


def plot_correlations_pooled_by_optimizer(
	df_periods: pd.DataFrame,
	*,
	plot_dir: str,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	labels: ExperimentDisplayLabels,
	corr_pairs: list[tuple[str, str, str, str]],
	corr_mat_cols: list[str],
	corr_mat_labels: list[str],
	show: bool = False,
	filename_suffix: str = "",
) -> None:
	mask = ~df_periods.apply(
		lambda r: is_control_experiment_id(r["experiment_id"], r["experiment"], labels),
		axis=1,
	)
	pool = df_periods[mask].copy()
	if pool.empty:
		return
	n_pairs = len(corr_pairs)
	opts = _sort_optimizer_ids(pool["optimizer_id"].unique().tolist())
	ncols = len(opts)
	inner_hspace = _corr_scatter_inner_hspace(n_pairs)
	fig = plt.figure(figsize=(4.5 * ncols, 1.65 * n_pairs + 2.4))
	gs_outer = fig.add_gridspec(
		2, ncols, height_ratios=[n_pairs * 0.95, 1.0], hspace=0.45, wspace=0.32
	)
	for j, oid in enumerate(opts):
		sub = pool[pool["optimizer_id"] == oid]
		color = _opt_color(oid)
		inner = gs_outer[0, j].subgridspec(n_pairs, 1, hspace=inner_hspace)
		for row, (xc, yc, xlab, ylab) in enumerate(corr_pairs):
			ax = fig.add_subplot(inner[row, 0])
			if row == 0:
				ax.set_title(_opt_display(oid), fontsize=11)
			_scatter_corr_pair(ax, sub, xc, yc, xlab, ylab, color, set_ylabel=(j == 0))
		ax_hm = fig.add_subplot(gs_outer[1, j])
		_plot_spearman_corr_matrix(ax_hm, sub, cols=corr_mat_cols, labels=corr_mat_labels)
	fig.suptitle(
		"Pooled period correlations by optimizer (non-control experiments)",
		fontsize=13,
		y=0.98,
	)
	fig.subplots_adjust(top=0.93)
	plt.savefig(
		Path(plot_dir)
		/ insert_filename_suffix(
			"additional_corr_pooled_by_optimizer_no_controls.pdf", filename_suffix
		),
		bbox_inches="tight",
	)
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_correlation_matrices_pooled_by_experiment(
	df_periods: pd.DataFrame,
	*,
	plot_dir: str,
	sort_experiment_names: Callable[[Iterable[str]], list[str]],
	corr_mat_cols: list[str],
	corr_mat_labels: list[str],
	show: bool = False,
	filename_suffix: str = "",
) -> None:
	if df_periods.empty:
		return
	experiments = sort_experiment_names(df_periods["experiment"].unique())
	n_exp = len(experiments)
	if n_exp == 0:
		return
	ncols = min(3, n_exp)
	nrows = (n_exp + ncols - 1) // ncols
	fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.9 * nrows), squeeze=False)
	for idx, exp in enumerate(experiments):
		ax = axes.flat[idx]
		sub = df_periods[df_periods["experiment"] == exp]
		_plot_spearman_corr_matrix(ax, sub, cols=corr_mat_cols, labels=corr_mat_labels)
		ax.set_title(exp, fontsize=10)
	for idx in range(n_exp, nrows * ncols):
		axes.flat[idx].set_visible(False)
	fig.suptitle(
		"Spearman correlations — pooled across optimizers per experiment",
		fontsize=13,
		y=1.02,
	)
	plt.tight_layout()
	plt.savefig(
		Path(plot_dir)
		/ insert_filename_suffix("additional_corr_matrix_by_experiment.pdf", filename_suffix),
		bbox_inches="tight",
	)
	if show:
		plt.show()
	else:
		plt.close(fig)


def _period_metrics_for_category(
	df_periods: pd.DataFrame, category: Literal["cat1", "cat2"]
) -> pd.DataFrame:
	if category == "cat1":
		mask = df_periods["experiment_id"].map(
			lambda e: is_pretrain_shuffle_mislabel_experiment(str(e))
		)
	else:
		mask = df_periods["experiment_id"].map(lambda e: is_cat2_sequence_experiment(str(e)))
	return df_periods.loc[mask].copy()


def plot_correlation_matrix_global_pool(
	df_periods: pd.DataFrame,
	*,
	plot_dir: str,
	labels: ExperimentDisplayLabels,
	corr_mat_cols: list[str],
	corr_mat_labels: list[str],
	show: bool = False,
	category: Literal["cat1", "cat2"] | None = None,
	exclude_controls: bool = True,
	filename_suffix: str = "",
) -> None:
	pool = df_periods.copy()
	if category is not None:
		pool = _period_metrics_for_category(pool, category)
	if exclude_controls:
		ctrl = pool.apply(
			lambda r: is_control_experiment_id(
				r["experiment_id"], r["experiment"], labels
			),
			axis=1,
		)
		pool = pool.loc[~ctrl].copy()
	if pool.empty:
		return
	if category == "cat1":
		title = (
			"Spearman correlations — Cat1 global pool"
			+ (" (non-control)" if exclude_controls else " (incl. controls)")
			+ ", all optimizers"
		)
		fname = "additional_corr_matrix_global_pool_cat1.pdf"
	elif category == "cat2":
		title = (
			"Spearman correlations — Cat2 global pool"
			+ (" (non-control)" if exclude_controls else " (incl. controls)")
			+ ", all optimizers"
		)
		fname = "additional_corr_matrix_global_pool_cat2.pdf"
	else:
		title = "Spearman correlations — global pool (non-control, all optimizers)"
		fname = "additional_corr_matrix_global_pool.pdf"
	fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.2))
	_plot_spearman_corr_matrix(ax, pool, cols=corr_mat_cols, labels=corr_mat_labels)
	fig.suptitle(title, fontsize=13, y=1.02)
	plt.tight_layout()
	plt.savefig(
		Path(plot_dir) / insert_filename_suffix(fname, filename_suffix),
		bbox_inches="tight",
	)
	if show:
		plt.show()
	else:
		plt.close(fig)


def first_digit_for_training_plot(
	eid: str, mid: str, rid: str, all_results: dict
) -> int | None:
	if not is_cat2_recover_reinforce_experiment(eid):
		return None
	for (e, m, _oid, r), payload in all_results.items():
		if e != eid or m != mid or r != rid:
			continue
		if payload.get("error"):
			continue
		df = payload.get("periods_df")
		if df is not None and not df.empty and "digitA" in df.columns:
			return int(df["digitA"].iloc[0])
	return parse_digit_a_from_config(eid, mid, rid)


def plot_train_loss_with_peaks_on_ax(
	ax,
	eid: str,
	mid: str,
	rid: str,
	oids: list[str],
	all_results: dict,
	*,
	x_max: float | None = None,
	find_peaks_kwargs: dict | None = None,
	line_alpha: float = 0.2,
	peak_marker_size: float = 22.0,
	peak_marker_linewidth: float = 0.4,
) -> tuple[dict | None, str]:
	first_m: dict | None = None
	x_mode = "iteration"
	for oid in oids:
		m = _metrics(eid, mid, oid, rid=rid, w_cache=False)
		if not m or "loss_all_train_mean" not in m:
			continue
		if first_m is None:
			first_m = m
		xs_full, x_mode = metrics_x_axis(m)
		mask = np.ones(xs_full.shape, dtype=bool)
		if x_max is not None:
			mask = xs_full <= x_max + 1e-9
		if not np.any(mask):
			continue
		xs = xs_full[mask]
		col = _opt_color(oid)
		lab = _opt_display(oid)
		y = np.asarray(m["loss_all_train_mean"], dtype=float)[mask]
		ax.plot(xs, y, color=col, label=lab, marker=".", ms=4, lw=1.6, alpha=line_alpha)
		kw = dict(find_peaks_kwargs or {})
		peak_idx, _ = find_peaks(y, **kw)
		if peak_idx.size:
			ax.scatter(
				xs[peak_idx],
				y[peak_idx],
				color=col,
				s=peak_marker_size,
				marker="o",
				edgecolors="0.15",
				linewidths=peak_marker_linewidth,
				zorder=5,
				label=None,
			)
	return first_m, x_mode


def plot_debug_training_loss_peaks(
	tree: dict,
	all_results: dict,
	*,
	rename_fn: Callable[[str], str],
	plot_dir: str,
	cat2_early_zoom_n_trials: int,
	cat2_early_zoom_max_iter: float,
	category_only: bool = True,
	show: bool = False,
	find_peaks_kwargs: dict | None = None,
	find_peaks_kwargs_control_global_shuffle_all: dict | None = None,
	line_alpha: float = 0.2,
	peak_marker_size: float = 22.0,
	peak_marker_linewidth: float = 0.4,
	filename_suffix: str = "",
	title_suffix: str = "",
) -> None:
	for eid, models in sorted(tree.items()):
		if category_only and not is_category_experiment(eid):
			continue
		fp_kw = find_peaks_kwargs_for_experiment(
			eid,
			find_peaks_kwargs=find_peaks_kwargs,
			find_peaks_kwargs_control_global_shuffle_all=find_peaks_kwargs_control_global_shuffle_all,
		)
		for mid, runs in sorted(models.items()):
			for rid, oids in sorted(runs.items()):
				oids = _sort_optimizer_ids(oids)
				with_zoom = is_cat2_sequence_experiment(eid)
				n_rows = 2 if with_zoom else 1
				fig, axes = plt.subplots(n_rows, 1, figsize=(9.5, 4.2 * n_rows), squeeze=False)
				first_m, x_mode = plot_train_loss_with_peaks_on_ax(
					axes[0, 0],
					eid,
					mid,
					rid,
					oids,
					all_results,
					find_peaks_kwargs=fp_kw,
					line_alpha=line_alpha,
					peak_marker_size=peak_marker_size,
					peak_marker_linewidth=peak_marker_linewidth,
				)
				if first_m is None:
					plt.close(fig)
					continue
				first_digit = None
				if is_cat2_sequence_experiment(eid):
					first_digit = first_digit_for_training_plot(eid, mid, rid, all_results)
					add_trial_boundaries_mpl(
						axes[0, 0], first_m, x_mode, report=True, first_digit=first_digit
					)
				x_label = "Iteration" if x_mode == "iteration" else "Checkpoint"
				axes[0, 0].set_ylabel("Training loss")
				axes[0, 0].legend(loc="best", fontsize=9, framealpha=0.92)
				if with_zoom:
					x_max = early_trials_x_max(
						first_m,
						x_mode,
						n_trials=cat2_early_zoom_n_trials,
						default_max=cat2_early_zoom_max_iter,
					)
					zoom_m, zoom_x_mode = plot_train_loss_with_peaks_on_ax(
						axes[1, 0],
						eid,
						mid,
						rid,
						oids,
						all_results,
						x_max=x_max,
						find_peaks_kwargs=fp_kw,
						line_alpha=line_alpha,
						peak_marker_size=peak_marker_size,
						peak_marker_linewidth=peak_marker_linewidth,
					)
					if zoom_m is not None:
						plot_x_full, _ = metrics_x_axis(zoom_m)
						zoom_end_idx = int(np.searchsorted(plot_x_full, x_max, side="right"))
						add_trial_boundaries_mpl(
							axes[1, 0],
							zoom_m,
							zoom_x_mode,
							plot_x_per_segment=plot_x_full,
							max_idx=zoom_end_idx,
							report=True,
							first_digit=first_digit,
						)
						axes[1, 0].set_xlim(
							float(plot_x_full[0]) if plot_x_full.size else 0.0, x_max
						)
						axes[1, 0].set_ylabel("Training loss")
						axes[1, 0].set_xlabel(x_label)
						axes[1, 0].legend(loc="best", fontsize=9, framealpha=0.92)
				else:
					axes[0, 0].set_xlabel(x_label)
				exp_label = rename_fn(eid)
				# fp_label = find_peaks_params_label(fp_kw)
				title = fr"{exp_label} ‒ training loss{title_suffix}"# (peaks, {fp_label})"
				if with_zoom:
					title += f" (first {cat2_early_zoom_n_trials} trials below)"
				fig.suptitle(title, y=1.01 if with_zoom else 1.02, fontsize=14)
				fig.tight_layout()
				from analysis.final.plot_dataset_context import insert_filename_suffix

				fname = insert_filename_suffix(
					f"additional_debug_training_loss_peaks_{safe_slug(eid)}.pdf",
					filename_suffix,
				)
				plt.savefig(Path(plot_dir) / fname, bbox_inches="tight")
				if show:
					plt.show()
				else:
					plt.close(fig)
