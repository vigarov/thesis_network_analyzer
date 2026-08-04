"""Plots for analysis Δw in neighborhood of t_start."""
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from scipy import stats

from analysis.dw_vicinity import (
	PERIOD_KEYS,
	filter_score_column,
	format_opt_display,
	plot_mean_tstart_vicinity_dw_heatmap,
	plot_mean_tstart_vicinity_dw_heatmap_diff_on_ax,
	plot_mean_tstart_vicinity_dw_heatmap_on_ax,
	plot_mean_tstart_vicinity_dw_heatmap_ratio_on_ax,
	plot_period_tstart_vicinity_dw_scatter_grid,
	select_dw_periods,
	shared_t_rel_xlim_from_dfs,
	shared_t_rel_xticks_from_xlim,
	shared_w_edges_from_dfs,
	vlim_from_df,
	vlim_from_diff_grid_pairs,
	vlim_from_ratio_grid_triples,
)

from analysis.final.plot_dataset_context import insert_filename_suffix


def _pdf_filename(name: str, filename_suffix: str) -> str:
	return insert_filename_suffix(name, filename_suffix)

STRATEGY_ORDER = ["angle", "length", "robustness", "total score"]

STRATEGY_COLORS = {
	"angle": "#4C72B0",
	"length": "#DD8452",
	"robustness": "#55A868",
	"total score": "#C44E52",
}

_WELCH_ALPHA = 0.05
_SEM_Z = 1.96
_BRACKET_PAD_FRAC = 0.06
_PREPOST_GROUP_GAP = 0.35
_PREPOST_BAR_W = 0.32
_PREPOST_BRACKET_ENGLOBE_PAD = 0.06


def slugify_label(label: str) -> str:
	return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "experiment"


def strategy_slug(score_column: str) -> str:
	return score_column.replace("_", "")


def topn_pool_by_experiments(
	df_dw: pd.DataFrame,
	exp_labels: list[str],
	top_n: int,
	score_column: str,
) -> pd.DataFrame:
	parts = [
		select_dw_periods(
			df_dw,
			experiment=exp_label,
			top_n=top_n,
			score_column=score_column,
		)
		for exp_label in exp_labels
	]
	return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_dw_scatter_all_experiments(
	df_dw: pd.DataFrame,
	all_periods_by_experiment: dict[str, pd.DataFrame],
	*,
	plot_dir: Path | str,
	sort_experiment_names: Callable[[Any], list[str]],
	parse_neuron_id: Callable[[str], str],
	ncols: int = 10,
	show: bool = True,
	filename_suffix: str = "",
) -> None:
	if df_dw.empty:
		return
	for exp_label in sort_experiment_names(all_periods_by_experiment.keys()):
		sel = select_dw_periods(df_dw, experiment=exp_label)
		print(f"{exp_label}: {len(sel)} plottable periods")
		fig_dw = plot_period_tstart_vicinity_dw_scatter_grid(
			sel,
			experiment_label=exp_label,
			ncols=ncols,
			parse_neuron_id=parse_neuron_id,
		)
		if fig_dw is not None:
			slug = slugify_label(exp_label)
			plt.savefig(
				Path(plot_dir)
				/ _pdf_filename(f"final_all_periods_dw_scatter_{slug}.pdf", filename_suffix),
				bbox_inches="tight",
			)
			if show:
				plt.show()


def plot_dw_heatmap_all_periods(
	df_dw: pd.DataFrame,
	all_periods_by_experiment: dict[str, pd.DataFrame],
	*,
	plot_dir: Path | str,
	sort_experiment_names: Callable[[Any], list[str]],
	two_row_cat12_from_sorted: Callable[[list[str]], tuple[list[str], list[str], list[str]]],
	show: bool = True,
	filename_suffix: str = "",
) -> None:
	if df_dw.empty:
		print("No period selections — skip aggregate Δw heatmaps")
		return

	exp_order = sort_experiment_names(all_periods_by_experiment.keys())
	cat1_exps, cat2_exps, _other_exps = two_row_cat12_from_sorted(exp_order)
	plot_dir = Path(plot_dir)

	for exp_label in exp_order:
		sel = select_dw_periods(df_dw, experiment=exp_label)
		print(f"{exp_label} aggregate: {len(sel)} periods (all)")
		fig_agg = plot_mean_tstart_vicinity_dw_heatmap(sel, experiment_label=exp_label)
		if fig_agg is not None:
			slug = slugify_label(exp_label)
			plt.savefig(
				plot_dir
				/ _pdf_filename(f"final_all_periods_dw_heatmap_mean_{slug}.pdf", filename_suffix),
				bbox_inches="tight",
			)
			if show:
				plt.show()

	all_sel = df_dw
	print(
		f"All experiments aggregate: {len(all_sel)} periods across"
		f" {all_sel['experiment'].nunique()} experiments"
	)
	fig_all = plot_mean_tstart_vicinity_dw_heatmap(all_sel, experiment_label="all experiments")
	if fig_all is not None:
		plt.savefig(
			plot_dir
			/ _pdf_filename(
				"final_all_periods_dw_heatmap_mean_all_experiments.pdf", filename_suffix
			),
			bbox_inches="tight",
		)
		if show:
			plt.show()

	for cat_label, cat_exps in (("Cat1", cat1_exps), ("Cat2", cat2_exps)):
		if not cat_exps:
			continue
		sel_cat = select_dw_periods(df_dw, experiments=cat_exps)
		print(f"{cat_label} aggregate: {len(sel_cat)} periods across {len(cat_exps)} experiments")
		fig_cat = plot_mean_tstart_vicinity_dw_heatmap(
			sel_cat,
			experiment_label=f"{cat_label} ‒ all experiments",
		)
		if fig_cat is not None:
			plt.savefig(
				plot_dir
				/ _pdf_filename(
					f"final_all_periods_dw_heatmap_mean_{cat_label.lower()}.pdf", filename_suffix
				),
				bbox_inches="tight",
			)
			if show:
				plt.show()


def plot_dw_heatmap_topn_per_experiment(
	df_dw: pd.DataFrame,
	all_periods_by_experiment: dict[str, pd.DataFrame],
	*,
	plot_dir: Path | str,
	sort_experiment_names: Callable[[Any], list[str]],
	top_n_per_experiment: int = 100,
	filter_strategy: str = "angle",
	show: bool = True,
	filename_suffix: str = "",
) -> None:
	if df_dw.empty:
		print("No period selections — skip per-experiment top-N Δw heatmaps")
		return

	plot_dir = Path(plot_dir)
	exp_order = sort_experiment_names(all_periods_by_experiment.keys())
	strategy_col = filter_score_column(filter_strategy, df_dw)
	filter_strat_slug = strategy_slug(strategy_col)

	for exp_label in exp_order:
		sel = select_dw_periods(
			df_dw,
			experiment=exp_label,
			top_n=top_n_per_experiment,
			score_column=strategy_col,
		)
		print(f"{exp_label}: {len(sel)} periods (top {top_n_per_experiment} by {filter_strategy})")
		fig_agg = plot_mean_tstart_vicinity_dw_heatmap(
			sel,
			experiment_label=fr"{exp_label} ‒ top {top_n_per_experiment} by {filter_strategy}",
		)
		if fig_agg is not None:
			slug = slugify_label(exp_label)
			plt.savefig(
				plot_dir
				/ _pdf_filename(
					f"final_top{top_n_per_experiment}_{filter_strat_slug}_dw_heatmap_mean_{slug}.pdf",
					filename_suffix,
				),
				bbox_inches="tight",
			)
			if show:
				plt.show()


def _restrict_dw_to_panel_periods(panel: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
	if panel.empty or df.empty:
		return pd.DataFrame()
	keys = panel[PERIOD_KEYS].drop_duplicates()
	return df.merge(keys, on=PERIOD_KEYS, how="inner")


def _parse_df_dw_baseline(
	df_dw_baseline: pd.DataFrame | list[pd.DataFrame] | None,
) -> tuple[
	Literal["plain", "diff", "ratio"],
	pd.DataFrame | None,
	pd.DataFrame | None,
	pd.DataFrame | None,
]:
	"""Return (mode, diff_baseline, full_df, inactive_df). `df_dw` is active in ratio mode."""
	if df_dw_baseline is None:
		return "plain", None, None, None
	if isinstance(df_dw_baseline, list):
		if len(df_dw_baseline) != 2:
			raise ValueError(
				f"df_dw_baseline list must have length 2 (full, inactive); got {len(df_dw_baseline)}"
			)
		df_full, df_inactive = df_dw_baseline
		if not isinstance(df_full, pd.DataFrame) or not isinstance(df_inactive, pd.DataFrame):
			raise TypeError("df_dw_baseline list elements must be pandas DataFrames")
		return "ratio", None, df_full, df_inactive
	if isinstance(df_dw_baseline, pd.DataFrame):
		return "diff", df_dw_baseline, None, None
	raise TypeError(
		f"df_dw_baseline must be None, a DataFrame, or a list of two DataFrames; got {type(df_dw_baseline)}"
	)


def plot_dw_heatmap_topn_strategy_grids(
	df_dw: pd.DataFrame,
	all_periods_by_experiment: dict[str, pd.DataFrame],
	*,
	plot_dir: Path | str,
	sort_experiment_names: Callable[[Any], list[str]],
	two_row_cat12_from_sorted: Callable[[list[str]], tuple[list[str], list[str], list[str]]],
	sort_optimizer_ids: Callable[[list[str]], list[str]],
	optimizer_ids: list[str],
	top_n_per_experiment: int = 100,
	top_n_per_optimizer: int = 10,
	strategy_order: list[str] | None = None,
	df_dw_baseline: pd.DataFrame | list[pd.DataFrame] | None = None,
	cmap: str = "RdBu_r",
	show: bool = True,
	filename_suffix: str = "",
) -> None:
	if df_dw.empty:
		print("No period selections — skip strategy-grid Δw heatmaps")
		return

	baseline_mode, df_diff_baseline, df_dw_full, df_dw_inactive = _parse_df_dw_baseline(df_dw_baseline)
	if baseline_mode == "diff" and df_diff_baseline is not None and df_diff_baseline.empty:
		print("Empty baseline df — skip strategy-grid Δw heatmaps")
		return
	if baseline_mode == "ratio":
		if df_dw_full is None or df_dw_inactive is None:
			raise RuntimeError("ratio mode requires full and inactive DataFrames")
		if df_dw_full.empty or df_dw_inactive.empty:
			print("Empty full or inactive df — skip strategy-grid Δw heatmaps")
			return

	diff_mode = baseline_mode == "diff"
	ratio_mode = baseline_mode == "ratio"
	if diff_mode:
		colorbar_label = r"mean $\Delta w_{ij}$ (full $-$ inactive-only)"
		file_tag = "_full_minus_inactive_diff"
		suptitle_metric_suffix = " ‒ full minus inactive-only presynapses"
		grid_suptitle_metric = "full minus inactive-only"
	elif ratio_mode:
		colorbar_label = (
			r"$(\langle\Delta w\rangle_\mathrm{active}"
			r" - \langle\Delta w\rangle_\mathrm{inactive})"
			r" / \langle\Delta w\rangle_\mathrm{full}$"
		)
		file_tag = "_active_minus_inactive_over_full"
		suptitle_metric_suffix = " ‒ (active minus inactive) over full presynapses"
		grid_suptitle_metric = "(active minus inactive) / full"
	else:
		colorbar_label = r"mean $\Delta w_{ij}$"
		file_tag = ""
		suptitle_metric_suffix = f" ‒ top {top_n_per_experiment} per experiment"
		grid_suptitle_metric = ""

	strategy_order = strategy_order or STRATEGY_ORDER
	plot_dir = Path(plot_dir)
	exp_order = sort_experiment_names(all_periods_by_experiment.keys())
	cat1_exps, cat2_exps, _other_exps = two_row_cat12_from_sorted(exp_order)
	cat_blocks = [("Cat1", cat1_exps), ("Cat2", cat2_exps)]
	opt_order = sort_optimizer_ids(optimizer_ids)
	n_opt = len(opt_order)

	n_strat = len(strategy_order)
	n_cat_rows = 1 + sum(1 for _, exps in cat_blocks if exps)
	row_specs = [("all experiments", exp_order)] + [
		(cat_label, exps) for cat_label, exps in cat_blocks if exps
	]

	all_panels: list[pd.DataFrame] = []
	for _row_label, row_exps in row_specs:
		for strat_label in strategy_order:
			score_col = filter_score_column(strat_label, df_dw)
			all_panels.append(
				topn_pool_by_experiments(df_dw, row_exps, top_n_per_experiment, score_col)
			)

	baseline_panels: list[pd.DataFrame] = []
	full_panels: list[pd.DataFrame] = []
	inactive_panels: list[pd.DataFrame] = []
	if diff_mode:
		for _row_label, row_exps in row_specs:
			for strat_label in strategy_order:
				score_col = filter_score_column(strat_label, df_diff_baseline)
				baseline_panels.append(
					topn_pool_by_experiments(
						df_diff_baseline, row_exps, top_n_per_experiment, score_col
					)
				)
	elif ratio_mode:
		for sel in all_panels:
			full_panels.append(_restrict_dw_to_panel_periods(sel, df_dw_full))
			inactive_panels.append(_restrict_dw_to_panel_periods(sel, df_dw_inactive))
	w_edges_all_row = shared_w_edges_from_dfs(all_panels)
	if diff_mode:
		vlim_all_row = (
			vlim_from_diff_grid_pairs(
				list(zip(all_panels, baseline_panels)), w_edges=w_edges_all_row
			)
			if all_panels
			else 1e-6
		)
	elif ratio_mode:
		vlim_all_row = (
			vlim_from_ratio_grid_triples(
				list(zip(all_panels, full_panels, inactive_panels)),
				w_edges=w_edges_all_row,
			)
			if all_panels
			else 1e-6
		)
	else:
		vlim_all_row = vlim_from_df(pd.concat(all_panels, ignore_index=True)) if all_panels else 1e-6
	fig_all_row, axes_all_row = plt.subplots(
		n_cat_rows,
		n_strat,
		figsize=(6.5 * n_strat, 4.5 * n_cat_rows),
		squeeze=False,
		sharey=True,
		constrained_layout=True,
	)
	im_all_last = None
	panel_idx = 0
	for row_i, (row_label, row_exps) in enumerate(row_specs):
		for col_j, strat_label in enumerate(strategy_order):
			ax = axes_all_row[row_i, col_j]
			sel = all_panels[panel_idx]
			sel_base = baseline_panels[panel_idx] if diff_mode else None
			sel_full = full_panels[panel_idx] if ratio_mode else None
			sel_inactive = inactive_panels[panel_idx] if ratio_mode else None
			panel_idx += 1
			if diff_mode:
				im_all_last = plot_mean_tstart_vicinity_dw_heatmap_diff_on_ax(
					ax,
					sel,
					sel_base,
					cmap=cmap,
					vlim=vlim_all_row,
					w_edges=w_edges_all_row,
					show_ylabel=(col_j == 0),
					ylabel_fontsize=26,
					ytick_fontsize=18,
				)
			elif ratio_mode:
				im_all_last = plot_mean_tstart_vicinity_dw_heatmap_ratio_on_ax(
					ax,
					sel,
					sel_full,
					sel_inactive,
					cmap=cmap,
					vlim=vlim_all_row,
					w_edges=w_edges_all_row,
					show_ylabel=(col_j == 0),
					ylabel_fontsize=26,
					ytick_fontsize=18,
				)
			else:
				im_all_last = plot_mean_tstart_vicinity_dw_heatmap_on_ax(
					ax,
					sel,
					cmap=cmap,
					vlim=vlim_all_row,
					w_edges=w_edges_all_row,
					show_ylabel=(col_j == 0),
					ylabel_fontsize=26,
					ytick_fontsize=18,
				)
			ax.set_title(
				f"{row_label} ‒ top {top_n_per_experiment} by {strat_label} (n={len(sel)} periods)",
				fontsize=12,
			)
			if col_j == 0:
				ax.text(
					-0.4,
					0.5,
					row_label,
					transform=ax.transAxes,
					rotation=90,
					va="center",
					ha="center",
					fontsize=20,
				)

	for ax in axes_all_row.ravel():
		if not ax.images:
			ax.axis("off")
	if im_all_last is not None:
		fig_all_row.colorbar(im_all_last, ax=axes_all_row, fraction=0.02, pad=0.02).set_label(
			colorbar_label
		)
	fig_all_row.suptitle(
		r"Mean $\Delta w$ around $t_\mathrm{start}$" + suptitle_metric_suffix,
		y=1.03,
	)
	plt.savefig(
		plot_dir
		/ _pdf_filename(
			f"final_top{top_n_per_experiment}_allstrategies_dw_heatmap_mean_all_experiments{file_tag}.pdf",
			filename_suffix,
		),
		bbox_inches="tight",
	)
	if show:
		plt.show()

	for strat_label in strategy_order:
		score_col = filter_score_column(strat_label, df_dw)
		strat_slug = strategy_slug(score_col)

		for cat_label, cat_exps in cat_blocks:
			if not cat_exps:
				continue
			cat_exp_order = [e for e in exp_order if e in cat_exps]
			n_exp_c = len(cat_exp_order)

			fig_grid, axes_grid = plt.subplots(
				n_exp_c,
				n_opt,
				figsize=(4 * n_opt, 2.4 * n_exp_c),
				squeeze=False,
				sharex=True,
				sharey=True,
				constrained_layout=True,
			)

			panel_dfs: list[pd.DataFrame] = []
			baseline_panel_dfs: list[pd.DataFrame] = []
			full_panel_dfs: list[pd.DataFrame] = []
			inactive_panel_dfs: list[pd.DataFrame] = []
			for exp_label in cat_exp_order:
				for oid in opt_order:
					panel_df = select_dw_periods(
						df_dw,
						experiment=exp_label,
						optimizer_id=oid,
						top_n=top_n_per_optimizer,
						score_column=score_col,
					)
					panel_dfs.append(panel_df)
					if diff_mode:
						baseline_panel_dfs.append(
							select_dw_periods(
								df_diff_baseline,
								experiment=exp_label,
								optimizer_id=oid,
								top_n=top_n_per_optimizer,
								score_column=score_col,
							)
						)
					elif ratio_mode:
						full_panel_dfs.append(_restrict_dw_to_panel_periods(panel_df, df_dw_full))
						inactive_panel_dfs.append(
							_restrict_dw_to_panel_periods(panel_df, df_dw_inactive)
						)

			w_edges_grid = shared_w_edges_from_dfs(panel_dfs)
			if diff_mode:
				vlim_grid = (
					vlim_from_diff_grid_pairs(
						list(zip(panel_dfs, baseline_panel_dfs)), w_edges=w_edges_grid
					)
					if panel_dfs
					else 1e-6
				)
			elif ratio_mode:
				vlim_grid = (
					vlim_from_ratio_grid_triples(
						list(zip(panel_dfs, full_panel_dfs, inactive_panel_dfs)),
						w_edges=w_edges_grid,
					)
					if panel_dfs
					else 1e-6
				)
			else:
				vlim_grid = vlim_from_df(pd.concat(panel_dfs, ignore_index=True)) if panel_dfs else 1e-6
			xlim_grid = shared_t_rel_xlim_from_dfs(panel_dfs)
			xticks_grid = (
				shared_t_rel_xticks_from_xlim(xlim_grid, n_per_side=4) if xlim_grid is not None else None
			)
			im_last = None
			idx = 0
			for i, exp_label in enumerate(cat_exp_order):
				for j, oid in enumerate(opt_order):
					ax = axes_grid[i, j]
					panel_df = panel_dfs[idx]
					panel_base = baseline_panel_dfs[idx] if diff_mode else None
					panel_full = full_panel_dfs[idx] if ratio_mode else None
					panel_inactive = inactive_panel_dfs[idx] if ratio_mode else None
					idx += 1
					if diff_mode:
						im_last = plot_mean_tstart_vicinity_dw_heatmap_diff_on_ax(
							ax,
							panel_df,
							panel_base,
							cmap=cmap,
							vlim=vlim_grid,
							w_edges=w_edges_grid,
							xlim=xlim_grid,
							xticks=xticks_grid,
							show_ylabel=(j == 0),
							ylabel_fontsize=16,
							ytick_fontsize=12,
						)
					elif ratio_mode:
						im_last = plot_mean_tstart_vicinity_dw_heatmap_ratio_on_ax(
							ax,
							panel_df,
							panel_full,
							panel_inactive,
							cmap=cmap,
							vlim=vlim_grid,
							w_edges=w_edges_grid,
							xlim=xlim_grid,
							xticks=xticks_grid,
							show_ylabel=(j == 0),
							ylabel_fontsize=16,
							ytick_fontsize=12,
						)
					else:
						im_last = plot_mean_tstart_vicinity_dw_heatmap_on_ax(
							ax,
							panel_df,
							cmap=cmap,
							vlim=vlim_grid,
							w_edges=w_edges_grid,
							xlim=xlim_grid,
							xticks=xticks_grid,
							show_ylabel=(j == 0),
							ylabel_fontsize=16,
							ytick_fontsize=12,
						)
					if i == 0:
						ax.set_title(format_opt_display(oid), fontsize=9)
					if j == 0:
						ax.text(
							-0.35,
							0.5,
							exp_label,
							transform=ax.transAxes,
							rotation=90,
							va="center",
							ha="center",
							fontsize=16,
						)
					if not panel_df.empty:
						ax.text(
							0.05,
							0.95,
							f"n={len(panel_df)}",
							transform=ax.transAxes,
							va="top",
							ha="left",
							fontsize=7,
							color="0.15",
						)

			for ax in axes_grid.ravel():
				if not ax.images:
					ax.axis("off")

			if im_last is not None:
				fig_grid.colorbar(im_last, ax=axes_grid, fraction=0.02, pad=0.02).set_label(
					colorbar_label
				)

			grid_suptitle = r"Mean $\Delta w$ around $t_\mathrm{start}$"
			if diff_mode or ratio_mode:
				grid_suptitle += (
					f" ‒ {cat_label}, {grid_suptitle_metric}"
					f" (top {top_n_per_optimizer} by {strat_label})"
				)
			else:
				grid_suptitle += (
					f" ‒ {cat_label}, top {top_n_per_optimizer} per optimizer by {strat_label}"
				)
			fig_grid.suptitle(grid_suptitle, fontsize=12, y=1.02)
			plt.savefig(
				plot_dir
				/ _pdf_filename(
					f"final_top{top_n_per_optimizer}peropt_{strat_slug}_{cat_label.lower()}_dw_heatmap_grid{file_tag}.pdf",
					filename_suffix,
				),
				bbox_inches="tight",
			)
			if show:
				plt.show()


def _period_mean_dw_side(
	t_rel: np.ndarray,
	delta_w: np.ndarray,
	*,
	post: bool,
) -> float | None:
	mask = (t_rel > 0) if post else (t_rel < 0)
	if not np.any(mask):
		return None
	return float(np.mean(delta_w[mask]))


def _period_pre_post_means_from_row(row) -> tuple[float | None, float | None]:
	t_rel = np.asarray(row.t_rel, dtype=np.float64)
	delta_w = np.asarray(row.delta_w, dtype=np.float64)
	pre = _period_mean_dw_side(t_rel, delta_w, post=False)
	post = _period_mean_dw_side(t_rel, delta_w, post=True)
	return pre, post


def _collect_period_means_pre_post(sel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
	pre_vals: list[float] = []
	post_vals: list[float] = []
	for row in sel.itertuples(index=False):
		pre, post = _period_pre_post_means_from_row(row)
		if pre is not None:
			pre_vals.append(pre)
		if post is not None:
			post_vals.append(post)
	return (
		np.asarray(pre_vals, dtype=np.float64),
		np.asarray(post_vals, dtype=np.float64),
	)


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
	"""Welch two-sample p-value for period-level mean arrays."""
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
	period_means_by_strategy: dict[str, np.ndarray],
	order: list[str],
) -> np.ndarray:
	"""Symmetric Holm-significant pairwise Welch flags over `order` (k×k)."""
	k = len(order)
	sig_mat = np.zeros((k, k), dtype=bool)
	pairs: list[tuple[int, int, float]] = []
	for i, si in enumerate(order):
		for j, sj in enumerate(order):
			if i >= j:
				continue
			ai = period_means_by_strategy.get(si)
			aj = period_means_by_strategy.get(sj)
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


def _triu_pair_count(sig: np.ndarray) -> int:
	return int(np.sum(np.triu(sig, k=1)))


def _n_pair_brackets(sig_pre: np.ndarray, sig_post: np.ndarray) -> int:
	"""Count strategy pairs with Holm significance in pre and/or post."""
	k = sig_pre.shape[0]
	n = 0
	for i in range(k):
		for j in range(i + 1, k):
			if sig_pre[i, j] or sig_post[i, j]:
				n += 1
	return n


def _holm_pair_bracket_breakdown(sig_pre: np.ndarray, sig_post: np.ndarray) -> dict[str, int]:
	"""Counts for upper-triangle strategy pairs after Holm (max 6 each side, ≤6 brackets)."""
	k = sig_pre.shape[0]
	pre_only = post_only = both = 0
	for i in range(k):
		for j in range(i + 1, k):
			p = bool(sig_pre[i, j])
			q = bool(sig_post[i, j])
			if p and q:
				both += 1
			elif p:
				pre_only += 1
			elif q:
				post_only += 1
	return {
		"n_sig_pre_pairs": _triu_pair_count(sig_pre),
		"n_sig_post_pairs": _triu_pair_count(sig_post),
		"n_brackets": pre_only + post_only + both,
		"n_pre_only": pre_only,
		"n_post_only": post_only,
		"n_both": both,
	}


def _print_pre_post_holm_summary(
	subplots: dict[tuple[str, str], "DwPrePostSubplot"],
	*,
	cat_label: str,
) -> None:
	"""Log Holm-significant strategy-pair counts per panel and category totals."""
	tot_pre = tot_post = tot_brackets = 0
	print(f"Pre/post Δw Holm pairwise — {cat_label}:")
	for (exp_label, oid), panel in sorted(subplots.items()):
		bd = _holm_pair_bracket_breakdown(panel.sig_pre_pairs, panel.sig_post_pairs)
		tot_pre += bd["n_sig_pre_pairs"]
		tot_post += bd["n_sig_post_pairs"]
		tot_brackets += bd["n_brackets"]
		print(
			f"  {exp_label} | {oid}: "
			f"brackets={bd['n_brackets']} "
			f"(pre-only={bd['n_pre_only']}, post-only={bd['n_post_only']}, both={bd['n_both']}) "
			f"| Holm sig pairs: pre={bd['n_sig_pre_pairs']}, post={bd['n_sig_post_pairs']}"
		)
	print(
		f"  {cat_label} totals: brackets={tot_brackets}, "
		f"Holm sig pairs pre={tot_pre}, post={tot_post} (max 6 per side per panel)"
	)


@dataclass
class DwStrategyPrePostBars:
	mean_pre: float
	mean_post: float
	sem_pre: float
	sem_post: float
	n_pre: int
	n_post: int
	period_means_pre: np.ndarray = field(repr=False)
	period_means_post: np.ndarray = field(repr=False)


@dataclass
class DwPrePostSubplot:
	"""One experiment×optimizer panel: pre/post bars per filter strategy."""
	strategies: list[str]
	bars: dict[str, DwStrategyPrePostBars]
	sig_pre_pairs: np.ndarray  # k×k Holm-significant Welch pairs (pre period means)
	sig_post_pairs: np.ndarray  # k×k Holm-significant Welch pairs (post period means)


@dataclass
class DwPrePostCategoryFigure:
	"""Layout and panel data for one Cat1 or Cat2 figure."""
	cat_label: str
	exp_order: list[str]
	opt_order: list[str]
	subplots: dict[tuple[str, str], DwPrePostSubplot]
	ylim: tuple[float, float]


@dataclass
class DwPrePostStrategyFacetGrid:
	"""Computed pre/post Δw bar data for all category figures (ready to plot)."""
	category_figures: list[DwPrePostCategoryFigure]
	top_n_per_optimizer: int
	strategy_order: list[str]


def _strategy_bar_stats(period_means_pre: np.ndarray, period_means_post: np.ndarray) -> DwStrategyPrePostBars:
	pre = period_means_pre[np.isfinite(period_means_pre)]
	post = period_means_post[np.isfinite(period_means_post)]
	mean_pre = float(np.mean(pre)) if pre.size else float("nan")
	mean_post = float(np.mean(post)) if post.size else float("nan")
	return DwStrategyPrePostBars(
		mean_pre=mean_pre,
		mean_post=mean_post,
		sem_pre=_sem_stderr(pre),
		sem_post=_sem_stderr(post),
		n_pre=int(pre.size),
		n_post=int(post.size),
		period_means_pre=pre,
		period_means_post=post,
	)


def _compute_panel_pre_post(
	df_dw: pd.DataFrame,
	*,
	experiment: str,
	optimizer_id: str,
	strategy_order: list[str],
	top_n_per_optimizer: int,
) -> DwPrePostSubplot:
	bars: dict[str, DwStrategyPrePostBars] = {}
	period_means_pre: dict[str, np.ndarray] = {}
	period_means_post: dict[str, np.ndarray] = {}

	for strat_label in strategy_order:
		score_col = filter_score_column(strat_label, df_dw)
		sel = select_dw_periods(
			df_dw,
			experiment=experiment,
			optimizer_id=optimizer_id,
			top_n=top_n_per_optimizer,
			score_column=score_col,
		)
		pre_arr, post_arr = _collect_period_means_pre_post(sel)
		bars[strat_label] = _strategy_bar_stats(pre_arr, post_arr)
		period_means_pre[strat_label] = pre_arr
		period_means_post[strat_label] = post_arr

	sig_pre_pairs = _welch_pairwise_holm_sig_matrix(period_means_pre, strategy_order)
	sig_post_pairs = _welch_pairwise_holm_sig_matrix(period_means_post, strategy_order)

	return DwPrePostSubplot(
		strategies=strategy_order,
		bars=bars,
		sig_pre_pairs=sig_pre_pairs,
		sig_post_pairs=sig_post_pairs,
	)


def _panel_ylim_extent(panel: DwPrePostSubplot) -> float:
	ext = 0.0
	for st in panel.bars.values():
		for mean, sem in ((st.mean_pre, st.sem_pre), (st.mean_post, st.sem_post)):
			if np.isfinite(mean):
				ext = max(ext, abs(mean) + _SEM_Z * sem)
	n_brackets = _n_pair_brackets(panel.sig_pre_pairs, panel.sig_post_pairs)
	if n_brackets:
		ext *= 1.0 + _BRACKET_PAD_FRAC * n_brackets
	return ext


def compute_dw_pre_post_strategy_facets(
	df_dw: pd.DataFrame,
	all_periods_by_experiment: dict[str, pd.DataFrame],
	*,
	sort_experiment_names: Callable[[Any], list[str]],
	two_row_cat12_from_sorted: Callable[[list[str]], tuple[list[str], list[str], list[str]]],
	sort_optimizer_ids: Callable[[list[str]], list[str]],
	optimizer_ids: list[str],
	top_n_per_optimizer: int = 10,
	strategy_order: list[str] | None = None,
) -> DwPrePostStrategyFacetGrid | None:
	if df_dw.empty:
		print("No period selections — skip pre/post Δw strategy facets")
		return None

	strategy_order = strategy_order or STRATEGY_ORDER
	exp_order = sort_experiment_names(all_periods_by_experiment.keys())
	cat1_exps, cat2_exps, _other_exps = two_row_cat12_from_sorted(exp_order)
	opt_order = sort_optimizer_ids(optimizer_ids)

	category_figures: list[DwPrePostCategoryFigure] = []
	for cat_label, cat_exps in (("Cat1", cat1_exps), ("Cat2", cat2_exps)):
		if not cat_exps:
			continue
		cat_exp_order = [e for e in exp_order if e in cat_exps]
		subplots: dict[tuple[str, str], DwPrePostSubplot] = {}
		for exp_label in cat_exp_order:
			for oid in opt_order:
				subplot = _compute_panel_pre_post(
					df_dw,
					experiment=exp_label,
					optimizer_id=oid,
					strategy_order=strategy_order,
					top_n_per_optimizer=top_n_per_optimizer,
				)
				subplots[(exp_label, oid)] = subplot

		extents = [_panel_ylim_extent(p) for p in subplots.values()]
		ylim_hi = max(extents) if extents else 1e-6
		if ylim_hi <= 0:
			ylim_hi = 1e-6

		_print_pre_post_holm_summary(subplots, cat_label=cat_label)

		category_figures.append(
			DwPrePostCategoryFigure(
				cat_label=cat_label,
				exp_order=cat_exp_order,
				opt_order=opt_order,
				subplots=subplots,
				ylim=(-float(ylim_hi), float(ylim_hi)),
			)
		)

	total_brackets = sum(
		_holm_pair_bracket_breakdown(p.sig_pre_pairs, p.sig_post_pairs)["n_brackets"]
		for cat in category_figures
		for p in cat.subplots.values()
	)
	print(
		f"Pre/post Δw Holm pairwise — all categories: {total_brackets} bracket(s) to draw "
		f"across {sum(len(c.subplots) for c in category_figures)} panel(s)"
	)

	return DwPrePostStrategyFacetGrid(
		category_figures=category_figures,
		top_n_per_optimizer=top_n_per_optimizer,
		strategy_order=strategy_order,
	)


def _annotate_bar_value(ax, x: float, mean: float, sem: float, *, color: str) -> None:
	if not np.isfinite(mean):
		return
	wh = _SEM_Z * sem
	y_tip = mean + wh if mean >= 0 else mean - wh
	va = "bottom" if mean >= 0 else "top"
	offset = 0.02 * max(abs(y_tip), 1e-12)
	y_text = y_tip + offset if mean >= 0 else y_tip - offset
	ax.text(
		x,
		y_text,
		f"{mean:.3g}",
		ha="center",
		va=va,
		fontsize=5.5,
		color=color,
	)


def _pair_bracket_label(sig_pre: bool, sig_post: bool) -> str:
	parts: list[str] = []
	if sig_pre:
		parts.append(r"$*_{\mathrm{pre}}$")
	if sig_post:
		parts.append(r"$*_{\mathrm{post}}$")
	return " ".join(parts)


def _draw_strategy_pair_brackets_pre_post(
	ax,
	x_centers: np.ndarray,
	sig_pre: np.ndarray,
	sig_post: np.ndarray,
	*,
	y_base: float,
	y_step: float,
	direction: Literal["up", "down"] = "up",
) -> None:
	"""One bracket per strategy pair; label pre/post/both from Holm pairwise tests."""
	k = len(x_centers)
	stack = 0
	pitch = max(y_step * 1.15, y_step)
	riser = y_step * 0.25
	sign = 1.0 if direction == "up" else -1.0
	label_va = "bottom" if direction == "up" else "top"
	for i in range(k):
		for j in range(i + 1, k):
			pre_ij = bool(sig_pre[i, j])
			post_ij = bool(sig_post[i, j])
			if not pre_ij and not post_ij:
				continue
			y = y_base + sign * stack * pitch
			stack += 1
			x1, x2 = float(x_centers[i]), float(x_centers[j])
			y_riser = y + sign * riser
			ax.plot(
				[x1, x1, x2, x2],
				[y, y_riser, y_riser, y],
				lw=1.2,
				color="0.15",
				clip_on=False,
			)
			ax.text(
				(x1 + x2) / 2.0,
				y_riser,
				_pair_bracket_label(pre_ij, post_ij),
				ha="center",
				va=label_va,
				fontsize=8,
				color="0.1",
				clip_on=False,
			)


def _pre_post_panel_x_centers(n_strat: int) -> np.ndarray:
	return np.arange(n_strat, dtype=float) * (
		2 * _PREPOST_BAR_W + _PREPOST_GROUP_GAP
	)


def _pre_post_strategy_x_span(x_center: float) -> tuple[float, float]:
	"""Left/right x in data coords spanning pre+post bars plus padding."""
	pad = _PREPOST_BRACKET_ENGLOBE_PAD
	return x_center - _PREPOST_BAR_W - pad, x_center + _PREPOST_BAR_W + pad


def _plot_pre_post_panel_on_ax(
	ax,
	panel: DwPrePostSubplot,
	*,
	ylim: tuple[float, float],
	bracket_direction: Literal["up", "down"] = "up",
	show_ylabel: bool = True,
	show_xticklabels: bool = True,
) -> np.ndarray | None:
	n_strat = len(panel.strategies)
	if n_strat == 0:
		ax.set_visible(False)
		return None

	bar_w = _PREPOST_BAR_W
	x_centers = _pre_post_panel_x_centers(n_strat)
	x_pre = x_centers - bar_w / 2.0
	x_post = x_centers + bar_w / 2.0

	for i, strat_label in enumerate(panel.strategies):
		st = panel.bars[strat_label]
		color = STRATEGY_COLORS.get(strat_label, "0.45")

		if np.isfinite(st.mean_pre):
			ax.bar(
				x_pre[i],
				st.mean_pre,
				width=bar_w,
				color=color,
				hatch="///",
				edgecolor="0.2",
				linewidth=0.5,
				zorder=2,
			)
			ax.errorbar(
				x_pre[i],
				st.mean_pre,
				yerr=_SEM_Z * st.sem_pre,
				fmt="none",
				ecolor="0.15",
				capsize=2,
				linewidth=0.9,
				zorder=3,
			)
			_annotate_bar_value(ax, x_pre[i], st.mean_pre, st.sem_pre, color="0.15")

		if np.isfinite(st.mean_post):
			ax.bar(
				x_post[i],
				st.mean_post,
				width=bar_w,
				color=color,
				edgecolor="0.2",
				linewidth=0.5,
				zorder=2,
			)
			ax.errorbar(
				x_post[i],
				st.mean_post,
				yerr=_SEM_Z * st.sem_post,
				fmt="none",
				ecolor="0.15",
				capsize=2,
				linewidth=0.9,
				zorder=3,
			)
			_annotate_bar_value(ax, x_post[i], st.mean_post, st.sem_post, color="0.15")

	ax.axhline(0.0, color="0.35", lw=0.8, zorder=1)
	ax.set_ylim(ylim)
	ax.set_xticks(x_centers)
	if show_xticklabels:
		ax.set_xticklabels(panel.strategies, rotation=25, ha="right", fontsize=7)
	else:
		ax.set_xticklabels([])
	if show_ylabel:
		ax.set_ylabel(r"mean $\Delta w$", fontsize=8, labelpad=6)
	ax.tick_params(axis="y", labelsize=7)

	span = ylim[1] - ylim[0]
	y_step = 0.1 * span
	if bracket_direction == "up":
		y_base = ylim[1] - 0.2 * span
	else:
		y_base = ylim[0] + 0.2 * span
	_draw_strategy_pair_brackets_pre_post(
		ax,
		x_centers,
		panel.sig_pre_pairs,
		panel.sig_post_pairs,
		y_base=y_base,
		y_step=y_step,
		direction=bracket_direction,
	)
	return x_centers


def _active_full_sig_per_strategy(
	active_panel: DwPrePostSubplot,
	full_panel: DwPrePostSubplot,
) -> tuple[np.ndarray, np.ndarray]:
	"""Uncorrected Welch active vs full per strategy (pre and post period means)."""
	k = len(active_panel.strategies)
	sig_pre = np.zeros(k, dtype=bool)
	sig_post = np.zeros(k, dtype=bool)
	for i, strat_label in enumerate(active_panel.strategies):
		act = active_panel.bars.get(strat_label)
		full = full_panel.bars.get(strat_label)
		if act is None or full is None:
			continue
		p_pre = _welch_t_pvalue_arrays(act.period_means_pre, full.period_means_pre)
		if np.isfinite(p_pre) and p_pre < _WELCH_ALPHA:
			sig_pre[i] = True
		p_post = _welch_t_pvalue_arrays(act.period_means_post, full.period_means_post)
		if np.isfinite(p_post) and p_post < _WELCH_ALPHA:
			sig_post[i] = True
	return sig_pre, sig_post


def _n_active_full_vertical_brackets(sig_pre: np.ndarray, sig_post: np.ndarray) -> int:
	return int(np.sum(sig_pre | sig_post))


def _print_active_full_summary(
	subplots_active: dict[tuple[str, str], DwPrePostSubplot],
	subplots_full: dict[tuple[str, str], DwPrePostSubplot],
	*,
	cat_label: str,
) -> None:
	tot = 0
	print(f"Pre/post Δw active↔full Welch (uncorrected) — {cat_label}:")
	for key in sorted(subplots_active.keys()):
		if key not in subplots_full:
			continue
		sig_pre, sig_post = _active_full_sig_per_strategy(
			subplots_active[key], subplots_full[key]
		)
		n = _n_active_full_vertical_brackets(sig_pre, sig_post)
		tot += n
		print(
			f"  {key[0]} | {key[1]}: vertical brackets={n} "
			f"(pre={int(np.sum(sig_pre))}, post={int(np.sum(sig_post))})"
		)
	print(f"  {cat_label} totals: vertical brackets={tot}")


def _blend_data_axes_to_fig(
	fig, ax, x_data: float, y_axes: float
) -> tuple[float, float]:
	trans = blended_transform_factory(ax.transData, ax.transAxes)
	disp = trans.transform((x_data, y_axes))
	fig_xy = fig.transFigure.inverted().transform(disp)
	return float(fig_xy[0]), float(fig_xy[1])


def _draw_englobing_cap(
	ax,
	x_left: float,
	x_right: float,
	y_base: float,
	*,
	direction: Literal["down", "up"],
	trans,
	lw: float,
	riser: float,
) -> float:
	"""Horizontal cap with end ticks (rotated bracket arm englobing a bar group)."""
	sign = -1.0 if direction == "down" else 1.0
	y_arm = y_base + sign * riser
	ax.plot(
		[x_left, x_left, x_right, x_right],
		[y_base, y_arm, y_arm, y_base],
		transform=trans,
		color="0.12",
		lw=lw,
		clip_on=False,
		zorder=12,
		solid_capstyle="round",
	)
	return y_arm


def _draw_active_full_caps(
	ax_top,
	ax_bottom,
	x_centers: np.ndarray,
	sig_pre: np.ndarray,
	sig_post: np.ndarray,
	*,
	y_top_cap: float,
	y_bottom_cap: float,
	riser: float,
	lw: float,
) -> None:
	"""Draw englobing caps only (connectors/labels need final figure layout)."""
	trans_top = blended_transform_factory(ax_top.transData, ax_top.transAxes)
	trans_bottom = blended_transform_factory(ax_bottom.transData, ax_bottom.transAxes)
	for i, x_center in enumerate(x_centers):
		if not (bool(sig_pre[i]) or bool(sig_post[i])):
			continue
		x_left, x_right = _pre_post_strategy_x_span(float(x_center))
		_draw_englobing_cap(
			ax_top,
			x_left,
			x_right,
			y_top_cap,
			direction="down",
			trans=trans_top,
			lw=lw,
			riser=riser,
		)
		_draw_englobing_cap(
			ax_bottom,
			x_left,
			x_right,
			y_bottom_cap,
			direction="up",
			trans=trans_bottom,
			lw=lw,
			riser=riser,
		)


def _draw_active_full_connectors_and_labels(
	fig,
	ax_top,
	ax_bottom,
	x_centers: np.ndarray,
	sig_pre: np.ndarray,
	sig_post: np.ndarray,
	*,
	y_top_cap: float,
	y_bottom_cap: float,
	riser: float,
	lw: float,
	connector_extend_frac: float,
	label_dx_frac: float,
	fontsize: float,
) -> None:
	"""Center connector + pre/post labels after layout (call after subplots_adjust)."""
	bbox_top = ax_top.get_position()
	bbox_bottom = ax_bottom.get_position()
	y_mid_fig = (bbox_top.y0 + bbox_bottom.y1) / 2.0
	y_top_arm = y_top_cap - riser
	y_bottom_arm = y_bottom_cap + riser
	extend = connector_extend_frac * riser
	y_top_conn = max(0.0, y_top_arm - extend)
	y_bottom_conn = min(1.0, y_bottom_arm + extend)

	for i, x_center in enumerate(x_centers):
		pre_i = bool(sig_pre[i])
		post_i = bool(sig_post[i])
		if not pre_i and not post_i:
			continue
		xf = float(x_center)
		x_left, x_right = _pre_post_strategy_x_span(xf)

		p_top = _blend_data_axes_to_fig(fig, ax_top, xf, y_top_conn)
		p_bottom = _blend_data_axes_to_fig(fig, ax_bottom, xf, y_bottom_conn)
		x_fig = 0.5 * (p_top[0] + p_bottom[0])
		y_lo = min(p_top[1], p_bottom[1])
		y_hi = max(p_top[1], p_bottom[1])
		if y_hi - y_lo < 1e-6:
			x_fig = _blend_data_axes_to_fig(fig, ax_top, xf, 0.5)[0]
			y_lo, y_hi = bbox_top.y0, bbox_bottom.y1

		fig.add_artist(
			Line2D(
				[x_fig, x_fig],
				[y_lo, y_hi],
				transform=fig.transFigure,
				color="0.12",
				lw=lw,
				clip_on=False,
				zorder=50,
			)
		)

		x_left_fig = _blend_data_axes_to_fig(fig, ax_top, x_left, 0.5)[0]
		x_right_fig = _blend_data_axes_to_fig(fig, ax_top, x_right, 0.5)[0]
		label_dx_fig = label_dx_frac * abs(x_right_fig - x_left_fig)

		if pre_i:
			t_pre = fig.text(
				x_fig - label_dx_fig,
				y_mid_fig,
				r"$*_{\mathrm{pre}}$",
				transform=fig.transFigure,
				ha="right",
				va="center",
				fontsize=fontsize,
				color="0.1",
				clip_on=False,
				zorder=51,
			)
			t_pre.set_in_layout(False)
		if post_i:
			t_post = fig.text(
				x_fig + label_dx_fig,
				y_mid_fig,
				r"$*_{\mathrm{post}}$",
				transform=fig.transFigure,
				ha="left",
				va="center",
				fontsize=fontsize,
				color="0.1",
				clip_on=False,
				zorder=51,
			)
			t_post.set_in_layout(False)


def _category_figure_by_label(
	grid: DwPrePostStrategyFacetGrid,
	cat_label: str,
) -> DwPrePostCategoryFigure | None:
	for cat in grid.category_figures:
		if cat.cat_label == cat_label:
			return cat
	return None


def _pre_post_facet_legend_handles(strategy_order: list[str]) -> list[Patch]:
	handles = [
		Patch(facecolor=STRATEGY_COLORS[s], edgecolor="0.2", label=s)
		for s in strategy_order
		if s in STRATEGY_COLORS
	]
	handles.append(
		Patch(
			facecolor="0.85",
			hatch="///",
			edgecolor="0.2",
			label=r"pre $t<t_\mathrm{start}$",
		)
	)
	handles.append(
		Patch(
			facecolor="0.85",
			edgecolor="0.2",
			label=r"post $t>t_\mathrm{start}$",
		)
	)
	return handles


def plot_dw_pre_post_strategy_facets_active_full(
	facet_grid_active: DwPrePostStrategyFacetGrid | None,
	facet_grid_full: DwPrePostStrategyFacetGrid | None,
	*,
	plot_dir: Path | str | None = None,
	show: bool = True,
	filename_suffix: str = "",
) -> None:
	"""Stack active-only (top) and full-surrounding (bottom) per panel; 2 figures (Cat1/Cat2)."""
	if facet_grid_active is None or not facet_grid_active.category_figures:
		print("Skip pre/post Δw strategy facets (no active grid)")
		return
	if facet_grid_full is None or not facet_grid_full.category_figures:
		print("Skip pre/post Δw strategy facets active+full (no full grid)")
		return

	plot_dir = Path(plot_dir) if plot_dir is not None else None

	for cat_active in facet_grid_active.category_figures:
		cat_full = _category_figure_by_label(facet_grid_full, cat_active.cat_label)
		if cat_full is None:
			print(f"Skip {cat_active.cat_label}: no matching full-surrounding category")
			continue

		n_exp = len(cat_active.exp_order)
		n_opt = len(cat_active.opt_order)
		if n_exp == 0 or n_opt == 0:
			continue

		_print_active_full_summary(
			cat_active.subplots,
			cat_full.subplots,
			cat_label=cat_active.cat_label,
		)

		# Squash active/full within each panel; reserve vertical gap between experiments.
		_inner_hspace = 0.02
		_outer_hspace = 0.52
		fig = plt.figure(
			figsize=(3.2 * n_opt, 2.9 * n_exp),
			constrained_layout=False,
		)
		outer_gs = GridSpec(
			n_exp, n_opt, figure=fig, wspace=0.28, hspace=_outer_hspace
		)

		top_axes: list[list] = [[None] * n_opt for _ in range(n_exp)]
		bottom_axes: list[list] = [[None] * n_opt for _ in range(n_exp)]
		active_full_bracket_jobs: list[tuple] = []
		cap_style = dict(y_top_cap=0.2, y_bottom_cap=0.8, riser=0.05, lw=1.6)
		connector_label_style = dict(
			connector_extend_frac=0.35,
			label_dx_frac=0.14,
			fontsize=7.5,
		)

		for i, exp_label in enumerate(cat_active.exp_order):
			for j, oid in enumerate(cat_active.opt_order):
				inner_gs = outer_gs[i, j].subgridspec(
					2, 1, height_ratios=[1, 1], hspace=_inner_hspace
				)
				ax_top = fig.add_subplot(inner_gs[0])
				ax_bottom = fig.add_subplot(inner_gs[1])
				top_axes[i][j] = ax_top
				bottom_axes[i][j] = ax_bottom

				active_sub = cat_active.subplots.get((exp_label, oid))
				full_sub = cat_full.subplots.get((exp_label, oid))
				if active_sub is None and full_sub is None:
					ax_top.set_visible(False)
					ax_bottom.set_visible(False)
					continue

				x_centers: np.ndarray | None = None
				if active_sub is not None:
					x_centers = _plot_pre_post_panel_on_ax(
						ax_top,
						active_sub,
						ylim=cat_active.ylim,
						bracket_direction="up",
						show_ylabel=(j == 0),
						show_xticklabels=False,
					)
					ax_top.tick_params(
						axis="x", which="both", bottom=False, labelbottom=False
					)
				else:
					ax_top.set_visible(False)

				if full_sub is not None:
					x_centers = _plot_pre_post_panel_on_ax(
						ax_bottom,
						full_sub,
						ylim=cat_full.ylim,
						bracket_direction="down",
						show_ylabel=(j == 0),
						show_xticklabels=True,
					)
					ax_bottom.tick_params(axis="x", which="both", top=False, labeltop=False)
				else:
					ax_bottom.set_visible(False)

				if (
					x_centers is not None
					and active_sub is not None
					and full_sub is not None
				):
					sig_af_pre, sig_af_post = _active_full_sig_per_strategy(
						active_sub, full_sub
					)
					_draw_active_full_caps(
						ax_top,
						ax_bottom,
						x_centers,
						sig_af_pre,
						sig_af_post,
						**cap_style,
					)
					active_full_bracket_jobs.append(
						(
							ax_top,
							ax_bottom,
							x_centers,
							sig_af_pre,
							sig_af_post,
						)
					)

				if i == 0:
					ax_top.set_title(format_opt_display(oid), fontsize=8)
				if j == 0:
					ax_top.text(
						-0.5,
						0.0,
						exp_label,
						transform=ax_top.transAxes,
						rotation=90,
						va="center",
						ha="center",
						fontsize=9,
					)
					ax_top.text(
						-0.25,
						0.5,
						"active only",
						transform=ax_top.transAxes,
						rotation=90,
						va="center",
						ha="center",
						fontsize=7,
						color="0.35",
					)
					ax_bottom.text(
						-0.25,
						0.5,
						"all surrounding",
						transform=ax_bottom.transAxes,
						rotation=90,
						va="center",
						ha="center",
						fontsize=7,
						color="0.35",
					)

		for i in range(n_exp):
			row_top_leader = top_axes[i][0]
			row_bot_leader = bottom_axes[i][0]
			for j in range(1, n_opt):
				if top_axes[i][j] is not None and row_top_leader is not None:
					top_axes[i][j].sharey(row_top_leader)
				if bottom_axes[i][j] is not None and row_bot_leader is not None:
					bottom_axes[i][j].sharey(row_bot_leader)

		handles = _pre_post_facet_legend_handles(facet_grid_active.strategy_order)
		fig.legend(
			handles=handles,
			loc="upper center",
			ncol=len(handles),
			fontsize=8,
			bbox_to_anchor=(0.5, 1.00),
		)
		suptitle = (
			r"Mean $\Delta w$ pre/post around $t_\mathrm{start}$"
			f" — {cat_active.cat_label} (active-only top, all surrounding bottom, "
			f"top {facet_grid_active.top_n_per_optimizer} per opt by strategy)"
		)
		fig.suptitle(suptitle, fontsize=11, y=1.03)
		fig.subplots_adjust(top=0.92, bottom=0.06, left=0.12, right=0.98)
		fig.canvas.draw()
		for ax_top, ax_bottom, x_centers, sig_af_pre, sig_af_post in active_full_bracket_jobs:
			_draw_active_full_connectors_and_labels(
				fig,
				ax_top,
				ax_bottom,
				x_centers,
				sig_af_pre,
				sig_af_post,
				**cap_style,
				**connector_label_style,
			)

		if plot_dir is not None:
			out = plot_dir / _pdf_filename(
				f"final_top{facet_grid_active.top_n_per_optimizer}peropt_prepost_strategy_active_and_full_{cat_active.cat_label.lower()}.pdf",
				filename_suffix,
			)
			plt.savefig(out, bbox_inches="tight")

		if show:
			plt.show()
		else:
			plt.close(fig)


def plot_dw_pre_post_strategy_facets(
	facet_grid: DwPrePostStrategyFacetGrid | None,
	*,
	dataset_label: str,
	plot_dir: Path | str | None = None,
	show: bool = True,
) -> None:
	if facet_grid is None or not facet_grid.category_figures:
		print(f"Skip pre/post Δw strategy facets ({dataset_label})")
		return

	plot_dir = Path(plot_dir) if plot_dir is not None else None
	dataset_slug = slugify_label(dataset_label)

	for cat in facet_grid.category_figures:
		n_exp = len(cat.exp_order)
		n_opt = len(cat.opt_order)
		if n_exp == 0 or n_opt == 0:
			continue

		fig, axes = plt.subplots(
			n_exp,
			n_opt,
			figsize=(3.2 * n_opt, 2.6 * n_exp),
			squeeze=False,
			sharey=True,
			constrained_layout=True,
		)

		for i, exp_label in enumerate(cat.exp_order):
			for j, oid in enumerate(cat.opt_order):
				ax = axes[i, j]
				subplot = cat.subplots.get((exp_label, oid))
				if subplot is None:
					ax.set_visible(False)
					continue
				_plot_pre_post_panel_on_ax(ax, subplot, ylim=cat.ylim)
				if i == 0:
					ax.set_title(format_opt_display(oid), fontsize=8)
				if j == 0:
					ax.text(
						-0.42,
						0.5,
						exp_label,
						transform=ax.transAxes,
						rotation=90,
						va="center",
						ha="center",
						fontsize=9,
					)

		handles = [
			Patch(facecolor=STRATEGY_COLORS[s], edgecolor="0.2", label=s)
			for s in facet_grid.strategy_order
			if s in STRATEGY_COLORS
		]
		handles.append(
			Patch(
				facecolor="0.85",
				hatch="///",
				edgecolor="0.2",
				label=r"pre $t<t_\mathrm{start}$",
			)
		)
		handles.append(
			Patch(
				facecolor="0.85",
				edgecolor="0.2",
				label=r"post $t>t_\mathrm{start}$",
			)
		)
		fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8,bbox_to_anchor=(0.5, 1.05))

		suptitle = (
			r"Mean $\Delta w$ pre/post around $t_\mathrm{start}$"
			f" — {cat.cat_label} ({dataset_label}, top {facet_grid.top_n_per_optimizer} per opt by strategy)"
		)
		fig.suptitle(suptitle, fontsize=11, y=1.1)
		if plot_dir is not None:
			out = (
				plot_dir
				/ f"final_top{facet_grid.top_n_per_optimizer}peropt_prepost_strategy_{dataset_slug}_{cat.cat_label.lower()}.pdf"
			)
			plt.savefig(out, bbox_inches="tight")

		if show:
			plt.show()
		else:
			plt.close(fig)
