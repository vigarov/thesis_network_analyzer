#!/usr/bin/env python3
"""Dash + Plotly visualization webapp for neural network analysis.

Run with::

	uv run network-analyzer-webapp
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from dash import (
	Dash,
	Input,
	Output,
	State,
	callback_context,
	clientside_callback,
	dcc,
	html,
	no_update,
)
from visualize_webapp.cache import _cache

# -- Paths --------------------------------------------------------------------
from visualize_webapp.common import (
	_compact_neuron_label,
	_opt_color,
	_opt_display,
	_opt_label,
	_opt_type,
	_sort_optimizer_ids,
)
from visualize_webapp.constants import (
	RESULTS,
	_APP_DIR,
	_NETWORK_EXPECTED_BATCH,
	_NETWORK_N_DIGITS,
	_NETWORK_SAMPLES_PER_DIGIT,
)
from visualize_webapp.io import _metrics, _nts, _pp_dead, _sigs, scan_results
from visualize_webapp.plot_helpers import _trial_boundaries_from_metrics
from models.unit_node_id import parse_unit_node_id


# -- Theme (kept in sync with assets/style.css) ------------------------------

FONT = "Space Grotesk, sans-serif"
C = {
	"bg": "#f4f1ec",
	"card": "#ffffff",
	"fg": "#1c1917",
	"muted": "#78716c",
	"accent": "#b45309",
	"border": "#d6d3d1",
	"grid": "#e7e5e4",
	"blue": "#2563eb",
	"green": "#16a34a",
	"red": "#dc2626",
	"purple": "#7c3aed",
	"orange": "#ea580c",
}

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
	h = hex_color.lstrip("#")
	r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
	return f"rgba({r},{g},{b},{alpha})"


def _sorted_run_ids(runs: dict[str, list[str]]) -> list[str]:
	return sorted(runs.keys())


# -- Data loading (module-level cache) ----------------------------------------

from visualize_webapp.post_processing import (
	_ever_dead_and_ppd_for_diagram,
	compute_recovery_cumsum_by_layer,
	dead_layer_inactive_trace_matrix,
	layer_inactive_count_lookup,
	layer_status_count_matrix,
)


def _training_config_path(eid: str, rid: str, mid: str) -> Path:
	return RESULTS / eid / rid / mid / "config.json"


def _format_training_config_text(
	eid: str | None, mid: str | None, rid: str | None
) -> str:
	"""Pretty-printed `config.json` for the selected run, or a short status message."""
	if not eid or not mid or rid is None:
		return "Select experiment, model, and run to view config.json."
	path = _training_config_path(eid, rid, mid)
	if not path.is_file():
		return f"No config.json found.\n{path}"
	try:
		raw = path.read_text(encoding="utf-8")
		data = json.loads(raw)
	except (OSError, json.JSONDecodeError) as e:
		return f"Could not read config.json: {e}"
	return json.dumps(data, indent=2, ensure_ascii=False, default=str)


# -- Plotly helpers -----------------------------------------------------------


def _style(fig: go.Figure, **kw: Any) -> go.Figure:
	defaults: dict[str, Any] = dict(
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
		font=dict(family=FONT, color=C["fg"], size=12),
		margin=dict(l=55, r=20, t=50, b=70),
		legend=dict(
			bgcolor="rgba(0,0,0,0.3)",
			bordercolor=C["border"],
			borderwidth=1,
			font=dict(size=10),
		),
	)
	defaults.update(kw)
	fig.update_layout(**defaults)
	fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["border"])
	fig.update_yaxes(gridcolor=C["grid"], zerolinecolor=C["border"])
	return fig


def _add_show_hide(fig: go.Figure) -> go.Figure:
	"""Inject Show-all / Hide-all buttons directly into the figure layout."""
	fig.update_layout(
		updatemenus=[
			dict(
				type="buttons",
				direction="left",
				buttons=[
					dict(
						label="Show all",
						method="restyle",
						args=[{"visible": True}],
					),
					dict(
						label="Hide all",
						method="restyle",
						args=[{"visible": "legendonly"}],
					),
				],
				pad={"r": 0, "t": 4},
				showactive=False,
				bgcolor=C["card"],
				bordercolor=C["border"],
				font=dict(size=11, color=C["fg"], family=FONT),
				x=1.0,
				xanchor="right",
				y=1.0,
				yanchor="bottom",
			)
		]
	)
	return fig


def _thin_ticks(
	vals: list, texts: list, max_ticks: int = 20
) -> tuple[list, list]:
	"""Return a thinned-down subset of tick positions and labels."""
	n = len(vals)
	if n <= max_ticks:
		return vals, texts
	step = max(1, n // max_ticks)
	idxs = list(range(0, n, step))
	if idxs[-1] != n - 1:
		idxs.append(n - 1)
	return [vals[i] for i in idxs], [texts[i] for i in idxs]


def _empty(msg: str = "", h: int = 200) -> go.Figure:
	fig = go.Figure()
	_style(fig, height=h)
	if msg:
		fig.add_annotation(
			x=0.5,
			y=0.5,
			xref="paper",
			yref="paper",
			text=msg,
			showarrow=False,
			font=dict(size=14, color=C["muted"]),
		)
	fig.update_xaxes(visible=False)
	fig.update_yaxes(visible=False)
	return fig


def _make_dual_y_axes_symmetrical(fig: go.Figure) -> None:
	"""For subplots with two y axes, set each axis range to [-max_abs, max_abs]
	so that 0 is aligned and both axes are symmetrical about 0."""
	# Collect y data per axis from traces
	axis_ys: dict[str, list[np.ndarray]] = {}
	for trace in fig.data:
		y = getattr(trace, "y", None)
		yaxis = getattr(trace, "yaxis", None)
		if y is None or yaxis is None:
			continue
		arr = np.asarray(y)
		if arr.size == 0 or np.all(np.isnan(arr)):
			continue
		axis_ys.setdefault(yaxis, []).append(arr)

	# Find dual-y pairs: either same domain, or one overlays the other
	layout = fig.layout
	paired_axes: set[str] = set()
	axis_pairs: list[tuple[str, str]] = []
	for i in range(1, 20):
		key = "yaxis" if i == 1 else f"yaxis{i}"
		yax = getattr(layout, key, None)
		if yax is None:
			break
		overlaying = getattr(yax, "overlaying", None)
		if overlaying:
			# This axis overlays another; overlaying is "y", "y2", etc.
			other_key = overlaying if overlaying == "yaxis" else f"yaxis{overlaying[1:]}"
			if overlaying == "y":
				other_key = "yaxis"
			else:
				other_key = f"yaxis{overlaying[1:]}"
			if key not in paired_axes and other_key not in paired_axes:
				axis_pairs.append((other_key, key))
				paired_axes.add(key)
				paired_axes.add(other_key)

	# Also group by shared domain (for make_subplots that set domain on both)
	domain_to_axes: dict[tuple[float, float], list[str]] = {}
	for i in range(1, 20):
		key = "yaxis" if i == 1 else f"yaxis{i}"
		yax = getattr(layout, key, None)
		if yax is None:
			break
		if key in paired_axes:
			continue
		domain = getattr(yax, "domain", None)
		if domain and len(domain) == 2:
			dom_tup = (float(domain[0]), float(domain[1]))
			domain_to_axes.setdefault(dom_tup, []).append(key)

	def _set_symmetrical_range(ax_key: str) -> None:
		trace_ref = "y" if ax_key == "yaxis" else ax_key.replace("yaxis", "y")
		y_arrays = axis_ys.get(trace_ref, [])
		if not y_arrays:
			return
		concat = np.concatenate([np.ravel(a) for a in y_arrays])
		valid = concat[~np.isnan(concat)]
		if len(valid) == 0:
			return
		max_abs = float(np.max(np.abs(valid)))
		if max_abs <= 0:
			max_abs = 1.0
		layout_update: dict[str, Any] = {ax_key: dict(range=[-max_abs, max_abs])}
		fig.update_layout(**layout_update)

	for ax1, ax2 in axis_pairs:
		_set_symmetrical_range(ax1)
		_set_symmetrical_range(ax2)

	for axes_keys in domain_to_axes.values():
		if len(axes_keys) < 2:
			continue
		for ax_key in axes_keys:
			_set_symmetrical_range(ax_key)


def _trial_switch_checkpoint_indices(metrics: dict[str, Any]) -> list[int]:
	"""Checkpoint indices at trial starts, inferred from `tsw_*_entry` tags (legacy: `sw_*_entry`)."""
	tags = metrics.get("checkpoint_tags")
	if tags is None:
		return []
	out: list[int] = []
	for i, t in enumerate(tags):
		s = str(t)
		if s.endswith("_entry") and (s.startswith("tsw_") or s.startswith("sw_")):
			out.append(i)
	return out


def _add_trial_boundaries(
	fig: go.Figure,
	metrics: dict[str, np.ndarray],
	x_mode: str,
	subplot_rows: list[tuple[str, int]] | None = None,
	subplot_xrefs: list[str] | None = None,
) -> go.Figure:
	"""Add vertical lines and training-trial labels to *fig*.

	x_mode: 'checkpoint' or 'iteration'.
	subplot_rows: optional list of (x_mode, row_num) for multi-row figures.
	  When provided, adds boundaries to each row with its x_mode. row_num is 1-based.
	subplot_xrefs: optional explicit Plotly x-axis refs (e.g. `x`, `x2`, …) for
	  multi-column grids; when set, overrides *subplot_rows* xref mapping.
	"""
	rows_config: list[tuple[str, str, str]] = []
	if subplot_xrefs is not None:
		for xref in subplot_xrefs:
			if xref == "x":
				yref = "y domain"
			else:
				yref = f"y{xref[1:]} domain"
			rows_config.append((x_mode, xref, yref))
	elif subplot_rows:
		for mode, r in subplot_rows:
			xref = "x" if r == 1 else f"x{r}"
			yref = "y domain" if r == 1 else f"y{r} domain"
			rows_config.append((mode, xref, yref))
	else:
		rows_config.append((x_mode, "x", "paper"))

	shapes = list(getattr(fig.layout, "shapes", None) or [])
	annotations = list(getattr(fig.layout, "annotations", None) or [])

	seen_modes: set[str] = set()
	for row_mode, xref, yref in rows_config:
		sb = _trial_boundaries_from_metrics(metrics, row_mode)
		if sb is None:
			continue
		trial_names, boundary_xs, use_iters = sb
		if not boundary_xs:
			continue

		for x in boundary_xs:
			shapes.append(
				dict(
					type="line",
					x0=x,
					x1=x,
					y0=0,
					y1=1,
					yref=yref,
					xref=xref,
					layer="above",
					line=dict(color="#000000", width=1.5, dash="dot"),
				)
			)

		# Trial labels: add only once per x_mode to avoid duplicates
		if row_mode in seen_modes:
			continue
		seen_modes.add(row_mode)
		cp_idxs = metrics.get(
			"trial_end_checkpoint_idxs", metrics.get("stage_end_checkpoint_idxs")
		)
		iters = metrics.get(
			"trial_end_iterations", metrics.get("stage_end_iterations")
		)
		n = len(trial_names)
		for i, name in enumerate(trial_names):
			if use_iters and iters is not None:
				start = 0 if i == 0 else float(iters[i - 1]) + 1
				end = float(iters[i])
			elif cp_idxs is not None:
				start = 0 if i == 0 else int(cp_idxs[i - 1]) + 1
				end = int(cp_idxs[i])
			else:
				continue
			mid = (start + end) / 2
			short = (
				name.replace("stage", "")
				.replace("trial", "")
				.replace("run", "r", 1)
				.replace("digit", "d")
				.replace("label", "l")
				.replace("_", "")
				.strip()
			)
			annotations.append(
				dict(
					x=mid,
					y=1.02,
					xref=xref,
					yref="paper",
					text=short,
					showarrow=False,
					font=dict(size=10, color=C["muted"], family=FONT),
					xanchor="center",
				)
			)

	fig.update_layout(shapes=shapes, annotations=annotations)
	return fig


# -- Training overview figures ------------------------------------------------


def _primary_loss_lname(metrics: dict[str, np.ndarray]) -> str | None:
	"""(Priority) pick which loss to show on the loss plot """
	lnames: list[str] = []
	prefix = "loss_"
	# Filter out the prefix
	for key in sorted(metrics):
		if not key.startswith(prefix):
			continue
		ln = key[len(prefix):]
		if ln == "all_train_se":
			continue
		lnames.append(ln)
	if not lnames:
		return None
	# Priority list
	for candidate in (
		"all_train_mean",
		"all_train",
		"all_test",
		"both_test",
		"eval_both_digits_test",
	):
		if candidate in lnames:
			return candidate
	return lnames[0]


def _build_loss(
	eid: str, mid: str, oids: list[str], *, rid: str
) -> go.Figure:
	fig = go.Figure()
	first_m = next(
		(_metrics(eid, mid, o, rid=rid, cache=_cache) for o in oids if _metrics(eid, mid, o, rid=rid, cache=_cache)),
		{},
	)
	tags = [str(t) for t in first_m.get("checkpoint_tags", [])]
	cp_iters = first_m.get("checkpoint_iterations")
	use_iter = cp_iters is not None and len(cp_iters) == len(tags)

	for oid in oids:
		m = _metrics(eid, mid, oid, rid=rid, cache=_cache)
		if not m:
			continue
		m_tags = [str(t) for t in m.get("checkpoint_tags", [])]
		m_cp = m.get("checkpoint_iterations")
		if m_cp is not None and len(m_cp) == len(m_tags):
			m_xs = np.array(m_cp, dtype=float)
		else:
			m_xs = np.arange(len(m_tags))

		col = _opt_color(oid)
		lab = _opt_label(oid)
		hover = [f"iter {int(x)} · {t}" for x, t in zip(m_xs, m_tags)]
		primary_lname = _primary_loss_lname(m)
		for key in sorted(m):
			if not key.startswith("loss_"):
				continue
			lname = key[5:]
			if lname == "all_train_se":
				# SE is used only for shaded band representing the CI of the mean
				continue
			is_primary = lname == primary_lname
			if not is_primary:
				continue
			group = f"{oid}_{lname}"
			# Confidence band: ±1.96 SE shaded region for all_train_mean
			se_key = "loss_all_train_se"
			if lname == "all_train_mean" and se_key in m:
				mean_vals = np.array(m[key], dtype=float)
				se_vals = np.array(m[se_key], dtype=float)
				assert len(mean_vals) == len(se_vals) and len(mean_vals) == len(m_xs)
				# 0 is nan
				se_vals[0] = 0.0
				lower = mean_vals - 1.96 * se_vals
				upper = mean_vals + 1.96 * se_vals
				fill_col = _hex_to_rgba(col, 0.5)
				ci_group = f"{oid}_{lname}_ci"
				xs_list = list(m_xs)
				fig.add_trace(
					go.Scatter(
						x=xs_list + xs_list[::-1],
						y=list(upper) + list(lower)[::-1],
						fill="tozerox",
						fillcolor=fill_col,
						line=dict(color="rgba(255,255,255,0)"),
						showlegend=True,
						hoverinfo="skip",
						legendgroup=ci_group,
						visible="legendonly",
						name=f"{lab} · {lname} - 95% CI",
					)
				)
			fig.add_trace(
				go.Scatter(
					x=m_xs,
					y=m[key],
					mode="lines+markers",
					name=f"{lab} · {lname}",
					text=hover,
					hoverinfo="text+y",
					line=dict(
						color=col,
						width=2,
						dash="solid" if "train" in lname else "dot",
					),
					marker=dict(size=4),
					legendgroup=group,
					showlegend=True,
					visible=True,
				)
			)

	_style(fig, title=dict(text="Loss", font=dict(size=15)), height=380)
	fig.update_layout(legend=dict(groupclick="togglegroup"))
	if tags:
		ref_xs = np.array(cp_iters, dtype=float) if use_iter else np.arange(len(tags))
		tick_labels = [str(int(x)) for x in ref_xs]
		tv, tt = _thin_ticks(list(ref_xs), tick_labels)
		fig.update_xaxes(
			tickvals=tv, ticktext=tt, tickangle=-40,
			title_text="Iteration" if use_iter else "Checkpoint",
		)
	fig.update_yaxes(title_text="Loss")
	_add_trial_boundaries(fig, first_m, "iteration" if use_iter else "checkpoint")
	_add_show_hide(fig)
	return fig


def _build_acc(
	eid: str, mid: str, oids: list[str], *, rid: str
) -> go.Figure:
	fig = go.Figure()
	first_m = next(
		(_metrics(eid, mid, o, rid=rid, cache=_cache) for o in oids if _metrics(eid, mid, o, rid=rid, cache=_cache)),
		{},
	)
	tags = [str(t) for t in first_m.get("checkpoint_tags", [])]
	cp_iters = first_m.get("checkpoint_iterations")
	use_iter = cp_iters is not None and len(cp_iters) == len(tags)

	for oid in oids:
		m = _metrics(eid, mid, oid, rid=rid, cache=_cache)
		if not m:
			continue
		m_tags = [str(t) for t in m.get("checkpoint_tags", [])]
		m_cp = m.get("checkpoint_iterations")
		if m_cp is not None and len(m_cp) == len(m_tags):
			m_xs = np.array(m_cp, dtype=float)
		else:
			m_xs = np.arange(len(m_tags))

		col = _opt_color(oid)
		lab = _opt_label(oid)
		hover = [f"iter {int(x)} · {t}" for x, t in zip(m_xs, m_tags)]
		for key in sorted(m):
			if not key.startswith("acc_"):
				continue
			aname = key[4:]
			fig.add_trace(
				go.Scatter(
					x=m_xs,
					y=m[key],
					mode="lines+markers",
					name=f"{lab} · {aname}",
					text=hover,
					hoverinfo="text+y",
					line=dict(
						color=col,
						width=2,
						dash="solid" if "test" in aname else "dot",
					),
					marker=dict(size=4),
					visible=True if aname in ("all_test") else "legendonly",
				)
			)

	_style(fig, title=dict(text="Accuracy", font=dict(size=15)), height=380)
	if tags:
		ref_xs = np.array(cp_iters, dtype=float) if use_iter else np.arange(len(tags))
		tick_labels = [str(int(x)) for x in ref_xs]
		tv, tt = _thin_ticks(list(ref_xs), tick_labels)
		fig.update_xaxes(
			tickvals=tv, ticktext=tt, tickangle=-40,
			title_text="Iteration" if use_iter else "Checkpoint",
		)
	fig.update_yaxes(title_text="Accuracy", range=[-0.02, 1.05])
	_add_trial_boundaries(fig, first_m, "iteration" if use_iter else "checkpoint")
	_add_show_hide(fig)
	return fig


# -- Model diagram ------------------------------------------------------------


def _parse_units(nts: dict[str, np.ndarray]) -> list[dict[str, Any]]:
	if "unit_node_ids" not in nts:
		return []
	out: list[dict[str, Any]] = []
	for nid in nts["unit_node_ids"]:
		parsed = parse_unit_node_id(str(nid))
		if parsed is not None:
			out.append(dict(parsed))
	return out


def _build_diagram(
	units: list[dict[str, Any]],
	mid: str,
	act_nts: dict[str, np.ndarray] | None = None,
	checkpoint_idx: int | None = None,
	show_hint: bool = False,
	dead_node_ids: set[str] | None = None,
	ppd_node_ids: set[str] | None = None,
) -> go.Figure:
	fig = go.Figure()
	if not units:
		return _empty("No model data", 120)

	layers: dict[str, list[dict[str, Any]]] = {}
	order: list[str] = []
	for u in units:
		ln = u["layer_name"]
		if ln not in layers:
			layers[ln] = []
			order.append(ln)
		layers[ln].append(u)

	n_cols = len(order) + 2
	max_n = max(len(v) for v in layers.values())
	default_dot_sz = max(4, min(7, 350 // max_n))

	# Compute log-scaled sizes per neuron from activation at checkpoint
	log_sizes: dict[str, float] = {}
	act_vals: dict[str, float] = {}
	if act_nts is not None and checkpoint_idx is not None:
		for u in units:
			safe = u["node_id"].replace(":", "__")
			key = f"act__{safe}"
			if key in act_nts:
				arr = act_nts[key]
				t = min(int(checkpoint_idx), len(arr) - 1)
				val = arr[t]
				act_vals[u["node_id"]] = float(
					np.mean(val) if isinstance(val, np.ndarray) else val
				)
		if act_vals:
			nids = list(act_vals.keys())
			raw = np.array([act_vals[n] for n in nids])
			log_raw = np.log1p(np.abs(raw))
			v_min, v_max = log_raw.min(), log_raw.max()
			rng = max(float(v_max - v_min), 1e-8)
			for i, nid in enumerate(nids):
				log_sizes[nid] = 4.0 + 6.0 * float(log_raw[i] - v_min) / rng

	# decorative input node
	fig.add_trace(
		go.Scatter(
			x=[0],
			y=[0.5],
			mode="markers+text",
			marker=dict(
				size=14,
				color=C["muted"],
				symbol="square",
				line=dict(width=1, color=C["border"]),
			),
			text=["In"],
			textposition="bottom center",
			textfont=dict(size=9, color=C["muted"], family=FONT),
			hoverinfo="skip",
			showlegend=False,
		)
	)

	for ci, ln in enumerate(order, 1):
		lu = layers[ln]
		n = len(lu)
		xs = [ci] * n
		# Vertically center each column: same inter-neuron spacing (1/max_n) as the
		# widest layer, but offset so short layers (e.g. head) sit mid-chart—not stuck
		# at the bottom.
		ys = [0.5 + (2 * i + 1 - n) / (2 * max_n) for i in range(n)]
		cdata = [u["node_id"] for u in lu]
		is_conv = lu[0]["unit_type"] == "channel"
		node_colors: list[str] = []
		for u in lu:
			nid = u["node_id"]
			if ppd_node_ids and nid in ppd_node_ids:
				node_colors.append("#000000")
			elif dead_node_ids and nid in dead_node_ids:
				node_colors.append("#808080")
			elif is_conv:
				node_colors.append(C["accent"])
			elif ln == "head":
				node_colors.append(C["green"])
			else:
				node_colors.append(C["blue"])
		sizes = [log_sizes.get(u["node_id"], default_dot_sz) for u in lu]
		if act_vals:
			htxt = [
				f"{ln} · {u['unit_type']} {u['unit_index']}"
				f"<br>activation: {act_vals.get(u['node_id'], 0.0):.4f}"
				+ (
					" [PPD]"
					if ppd_node_ids and u["node_id"] in ppd_node_ids
					else (" [DEAD]" if dead_node_ids and u["node_id"] in dead_node_ids else "")
				)
				for u in lu
			]
		else:
			htxt = [f"{ln} · {u['unit_type']} {u['unit_index']}" for u in lu]

		fig.add_trace(
			go.Scatter(
				x=xs,
				y=ys,
				mode="markers",
				marker=dict(
					size=sizes,
					color=node_colors,
					line=dict(width=0),
				),
				customdata=cdata,
				hovertext=htxt,
				hoverinfo="text",
				showlegend=False,
			)
		)
		display = (
			ln.replace("hidden.", "H")
			.replace("conv", "Conv")
			.replace("fc", "FC")
			.replace("head", "Head")
		)
		fig.add_annotation(
			x=ci,
			y=-0.07,
			text=display,
			showarrow=False,
			font=dict(
				size=10,
				color=C["green"] if ln == "head" else C["muted"],
				family=FONT,
			),
		)

	# decorative output node
	fig.add_trace(
		go.Scatter(
			x=[n_cols - 1],
			y=[0.5],
			mode="markers+text",
			marker=dict(
				size=14,
				color=C["muted"],
				symbol="square",
				line=dict(width=1, color=C["border"]),
			),
			text=["Out"],
			textposition="bottom center",
			textfont=dict(size=9, color=C["muted"], family=FONT),
			hoverinfo="skip",
			showlegend=False,
		)
	)

	if show_hint:
		fig.add_annotation(
			x=0.5,
			y=1.06,
			xref="paper",
			yref="paper",
			text="Select an optimizer above to enable node inspection and activation sizing",
			showarrow=False,
			font=dict(size=11, color=C["muted"], family=FONT),
			bgcolor="rgba(246,241,230,0.9)",
			bordercolor=C["border"],
			borderwidth=1,
		)

	max_sz = max(log_sizes.values()) if log_sizes else default_dot_sz
	height = max(420, max_n * (max_sz + 3) + 160)
	_style(
		fig,
		height=height,
		title=dict(text=f"Architecture — {mid}", font=dict(size=14)),
	)
	fig.update_xaxes(
		showgrid=False,
		zeroline=False,
		showticklabels=False,
		range=[-0.5, n_cols - 0.5],
	)
	fig.update_yaxes(
		showgrid=False,
		zeroline=False,
		showticklabels=False,
		range=[-0.15, 1.1],
	)
	fig.update_layout(clickmode="event")
	return fig


# -- Neuron detail ------------------------------------------------------------

# Grad norm only (row 3)
_GRAD_NORM_SIG = ("grad_norm", "Grad norm", C["red"])

# Cosine similarity row: grad_cosine_sim (all); for Adam also 1st moment cos sim (secondary y)
_COSINE_SIM_SIGS: dict[str, list[tuple[str, str, str, bool]]] = {
	"sgd": [("grad_cosine_sim", "Grad cos. sim.", C["purple"], False)],
	"adam": [
		("grad_cosine_sim", "Grad cos. sim.", C["purple"], False),
		("moment_cosine_sim", "1st moment cos. sim.", C["green"], True),  # secondary y
	],
	"adagrad": [("grad_cosine_sim", "Grad cos. sim.", C["purple"], False)],
	"pure_shampoo": [("grad_cosine_sim", "Grad cos. sim.", C["purple"], False)],
	"grafted_shampoo": [("grad_cosine_sim", "Grad cos. sim.", C["purple"], False)],
}

# Adam moments (exp_avg, exp_avg_sq per-weight vectors) - only for Adam, two y axes
_ADAM_MOMENTS_SIGS: list[tuple[str, str, str, bool]] = [
	("exp_avg", "1st moment (mean)", C["blue"], False),
	("exp_avg_sq", "2nd moment (mean)", C["orange"], True),  # secondary y
]

# Adagrad / grafted Shampoo: effective step scale (per-weight LR-style)
_ADAGRAD_LR_SIG = ("effective_lr", "Effective LR", C["purple"])

# Shampoo: global preconditioner inverse Frobenius norm (replicated per unit in logs)
_H_INV_NORM_SIG = ("h_inv_norm", "Precond. inv. norm", "#0d9488")


def _signal_at_checkpoints(
	sigs: dict[str, np.ndarray],
	sig_name: str,
	col_idx: int,
	cp_iters: np.ndarray,
	unit_safe: str | None = None,
) -> np.ndarray:
	"""Sample signal values at each checkpoint. Returns NaN where out of range.

	For scalar signals: uses sigs[sig_name] 2D matrix (n_iters, n_units).
	For array-valued per-unit signals: uses sigs[f"{sig_name}__{unit_safe}"]
	(n_iters, n_vals) and computes mean over the value dimension for display.
	"""
	if unit_safe is not None:
		per_unit_key = f"{sig_name}__{unit_safe}"
		if per_unit_key in sigs:
			arr = sigs[per_unit_key]
			if arr.ndim >= 2:
				# Compute mean over value dimension for display
				arr = np.mean(arr, axis=tuple(range(1, arr.ndim)))
			n = len(arr)
			vals = []
			for it in cp_iters:
				idx = int(it)
				if 0 <= idx < n:
					vals.append(float(arr[idx]))
				else:
					vals.append(np.nan)
			return np.array(vals)
	if sig_name not in sigs:
		return np.full(len(cp_iters), np.nan)
	arr = sigs[sig_name]
	if arr.ndim != 2 or col_idx >= arr.shape[1]:
		return np.full(len(cp_iters), np.nan)
	n = len(arr)
	vals = []
	for it in cp_iters:
		idx = int(it)
		if 0 <= idx < n:
			vals.append(float(arr[idx, col_idx]))
		else:
			vals.append(np.nan)
	return np.array(vals)


def _rolling_mean_95ci(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Centered rolling mean over exactly `window` checkpoints and 95% CI for that mean.

	Indices where a full window does not fit (series ends) or any value in the window
	is non-finite are left NaN so those checkpoints are not plotted.
	CI uses mean ± 1.96 * s / sqrt(w) (sample std s, w points in window).
	"""
	y = np.asarray(y, dtype=float)
	n = len(y)
	w = max(1, int(window))
	mean = np.full(n, np.nan)
	lower = np.full(n, np.nan)
	upper = np.full(n, np.nan)
	z = 1.96
	if w == 1:
		mean = y.copy()
		lower = np.where(np.isfinite(y), y, np.nan)
		upper = lower.copy()
		return mean, lower, upper

	half_lo = (w - 1) // 2
	half_hi = w - 1 - half_lo
	for i in range(n):
		j0 = i - half_lo
		j1 = i + half_hi + 1
		if j0 < 0 or j1 > n:
			continue
		seg = y[j0:j1]
		if seg.shape[0] != w or np.any(~np.isfinite(seg)):
			continue
		m = float(np.mean(seg))
		mean[i] = m
		sd = float(np.std(seg, ddof=1))
		half = z * sd / np.sqrt(w)
		lower[i] = m - half
		upper[i] = m + half
	return mean, lower, upper


def _true_index_runs(mask: np.ndarray) -> list[tuple[int, int]]:
	"""Contiguous [start, end) index intervals where mask is True."""
	idx = np.flatnonzero(mask)
	if idx.size == 0:
		return []
	runs: list[tuple[int, int]] = []
	s = int(idx[0])
	prev = s
	for k in idx[1:]:
		k = int(k)
		if k != prev + 1:
			runs.append((s, prev + 1))
			s = k
		prev = k
	runs.append((s, prev + 1))
	return runs


def _add_neuron_smoothed_series(
	fig: go.Figure,
	*,
	row: int,
	col: int = 1,
	secondary_y: bool,
	x: np.ndarray,
	y: np.ndarray,
	name: str,
	color: str,
	hover: list[str],
	legend: str,
	smooth_window: int,
) -> None:
	y_arr = np.asarray(y, dtype=float)
	if np.all(np.isnan(y_arr)):
		return
	w = max(1, int(smooth_window))
	group = f"neuron_ts_r{row}_{name}"
	group = "".join(c if c.isalnum() else "_" for c in group)

	trace_kw: dict[str, Any] = dict(row=row, col=col)
	if secondary_y:
		trace_kw["secondary_y"] = True

	if w == 1:
		fig.add_trace(
			go.Scatter(
				x=x,
				y=y_arr,
				mode="lines+markers",
				name=name,
				legend=legend,
				legendgroup=group,
				text=hover,
				hoverinfo="text+y",
				line=dict(color=color, width=2),
				marker=dict(size=5),
			),
			**trace_kw,
		)
		return

	y_mean, y_lo, y_hi = _rolling_mean_95ci(y_arr, w)
	mask = np.isfinite(y_mean)
	if not np.any(mask):
		return
	fill_col = _hex_to_rgba(color, 0.5)
	ci_name = f"{name} - 95% CI"
	for run_i, (a, b) in enumerate(_true_index_runs(mask)):
		sl = slice(a, b)
		x_seg = x[sl]
		y_m = y_mean[sl]
		y_hi_s = y_hi[sl]
		y_lo_s = y_lo[sl]
		xs_list = list(x_seg)
		h_seg = hover[a:b]
		fig.add_trace(
			go.Scatter(
				x=xs_list + xs_list[::-1],
				y=list(y_hi_s) + list(y_lo_s)[::-1],
				fill="tozerox",
				fillcolor=fill_col,
				line=dict(color="rgba(255,255,255,0)"),
				showlegend=run_i == 0,
				hoverinfo="skip",
				legendgroup=group,
				name=ci_name,
				legend=legend,
			),
			**trace_kw,
		)
		fig.add_trace(
			go.Scatter(
				x=x_seg,
				y=y_m,
				mode="lines+markers",
				name=name,
				legend=legend,
				legendgroup=group,
				showlegend=run_i == 0,
				text=h_seg,
				hoverinfo="text+y",
				line=dict(color=color, width=2),
				marker=dict(size=5),
			),
			**trace_kw,
		)


def _add_neuron_dw_over_w_mean_se(
	fig: go.Figure,
	*,
	row: int,
	col: int = 1,
	secondary_y: bool,
	x: np.ndarray,
	y_mean: np.ndarray,
	y_se: np.ndarray,
	name: str,
	color: str,
	hover: list[str],
	legend: str,
	showlegend: bool = True,
) -> None:
	"""Mean dw/w line with ±1.96·SE band (across weights); no rolling smooth over time."""
	y_m = np.asarray(y_mean, dtype=float)
	y_s = np.asarray(y_se, dtype=float)
	if np.all(~np.isfinite(y_m)):
		return
	z = 1.96
	y_lo = y_m - z * y_s
	y_hi = y_m + z * y_s
	mask = np.isfinite(y_m) & np.isfinite(y_lo) & np.isfinite(y_hi)
	if not np.any(mask):
		return
	group = f"neuron_ts_r{row}_{name}"
	group = "".join(c if c.isalnum() else "_" for c in group)
	trace_kw: dict[str, Any] = dict(row=row, col=col)
	if secondary_y:
		trace_kw["secondary_y"] = True
	fill_col = _hex_to_rgba(color, 0.5)
	ci_name = f"{name} - 95% SE"
	for run_i, (a, b) in enumerate(_true_index_runs(mask)):
		sl = slice(a, b)
		x_seg = x[sl]
		y_m_s = y_m[sl]
		y_hi_s = y_hi[sl]
		y_lo_s = y_lo[sl]
		xs_list = list(x_seg)
		h_seg = hover[a:b]
		fig.add_trace(
			go.Scatter(
				x=xs_list + xs_list[::-1],
				y=list(y_hi_s) + list(y_lo_s)[::-1],
				fill="tozerox",
				fillcolor=fill_col,
				line=dict(color="rgba(255,255,255,0)"),
				showlegend=showlegend and run_i == 0,
				hoverinfo="skip",
				legendgroup=group,
				name=ci_name,
				legend=legend,
			),
			**trace_kw,
		)
		fig.add_trace(
			go.Scatter(
				x=x_seg,
				y=y_m_s,
				mode="lines+markers",
				name=name,
				legend=legend,
				legendgroup=group,
				showlegend=showlegend and run_i == 0,
				text=h_seg,
				hoverinfo="text+y",
				line=dict(color=color, width=2),
				marker=dict(size=5),
			),
			**trace_kw,
		)


_NEURON_ACT_GRID_ROWS = 5
_NEURON_ACT_GRID_COLS = 2


def _activation_sample_matrix(
	nts: dict[str, np.ndarray], nid: str
) -> np.ndarray | None:
	"""Return shape (n_checkpoints, n_samples) float array, or None."""
	safe = nid.replace(":", "__")
	key = f"act__{safe}"
	if key not in nts:
		return None
	act_arr = nts[key]
	if act_arr.dtype == object:
		raise ValueError("act_arr.dtype == object")
		# rows: list[np.ndarray] = []
		# for t in range(len(act_arr)):
		#     row = np.asarray(act_arr[t], dtype=np.float64).ravel()
		#     rows.append(row)
		# if not rows:
		#     return None
		# n_s = rows[0].shape[0]
		# if any(r.shape[0] != n_s for r in rows):
		#     return None
		# return np.stack(rows, axis=0)
	if act_arr.ndim < 2:
		return None
	return np.asarray(act_arr, dtype=np.float64)

def _weights_sample_matrix(
	nts: dict[str, np.ndarray], nid: str
) -> np.ndarray | None:
	"""Return shape (n_checkpoints, n_samples) float array, or None."""
	safe = nid.replace(":", "__")
	key = f"weights__{safe}"
	if key not in nts:
		return None
	weights_arr = nts[key]
	if weights_arr.ndim < 2:
		return None
	return np.asarray(weights_arr, dtype=np.float64)



def _head_layer_from_nts(nts: dict[str, np.ndarray]) -> str | None:
	"""Return the head (last) layer name from the unit_node_ids in nts."""
	if "unit_node_ids" not in nts:
		return None
	layer_order: list[str] = []
	for nid in nts["unit_node_ids"]:
		parsed = parse_unit_node_id(str(nid))
		if parsed is None:
			continue
		ln = parsed["layer_name"]
		if ln not in layer_order:
			layer_order.append(ln)
	return layer_order[-1] if layer_order else None


def _activation_sample_matrix_post_nl(
	nts: dict[str, np.ndarray],
	nid: str,
) -> np.ndarray | None:
	"""Like _activation_sample_matrix but with post-nonlinearity transform applied.

	Hidden layers: ReLU.  Head layer: softmax across all head units per sample.
	"""
	A = _activation_sample_matrix(nts, nid)
	if A is None:
		return None
	parsed = parse_unit_node_id(nid)
	if parsed is None:
		return F.relu(torch.from_numpy(A)).numpy()

	head_layer = _head_layer_from_nts(nts)
	if parsed["layer_name"] != head_layer:
		return F.relu(torch.from_numpy(A)).numpy()

	head_nids: list[str] = []
	for uid in nts["unit_node_ids"]:
		p = parse_unit_node_id(str(uid))
		if p and p["layer_name"] == head_layer:
			head_nids.append(str(uid))

	head_acts: list[np.ndarray] = []
	for hn in head_nids:
		ha = _activation_sample_matrix(nts, hn)
		if ha is None:
			return F.relu(torch.from_numpy(A)).numpy()
		head_acts.append(ha)

	stacked = np.stack(head_acts, axis=-1)  # (n_cp, n_samples, n_head)
	sm = torch.softmax(torch.from_numpy(stacked), dim=-1).numpy()
	idx = head_nids.index(nid)
	return sm[:, :, idx]


def _build_neuron_detail_figure(
	nid: str,
	eid: str,
	mid: str,
	oid: str,
	*,
	rid: str,
	smooth_window: int = 1,
	post_nonlinearity: bool = False,
) -> go.Figure:
	"""Build one combined figure: 10-digit activation grid (one trace per eval sample, K=5),
	then shared x-axis rows for dw/w, grad norm, cosine sim, Adam / Shampoo / LR signals.
	"""
	nts = _nts(eid, mid, oid, rid=rid, cache=_cache)
	sigs = _sigs(eid, mid, oid, rid=rid, cache=_cache)
	metrics = _metrics(eid, mid, oid, rid=rid, cache=_cache)
	otype = _opt_type(oid)

	tags = [str(t) for t in nts.get("checkpoint_tags", [])]
	cp_iters = metrics.get("checkpoint_iterations")
	if cp_iters is None or len(cp_iters) != len(tags):
		return _empty("Checkpoint iterations not available (re-run training)", 200)
	x_iters = np.array(cp_iters, dtype=float)

	has_sigs = bool(sigs) and "unit_node_ids" in sigs
	uid_list = [str(n) for n in sigs.get("unit_node_ids", [])] if has_sigs else []
	ci = uid_list.index(nid) if nid in uid_list else -1

	safe = nid.replace(":", "__")
	if post_nonlinearity:
		A = _activation_sample_matrix_post_nl(nts, nid)
	else:
		A = _activation_sample_matrix(nts, nid)
	if A is None or A.shape[1] != _NETWORK_EXPECTED_BATCH:
		return _empty(
			f"Neuron detail needs {_NETWORK_EXPECTED_BATCH} eval samples "
			f"(10 digits x 5); this run has "
			f"{0 if A is None else A.shape[1]}.",
			200,
		)

	has_adam_moments = has_sigs and otype == "adam"
	has_shampoo_h_inv = has_sigs and otype in ("pure_shampoo", "grafted_shampoo")
	has_eff_lr = has_sigs and otype in ("adagrad", "grafted_shampoo")
	has_grad_cos = has_sigs  # grad norm + cosine sim rows

	n_metric_rows = (
		1
		+ (2 if has_grad_cos else 0)
		+ (1 if has_adam_moments else 0)
		+ (1 if has_shampoo_h_inv else 0)
		+ (1 if has_eff_lr else 0)
	)
	total_fig_rows = _NEURON_ACT_GRID_ROWS + n_metric_rows

	specs: list[list[dict | None]] = [
		[{}, {}] for _ in range(_NEURON_ACT_GRID_ROWS)
	]
	specs.append([{"colspan": 2}, None])
	if has_grad_cos:
		specs.append([{"colspan": 2}, None])
		specs.append(
			[{"secondary_y": True, "colspan": 2}, None]
			if otype == "adam"
			else [{"colspan": 2}, None]
		)
	if has_adam_moments:
		specs.append([{"secondary_y": True, "colspan": 2}, None])
	if has_shampoo_h_inv:
		specs.append([{"colspan": 2}, None])
	if has_eff_lr:
		specs.append([{"colspan": 2}, None])

	subplot_titles: list[str | None] = [f"Digit {d}" for d in range(_NETWORK_N_DIGITS)]
	subplot_titles.append(
		r"$\frac{|\Delta w|}{|w|} \text{(mean ± 95% SE over weights)}$"
	)
	if has_grad_cos:
		subplot_titles.extend(["Grad norm", "Cosine similarity"])
	if has_adam_moments:
		subplot_titles.append("Adam moments")
	if has_shampoo_h_inv:
		subplot_titles.append("Precond. inv. norm")
	if has_eff_lr:
		subplot_titles.append("Effective LR")

	row_heights: list[float] | None = [0.11] * _NEURON_ACT_GRID_ROWS + [
		0.22
	] * n_metric_rows

	fig = make_subplots(
		rows=total_fig_rows,
		cols=2,
		shared_xaxes=True,
		shared_yaxes=False,
		specs=specs,
		vertical_spacing=0.05,
		horizontal_spacing=0.06,
		subplot_titles=tuple(subplot_titles),
		row_heights=row_heights,
	)

	hover = [f"iter {int(x)} · {t}" for x, t in zip(x_iters, tags)]
	tv, tt = _thin_ticks(list(x_iters), [str(int(x)) for x in x_iters])

	k_samp = _NETWORK_SAMPLES_PER_DIGIT
	for d in range(_NETWORK_N_DIGITS):
		block = A[:, d * k_samp : (d + 1) * k_samp].astype(np.float64, copy=False)
		r_1b = d // _NEURON_ACT_GRID_COLS + 1
		c_1b = d % _NEURON_ACT_GRID_COLS + 1
		trace_kw: dict[str, Any] = dict(row=r_1b, col=c_1b)
		subplot_legend = "legend" if d == 0 else f"legend{d + 1}"
		for k in range(k_samp):
			y_k = block[:, k]
			fig.add_trace(
				go.Scatter(
					x=x_iters,
					y=y_k,
					mode="lines+markers",
					name=f"K={k + 1}",
					legend=subplot_legend,
					legendgroup=f"eval_k_{k}",
					showlegend=(d == 0),
					text=hover,
					hoverinfo="text+y",
					line=dict(color=_NETWORK_EVAL_K_LINE_COLORS[k], width=1.5),
					marker=dict(size=3),
				),
				**trace_kw,
			)

	def _legend_for_subplot(si: int) -> str:
		return "legend" if si == 1 else f"legend{si}"

	leg_si = _NETWORK_N_DIGITS + 1
	r_m = _NEURON_ACT_GRID_ROWS + 1
	r_dw = r_m
	r_grad: int | None = None
	r_cos: int | None = None
	r_adam: int | None = None
	r_sh: int | None = None
	r_lr: int | None = None

	weights_key = f"weights__{safe}"
	if weights_key in nts:
		weights = nts[weights_key]
		assert weights.ndim == 2, (
			f"Expected weights shape (num_checkpoints, weight_vector_size), got {weights.shape}"
		)
		w_arr = weights.astype(np.float64, copy=False)
		n_w = int(w_arr.shape[1])
		num = np.abs(np.diff(w_arr, axis=0))
		denom = np.abs(w_arr[:-1])
		dw_over_w = np.where(denom > 1e-12, num / denom, 0.0)
		mean_dw = np.mean(dw_over_w, axis=1)
		std_dw = np.std(dw_over_w, axis=1, ddof=1)
		if n_w <= 1:
			se_dw = np.zeros_like(mean_dw)
		else:
			se_dw = std_dw / np.sqrt(n_w)
		x_w = x_iters[1:]
		hover_w = hover[1:]
		_add_neuron_dw_over_w_mean_se(
			fig,
			row=r_dw,
			col=1,
			secondary_y=False,
			x=x_w,
			y_mean=mean_dw,
			y_se=se_dw,
			name=r"Mean $|\Delta w|/|w|$",
			color=C["accent"],
			hover=hover_w,
			legend=_legend_for_subplot(leg_si),
		)
	r_m += 1
	leg_si += 1

	if has_grad_cos:
		r_grad = int(r_m)
		if ci >= 0:
			sname, slabel, scolor = _GRAD_NORM_SIG
			y = _signal_at_checkpoints(sigs, sname, ci, x_iters)
			_add_neuron_smoothed_series(
				fig,
				row=r_grad,
				col=1,
				secondary_y=False,
				x=x_iters,
				y=y,
				name=slabel,
				color=scolor,
				hover=hover,
				legend=_legend_for_subplot(leg_si),
				smooth_window=smooth_window,
			)
		r_m += 1
		leg_si += 1

		r_cos = int(r_m)
		if ci >= 0:
			for sname, slabel, scolor, sec_y in _COSINE_SIM_SIGS.get(otype, []):
				y = _signal_at_checkpoints(sigs, sname, ci, x_iters)
				_add_neuron_smoothed_series(
					fig,
					row=r_cos,
					col=1,
					secondary_y=sec_y,
					x=x_iters,
					y=y,
					name=slabel,
					color=scolor,
					hover=hover,
					legend=_legend_for_subplot(leg_si),
					smooth_window=smooth_window,
				)
		r_m += 1
		leg_si += 1

	if has_adam_moments and ci >= 0:
		r_adam = int(r_m)
		for sname, slabel, scolor, sec_y in _ADAM_MOMENTS_SIGS:
			y = _signal_at_checkpoints(
				sigs, sname, ci, x_iters, unit_safe=safe,
			)
			_add_neuron_smoothed_series(
				fig,
				row=r_adam,
				col=1,
				secondary_y=sec_y,
				x=x_iters,
				y=y,
				name=slabel,
				color=scolor,
				hover=hover,
				legend=_legend_for_subplot(leg_si),
				smooth_window=smooth_window,
			)
		r_m += 1
		leg_si += 1

	if has_shampoo_h_inv and ci >= 0:
		r_sh = int(r_m)
		sname, slabel, scolor = _H_INV_NORM_SIG
		y = _signal_at_checkpoints(sigs, sname, ci, x_iters)
		_add_neuron_smoothed_series(
			fig,
			row=r_sh,
			col=1,
			secondary_y=False,
			x=x_iters,
			y=y,
			name=slabel,
			color=scolor,
			hover=hover,
			legend=_legend_for_subplot(leg_si),
			smooth_window=smooth_window,
		)
		r_m += 1
		leg_si += 1

	if has_eff_lr and ci >= 0:
		r_lr = int(r_m)
		sname, slabel, scolor = _ADAGRAD_LR_SIG
		y = _signal_at_checkpoints(sigs, sname, ci, x_iters, unit_safe=safe)
		if np.all(np.isnan(y)) and "effective_lr_mean" in sigs:
			y = _signal_at_checkpoints(sigs, "effective_lr_mean", ci, x_iters)
		_add_neuron_smoothed_series(
			fig,
			row=r_lr,
			col=1,
			secondary_y=False,
			x=x_iters,
			y=y,
			name=slabel,
			color=scolor,
			hover=hover,
			legend=_legend_for_subplot(leg_si),
			smooth_window=smooth_window,
		)

	# X axes: ticks on all; "Iteration" title on bottom row only
	for r in range(1, _NEURON_ACT_GRID_ROWS + 1):
		for c in range(1, _NEURON_ACT_GRID_COLS + 1):
			fig.update_xaxes(
				tickvals=tv,
				ticktext=tt,
				tickangle=-40,
				title_text="",
				showticklabels=True,
				row=r,
				col=c,
			)
	for r in range(_NEURON_ACT_GRID_ROWS + 1, total_fig_rows + 1):
		fig.update_xaxes(
			tickvals=tv,
			ticktext=tt,
			tickangle=-40,
			title_text="Iteration" if r == total_fig_rows else "",
			showticklabels=True,
			row=r,
			col=1,
		)

	for r in range(1, _NEURON_ACT_GRID_ROWS + 1):
		for c in range(1, _NEURON_ACT_GRID_COLS + 1):
			fig.update_yaxes(title_text="Activation", row=r, col=c)

	fig.update_yaxes(title_text=r"$|\Delta w|/|w|$", row=r_dw, col=1)
	if r_grad is not None:
		fig.update_yaxes(title_text="Norm", row=r_grad, col=1)
	if r_cos is not None:
		fig.update_yaxes(title_text="Grad cos. sim.", row=r_cos, col=1)
		if otype == "adam":
			fig.update_yaxes(
				title_text="1st moment cos. sim.",
				row=r_cos,
				col=1,
				secondary_y=True,
			)
	if has_adam_moments and r_adam is not None:
		fig.update_yaxes(title_text="1st moment (mean)", row=r_adam, col=1)
		fig.update_yaxes(
			title_text="2nd moment (mean)", row=r_adam, col=1, secondary_y=True
		)
	if has_shampoo_h_inv and r_sh is not None:
		fig.update_yaxes(title_text="Precond. inv. norm", row=r_sh, col=1)
	if has_eff_lr and r_lr is not None:
		fig.update_yaxes(title_text="Effective LR", row=r_lr, col=1)

	height = int(_NEURON_ACT_GRID_ROWS * 150 + n_metric_rows * 280)
	_style(fig, height=height)

	trial_boundary_xrefs: list[str] = ["x"] + [f"x{i}" for i in range(2, _NETWORK_N_DIGITS + 1)]
	for j in range(_NETWORK_N_DIGITS + 1, _NETWORK_N_DIGITS + 1 + n_metric_rows):
		trial_boundary_xrefs.append(f"x{j}")
	_add_trial_boundaries(
		fig,
		metrics,
		"iteration",
		subplot_xrefs=trial_boundary_xrefs,
	)

	_legend_style: dict[str, Any] = dict(
		bgcolor="rgba(0,0,0,0.3)",
		bordercolor=C["border"],
		borderwidth=1,
		font=dict(size=10),
		xanchor="right",
		x=1.0,
		xref="paper",
		yref="paper",
		visible=True,
	)
	seen_domains: list[tuple[float, float]] = []
	for i in range(1, 64):
		yax_key = "yaxis" if i == 1 else f"yaxis{i}"
		try:
			yax = getattr(fig.layout, yax_key, None)
		except AttributeError:
			break
		if yax is None:
			break
		domain = getattr(yax, "domain", None)
		if domain and len(domain) == 2:
			dom_tup = (float(domain[0]), float(domain[1]))
			if dom_tup not in seen_domains:
				seen_domains.append(dom_tup)
	n_leg_panels = _NETWORK_N_DIGITS + n_metric_rows
	legend_updates: dict[str, Any] = {}
	for si in range(1, n_leg_panels + 1):
		if si <= _NETWORK_N_DIGITS and si > 1:
			continue
		leg_key = "legend" if si == 1 else f"legend{si}"
		if si <= len(seen_domains):
			y_center = (seen_domains[si - 1][0] + seen_domains[si - 1][1]) / 2
			legend_updates[leg_key] = dict(
				**_legend_style,
				y=y_center,
				yanchor="middle",
			)
	if legend_updates:
		fig.update_layout(**legend_updates)

	_make_dual_y_axes_symmetrical(fig)
	_add_show_hide(fig)
	return fig


# -- Network analysis (eval-batch activation scatter) -------------------------

# Plotly sequential colorscales: one per eval-sample index k in {0..4}.
# Plotly built-in names (`speed` is lowercase in plotly.express.colors.sequential).
_NETWORK_EVAL_K_COLORSCALES: tuple[str, ...] = (
	"Teal",
	"speed",
	"Purp",
	"Brwnyl",
	"Greys_r",
)

# Solid line colors for eval-sample index k (matches K order above; see MNIST eval_ds: 5 per digit).
_NETWORK_EVAL_K_LINE_COLORS: tuple[str, ...] = (
	"#14b8a6",
	"#2563eb",
	"#7c3aed",
	"#a16207",
	"#57534e",
)


def _network_linear_regression_line(
	xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
	"""Return `(x_line, y_line)` for a least-squares fit, or None if ill-defined."""
	xs = np.asarray(xs, dtype=np.float64).ravel()
	ys = np.asarray(ys, dtype=np.float64).ravel()
	m = np.isfinite(xs) & np.isfinite(ys)
	if m.sum() < 2:
		return None
	xs = xs[m]
	ys = ys[m]
	coef = np.polyfit(xs, ys, 1)
	slope, icept = float(coef[0]), float(coef[1])
	x0, x1 = float(xs.min()), float(xs.max())
	if x0 == x1:
		x0 -= 1e-6
		x1 += 1e-6
	xl = np.array([x0, x1], dtype=np.float64)
	yl = slope * xl + icept
	return xl, yl


def _build_network_activation_figure(
	nid: str,
	eid: str,
	mid: str,
	oid: str,
	*,
	rid: str,
	k_visible: list[bool] | tuple[bool, ...] | None = None,
	post_nonlinearity: bool = False,
) -> go.Figure:
	"""10 subplots (2x5): per digit, scatter x=prev activation, y=consecutive delta-act.

	Eval batch layout matches MNIST base: 50 samples, index `5*d + k` for digit `d`, sample `k`.
	"""
	kv: list[bool]
	if k_visible is None or len(k_visible) != _NETWORK_SAMPLES_PER_DIGIT:
		kv = [True] * _NETWORK_SAMPLES_PER_DIGIT
	else:
		kv = [bool(x) for x in k_visible]

	nts = _nts(eid, mid, oid, rid=rid, cache=_cache)
	metrics = _metrics(eid, mid, oid, rid=rid, cache=_cache)
	if post_nonlinearity:
		A = _activation_sample_matrix_post_nl(nts, nid)
	else:
		A = _activation_sample_matrix(nts, nid)
	if A is None:
		return _empty("No activation timeseries for this unit.", h=400)

	n_cp, n_samp = A.shape
	if n_cp < 2:
		return _empty("Need at least 2 checkpoints for consecutive Δ.", h=400)
	if n_samp != _NETWORK_EXPECTED_BATCH:
		return _empty(
			f"Network view expects { _NETWORK_EXPECTED_BATCH } eval samples "
			f"(10 digits x 5); this run has {n_samp}.",
			h=400,
		)

	tags = [str(t) for t in nts.get("checkpoint_tags", [])]
	cp_iters = metrics.get("checkpoint_iterations")
	if cp_iters is not None and len(cp_iters) == len(tags):
		it_list = [int(x) for x in cp_iters]
	else:
		it_list = list(range(len(tags)))

	subplot_titles = [f"Digit {d}" for d in range(_NETWORK_N_DIGITS)]
	fig = make_subplots(
		rows=5,
		cols=2,
		subplot_titles=subplot_titles,
		vertical_spacing=0.07,
		horizontal_spacing=0.14,
	)

	cmin = 1
	cmax = max(1, n_cp - 1)

	# Per-digit bounds for symmetric axes around 0 (visible points only): x from prev act, y from Δact.
	axis_bound_x: list[float] = []
	axis_bound_y: list[float] = []

	for d in range(_NETWORK_N_DIGITS):
		row = d // 2 + 1
		col = d % 2 + 1
		xs_digit: list[float] = []
		ys_digit: list[float] = []
		for k in range(_NETWORK_SAMPLES_PER_DIGIT):
			if not kv[k]:
				continue
			i = _NETWORK_SAMPLES_PER_DIGIT * d + k
			for t in range(1, n_cp):
				prev_a = float(A[t - 1, i])
				cur_a = float(A[t, i])
				xs_digit.append(prev_a)
				ys_digit.append(cur_a - prev_a)

		if xs_digit:
			arr_x = np.asarray(xs_digit, dtype=np.float64)
			arr_y = np.asarray(ys_digit, dtype=np.float64)
			bx = float(np.nanmax(np.abs(arr_x)))
			by = float(np.nanmax(np.abs(arr_y)))
			if not np.isfinite(bx) or bx <= 0:
				bx = 1e-6
			if not np.isfinite(by) or by <= 0:
				by = 1e-6
			axis_bound_x.append(bx * 1.05)
			axis_bound_y.append(by * 1.05)
		else:
			axis_bound_x.append(1.0)
			axis_bound_y.append(1.0)

		for k in range(_NETWORK_SAMPLES_PER_DIGIT):
			i = _NETWORK_SAMPLES_PER_DIGIT * d + k
			xs: list[float] = []
			ys: list[float] = []
			colors: list[float] = []
			htext: list[str] = []
			for t in range(1, n_cp):
				prev_a = float(A[t - 1, i])
				cur_a = float(A[t, i])
				xs.append(prev_a)
				ys.append(cur_a - prev_a)
				colors.append(float(t))
				tag = tags[t] if t < len(tags) else "?"
				it = it_list[t] if t < len(it_list) else t
				htext.append(
					f"digit {d} · K={k + 1} · step t={t}<br>"
					f"iter {it} · {tag}<br>"
					f"prev act: {prev_a:.5f}<br>"
					f"Δact: {cur_a - prev_a:.5f}"
				)

			cs = _NETWORK_EVAL_K_COLORSCALES[k]
			show_leg = d == 0
			trace_name = f"K={k + 1} ({cs})"
			first_trace = d == 0 and k == 0
			mk: dict[str, Any] = dict(
				size=7,
				opacity=0.6,
				color=colors,
				colorscale=cs,
				cmin=cmin,
				cmax=cmax,
			)
			if first_trace:
				mk["showscale"] = True
				mk["colorbar"] = dict(
					title=dict(text="Checkpoint step t", font=dict(size=10)),
					len=0.28,
					thickness=12,
					x=1.02,
					xanchor="left",
					y=1.0,
					yanchor="top",
				)
			else:
				mk["showscale"] = False

			fig.add_trace(
				go.Scatter(
					x=xs,
					y=ys,
					mode="markers",
					marker=mk,
					text=htext,
					hoverinfo="text",
					name=trace_name,
					legendgroup=f"k{k}",
					showlegend=show_leg,
					visible=kv[k],
				),
				row=row,
				col=col,
			)

		# Linear regression on visible points only for this digit.
		xs_reg = np.asarray(xs_digit, dtype=np.float64)
		ys_reg = np.asarray(ys_digit, dtype=np.float64)
		reg = _network_linear_regression_line(xs_reg, ys_reg)
		if reg is not None:
			xl, yl = reg
			fig.add_trace(
				go.Scatter(
					x=xl,
					y=yl,
					mode="lines",
					line=dict(color="rgba(28,25,23,0.85)", width=2, dash="dash"),
					name="OLS fit",
					legendgroup="reg",
					showlegend=False,
					hoverinfo="skip",
				),
				row=row,
				col=col,
			)

	x_title = r"$\text{Activation at } t-1$"
	y_title = r"$\Delta \text{activation}$"
	for i_d, d in enumerate(range(_NETWORK_N_DIGITS)):
		ri = d // 2 + 1
		ci = d % 2 + 1
		bndx = axis_bound_x[i_d]
		bndy = axis_bound_y[i_d]
		fig.update_xaxes(
			title_text=x_title,
			range=[-bndx, bndx],
			zeroline=True,
			zerolinewidth=1,
			zerolinecolor=C["border"],
			row=ri,
			col=ci,
		)
		fig.update_yaxes(
			title_text=y_title,
			range=[-bndy, bndy],
			zeroline=True,
			zerolinewidth=1,
			zerolinecolor=C["border"],
			row=ri,
			col=ci,
		)

	total_h = 260 * 5
	_style(
		fig,
		height=total_h,
		margin=dict(l=72, r=120, t=88, b=48),
		title=dict(
			text=f"Eval-batch activation deltas — {nid}",
			font=dict(size=14),
		),
		legend=dict(
			x=0.01,
			y=1.0,
			xanchor="left",
			yanchor="top",
			bgcolor="rgba(255,255,255,0.92)",
			bordercolor=C["border"],
			borderwidth=1,
			font=dict(size=10),
			itemclick=False,
			itemdoubleclick=False,
		),
	)
	return fig


# -- Network tab builder functions --------------------------------------------

_DIGIT_COLORS = (
	"#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
	"#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#469990",
)

_LAYER_COMBINED_COLORS = (
	"#636efa",
	"#ef553b",
	"#00cc96",
	"#ab63fa",
	"#ffa15a",
	"#19d3f3",
	"#ff6692",
	"#b6e880",
	"#ff97ff",
	"#fecb52",
)


def _build_recovery_cumsum_plot(
	df_dead: pd.DataFrame,
	metrics: dict[str, np.ndarray],
	layer_order: list[str],
) -> go.Figure:
	"""Line chart: cumulative dead→active recoveries per hidden layer (not head)."""
	hidden = [ln for ln in layer_order if ln != "head"]
	all_cp_idxs, by_layer = compute_recovery_cumsum_by_layer(df_dead, layer_order)
	if not hidden or not all_cp_idxs:
		return _empty("No recovery data (need dead-neuron table and hidden layers).", h=220)

	tags = [str(t) for t in metrics.get("checkpoint_tags", [])]
	cp_iters = metrics.get("checkpoint_iterations")
	use_iter = cp_iters is not None and len(cp_iters) == len(tags)

	fig = go.Figure()
	any_trace = False
	for li, layer_name in enumerate(hidden):
		cum = by_layer.get(layer_name)
		if cum is None or len(cum) != len(all_cp_idxs):
			continue
		color = _LAYER_COMBINED_COLORS[li % len(_LAYER_COMBINED_COLORS)]
		x_vals = [
			float(cp_iters[ci]) if use_iter and cp_iters is not None else float(ci)
			for ci in all_cp_idxs
		]
		display_name = layer_name.replace("hidden.", "H").replace("head", "Head")
		fig.add_trace(
			go.Scatter(
				x=x_vals,
				y=cum.tolist(),
				mode="lines+markers",
				name=display_name,
				line=dict(width=1.5, color=color),
				marker=dict(size=5, color=color),
			)
		)
		any_trace = True

	if not any_trace:
		return _empty("No recovery traces to plot.", h=220)

	_style(
		fig,
		title=dict(
			text="Cumulative recovery count (dead -> active) — hidden layers",
			font=dict(size=13),
		),
		height=380,
	)
	fig.update_xaxes(title_text="Iteration" if use_iter else "Checkpoint")
	fig.update_yaxes(title_text="Cumulative recoveries")
	_add_trial_boundaries(fig, metrics, "iteration" if use_iter else "checkpoint")
	return fig


def _build_layer_inactive_count(
	layer_name: str,
	df_nd: pd.DataFrame,
	metrics: dict[str, np.ndarray],
) -> go.Figure:
	"""Line chart: number of inactive neurons per digit over checkpoints."""
	all_cp_idxs, lookup = layer_inactive_count_lookup(df_nd, layer_name)

	tags = [str(t) for t in metrics.get("checkpoint_tags", [])]
	cp_iters = metrics.get("checkpoint_iterations")
	use_iter = cp_iters is not None and len(cp_iters) == len(tags)

	fig = go.Figure()
	for d in range(_NETWORK_N_DIGITS):
		x_full: list[float] = []
		y_full: list[int] = []
		for ci in all_cp_idxs:
			x_full.append(
				float(cp_iters[ci]) if use_iter and cp_iters is not None else float(ci)
			)
			y_full.append(int(lookup.get((ci, d), 0)))
		fig.add_trace(
			go.Scatter(
				x=x_full,
				y=y_full,
				mode="lines+markers",
				name=f"Digit {d}",
				line=dict(color=_DIGIT_COLORS[d], width=1.5),
				marker=dict(size=3),
			)
		)

	display_name = (
		layer_name.replace("hidden.", "H").replace("head", "Head")
	)
	_style(
		fig,
		title=dict(
			text=f"Inactive neuron count per digit — {display_name}",
			font=dict(size=13),
		),
		height=300,
	)
	fig.update_xaxes(title_text="Iteration" if use_iter else "Checkpoint")
	fig.update_yaxes(title_text="Count")
	_add_trial_boundaries(fig, metrics, "iteration" if use_iter else "checkpoint")
	return fig


def _build_layer_digit_count_plot(
	layer_name: str,
	df_nd: pd.DataFrame,
	metrics: dict[str, np.ndarray],
	status: str,
	title_verb: str,
	*,
	boxplot: bool = False,
) -> go.Figure:
	"""Per-neuron digit-count for *status* across checkpoints.

	For each (neuron, checkpoint), count how many digits have the given *status*.

	* `boxplot=True`: boxplot of the distribution across neurons (legacy).
	* `boxplot=False` (default): mean ± 1.96·SE ribbon across neurons, plus mean
	  (line+markers) and max (markers) per checkpoint.
	"""
	all_cp_idxs, _all_neurons, mat = layer_status_count_matrix(df_nd, layer_name, status)

	tags = [str(t) for t in metrics.get("checkpoint_tags", [])]
	cp_iters = metrics.get("checkpoint_iterations")
	use_iter = cp_iters is not None and len(cp_iters) == len(tags)

	display_name = (
		layer_name.replace("hidden.", "H").replace("head", "Head")
	)
	base_color = C["blue"] if status == "assigned" else C["red"]

	if boxplot:
		all_x: list[str] = []
		all_y: list[int] = []
		for cp_i, ci in enumerate(all_cp_idxs):
			label = (
				str(int(cp_iters[ci])) if use_iter and cp_iters is not None else str(ci)
			)
			vals = mat[:, cp_i]
			all_x.extend([label] * len(vals))
			all_y.extend(vals.tolist())

		fig = go.Figure()
		fig.add_trace(
			go.Box(
				x=all_x,
				y=all_y,
				boxpoints="outliers",
				marker_color=base_color,
				showlegend=False,
			)
		)
		title_text = f"{title_verb} digit count distribution — {display_name}"
	else:
		x_labels: list[str] = []
		means: list[float] = []
		ses: list[float] = []
		maxs: list[float] = []
		for cp_i, ci in enumerate(all_cp_idxs):
			label = (
				str(int(cp_iters[ci])) if use_iter and cp_iters is not None else str(ci)
			)
			x_labels.append(label)
			vals = mat[:, cp_i].astype(np.float64)
			n = int(vals.size)
			m = float(np.mean(vals)) if n else 0.0
			if n > 1:
				se = float(np.std(vals, ddof=1) / np.sqrt(n))
			else:
				se = 0.0
			means.append(m)
			ses.append(se)
			maxs.append(float(np.max(vals)) if n else 0.0)

		upper = [m + 1.96 * s for m, s in zip(means, ses)]
		lower = [m - 1.96 * s for m, s in zip(means, ses)]
		fill_rgba = _hex_to_rgba(base_color, 0.22)

		fig = go.Figure()
		fig.add_trace(
			go.Scatter(
				x=x_labels,
				y=upper,
				mode="lines",
				line=dict(width=0),
				showlegend=False,
				hoverinfo="skip",
			)
		)
		fig.add_trace(
			go.Scatter(
				x=x_labels,
				y=lower,
				mode="lines",
				line=dict(width=0),
				fillcolor=fill_rgba,
				fill="tonexty",
				name="Mean ± 1.96·SE",
				hoverinfo="skip",
			)
		)
		fig.add_trace(
			go.Scatter(
				x=x_labels,
				y=means,
				mode="lines+markers",
				name="Mean",
				line=dict(width=1.5, color=base_color),
				marker=dict(size=6, color=base_color),
				hovertemplate="%{x}<br>mean %{y:.2f}<extra></extra>",
			)
		)
		fig.add_trace(
			go.Scatter(
				x=x_labels,
				y=maxs,
				mode="markers",
				name="Max",
				marker=dict(size=8, color=base_color, symbol="x", line=dict(width=1)),
				hovertemplate="%{x}<br>max %{y:.0f}<extra></extra>",
			)
		)
		title_text = f"{title_verb} digit count (mean ± 1.96·SE) — {display_name}"

	_style(
		fig,
		title=dict(
			text=title_text,
			font=dict(size=13),
		),
		height=300,
	)
	fig.update_xaxes(title_text="Iteration" if use_iter else "Checkpoint")
	fig.update_yaxes(title_text=f"# {status} digits", range=[-0.5, _NETWORK_N_DIGITS + 0.5])
	_add_trial_boundaries(fig, metrics, "iteration" if use_iter else "checkpoint")
	return fig


def _build_layer_digit_count_non_head_combined(
	df_nd: pd.DataFrame,
	metrics: dict[str, np.ndarray],
	layer_order: list[str],
	status: str,
	title_verb: str,
) -> go.Figure:
	"""Mean ± 1.96·SE ribbon per non-head layer on one figure (numeric x-axis)."""
	non_head = [ln for ln in layer_order if ln != "head"]
	if not non_head:
		return _empty("No non-head layers.", h=200)

	tags = [str(t) for t in metrics.get("checkpoint_tags", [])]
	cp_iters = metrics.get("checkpoint_iterations")
	use_iter = cp_iters is not None and len(cp_iters) == len(tags)

	fig = go.Figure()
	any_trace = False
	for li, layer_name in enumerate(non_head):
		color = _LAYER_COMBINED_COLORS[li % len(_LAYER_COMBINED_COLORS)]
		all_cp_idxs, _all_neurons, mat = layer_status_count_matrix(df_nd, layer_name, status)
		if not len(all_cp_idxs):
			continue

		x_vals: list[float] = []
		means: list[float] = []
		ses: list[float] = []
		for cp_i, ci in enumerate(all_cp_idxs):
			x_vals.append(
				float(cp_iters[ci]) if use_iter and cp_iters is not None else float(ci)
			)
			vals = mat[:, cp_i].astype(np.float64)
			n = int(vals.size)
			m = float(np.mean(vals)) if n else 0.0
			if n > 1:
				se = float(np.std(vals, ddof=1) / np.sqrt(n))
			else:
				se = 0.0
			means.append(m)
			ses.append(se)

		upper = [m + 1.96 * s for m, s in zip(means, ses)]
		lower = [m - 1.96 * s for m, s in zip(means, ses)]
		fill_rgba = _hex_to_rgba(color, 0.22)
		display_name = layer_name.replace("hidden.", "H").replace("head", "Head")

		x_poly = x_vals + x_vals[::-1]
		y_poly = upper + lower[::-1]
		fig.add_trace(
			go.Scatter(
				x=x_poly,
				y=y_poly,
				fill="toself",
				fillcolor=fill_rgba,
				line=dict(width=0),
				mode="lines",
				showlegend=False,
				legendgroup=display_name,
				hoverinfo="skip",
			)
		)
		fig.add_trace(
			go.Scatter(
				x=x_vals,
				y=means,
				mode="lines+markers",
				name=display_name,
				legendgroup=display_name,
				line=dict(width=1.5, color=color),
				marker=dict(size=5, color=color),
			)
		)
		any_trace = True

	if not any_trace:
		return _empty("No checkpoint data for non-head layers.", h=200)

	title_text = f"{title_verb} digit count (mean ± 1.96·SE) — all layers except Head"
	_style(
		fig,
		title=dict(
			text=title_text,
			font=dict(size=13),
		),
		height=400,
	)
	fig.update_xaxes(title_text="Iteration" if use_iter else "Checkpoint")
	fig.update_yaxes(
		title_text=f"# {status} digits",
		range=[-0.5, _NETWORK_N_DIGITS + 0.5],
	)
	_add_trial_boundaries(fig, metrics, "iteration" if use_iter else "checkpoint")
	return fig


def _build_dead_layer_traces(
	layer_name: str,
	df_nd: pd.DataFrame,
	df_dead: pd.DataFrame,
	metrics: dict[str, np.ndarray],
	*,
	neuron_ids: set[str] | frozenset[str] | list[str] | None = None,
	title: str | None = None,
	empty_message: str | None = None,
) -> go.Figure:
	"""Per dead neuron: inactive-digit count over time (one trace per dead neuron)."""
	all_cp_idxs, ever_dead_nids, traces = dead_layer_inactive_trace_matrix(
		layer_name,
		df_nd,
		df_dead,
		neuron_ids=neuron_ids,
	)

	empty_default = "No dead neurons in this layer."
	if len(ever_dead_nids) == 0:
		return _empty(empty_message if empty_message is not None else empty_default, h=200)

	tags = [str(t) for t in metrics.get("checkpoint_tags", [])]
	cp_iters = metrics.get("checkpoint_iterations")
	use_iter = cp_iters is not None and len(cp_iters) == len(tags)

	fig = go.Figure()
	for row_i, nid in enumerate(ever_dead_nids):
		xs: list[float] = []
		ys: list[int] = []
		for cp_i, ci in enumerate(all_cp_idxs):
			xs.append(
				float(cp_iters[ci]) if use_iter and cp_iters is not None else float(ci)
			)
			ys.append(int(traces[row_i, cp_i]))
		fig.add_trace(
			go.Scatter(
				x=xs,
				y=ys,
				mode="lines+markers",
				name=_compact_neuron_label(nid),
				marker=dict(size=3),
				line=dict(width=1.5),
			)
		)

	display_name = (
		layer_name.replace("hidden.", "H").replace("head", "Head")
	)
	head = title if title is not None else "Dead neurons"
	_style(
		fig,
		title=dict(
			text=f"{head} — inactive digit count — {display_name}",
			font=dict(size=13),
		),
		height=320,
	)
	fig.update_xaxes(title_text="Iteration" if use_iter else "Checkpoint")
	fig.update_yaxes(
		title_text="# inactive digits",
		range=[-0.5, _NETWORK_N_DIGITS + 0.5],
	)
	_add_trial_boundaries(fig, metrics, "iteration" if use_iter else "checkpoint")
	_add_show_hide(fig)
	return fig


def _build_dead_neuron_pre_nl_grid(
	nid: str,
	eid: str,
	mid: str,
	oid: str,
	*,
	rid: str,
) -> go.Figure:
	"""2x5 grid of pre-NL activation over time per digit (N=10, K=5 eval samples each).

	Batch layout matches `MNISTWrapper.evaluation_inputs` / checkpoint capture:
	sample index `5*d + k` for digit `d` and eval-sample slot `k` in `{0..4}`.
	"""
	nts = _nts(eid, mid, oid, rid=rid, cache=_cache)
	metrics = _metrics(eid, mid, oid, rid=rid, cache=_cache)
	A = _activation_sample_matrix(nts, nid)

	tags = [str(t) for t in nts.get("checkpoint_tags", [])]
	cp_iters_raw = metrics.get("checkpoint_iterations")
	if cp_iters_raw is None or len(cp_iters_raw) != len(tags):
		return _empty("Checkpoint iterations not available", 200)
	x_iters = np.array(cp_iters_raw, dtype=float)

	if A is None or A.shape[1] != _NETWORK_EXPECTED_BATCH:
		return _empty("Activation data unavailable for this neuron.", 200)

	subplot_titles = [f"Digit {d}" for d in range(_NETWORK_N_DIGITS)]
	fig = make_subplots(
		rows=_NEURON_ACT_GRID_ROWS,
		cols=_NEURON_ACT_GRID_COLS,
		subplot_titles=subplot_titles,
		vertical_spacing=0.08,
		horizontal_spacing=0.08,
	)

	for d in range(_NETWORK_N_DIGITS):
		r = d // _NEURON_ACT_GRID_COLS + 1
		c = d % _NEURON_ACT_GRID_COLS + 1
		block = A[:, d * _NETWORK_SAMPLES_PER_DIGIT : (d + 1) * _NETWORK_SAMPLES_PER_DIGIT]
		for k in range(_NETWORK_SAMPLES_PER_DIGIT):
			y_k = block[:, k]
			fig.add_trace(
				go.Scatter(
					x=x_iters,
					y=y_k,
					mode="lines+markers",
					name=f"K={k + 1}",
					line=dict(color=_NETWORK_EVAL_K_LINE_COLORS[k], width=1.5),
					marker=dict(size=3),
					showlegend=(d == 0),
				),
				row=r,
				col=c,
			)
		y_min = float(np.nanmin(block)) if np.any(np.isfinite(block)) else -1.0
		fig.update_yaxes(
			range=[y_min * 1.1 if y_min < 0 else -0.1, 0.0],
			row=r,
			col=c,
		)
		fig.update_xaxes(title_text="Iteration" if r == _NEURON_ACT_GRID_ROWS else "", row=r, col=c)

	_style(
		fig,
		title=dict(
			text=f"Pre-NL activation (dead neuron {_compact_neuron_label(nid)})",
			font=dict(size=13),
		),
		height=_NEURON_ACT_GRID_ROWS * 160,
	)
	_add_trial_boundaries(fig, metrics, "iteration")
	return fig


def _build_dead_neuron_inactivity_bar(
	nid: str,
	pp_ta: dict[str, np.ndarray],
) -> go.Figure:
	"""Bar chart: per-digit inactivity ratio from the full training-set forward pass."""
	safe = nid.replace(":", "__")
	key = f"act__{safe}"
	labels = pp_ta.get("digit_labels")
	acts = pp_ta.get(key)

	if labels is None or acts is None:
		return _empty("Training-set activation data not available.", 200)

	labels = np.asarray(labels, dtype=int)
	acts = np.asarray(acts, dtype=np.float64)
	post_nl = F.relu(torch.from_numpy(acts)).numpy()

	digits = list(range(_NETWORK_N_DIGITS))
	ratios: list[float] = []
	for d in digits:
		mask = labels == d
		n_total = int(mask.sum())
		if n_total == 0:
			ratios.append(0.0)
			continue
		n_inactive = int(np.sum(post_nl[mask] == 0))
		ratios.append(n_inactive / n_total)

	fig = go.Figure()
	fig.add_trace(
		go.Bar(
			x=[str(d) for d in digits],
			y=ratios,
			marker_color=[_DIGIT_COLORS[d] for d in digits],
			showlegend=False,
		)
	)

	_style(
		fig,
		title=dict(
			text=f"Inactivity ratio on training set — {_compact_neuron_label(nid)}",
			font=dict(size=13),
		),
		height=300,
	)
	fig.update_xaxes(title_text="Digit")
	fig.update_yaxes(title_text="Inactivity ratio", range=[0, 1.05])
	return fig


# -- Dash app -----------------------------------------------------------------

app = Dash(
	__name__,
	title="Network Analyzer",
	assets_folder=str(_APP_DIR / "assets"),
	external_stylesheets=[
		"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap",
	],
	suppress_callback_exceptions=True,
)


def _serve_layout() -> html.Div:
	"""Regenerate layout on each page load for a fresh results scan."""
	tree = scan_results(cache=_cache, refresh=True)
	exp_opts = [{"label": e, "value": e} for e in sorted(tree)]

	return html.Div(
		className="app-shell",
		children=[
			# -- header --
			html.Header(
				className="app-header",
				children=[
					html.H1("Network Analyzer")
				],
			),
			# -- top-level controls --
			html.Section(
				className="controls-row",
				children=[
					html.Div(
						[
							html.Label("Experiment"),
							dcc.Dropdown(
								id="dd-experiment",
								options=exp_opts,
								placeholder="Select experiment…",
								clearable=False,
							),
						],
						className="ctrl",
					),
					html.Div(
						[
							html.Label("Model"),
							dcc.Dropdown(
								id="dd-model",
								placeholder="Select model…",
								clearable=False,
							),
						],
						className="ctrl",
					),
					html.Div(
						[
							html.Label("Run"),
							dcc.Dropdown(
								id="dd-run",
								placeholder="Select run…",
								clearable=False,
							),
						],
						className="ctrl",
					),
				],
			),
			html.Section(
				className="section section-config",
				children=[
					html.Details(
						className="config-details",
						open=False,
						children=[
							html.Summary("Training config (config.json)"),
							html.Pre(
								id="pre-training-config",
								className="config-json-pre",
								children=_format_training_config_text(None, None, None),
							),
						],
					),
				],
			),
			# -- section 1: training overview --
			html.Section(
				className="section",
				children=[
					html.H2("Training Overview"),
					html.Div(
						className="graph-row",
						children=[
							dcc.Graph(
								id="graph-loss",
								figure=_empty(
									"Select experiment, model & run", 380
								),
								mathjax=True,
								className="graph-half",
							),
							dcc.Graph(
								id="graph-acc",
								figure=_empty("", 380),
								mathjax=True,
								className="graph-half",
							),
						],
					),
				],
			),
			# -- section 2: neuron / network analysis --
			html.Section(
				className="section",
				children=[
					html.Div(
						className="controls-row",
						children=[
							html.Div(
								[
									html.Label("Optimizer"),
									dcc.Dropdown(
										id="dd-optimizer",
										placeholder="Select optimizer…",
										clearable=False,
									),
								],
								className="ctrl",
							),
						],
					),
					dcc.Graph(
						id="graph-diagram",
						figure=_empty(
							"Select an experiment & model to view architecture",
							200,
						),
						mathjax=True,
					),
					html.Div(
						className="slider-container",
						children=[
							html.Label(
								"Training progress",
								className="slider-label",
							),
							html.Div(
								className="slider-with-play",
								children=[
									html.Button(
										"▶ Play",
										id="btn-slider-autoplay",
										className="slider-play-btn",
										type="button",
										disabled=True,
										title="Animate checkpoints (5s per full sweep, loops until paused)",
										n_clicks=0,
									),
									html.Div(
										className="slider-track-wrap",
										children=[
											dcc.Slider(
												id="slider-checkpoint",
												min=0,
												max=0,
												step=1,
												value=0,
												marks={},
												disabled=True,
												updatemode="drag",
												tooltip={
													"placement": "bottom",
													"always_visible": False,
												},
											),
										],
									),
								],
							),
							html.Div(
								className="neuron-smooth-row",
								title="Rolling mean window in checkpoints (≥1). "
								"Endpoints are omitted until a full window fits; shaded band is 95% CI.",
								children=[
									html.Label(
										"Smooth",
										htmlFor="input-neuron-smooth",
										className="neuron-smooth-label",
									),
									html.Div(
										className="neuron-smooth-control",
										children=[
											html.Div(
												className="neuron-smooth-input-wrap",
												children=[
													dcc.Input(
														id="input-neuron-smooth",
														type="number",
														min=1,
														step=1,
														value=1,
														debounce=True,
														className="neuron-smooth-input",
													),
												],
											),
											html.Div(
												className="neuron-spin-stack",
												children=[
													html.Button(
														"▴",
														id="btn-neuron-smooth-inc",
														className="neuron-spin-btn",
														type="button",
														title="Increase window",
														n_clicks=0,
													),
													html.Button(
														"▾",
														id="btn-neuron-smooth-dec",
														className="neuron-spin-btn",
														type="button",
														title="Decrease window",
														n_clicks=0,
													),
												],
											),
										],
									),
								],
							),
						],
					),
					dcc.Store(id="store-slider-autoplay", data=False),
					dcc.Interval(
						id="interval-slider-autoplay",
						interval=2500,
						n_intervals=0,
						disabled=True,
						max_intervals=-1,
					),
					html.Div(
						id="panel-neuron-analysis",
						style={"display": "block"},
						children=[
							html.H2("Neuron Analysis"),
							html.Div(
								className="neuron-viz-toggles",
								children=[
									html.Div(
										[
											html.Label("Viz type"),
											dcc.RadioItems(
												id="radio-viz-type",
												options=[
													{"label": "Timeseries", "value": "timeseries"},
													{"label": "Scatter", "value": "scatter"},
												],
												value="timeseries",
												inline=True,
												className="analysis-mode-radio",
												inputClassName="analysis-mode-radio-input",
												labelClassName="analysis-mode-radio-label",
											),
										],
										className="ctrl-inline",
									),
									html.Div(
										[
											html.Label("Non-linearity"),
											dcc.RadioItems(
												id="radio-nonlinearity",
												options=[
													{"label": "Pre NL", "value": "pre"},
													{"label": "Post NL", "value": "post"},
												],
												value="pre",
												inline=True,
												className="analysis-mode-radio",
												inputClassName="analysis-mode-radio-input",
												labelClassName="analysis-mode-radio-label",
											),
										],
										className="ctrl-inline",
									),
								],
							),
							html.Div(
								id="k-checklist-wrap",
								className="network-k-checklist-wrap",
								style={"display": "none"},
								children=[
									html.Label(
										"Visible K",
										htmlFor="checklist-network-k-visible",
										className="network-k-checklist-label",
									),
									dcc.Checklist(
										id="checklist-network-k-visible",
										className="network-k-checklist",
										options=[
											{"label": "K=1 (Teal)", "value": "0"},
											{"label": "K=2 (speed)", "value": "1"},
											{"label": "K=3 (Purp)", "value": "2"},
											{"label": "K=4 (Brwnyl)", "value": "3"},
											{"label": "K=5 (Greys_r)", "value": "4"},
										],
										value=["0", "1", "2", "3", "4"],
										inline=True,
										inputClassName="network-k-check-input",
										labelClassName="network-k-check-label",
									),
								],
							),
							html.Div(
								className="neuron-detail",
								children=[
									html.P(
										"Click a node in the diagram above to inspect it.",
										id="hint-text",
										className="hint-text",
									),
									dcc.Graph(
										id="graph-neuron-detail",
										figure=_empty("", 100),
										mathjax=True,
										className="neuron-graph",
									),
								],
							),
						],
					),
				],
			),
			# client-side store for the last-clicked node id
			dcc.Store(id="store-node"),
			# list of checkpoint iteration numbers (int) for the current optimizer,
			# or null when falling back to checkpoint-index mode
			dcc.Store(id="store-cp-iters", data=None),
			# Prefetched diagram figures (current + next checkpoints) for client-side hits
			dcc.Store(id="store-diagram-cache", data=None),
		],
	)


app.layout = _serve_layout


# -- Callbacks ----------------------------------------------------------------


@app.callback(
	Output("dd-model", "options"),
	Output("dd-model", "value"),
	Input("dd-experiment", "value"),
)
def _cb_models(eid: str | None):
	if not eid:
		return [], None
	tree = scan_results(cache=_cache)
	ms = sorted(tree.get(eid, {}))
	opts = [{"label": m, "value": m} for m in ms]
	return opts, (ms[0] if len(ms) == 1 else None)


@app.callback(
	Output("dd-run", "options"),
	Output("dd-run", "value"),
	Input("dd-experiment", "value"),
	Input("dd-model", "value"),
)
def _cb_run(eid: str | None, mid: str | None):
	if not eid or not mid:
		return [], None
	tree = scan_results(cache=_cache)
	runs = tree.get(eid, {}).get(mid, {})
	rids = _sorted_run_ids(runs)
	if not rids:
		return [], None
	opts = [{"label": r, "value": r} for r in rids]
	return opts, (rids[0] if len(rids) == 1 else None)


@app.callback(
	Output("pre-training-config", "children"),
	Input("dd-experiment", "value"),
	Input("dd-model", "value"),
	Input("dd-run", "value"),
)
def _cb_training_config_preview(
	eid: str | None, mid: str | None, rid: str | None
):
	return _format_training_config_text(eid, mid, rid)


@app.callback(
	Output("graph-loss", "figure"),
	Output("graph-acc", "figure"),
	Output("dd-optimizer", "options"),
	Output("dd-optimizer", "value"),
	Input("dd-experiment", "value"),
	Input("dd-model", "value"),
	Input("dd-run", "value"),
)
def _cb_training(eid: str | None, mid: str | None, rid: str | None):
	if not eid or not mid or rid is None:
		return (
			_empty("Select experiment, model & run", 380),
			_empty("", 380),
			[],
			None,
		)
	tree = scan_results(cache=_cache)
	oids = tree.get(eid, {}).get(mid, {}).get(rid, [])
	if not oids:
		return (
			_empty("No results yet", 380),
			_empty("", 380),
			[],
			None,
		)
	opts = [{"label": o, "value": o} for o in oids]
	return (
		_build_loss(eid, mid, oids, rid=rid),
		_build_acc(eid, mid, oids, rid=rid),
		opts,
		oids[0] if len(oids) == 1 else None,
	)


@app.callback(
	Output("slider-checkpoint", "max"),
	Output("slider-checkpoint", "marks"),
	Output("slider-checkpoint", "value"),
	Output("slider-checkpoint", "disabled"),
	Output("slider-checkpoint", "step"),
	Output("store-cp-iters", "data"),
	Input("dd-experiment", "value"),
	Input("dd-model", "value"),
	Input("dd-run", "value"),
	Input("dd-optimizer", "value"),
)
def _cb_slider(
	eid: str | None, mid: str | None, rid: str | None, oid: str | None
):
	if not eid or not mid or rid is None or not oid:
		return 0, {}, 0, True, 1, None
	nts = _nts(eid, mid, oid, rid=rid, cache=_cache)
	metrics = _metrics(eid, mid, oid, rid=rid, cache=_cache)
	tags = [str(t) for t in nts.get("checkpoint_tags", [])]
	n = len(tags)
	if n == 0:
		return 0, {}, 0, True, 1, None

	cp_iters_raw = metrics.get("checkpoint_iterations")
	use_iter = cp_iters_raw is not None and len(cp_iters_raw) == n

	# Choose which checkpoint indices get a visible label (uniformly sampled).
	MAX_MARKS = 40
	if n <= MAX_MARKS:
		labeled_idxs = list(range(n))
	else:
		step = max(1, n // MAX_MARKS)
		labeled_idxs = list(range(0, n, step))
		if labeled_idxs[-1] != n - 1:
			labeled_idxs.append(n - 1)

	trial_tick = "\u275A"
	trial_tick_style = {"fontSize": "14px", "color": C["accent"], "fontWeight": "700"}
	label_style = {"fontSize": "10px", "color": C["muted"]}

	if use_iter and cp_iters_raw is not None:
		# Slider value space = actual iteration numbers so the tooltip is meaningful.
		cp_list: list[int] = [int(x) for x in cp_iters_raw]
		slider_max = cp_list[-1]
		slider_val = 0 if 0 in set(cp_list) else cp_list[0]
		labeled_set = {cp_list[i] for i in labeled_idxs}


		trial_switch_indices = _trial_switch_checkpoint_indices(metrics)
		trial_switch_iters = {cp_list[sw] for sw in trial_switch_indices if 0 <= sw < n}
		visible_iters = set(labeled_set) | trial_switch_iters

		marks: dict[int, Any] = {}
		for it in visible_iters:
			if it in trial_switch_iters:
				marks[it] = {"label": trial_tick, "style": trial_tick_style}
			else:
				marks[it] = {"label": str(it), "style": label_style}

		return slider_max, marks, slider_val, False, None, cp_list

	else:
		# Fallback: slider value = checkpoint index (0..n-1), step=1.
		marks_idx: dict[int, Any] = {
			i: {"label": tags[i], "style": label_style}
			for i in labeled_idxs
		}
		for sw in _trial_switch_checkpoint_indices(metrics):
			if sw < 0 or sw >= n:
				continue
			if sw in marks_idx:
				m = marks_idx[sw]
				lab = m.get("label", "") if isinstance(m, dict) else str(m)
				st = dict(m.get("style") or {}) if isinstance(m, dict) else {}
				label_text = f"{trial_tick} {lab}" if lab else trial_tick
				marks_idx[sw] = {"label": label_text, "style": {**st, **trial_tick_style}}
			else:
				marks_idx[sw] = {"label": trial_tick, "style": trial_tick_style}
		# Start at checkpoint index 0 when optimizer changes.
		return n - 1, marks_idx, 0, False, 1, None


def _autoplay_step_ms(n_checkpoints: int) -> int:
	"""Milliseconds between checkpoint steps so a full sweep takes ~2.5s."""
	if n_checkpoints <= 1:
		return 2500
	return max(1, int(round(2500.0 / (n_checkpoints - 1))))


@app.callback(
	Output("btn-slider-autoplay", "disabled"),
	Input("slider-checkpoint", "max"),
)
def _cb_autoplay_btn_disabled(smax: int):
	return smax <= 0


@app.callback(
	Output("store-slider-autoplay", "data"),
	Output("interval-slider-autoplay", "disabled"),
	Output("interval-slider-autoplay", "interval"),
	Output("btn-slider-autoplay", "children"),
	Output("slider-checkpoint", "value", allow_duplicate=True),
	Input("btn-slider-autoplay", "n_clicks"),
	Input("dd-experiment", "value"),
	Input("dd-model", "value"),
	Input("dd-run", "value"),
	Input("dd-optimizer", "value"),
	Input("slider-checkpoint", "max"),
	State("store-slider-autoplay", "data"),
	State("store-cp-iters", "data"),
	prevent_initial_call=True,
)
def _cb_autoplay_control(
	n_clicks: int | None,
	eid: str | None,
	mid: str | None,
	rid: str | None,
	oid: str | None,
	smax: int,
	playing: bool,
	cp_list: list[int] | None,
):
	trig_comp = (
		callback_context.triggered[0]["prop_id"].rsplit(".", 1)[0]
		if callback_context.triggered
		else None
	)

	n_cp = len(cp_list) if cp_list else (smax + 1)
	step_ms = _autoplay_step_ms(n_cp)

	if trig_comp == "btn-slider-autoplay":
		if smax <= 0:
			return False, True, step_ms, "▶ Play", no_update
		nxt = not playing
		if nxt:
			return True, False, step_ms, "⏸ Pause", no_update
		return False, True, step_ms, "▶ Play", no_update

	# Experiment / model / optimizer / max changed: stop autoplay and sync timing
	return False, True, step_ms, "▶ Play", no_update


@app.callback(
	Output("slider-checkpoint", "value", allow_duplicate=True),
	Input("interval-slider-autoplay", "n_intervals"),
	State("store-slider-autoplay", "data"),
	State("slider-checkpoint", "max"),
	State("slider-checkpoint", "value"),
	State("store-cp-iters", "data"),
	prevent_initial_call=True,
)
def _cb_autoplay_tick(
	_n: int,
	playing: bool,
	smax: int,
	val: int | None,
	cp_list: list[int] | None,
):
	if not playing or smax <= 0 or val is None:
		return no_update
	if cp_list:
		v = int(val)
		for it in cp_list:
			if it > v:
				return it
		return cp_list[0]  # wrap around to the beginning
	# Fallback: checkpoint-index mode — advance by 1.
	v = int(val)
	return v + 1 if v < smax else 0


def _slider_val_to_cp_idx(
	slider_val: int | None,
	cp_iters: np.ndarray | None,
	n: int,
) -> int | None:
	"""Convert a slider value to a checkpoint array index.

	When cp_iters is available the slider value is an iteration number; we
	find the checkpoint whose iteration is closest (exact match preferred).
	Otherwise the slider value is already the checkpoint index.
	"""
	if slider_val is None:
		return None
	if cp_iters is None or len(cp_iters) != n:
		return int(slider_val)
	v = int(slider_val)
	cp_list = [int(x) for x in cp_iters]
	try:
		return cp_list.index(v)
	except ValueError:
		dists = [abs(x - v) for x in cp_list]
		return dists.index(min(dists))


# Diagram slider: prefetch this many checkpoints (including current) per server response.
_DIAGRAM_PREFETCH_CP = 10


def _slider_value_for_checkpoint_index(
	cp_idx: int,
	cp_iters: np.ndarray | None,
	n_checkpoints: int,
) -> int:
	"""Slider `value` that selects checkpoint index `cp_idx` (iteration or index mode)."""
	if cp_iters is not None and len(cp_iters) == n_checkpoints:
		return int(cp_iters[cp_idx])
	return int(cp_idx)


_DIAGRAM_CACHE_CLIENT_JS = r"""
function(sv, cache, eid, mid, runId, oid) {
	if (cache === null || cache === undefined) return window.dash_clientside.no_update;
	if (!cache.figures || !cache.ctx) return window.dash_clientside.no_update;
	var c = cache.ctx;
	if (c.e !== eid || c.m !== mid || c.run !== runId) return window.dash_clientside.no_update;
	var implicit = (oid === null || oid === undefined);
	if (implicit) {
		if (!c.implicit) return window.dash_clientside.no_update;
	} else if (c.r !== oid) {
		return window.dash_clientside.no_update;
	}
	if (sv === null || sv === undefined) return window.dash_clientside.no_update;
	var k = String(Math.round(Number(sv)));
	if (Object.prototype.hasOwnProperty.call(cache.figures, k)) {
		return cache.figures[k];
	}
	return window.dash_clientside.no_update;
}
"""


@app.callback(
	Output("graph-diagram", "figure"),
	Output("store-diagram-cache", "data"),
	Input("dd-experiment", "value"),
	Input("dd-model", "value"),
	Input("dd-run", "value"),
	Input("dd-optimizer", "value"),
	Input("slider-checkpoint", "value"),
	State("store-diagram-cache", "data"),
)
def _cb_diagram(
	eid: str | None,
	mid: str | None,
	rid: str | None,
	oid: str | None,
	slider_val: int | None,
	prev_store: dict[str, Any] | None,
):
	if eid is None or mid is None or rid is None:
		return _empty("Select an experiment, model & run to view architecture", 200), None
	tree = scan_results(cache=_cache)
	oids_avail = tree.get(eid, {}).get(mid, {}).get(rid, [])
	if not oids_avail:
		return _empty("No data available", 200), None
	ref_oid = oid if oid else oids_avail[0]
	nts = _nts(eid, mid, ref_oid, rid=rid, cache=_cache)
	units = _parse_units(nts)
	n_cp = len(nts.get("checkpoint_tags", []))
	m = _metrics(eid, mid, ref_oid, rid=rid, cache=_cache)
	cp_iters = m.get("checkpoint_iterations") if m else None

	df_dead = _pp_dead(eid, mid, ref_oid, rid=rid, cache=_cache) if oid else None
	gray_dead, ever_ppd = _ever_dead_and_ppd_for_diagram(df_dead)

	def _fig_at_checkpoint(ci: int | None) -> go.Figure:
		return _build_diagram(
			units,
			mid,
			act_nts=nts if oid else None,
			checkpoint_idx=ci,
			show_hint=(oid is None),
			dead_node_ids=gray_dead,
			ppd_node_ids=ever_ppd,
		)

	if n_cp == 0 or slider_val is None:
		return _fig_at_checkpoint(None), None

	cp_idx = _slider_val_to_cp_idx(slider_val, cp_iters, n_cp)
	if cp_idx is None:
		return _fig_at_checkpoint(None), None
	new_ctx = {
		"e": eid,
		"m": mid,
		"run": rid,
		"r": ref_oid,
		"implicit": oid is None,
	}
	prev = prev_store or {}
	prev_figures: dict[str, Any] = prev.get("figures") or {}
	ctx_match = prev.get("ctx") == new_ctx

	figures_json: dict[str, Any] = {}
	for k in range(min(_DIAGRAM_PREFETCH_CP, n_cp)):
		idx = (cp_idx + k) % n_cp
		sv_key = _slider_value_for_checkpoint_index(idx, cp_iters, n_cp)
		key = str(int(sv_key))
		if ctx_match and key in prev_figures:
			payload = prev_figures[key]
		else:
			payload = _fig_at_checkpoint(idx).to_plotly_json()
		figures_json[key] = payload

	canon_key = str(int(_slider_value_for_checkpoint_index(cp_idx, cp_iters, n_cp)))
	skid = str(int(slider_val))
	if skid != canon_key and canon_key in figures_json:
		figures_json[skid] = figures_json[canon_key]

	current_fig = go.Figure(figures_json[skid if skid in figures_json else canon_key])
	store = {"ctx": new_ctx, "figures": figures_json}
	return current_fig, store


clientside_callback(
	_DIAGRAM_CACHE_CLIENT_JS,
	Output("graph-diagram", "figure", allow_duplicate=True),
	Input("slider-checkpoint", "value"),
	Input("store-diagram-cache", "data"),
	State("dd-experiment", "value"),
	State("dd-model", "value"),
	State("dd-run", "value"),
	State("dd-optimizer", "value"),
	prevent_initial_call=True,
)


@app.callback(
	Output("store-node", "data"),
	Input("graph-diagram", "clickData"),
	prevent_initial_call=True,
)
def _cb_click(click_data: dict | None):
	if not click_data:
		return no_update
	pts = click_data.get("points", [])
	if not pts:
		return no_update
	val = pts[0].get("customdata")
	if val is None:
		return no_update
	return val


def _parse_neuron_smooth_window(raw: Any) -> int:
	try:
		if raw is None or raw == "":
			return 1
		w = int(float(raw))
	except (TypeError, ValueError):
		return 1
	return max(1, w)


@app.callback(
	Output("input-neuron-smooth", "value"),
	Input("btn-neuron-smooth-inc", "n_clicks"),
	Input("btn-neuron-smooth-dec", "n_clicks"),
	State("input-neuron-smooth", "value"),
	prevent_initial_call=True,
)
def _cb_neuron_smooth_spin(
	_n_inc: int | None,
	_n_dec: int | None,
	raw: Any,
):
	if not callback_context.triggered:
		return no_update
	prop = callback_context.triggered[0]["prop_id"]
	btn_id = prop.split(".")[0] if prop else ""
	w = _parse_neuron_smooth_window(raw)
	if btn_id == "btn-neuron-smooth-inc":
		return w + 1
	if btn_id == "btn-neuron-smooth-dec":
		return max(1, w - 1)
	return no_update


def _neuron_should_reset_k() -> bool:
	"""True unless the only trigger was the K checklist."""
	t = callback_context.triggered
	if not t:
		return True
	if len(t) == 1 and t[0].get("prop_id", "").startswith(
		"checklist-network-k-visible"
	):
		return False
	return True


@app.callback(
	Output("graph-neuron-detail", "figure"),
	Output("hint-text", "children"),
	Output("k-checklist-wrap", "style"),
	Output("checklist-network-k-visible", "value"),
	Input("store-node", "data"),
	Input("dd-optimizer", "value"),
	Input("dd-experiment", "value"),
	Input("dd-model", "value"),
	Input("dd-run", "value"),
	Input("input-neuron-smooth", "value"),
	Input("radio-viz-type", "value"),
	Input("radio-nonlinearity", "value"),
	Input("checklist-network-k-visible", "value"),
)
def _cb_neuron(
	nid: str | None,
	oid: str | None,
	eid: str | None,
	mid: str | None,
	rid: str | None,
	smooth_raw: Any,
	viz_type: str | None,
	nonlinearity: str | None,
	k_checklist: list[str] | None,
):
	hint_default = "Click a node in the diagram above to inspect it."
	empty = _empty("", 100)
	smooth_w = _parse_neuron_smooth_window(smooth_raw)
	is_scatter = viz_type == "scatter"
	post_nl = nonlinearity == "post"

	k_wrap_style = {"display": "block"} if is_scatter else {"display": "none"}

	all_k = [str(k) for k in range(_NETWORK_SAMPLES_PER_DIGIT)]
	reset_k = _neuron_should_reset_k()
	if reset_k:
		k_sel = set(all_k)
		checklist_out: Any = list(all_k)
	else:
		checklist_out = no_update
		k_sel = set(k_checklist) if k_checklist else set(all_k)

	if nid is None or eid is None or mid is None or rid is None:
		return empty, hint_default, k_wrap_style, checklist_out
	if oid is None:
		return (
			empty,
			"Select an optimizer above first, then click a node.",
			k_wrap_style,
			checklist_out,
		)

	nts = _nts(eid, mid, oid, rid=rid, cache=_cache)
	safe = nid.replace(":", "__")
	if f"act__{safe}" not in nts:
		return empty, hint_default, k_wrap_style, checklist_out

	if is_scatter:
		k_vis = [str(k) in k_sel for k in range(_NETWORK_SAMPLES_PER_DIGIT)]
		fig = _build_network_activation_figure(
			nid, eid, mid, oid,
			rid=rid,
			k_visible=k_vis,
			post_nonlinearity=post_nl,
		)
	else:
		fig = _build_neuron_detail_figure(
			nid, eid, mid, oid,
			rid=rid,
			smooth_window=smooth_w,
			post_nonlinearity=post_nl,
		)
	return fig, f"Selected: {nid}", k_wrap_style, checklist_out




# -- Entry point --------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(
		description="Run the Network Analyzer Dash webapp."
	)
	p.add_argument("--host", default="0.0.0.0")
	p.add_argument("--port", type=int, default=8050)
	p.add_argument(
		"--debug",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Enable Dash debug mode (default: true).",
	)
	args = p.parse_args(argv)

	app.run(debug=args.debug, host=args.host, port=args.port)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
