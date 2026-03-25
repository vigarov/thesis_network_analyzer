#!/usr/bin/env python3
"""Dash + Plotly visualization webapp for neural network analysis.

Run with::

    uv run network-analyzer-webapp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
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
# -- Paths --------------------------------------------------------------------

_APP_DIR = Path(__file__).resolve().parent
_PROJECT = _APP_DIR.parent
RESULTS = _PROJECT / "results"

if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

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

_OPT_PAL = {
    "sgd": C["blue"],
    "adam": C["accent"],
    "adagrad": C["green"],
    "pure_shampoo": "#0891b2",
    "grafted_shampoo": "#c026d3",
}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _opt_color(oid: str) -> str:
    for prefix, color in _OPT_PAL.items():
        if oid.startswith(prefix):
            return color
    return C["purple"]


def _opt_label(oid: str) -> str:
    return oid.split("_lr")[0].upper() if "_lr" in oid else oid


# Dropdown / plot legend order (unknown families sort last, then by id)
_OPT_ORDER_PREFIXES = (
    "sgd",
    "adagrad",
    "adam",
    "pure_shampoo",
    "grafted_shampoo",
)


def _sort_optimizer_ids(oids: list[str]) -> list[str]:
    def _key(oid: str) -> tuple[int, str]:
        for i, prefix in enumerate(_OPT_ORDER_PREFIXES):
            if oid.startswith(prefix):
                return (i, oid)
        return (len(_OPT_ORDER_PREFIXES), oid)

    return sorted(oids, key=_key)


def _sorted_run_ids(runs: dict[str, list[str]]) -> list[str]:
    return sorted(runs.keys())


# -- Data loading (module-level cache) ----------------------------------------

_cache: dict[tuple, Any] = {}


def _oids_from_optimizers_dir(opt_base: Path) -> list[str]:
    if not opt_base.is_dir():
        return []
    return _sort_optimizer_ids(
        [
            d.name
            for d in opt_base.iterdir()
            if d.is_dir() and (d / "training_metrics.npz").exists()
        ]
    )


def scan_results() -> dict[str, dict[str, dict[str, list[str]]]]:
    """Return ``{experiment_id: {model_id: {run_id: [optimizer_id, …]}}}``.

    Layout: ``results/<experiment_id>/<run_id>/<model_id>/optimizers/<optimizer_id>/``.
    """
    key = ("__scan__",)
    if key in _cache:
        return _cache[key]

    tree: dict[str, dict[str, dict[str, list[str]]]] = {}
    if not RESULTS.exists():
        _cache[key] = tree
        return tree

    for exp_dir in sorted(RESULTS.iterdir()):
        if not exp_dir.is_dir():
            continue
        eid = exp_dir.name
        for rid_dir in sorted(exp_dir.iterdir()):
            if not rid_dir.is_dir():
                continue
            for mid_dir in sorted(rid_dir.iterdir()):
                if not mid_dir.is_dir():
                    continue
                opt_base = mid_dir / "optimizers"
                if not opt_base.is_dir():
                    continue
                oids = _oids_from_optimizers_dir(opt_base)
                if oids:
                    rid = rid_dir.name
                    mid = mid_dir.name
                    tree.setdefault(eid, {}).setdefault(mid, {})
                    tree[eid][mid][rid] = oids

    _cache[key] = tree
    return tree


def _optimizer_npz_path(eid: str, rid: str, mid: str, oid: str, fname: str) -> Path:
    return RESULTS / eid / rid / mid / "optimizers" / oid / fname


def _training_config_path(eid: str, rid: str, mid: str) -> Path:
    return RESULTS / eid / rid / mid / "config.json"


def _format_training_config_text(
    eid: str | None, mid: str | None, rid: str | None
) -> str:
    """Pretty-printed ``config.json`` for the selected run, or a short status message."""
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


def _npz(
    eid: str, mid: str, oid: str, fname: str, *, rid: str
) -> dict[str, np.ndarray]:
    key = (fname, eid, rid, mid, oid)
    if key in _cache:
        return _cache[key]
    path = _optimizer_npz_path(eid, rid, mid, oid, fname)
    if not path.exists():
        return {}
    data = dict(np.load(str(path), allow_pickle=True))
    _cache[key] = data
    return data


def _metrics(e: str, m: str, o: str, *, rid: str) -> dict[str, np.ndarray]:
    return _npz(e, m, o, "training_metrics.npz", rid=rid)


def _nts(e: str, m: str, o: str, *, rid: str) -> dict[str, np.ndarray]:
    return _npz(e, m, o, "neuron_timeseries.npz", rid=rid)


def _sigs(e: str, m: str, o: str, *, rid: str) -> dict[str, np.ndarray]:
    return _npz(e, m, o, "signals.npz", rid=rid)


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


def _stage_boundaries_from_metrics(
    metrics: dict[str, np.ndarray],
    x_mode: str,
) -> tuple[list[str], list[float], bool] | None:
    """Extract (stage_names, boundary_x_positions, use_iterations) for plotting.

    Returns None if stage data is missing (backward compat with old runs).
    x_mode: 'checkpoint' or 'iteration'.
    """
    names = metrics.get("stage_names")
    cp_idxs = metrics.get("stage_end_checkpoint_idxs")
    iters = metrics.get("stage_end_iterations")
    if names is None or cp_idxs is None:
        return None
    names = [str(n) for n in names]
    if len(names) < 1:
        return None
    use_iters = (
        x_mode == "iteration"
        and iters is not None
        and len(iters) == len(names)
    )
    boundaries: list[float] = []
    for i in range(len(names) - 1):
        if use_iters and iters is not None:
            boundaries.append(float(iters[i]) + 0.5)
        elif cp_idxs is not None:
            boundaries.append(float(cp_idxs[i]) + 0.5)
    return (names, boundaries, use_iters)


def _stage_switch_checkpoint_indices(metrics: dict[str, Any]) -> list[int]:
    """Checkpoint indices at stage starts, inferred from training_loop tags ``sw_*_entry``."""
    tags = metrics.get("checkpoint_tags")
    if tags is None:
        return []
    out: list[int] = []
    for i, t in enumerate(tags):
        s = str(t)
        if s.startswith("sw_") and s.endswith("_entry"):
            out.append(i)
    return out


def _add_stage_boundaries(
    fig: go.Figure,
    metrics: dict[str, np.ndarray],
    x_mode: str,
    subplot_rows: list[tuple[str, int]] | None = None,
) -> go.Figure:
    """Add vertical lines and stage labels to *fig*.

    x_mode: 'checkpoint' or 'iteration'.
    subplot_rows: optional list of (x_mode, row_num) for multi-row figures.
      When provided, adds boundaries to each row with its x_mode. row_num is 1-based.
    """
    rows_config: list[tuple[str, str, str]] = []
    if subplot_rows:
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
        sb = _stage_boundaries_from_metrics(metrics, row_mode)
        if sb is None:
            continue
        stage_names, boundary_xs, use_iters = sb
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

        # Stage labels: add only once per x_mode to avoid duplicates
        if row_mode in seen_modes:
            continue
        seen_modes.add(row_mode)
        cp_idxs = metrics.get("stage_end_checkpoint_idxs")
        iters = metrics.get("stage_end_iterations")
        n = len(stage_names)
        for i, name in enumerate(stage_names):
            if use_iters and iters is not None:
                start = 0 if i == 0 else float(iters[i - 1]) + 1
                end = float(iters[i])
            elif cp_idxs is not None:
                start = 0 if i == 0 else int(cp_idxs[i - 1]) + 1
                end = int(cp_idxs[i])
            else:
                continue
            mid = (start + end) / 2
            short = name.replace("stage", "").replace("_", " ").strip()
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
    ):
        if candidate in lnames:
            return candidate
    return lnames[0]


def _build_loss(
    eid: str, mid: str, oids: list[str], *, rid: str
) -> go.Figure:
    fig = go.Figure()
    first_m = next(
        (_metrics(eid, mid, o, rid=rid) for o in oids if _metrics(eid, mid, o, rid=rid)),
        {},
    )
    tags = [str(t) for t in first_m.get("checkpoint_tags", [])]
    cp_iters = first_m.get("checkpoint_iterations")
    use_iter = cp_iters is not None and len(cp_iters) == len(tags)

    for oid in oids:
        m = _metrics(eid, mid, oid, rid=rid)
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
    _add_stage_boundaries(fig, first_m, "iteration" if use_iter else "checkpoint")
    _add_show_hide(fig)
    return fig


def _build_acc(
    eid: str, mid: str, oids: list[str], *, rid: str
) -> go.Figure:
    fig = go.Figure()
    first_m = next(
        (_metrics(eid, mid, o, rid=rid) for o in oids if _metrics(eid, mid, o, rid=rid)),
        {},
    )
    tags = [str(t) for t in first_m.get("checkpoint_tags", [])]
    cp_iters = first_m.get("checkpoint_iterations")
    use_iter = cp_iters is not None and len(cp_iters) == len(tags)

    for oid in oids:
        m = _metrics(eid, mid, oid, rid=rid)
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
    _add_stage_boundaries(fig, first_m, "iteration" if use_iter else "checkpoint")
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
        if is_conv:
            node_color = C["accent"]
        elif ln == "head":
            node_color = C["green"]
        else:
            node_color = C["blue"]
        sizes = [log_sizes.get(u["node_id"], default_dot_sz) for u in lu]
        if act_vals:
            htxt = [
                f"{ln} · {u['unit_type']} {u['unit_index']}"
                f"<br>activation: {act_vals.get(u['node_id'], 0.0):.4f}"
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
                    color=node_color,
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

# Adam moments (exp_avg_norm, exp_avg_sq_norm) - only for Adam, two y axes
_ADAM_MOMENTS_SIGS: list[tuple[str, str, str, bool]] = [
    ("exp_avg_norm", "1st moment norm", C["blue"], False),
    ("exp_avg_sq_norm", "2nd moment norm", C["orange"], True),  # secondary y
]

# Adagrad / grafted Shampoo: effective step scale (per-weight LR-style)
_ADAGRAD_LR_SIG = ("effective_lr", "Effective LR", C["purple"])

# Shampoo: global preconditioner inverse Frobenius norm (replicated per unit in logs)
_H_INV_NORM_SIG = ("h_inv_norm", "Precond. inv. norm", "#0d9488")


def _opt_type(oid: str) -> str:
    if oid.startswith("pure_shampoo"):
        return "pure_shampoo"
    if oid.startswith("grafted_shampoo"):
        return "grafted_shampoo"
    for t in ("sgd", "adam", "adagrad"):
        if oid.startswith(t):
            return t
    return "unknown"


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


def _build_neuron_detail_figure(
    nid: str, eid: str, mid: str, oid: str, *, rid: str
) -> go.Figure:
    """Build one combined figure with shared x-axis: Activation, Weight, Grad norm,
    Cosine similarity (dual y for Adam), Adam moments (Adam only),
    Shampoo preconditioner norm (Shampoo only), Effective LR (Adagrad / grafted Shampoo).
    """
    nts = _nts(eid, mid, oid, rid=rid)
    sigs = _sigs(eid, mid, oid, rid=rid)
    metrics = _metrics(eid, mid, oid, rid=rid)
    otype = _opt_type(oid)

    tags = [str(t) for t in nts.get("checkpoint_tags", [])]
    cp_iters = metrics.get("checkpoint_iterations")
    if cp_iters is None or len(cp_iters) != len(tags):
        return _empty("Checkpoint iterations not available (re-run training)", 200)
    x_iters = np.array(cp_iters, dtype=float)

    has_sigs = bool(sigs) and "unit_node_ids" in sigs
    uid_list = [str(n) for n in sigs.get("unit_node_ids", [])] if has_sigs else []
    ci = uid_list.index(nid) if nid in uid_list else -1

    # Row layout: 1=Activation, 2=Weight, 3=Grad norm, 4=Cosine sim (when has_sigs),
    # 5=Adam moments (Adam only), then Shampoo precond. norm, then Effective LR
    has_adam_moments = has_sigs and otype == "adam"
    has_shampoo_h_inv = has_sigs and otype in ("pure_shampoo", "grafted_shampoo")
    has_eff_lr = has_sigs and otype in ("adagrad", "grafted_shampoo")
    has_grad_cos = has_sigs  # grad norm + cosine sim rows

    n_rows = (
        2
        + (2 if has_grad_cos else 0)
        + (1 if has_adam_moments else 0)
        + (1 if has_shampoo_h_inv else 0)
        + (1 if has_eff_lr else 0)
    )
    specs: list[list[dict]] = [[{}], [{}]]
    if has_grad_cos:
        specs.append([{}])
        specs.append([{"secondary_y": True}] if otype == "adam" else [{}])
    if has_adam_moments:
        specs.append([{"secondary_y": True}])
    if has_shampoo_h_inv:
        specs.append([{}])
    if has_eff_lr:
        specs.append([{}])

    subplot_titles = ["Activation", "Weight Evolution"]
    if has_grad_cos:
        subplot_titles.extend(["Grad norm", "Cosine similarity"])
    if has_adam_moments:
        subplot_titles.append("Adam moments")
    if has_shampoo_h_inv:
        subplot_titles.append("Precond. inv. norm")
    if has_eff_lr:
        subplot_titles.append("Effective LR")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        specs=specs,
        vertical_spacing=0.06,
        subplot_titles=subplot_titles,
    )

    hover = [f"iter {int(x)} · {t}" for x, t in zip(x_iters, tags)]
    tv, tt = _thin_ticks(list(x_iters), [str(int(x)) for x in x_iters])
    row = 1

    # Assign traces to per-subplot legends (legend for row 1, legend2 for row 2, etc.)
    def _legend_for_row(r: int) -> str:
        return "legend" if r == 1 else f"legend{r}"

    # Row 1: Activation
    safe = nid.replace(":", "__")
    if f"act__{safe}" in nts:
        act_arr = nts[f"act__{safe}"]
        # Compute mean over sample/value dimensions for display
        if act_arr.dtype == object:
            act_1d = np.array([float(np.mean(act_arr[t])) for t in range(len(act_arr))])
        elif act_arr.ndim > 1:
            act_1d = np.mean(act_arr, axis=tuple(range(1, act_arr.ndim)))
        else:
            act_1d = act_arr
        fig.add_trace(
            go.Scatter(
                x=x_iters,
                y=act_1d,
                mode="lines+markers",
                name="Activation",
                legend=_legend_for_row(row),
                text=hover,
                hoverinfo="text+y",
                line=dict(color=C["blue"], width=2),
                marker=dict(size=5),
            ),
            row=row,
            col=1,
        )
    row += 1

    # Row 2: Weight
    if f"wnorm__{safe}" in nts:
        fig.add_trace(
            go.Scatter(
                x=x_iters,
                y=nts[f"wnorm__{safe}"],
                mode="lines+markers",
                name="Weight norm",
                legend=_legend_for_row(row),
                text=hover,
                hoverinfo="text+y",
                line=dict(color=C["accent"], width=2),
                marker=dict(size=5),
            ),
            row=row,
            col=1,
        )
    row += 1

    # Row 3: Grad norm (only when has_grad_cos)
    if has_grad_cos:
        if ci >= 0:
            sname, slabel, scolor = _GRAD_NORM_SIG
            y = _signal_at_checkpoints(sigs, sname, ci, x_iters)
            if not np.all(np.isnan(y)):
                fig.add_trace(
                    go.Scatter(
                        x=x_iters,
                        y=y,
                        mode="lines+markers",
                        name=slabel,
                        legend=_legend_for_row(row),
                        text=hover,
                        hoverinfo="text+y",
                        line=dict(color=scolor, width=2),
                        marker=dict(size=5),
                    ),
                    row=row,
                    col=1,
                )
        row += 1

    # Row 4: Cosine similarity (grad + 1st moment for Adam with secondary y)
    if has_grad_cos:
        if ci >= 0:
            for sname, slabel, scolor, sec_y in _COSINE_SIM_SIGS.get(otype, []):
                y = _signal_at_checkpoints(sigs, sname, ci, x_iters)
                if not np.all(np.isnan(y)):
                    fig.add_trace(
                        go.Scatter(
                            x=x_iters,
                            y=y,
                            mode="lines+markers",
                            name=slabel,
                            legend=_legend_for_row(row),
                            text=hover,
                            hoverinfo="text+y",
                            line=dict(color=scolor, width=2),
                            marker=dict(size=5),
                        ),
                        row=row,
                        col=1,
                        secondary_y=sec_y,
                    )
        row += 1

    # Row 5: Adam moments (only when Adam) - two y axes
    if has_adam_moments and ci >= 0:
        for sname, slabel, scolor, sec_y in _ADAM_MOMENTS_SIGS:
            y = _signal_at_checkpoints(sigs, sname, ci, x_iters)
            if not np.all(np.isnan(y)):
                fig.add_trace(
                    go.Scatter(
                        x=x_iters,
                        y=y,
                        mode="lines+markers",
                        name=slabel,
                        legend=_legend_for_row(row),
                        text=hover,
                        hoverinfo="text+y",
                        line=dict(color=scolor, width=2),
                        marker=dict(size=5),
                    ),
                    row=row,
                    col=1,
                    secondary_y=sec_y,
                )
        row += 1

    # Preconditioner inverse norm (Pure / Grafted Shampoo)
    if has_shampoo_h_inv and ci >= 0:
        sname, slabel, scolor = _H_INV_NORM_SIG
        y = _signal_at_checkpoints(sigs, sname, ci, x_iters)
        if not np.all(np.isnan(y)):
            fig.add_trace(
                go.Scatter(
                    x=x_iters,
                    y=y,
                    mode="lines+markers",
                    name=slabel,
                    legend=_legend_for_row(row),
                    text=hover,
                    hoverinfo="text+y",
                    line=dict(color=scolor, width=2),
                    marker=dict(size=5),
                ),
                row=row,
                col=1,
            )
        row += 1

    # Effective LR (Adagrad or grafted Shampoo)
    if has_eff_lr and ci >= 0:
        sname, slabel, scolor = _ADAGRAD_LR_SIG
        y = _signal_at_checkpoints(sigs, sname, ci, x_iters, unit_safe=safe)
        # Backward compat: old data used effective_lr_mean as 2D matrix
        if np.all(np.isnan(y)) and "effective_lr_mean" in sigs:
            y = _signal_at_checkpoints(sigs, "effective_lr_mean", ci, x_iters)
        if not np.all(np.isnan(y)):
            fig.add_trace(
                go.Scatter(
                    x=x_iters,
                    y=y,
                    mode="lines+markers",
                    name=slabel,
                    legend=_legend_for_row(row),
                    text=hover,
                    hoverinfo="text+y",
                    line=dict(color=scolor, width=2),
                    marker=dict(size=5),
                ),
                row=row,
                col=1,
            )

    # Shared x-axis: tick labels on all subplots (fig.update_layout)
    for i in range(1, n_rows + 1):
        fig.update_xaxes(
            tickvals=tv,
            ticktext=tt,
            tickangle=-40,
            title_text="Iteration" if i == n_rows else "",
            showticklabels=True,
            row=i,
            col=1,
        )

    # Y-axis titles
    fig.update_yaxes(title_text="Activation", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    if has_grad_cos:
        fig.update_yaxes(title_text="Norm", row=3, col=1)
        fig.update_yaxes(title_text="Grad cos. sim.", row=4, col=1)
        if otype == "adam":
            fig.update_yaxes(title_text="1st moment cos. sim.", row=4, col=1, secondary_y=True)
    adam_row = 3 + (2 if has_grad_cos else 0)
    if has_adam_moments:
        fig.update_yaxes(title_text="1st moment norm", row=adam_row, col=1)
        fig.update_yaxes(title_text="2nd moment norm", row=adam_row, col=1, secondary_y=True)
    if has_shampoo_h_inv:
        h_inv_row = n_rows - (1 if has_eff_lr else 0)
        fig.update_yaxes(title_text="Precond. inv. norm", row=h_inv_row, col=1)
    if has_eff_lr:
        fig.update_yaxes(title_text="Effective LR", row=n_rows, col=1)

    height = 280 * n_rows
    _style(fig, height=height)
    _add_stage_boundaries(
        fig,
        metrics,
        "iteration",
        subplot_rows=[("iteration", r) for r in range(1, n_rows + 1)],
    )

    # Position one legend per subplot at the level of the relevant row.
    # With secondary_y (e.g. Adam cosine row), yaxis4 and yaxis5 share a domain,
    # so we map row index to unique subplot domains.
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
    for i in range(1, 20):
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
    legend_updates: dict[str, Any] = {}
    for r in range(1, n_rows + 1):
        leg_key = "legend" if r == 1 else f"legend{r}"
        if r <= len(seen_domains):
            y_center = (seen_domains[r - 1][0] + seen_domains[r - 1][1]) / 2
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
    _cache.pop(("__scan__",), None)
    tree = scan_results()
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
                                className="graph-half",
                            ),
                            dcc.Graph(
                                id="graph-acc",
                                figure=_empty("", 380),
                                className="graph-half",
                            ),
                        ],
                    ),
                ],
            ),
            # -- section 2: neuron analysis --
            html.Section(
                className="section",
                children=[
                    html.H2("Neuron Analysis"),
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
                                className="neuron-graph",
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
    tree = scan_results()
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
    tree = scan_results()
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
    tree = scan_results()
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
    nts = _nts(eid, mid, oid, rid=rid)
    metrics = _metrics(eid, mid, oid, rid=rid)
    tags = [str(t) for t in nts.get("checkpoint_tags", [])]
    n = len(tags)
    if n == 0:
        return 0, {}, 0, True, 1, None

    cp_iters_raw = metrics.get("checkpoint_iterations")
    use_iter = cp_iters_raw is not None and len(cp_iters_raw) == n

    # Choose which checkpoint indices get a visible label (uniformly sampled).
    MAX_MARKS = 12
    if n <= MAX_MARKS:
        labeled_idxs = list(range(n))
    else:
        step = max(1, n // MAX_MARKS)
        labeled_idxs = list(range(0, n, step))
        if labeled_idxs[-1] != n - 1:
            labeled_idxs.append(n - 1)

    stage_tick = "\u275A"
    stage_style = {"fontSize": "14px", "color": C["accent"], "fontWeight": "700"}
    label_style = {"fontSize": "10px", "color": C["muted"]}

    if use_iter and cp_iters_raw is not None:
        # Slider value space = actual iteration numbers so the tooltip is meaningful.
        cp_list: list[int] = [int(x) for x in cp_iters_raw]
        slider_max = cp_list[-1]
        slider_val = cp_list[-1]
        labeled_set = {cp_list[i] for i in labeled_idxs}

        # All checkpoint positions become marks so step=None can snap to them.
        # Only the sampled subset gets a visible text label.
        marks: dict[int, Any] = {}
        for it in cp_list:
            if it in labeled_set:
                marks[it] = {"label": str(it), "style": label_style}
            else:
                marks[it] = {"label": "", "style": {"fontSize": "0px"}}

        # Stage-switch ticks: decorate the mark at the matching iteration.
        for sw in _stage_switch_checkpoint_indices(metrics):
            if sw < 0 or sw >= n:
                continue
            iter_val = cp_list[sw]
            existing = marks.get(iter_val, {})
            lab = existing.get("label", "") if isinstance(existing, dict) else ""
            st = dict(existing.get("style") or {}) if isinstance(existing, dict) else {}
            label_text = f"{stage_tick} {lab}".strip() if lab else stage_tick
            marks[iter_val] = {"label": label_text, "style": {**st, **stage_style}}

        return slider_max, marks, slider_val, False, None, cp_list

    else:
        # Fallback: slider value = checkpoint index (0..n-1), step=1.
        marks_idx: dict[int, Any] = {
            i: {"label": tags[i], "style": label_style}
            for i in labeled_idxs
        }
        for sw in _stage_switch_checkpoint_indices(metrics):
            if sw < 0 or sw >= n:
                continue
            if sw in marks_idx:
                m = marks_idx[sw]
                lab = m.get("label", "") if isinstance(m, dict) else str(m)
                st = dict(m.get("style") or {}) if isinstance(m, dict) else {}
                label_text = f"{stage_tick} {lab}" if lab else stage_tick
                marks_idx[sw] = {"label": label_text, "style": {**st, **stage_style}}
            else:
                marks_idx[sw] = {"label": stage_tick, "style": stage_style}
        return n - 1, marks_idx, n - 1, False, 1, None


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
    """Slider `value` that selects checkpoint index ``cp_idx`` (iteration or index mode)."""
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
    tree = scan_results()
    oids_avail = tree.get(eid, {}).get(mid, {}).get(rid, [])
    if not oids_avail:
        return _empty("No data available", 200), None
    ref_oid = oid if oid else oids_avail[0]
    nts = _nts(eid, mid, ref_oid, rid=rid)
    units = _parse_units(nts)
    n_cp = len(nts.get("checkpoint_tags", []))
    m = _metrics(eid, mid, ref_oid, rid=rid)
    cp_iters = m.get("checkpoint_iterations") if m else None

    def _fig_at_checkpoint(ci: int | None) -> go.Figure:
        return _build_diagram(
            units,
            mid,
            act_nts=nts if oid else None,
            checkpoint_idx=ci,
            show_hint=(oid is None),
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


@app.callback(
    Output("graph-neuron-detail", "figure"),
    Output("hint-text", "children"),
    Input("store-node", "data"),
    Input("dd-optimizer", "value"),
    Input("dd-experiment", "value"),
    Input("dd-model", "value"),
    Input("dd-run", "value"),
)
def _cb_neuron(
    nid: str | None,
    oid: str | None,
    eid: str | None,
    mid: str | None,
    rid: str | None,
):
    hint_default = "Click a node in the diagram above to inspect it."
    empty = _empty("", 100)
    if nid is None or eid is None or mid is None or rid is None:
        return empty, hint_default
    if oid is None:
        return empty, "⚠ Select an optimizer above first, then click a node."

    nts = _nts(eid, mid, oid, rid=rid)
    safe = nid.replace(":", "__")
    if f"act__{safe}" not in nts:
        return empty, hint_default

    fig = _build_neuron_detail_figure(nid, eid, mid, oid, rid=rid)
    return fig, f"Selected: {nid}"


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
