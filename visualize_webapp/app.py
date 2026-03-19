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
from dash import Dash, Input, Output, dcc, html, no_update
from plotly.subplots import make_subplots

# -- Paths --------------------------------------------------------------------

_APP_DIR = Path(__file__).resolve().parent
_PROJECT = _APP_DIR.parent
RESULTS = _PROJECT / "results"

if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

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

_OPT_PAL = {"sgd": C["blue"], "adam": C["accent"], "adagrad": C["green"]}


def _opt_color(oid: str) -> str:
    for prefix, color in _OPT_PAL.items():
        if oid.startswith(prefix):
            return color
    return C["purple"]


def _opt_label(oid: str) -> str:
    return oid.split("_lr")[0].upper() if "_lr" in oid else oid


# -- Data loading (module-level cache) ----------------------------------------

_cache: dict[tuple, Any] = {}


def scan_results() -> dict[str, dict[str, list[str]]]:
    """Return ``{experiment_id: {model_id: [optimizer_id, …]}}``."""
    key = ("__scan__",)
    if key in _cache:
        return _cache[key]

    tree: dict[str, dict[str, list[str]]] = {}
    if not RESULTS.exists():
        _cache[key] = tree
        return tree

    for exp_dir in sorted(RESULTS.iterdir()):
        if not exp_dir.is_dir():
            continue
        for model_dir in sorted(exp_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            opt_base = model_dir / "optimizers"
            if not opt_base.exists():
                continue
            oids = sorted(
                d.name
                for d in opt_base.iterdir()
                if d.is_dir() and (d / "training_metrics.npz").exists()
            )
            if oids:
                tree.setdefault(exp_dir.name, {})[model_dir.name] = oids

    _cache[key] = tree
    return tree


def _npz(eid: str, mid: str, oid: str, fname: str) -> dict[str, np.ndarray]:
    key = (fname, eid, mid, oid)
    if key in _cache:
        return _cache[key]
    path = RESULTS / eid / mid / "optimizers" / oid / fname
    if not path.exists():
        return {}
    data = dict(np.load(str(path), allow_pickle=True))
    _cache[key] = data
    return data


def _metrics(e: str, m: str, o: str) -> dict[str, np.ndarray]:
    return _npz(e, m, o, "training_metrics.npz")


def _nts(e: str, m: str, o: str) -> dict[str, np.ndarray]:
    return _npz(e, m, o, "neuron_timeseries.npz")


def _sigs(e: str, m: str, o: str) -> dict[str, np.ndarray]:
    return _npz(e, m, o, "signals.npz")


# -- Plotly helpers -----------------------------------------------------------


def _style(fig: go.Figure, **kw: Any) -> go.Figure:
    """Apply dark-theme defaults to *fig*.  Keyword args override defaults."""
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


# -- Training overview figures ------------------------------------------------


def _build_loss(eid: str, mid: str, oids: list[str]) -> go.Figure:
    fig = go.Figure()
    tvals: list[int] = []
    ttxt: list[str] = []

    for oid in oids:
        m = _metrics(eid, mid, oid)
        if not m:
            continue
        tags = [str(t) for t in m.get("checkpoint_tags", [])]
        xs = list(range(len(tags)))
        if not tvals:
            tvals, ttxt = xs, tags

        col = _opt_color(oid)
        lab = _opt_label(oid)
        for key in sorted(m):
            if not key.startswith("loss_"):
                continue
            lname = key[5:]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=m[key],
                    mode="lines+markers",
                    name=f"{lab} · {lname}",
                    line=dict(
                        color=col,
                        width=2,
                        dash="solid" if "test" in lname else "dot",
                    ),
                    marker=dict(size=4),
                    visible=True
                    if lname in ("all_test", "all_train")
                    else "legendonly",
                )
            )

    _style(fig, title=dict(text="Loss", font=dict(size=15)), height=380)
    if tvals:
        tv, tt = _thin_ticks(tvals, ttxt)
        fig.update_xaxes(
            tickvals=tv, ticktext=tt, tickangle=-40, title_text="Checkpoint"
        )
    fig.update_yaxes(title_text="Loss")
    _add_show_hide(fig)
    return fig


def _build_acc(eid: str, mid: str, oids: list[str]) -> go.Figure:
    fig = go.Figure()
    tvals: list[int] = []
    ttxt: list[str] = []

    for oid in oids:
        m = _metrics(eid, mid, oid)
        if not m:
            continue
        tags = [str(t) for t in m.get("checkpoint_tags", [])]
        xs = list(range(len(tags)))
        if not tvals:
            tvals, ttxt = xs, tags

        col = _opt_color(oid)
        lab = _opt_label(oid)
        for key in sorted(m):
            if not key.startswith("acc_"):
                continue
            aname = key[4:]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=m[key],
                    mode="lines+markers",
                    name=f"{lab} · {aname}",
                    line=dict(
                        color=col,
                        width=2,
                        dash="solid" if "test" in aname else "dot",
                    ),
                    marker=dict(size=4),
                    visible=True if aname == "all_test" else "legendonly",
                )
            )

    _style(fig, title=dict(text="Accuracy", font=dict(size=15)), height=380)
    if tvals:
        tv, tt = _thin_ticks(tvals, ttxt)
        fig.update_xaxes(
            tickvals=tv, ticktext=tt, tickangle=-40, title_text="Checkpoint"
        )
    fig.update_yaxes(title_text="Accuracy", range=[-0.02, 1.05])
    _add_show_hide(fig)
    return fig


# -- Model diagram ------------------------------------------------------------


def _parse_units(nts: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    if "unit_node_ids" not in nts or "units_meta" not in nts:
        return []
    out: list[dict[str, Any]] = []
    for nid, meta in zip(nts["unit_node_ids"], nts["units_meta"]):
        layer, idx, utype = str(meta).split("|")
        out.append(
            dict(
                node_id=str(nid),
                layer_name=layer,
                unit_index=int(idx),
                unit_type=utype,
            )
        )
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
                act_vals[u["node_id"]] = float(arr[t])
        if act_vals:
            nids = list(act_vals.keys())
            raw = np.array([act_vals[n] for n in nids])
            log_raw = np.log1p(np.abs(raw))
            v_min, v_max = log_raw.min(), log_raw.max()
            rng = max(float(v_max - v_min), 1e-8)
            for i, nid in enumerate(nids):
                log_sizes[nid] = 4.0 + 4.0 * float(log_raw[i] - v_min) / rng

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
        ys = [(i + 0.5) / max_n for i in range(n)]
        cdata = [u["node_id"] for u in lu]
        is_conv = lu[0]["unit_type"] == "channel"
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
                    color=C["accent"] if is_conv else C["blue"],
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
        )
        fig.add_annotation(
            x=ci,
            y=-0.07,
            text=display,
            showarrow=False,
            font=dict(size=10, color=C["muted"], family=FONT),
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

_SIG_TITLE = {
    "sgd": "Gradient Signals",
    "adam": "Adam Moments",
    "adagrad": "Gradient Norm",
}

_SIG_MAP: dict[str, list[tuple[str, str, str]]] = {
    "sgd": [
        ("grad_norm", "Grad norm", C["red"]),
        ("grad_cosine_sim", "Grad cos. sim.", C["purple"]),
    ],
    "adam": [
        ("grad_norm", "Grad norm", C["red"]),
        ("exp_avg_norm", "1st moment norm", C["purple"]),
        ("exp_avg_sq_norm", "2nd moment norm", C["orange"]),
        ("moment_cosine_sim", "Moment cos. sim.", C["green"]),
    ],
    "adagrad": [
        ("grad_norm", "Grad norm", C["red"]),
    ],
}

# Adagrad gets a dedicated 4th row for the effective LR (very different scale)
_ADAGRAD_LR_SIG = ("effective_lr_mean", "Effective LR", C["purple"])


def _opt_type(oid: str) -> str:
    for t in ("sgd", "adam", "adagrad"):
        if oid.startswith(t):
            return t
    return "unknown"


def _add_sig_trace(
    fig: go.Figure,
    sigs: dict[str, np.ndarray],
    sig_name: str,
    col_idx: int,
    iters: np.ndarray,
    label: str,
    color: str,
    row: int,
) -> None:
    if sig_name not in sigs:
        return
    arr = sigs[sig_name]
    if arr.ndim != 2 or col_idx >= arr.shape[1]:
        return
    y = arr[:, col_idx].astype(float)
    x = iters
    if len(y) > 2000:
        step = max(1, len(y) // 2000)
        y, x = y[::step], x[::step]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=label,
            line=dict(color=color, width=1.5),
            opacity=0.85,
        ),
        row=row,
        col=1,
    )


def _build_detail(nid: str, eid: str, mid: str, oid: str) -> go.Figure:
    nts = _nts(eid, mid, oid)
    sigs = _sigs(eid, mid, oid)
    otype = _opt_type(oid)
    safe = nid.replace(":", "__")

    tags = [str(t) for t in nts.get("checkpoint_tags", [])]
    cx = list(range(len(tags)))

    has_sigs = bool(sigs) and "iteration" in sigs
    adagrad_split = has_sigs and otype == "adagrad"
    n_rows = 2
    subtitles = ["Activation", "Weight Evolution"]
    if has_sigs:
        n_rows += 1
        subtitles.append(_SIG_TITLE.get(otype, "Optimizer Signals"))
    if adagrad_split:
        n_rows += 1
        subtitles.append("Effective Learning Rate")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=subtitles,
        vertical_spacing=0.10,
    )

    # row 1 - activation
    ak = f"act__{safe}"
    if ak in nts:
        fig.add_trace(
            go.Scatter(
                x=cx,
                y=nts[ak],
                mode="lines+markers",
                name="Activation",
                line=dict(color=C["blue"], width=2),
                marker=dict(size=5),
            ),
            row=1,
            col=1,
        )

    # row 2 - weight norm & cosine similarity
    for key, label, color in [
        (f"wnorm__{safe}", "Weight norm", C["accent"]),
        (f"wcos__{safe}", "Weight cos. sim.", C["green"]),
    ]:
        if key in nts:
            fig.add_trace(
                go.Scatter(
                    x=cx,
                    y=nts[key],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                ),
                row=2,
                col=1,
            )

    # row 3 - optimizer-specific per-iteration signals
    if has_sigs:
        iters = sigs["iteration"]
        uid_list = [str(n) for n in sigs.get("unit_node_ids", [])]
        if nid in uid_list:
            ci = uid_list.index(nid)
            for sname, slabel, scolor in _SIG_MAP.get(otype, []):
                _add_sig_trace(fig, sigs, sname, ci, iters, slabel, scolor, 3)

    # row 4 (adagrad only) - effective learning rate on its own scale
    if adagrad_split:
        iters = sigs["iteration"]
        uid_list = [str(n) for n in sigs.get("unit_node_ids", [])]
        if nid in uid_list:
            ci = uid_list.index(nid)
            sname, slabel, scolor = _ADAGRAD_LR_SIG
            _add_sig_trace(fig, sigs, sname, ci, iters, slabel, scolor, 4)

    for r in (1, 2):
        fig.update_xaxes(tickvals=cx, ticktext=tags, tickangle=-40, row=r, col=1)
    if has_sigs:
        fig.update_xaxes(title_text="Iteration", row=3, col=1)
    if adagrad_split:
        fig.update_xaxes(title_text="Iteration", row=4, col=1)

    _style(
        fig,
        height=240 * n_rows + 60,
        title=dict(text=f"Unit: {nid}", font=dict(size=14)),
        margin=dict(l=55, r=20, t=70, b=50),
    )
    fig.update_annotations(font=dict(color=C["muted"], family=FONT, size=12))
    for r in range(1, n_rows + 1):
        fig.update_xaxes(
            gridcolor=C["grid"], zerolinecolor=C["border"], row=r, col=1
        )
        fig.update_yaxes(
            gridcolor=C["grid"], zerolinecolor=C["border"], row=r, col=1
        )
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
                    html.H1("Network Analyzer"),
                    html.P(
                        "Neuron & synapse behavior explorer",
                        className="subtitle",
                    ),
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
                                    "Select experiment & model", 380
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
                                "Time (checkpoint)",
                                className="slider-label",
                            ),
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
                    html.Div(
                        className="neuron-detail",
                        children=[
                            html.P(
                                "Click a node in the diagram above to inspect it.",
                                id="hint-text",
                                className="hint-text",
                            ),
                            dcc.Graph(
                                id="graph-neuron",
                                figure=_empty("", 100),
                            ),
                        ],
                    ),
                ],
            ),
            # client-side store for the last-clicked node id
            dcc.Store(id="store-node"),
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
    Output("graph-loss", "figure"),
    Output("graph-acc", "figure"),
    Output("dd-optimizer", "options"),
    Output("dd-optimizer", "value"),
    Input("dd-experiment", "value"),
    Input("dd-model", "value"),
)
def _cb_training(eid: str | None, mid: str | None):
    if not eid or not mid:
        return (
            _empty("Select experiment & model", 380),
            _empty("", 380),
            [],
            None,
        )
    tree = scan_results()
    oids = tree.get(eid, {}).get(mid, [])
    if not oids:
        return (
            _empty("No results yet", 380),
            _empty("", 380),
            [],
            None,
        )
    opts = [{"label": o, "value": o} for o in oids]
    return (
        _build_loss(eid, mid, oids),
        _build_acc(eid, mid, oids),
        opts,
        oids[0] if len(oids) == 1 else None,
    )


@app.callback(
    Output("slider-checkpoint", "max"),
    Output("slider-checkpoint", "marks"),
    Output("slider-checkpoint", "value"),
    Output("slider-checkpoint", "disabled"),
    Input("dd-experiment", "value"),
    Input("dd-model", "value"),
    Input("dd-optimizer", "value"),
)
def _cb_slider(
    eid: str | None, mid: str | None, oid: str | None
):
    if not eid or not mid or not oid:
        return 0, {}, 0, True
    nts = _nts(eid, mid, oid)
    tags = [str(t) for t in nts.get("checkpoint_tags", [])]
    n = len(tags)
    if n == 0:
        return 0, {}, 0, True
    MAX_MARKS = 12
    if n <= MAX_MARKS:
        idxs = list(range(n))
    else:
        step = max(1, n // MAX_MARKS)
        idxs = list(range(0, n, step))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
    marks = {
        i: {"label": tags[i], "style": {"fontSize": "10px", "color": C["muted"]}}
        for i in idxs
    }
    return n - 1, marks, n - 1, False


@app.callback(
    Output("graph-diagram", "figure"),
    Input("dd-experiment", "value"),
    Input("dd-model", "value"),
    Input("dd-optimizer", "value"),
    Input("slider-checkpoint", "value"),
)
def _cb_diagram(
    eid: str | None,
    mid: str | None,
    oid: str | None,
    checkpoint_idx: int | None,
):
    if eid is None or mid is None:
        return _empty("Select an experiment & model to view architecture", 200)
    tree = scan_results()
    oids_avail = tree.get(eid, {}).get(mid, [])
    if not oids_avail:
        return _empty("No data available", 200)
    ref_oid = oid if oid else oids_avail[0]
    nts = _nts(eid, mid, ref_oid)
    units = _parse_units(nts)
    return _build_diagram(
        units,
        mid,
        act_nts=nts if oid else None,
        checkpoint_idx=checkpoint_idx,
        show_hint=(oid is None),
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
    Output("graph-neuron", "figure"),
    Output("hint-text", "children"),
    Input("store-node", "data"),
    Input("dd-optimizer", "value"),
    Input("dd-experiment", "value"),
    Input("dd-model", "value"),
)
def _cb_neuron(
    nid: str | None,
    oid: str | None,
    eid: str | None,
    mid: str | None,
):
    hint_default = "Click a node in the diagram above to inspect it."
    if nid is None or eid is None or mid is None:
        return _empty("Click a node to see details", 100), hint_default
    if oid is None:
        return (
            _empty("Select an optimizer first to inspect nodes", 100),
            "⚠ Select an optimizer above first, then click a node.",
        )

    nts = _nts(eid, mid, oid)
    safe = nid.replace(":", "__")
    if f"act__{safe}" not in nts:
        return _empty("Select a node from the diagram", 100), hint_default

    return _build_detail(nid, eid, mid, oid), f"Selected: {nid}"


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
