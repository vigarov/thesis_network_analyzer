"""Matplotlib variants of Network Analysis panel plots (see network_analysis.ipynb)."""

from __future__ import annotations

from typing import Iterable

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory

from visualize_webapp.app import (
    C,
    _DIGIT_COLORS,
    _NETWORK_N_DIGITS,
    _compact_neuron_label,
    _trial_boundaries_from_metrics,
    compute_recovery_cumsum_by_layer,
)


def _rgba_tuple(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    """Matplotlib-compatible RGBA; avoid CSS ``rgba(...)`` strings for *facecolor*."""
    return mcolors.to_rgba(hex_color, alpha)

_AX_TITLE_PAD = 20.0

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


def _x_mode_from_metrics(metrics: dict[str, np.ndarray]) -> tuple[bool, np.ndarray | None]:
    tags = [str(t) for t in metrics.get("checkpoint_tags", [])]
    cp_iters = metrics.get("checkpoint_iterations")
    use_iter = cp_iters is not None and len(cp_iters) == len(tags)
    return use_iter, cp_iters if use_iter else None


def _x_values_for_checkpoints(
    all_cp_idxs: list[int],
    metrics: dict[str, np.ndarray],
) -> tuple[list[float], bool, str]:
    use_iter, cp_iters = _x_mode_from_metrics(metrics)
    xs: list[float] = []
    for ci in all_cp_idxs:
        xs.append(
            float(cp_iters[ci]) if use_iter and cp_iters is not None else float(ci)
        )
    x_mode = "iteration" if use_iter else "checkpoint"
    return xs, use_iter, x_mode


def add_trial_boundaries_mpl(ax: mpl.axes.Axes, metrics: dict[str, np.ndarray], x_mode: str) -> None:
    """Vertical dotted lines and trial labels (matches Plotly _add_trial_boundaries for a single axes)."""
    sb = _trial_boundaries_from_metrics(metrics, x_mode)
    if sb is None:
        return
    trial_names, boundary_xs, use_iters = sb
    if not boundary_xs:
        return

    for x in boundary_xs:
        ax.axvline(x, color="#000000", linestyle=":", linewidth=1.5, zorder=1)

    # Use *trial_names* from sb (already normalized). Do not use
    # ``metrics.get("trial_names") or []``: numpy arrays are truthy-ambiguous.
    cp_idxs = metrics.get(
        "trial_end_checkpoint_idxs", metrics.get("stage_end_checkpoint_idxs")
    )
    iters = metrics.get(
        "trial_end_iterations", metrics.get("stage_end_iterations")
    )
    if cp_idxs is None:
        return

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, name in enumerate(trial_names):
        if use_iters and iters is not None:
            start = 0.0 if i == 0 else float(iters[i - 1]) + 1.0
            end = float(iters[i])
        else:
            start = 0.0 if i == 0 else float(int(cp_idxs[i - 1]) + 1)
            end = float(int(cp_idxs[i]))
        mid = (start + end) / 2.0
        short = (
            name.replace("stage", "")
            .replace("trial", "")
            .replace("_", " ")
            .replace("run", "r", 1)
            .replace("digit", "d")
            .replace("label", "l")
            .strip()
        )
        ax.text(
            mid,
            1.02,
            short,
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=10,
            color=C["muted"],
            clip_on=False,
        )


def _apply_figure_style(fig: Figure) -> None:
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.grid(True, color=C["grid"], linestyle="-", linewidth=0.8)
        ax.tick_params(colors=C["fg"])
        for spine in ax.spines.values():
            spine.set_color(C["border"])


def mpl_layer_inactive_count(
    layer_name: str,
    df_nd: pd.DataFrame,
    metrics: dict[str, np.ndarray],
) -> Figure:
    layer_df = df_nd[df_nd["layer_name"] == layer_name]
    all_cp_idxs = sorted(layer_df["checkpoint_idx"].unique())

    inactive = layer_df[layer_df["status"] == "inactive"]
    counts = (
        inactive.groupby(["checkpoint_idx", "digit"])
        .size()
        .reset_index(name="count")
    )
    lookup = counts.set_index(["checkpoint_idx", "digit"])["count"]

    xs, use_iter, x_mode = _x_values_for_checkpoints(all_cp_idxs, metrics)

    fig, ax = plt.subplots(figsize=(8, 3.75), dpi=100)
    for d in range(_NETWORK_N_DIGITS):
        ys = [int(lookup.get((ci, d), 0)) for ci in all_cp_idxs]
        ax.plot(
            xs,
            ys,
            marker=".",
            ms=3,
            lw=1.5,
            color=_DIGIT_COLORS[d],
            label=f"Digit {d}",
        )

    display_name = layer_name.replace("hidden.", "H").replace("head", "Head")
    ax.set_title(
        f"Inactive neuron count per digit — {display_name}",
        fontsize=13,
        color=C["fg"],
        pad=_AX_TITLE_PAD,
    )
    ax.set_xlabel("Iteration" if use_iter else "Checkpoint")
    ax.set_ylabel("Count")
    add_trial_boundaries_mpl(ax, metrics, x_mode)
    ax.legend(loc="best", fontsize=9, framealpha=0.92, edgecolor=C["border"])
    fig.tight_layout()
    _apply_figure_style(fig)
    return fig


def mpl_layer_digit_count_plot(
    layer_name: str,
    df_nd: pd.DataFrame,
    metrics: dict[str, np.ndarray],
    status: str,
    title_verb: str,
    *,
    boxplot: bool = False,
) -> Figure:
    layer_df = df_nd[df_nd["layer_name"] == layer_name]
    all_neurons = layer_df["neuron_id"].unique()
    all_cp_idxs = sorted(layer_df["checkpoint_idx"].unique())

    matched = layer_df[layer_df["status"] == status]
    per_nc = (
        matched.groupby(["neuron_id", "checkpoint_idx"])
        .size()
        .reset_index(name="n")
    )
    full_idx = pd.DataFrame(
        [(n, c) for n in all_neurons for c in all_cp_idxs],
        columns=["neuron_id", "checkpoint_idx"],
    )
    per_nc = full_idx.merge(per_nc, on=["neuron_id", "checkpoint_idx"], how="left")
    per_nc["n"] = per_nc["n"].fillna(0).astype(int)

    display_name = layer_name.replace("hidden.", "H").replace("head", "Head")
    base_color = C["blue"] if status == "assigned" else C["red"]

    use_iter, cp_iters = _x_mode_from_metrics(metrics)

    if boxplot:
        tick_labels: list[str] = []
        data: list[np.ndarray] = []
        for ci in all_cp_idxs:
            label = (
                str(int(cp_iters[ci])) if use_iter and cp_iters is not None else str(ci)
            )
            tick_labels.append(label)
            vals = per_nc.loc[per_nc["checkpoint_idx"] == ci, "n"].values
            data.append(vals.astype(float))

        xs_arr = np.array(_x_values_for_checkpoints(all_cp_idxs, metrics)[0], dtype=float)
        if xs_arr.size > 1:
            xs_sorted = np.sort(xs_arr)
            box_w = float(np.min(np.diff(xs_sorted))) * 0.45
        else:
            box_w = 0.8

        fig, ax = plt.subplots(figsize=(8, 3.75), dpi=100)
        bp = ax.boxplot(
            data,
            positions=xs_arr,
            widths=box_w,
            patch_artist=True,
            showfliers=True,
        )
        for box in bp["boxes"]:
            box.set(facecolor=_rgba_tuple(base_color, 0.35), edgecolor=base_color)
        for whisker in bp["whiskers"]:
            whisker.set(color=base_color)
        for cap in bp["caps"]:
            cap.set(color=base_color)
        for med in bp["medians"]:
            med.set(color=C["fg"])
        ax.set_xticks(xs_arr)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=9)
        title_text = f"{title_verb} digit count distribution — {display_name}"
        x_mode = "iteration" if use_iter else "checkpoint"
    else:
        xs, use_iter2, x_mode = _x_values_for_checkpoints(all_cp_idxs, metrics)
        assert use_iter2 == use_iter

        means: list[float] = []
        ses: list[float] = []
        maxs: list[float] = []
        for ci in all_cp_idxs:
            vals = per_nc.loc[per_nc["checkpoint_idx"] == ci, "n"].values.astype(np.float64)
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
        fill_c = _rgba_tuple(base_color, 0.22)

        fig, ax = plt.subplots(figsize=(8, 3.75), dpi=100)
        ax.fill_between(xs, lower, upper, facecolor=fill_c, edgecolor="none", linewidth=0, zorder=1)
        ax.plot(xs, means, marker=".", ms=6, lw=1.5, color=base_color, label="Mean", zorder=2)
        ax.scatter(xs, maxs, s=50, marker="x", color=base_color, linewidths=1, label="Max", zorder=3)
        title_text = f"{title_verb} digit count (mean ± 1.96·SE) — {display_name}"
        ax.legend(loc="best", fontsize=9, framealpha=0.92, edgecolor=C["border"])

    ax.set_title(title_text, fontsize=13, color=C["fg"], pad=_AX_TITLE_PAD)
    ax.set_xlabel("Iteration" if use_iter else "Checkpoint")
    ax.set_ylabel(f"# {status} digits")
    ax.set_ylim(-0.5, _NETWORK_N_DIGITS + 0.5)
    add_trial_boundaries_mpl(ax, metrics, x_mode)
    fig.tight_layout()
    _apply_figure_style(fig)
    return fig


def mpl_layer_digit_count_non_head_combined(
    df_nd: pd.DataFrame,
    metrics: dict[str, np.ndarray],
    layer_order: list[str],
    status: str,
    title_verb: str,
) -> Figure:
    non_head = [ln for ln in layer_order if ln != "head"]
    if not non_head:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        ax.text(0.5, 0.5, "No non-head layers.", ha="center", va="center", transform=ax.transAxes, color=C["muted"])
        ax.axis("off")
        _apply_figure_style(fig)
        return fig

    use_iter, _ = _x_mode_from_metrics(metrics)
    x_mode = "iteration" if use_iter else "checkpoint"
    fig, ax = plt.subplots(figsize=(9, 4), dpi=100)
    n_plotted = 0

    for li, layer_name in enumerate(non_head):
        color = _LAYER_COMBINED_COLORS[li % len(_LAYER_COMBINED_COLORS)]
        layer_df = df_nd[df_nd["layer_name"] == layer_name]
        all_neurons = layer_df["neuron_id"].unique()
        all_cp_idxs = sorted(layer_df["checkpoint_idx"].unique())
        if not len(all_cp_idxs):
            continue

        matched = layer_df[layer_df["status"] == status]
        per_nc = (
            matched.groupby(["neuron_id", "checkpoint_idx"])
            .size()
            .reset_index(name="n")
        )
        full_idx = pd.DataFrame(
            [(n, c) for n in all_neurons for c in all_cp_idxs],
            columns=["neuron_id", "checkpoint_idx"],
        )
        per_nc = full_idx.merge(per_nc, on=["neuron_id", "checkpoint_idx"], how="left")
        per_nc["n"] = per_nc["n"].fillna(0).astype(int)

        xs, _, _ = _x_values_for_checkpoints(all_cp_idxs, metrics)
        means: list[float] = []
        ses: list[float] = []
        for ci in all_cp_idxs:
            vals = per_nc.loc[per_nc["checkpoint_idx"] == ci, "n"].values.astype(np.float64)
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
        fill_c = _rgba_tuple(color, 0.22)
        display_name = layer_name.replace("hidden.", "H").replace("head", "Head")

        ax.fill_between(xs, lower, upper, facecolor=fill_c, edgecolor="none", linewidth=0, zorder=li)
        ax.plot(
            xs,
            means,
            marker=".",
            ms=2.5,
            lw=1.5,
            color=color,
            label=display_name,
            zorder=li + 100,
            alpha=0.5,
        )
        n_plotted += 1

    if n_plotted == 0:
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        ax.text(
            0.5,
            0.5,
            "No checkpoint data for non-head layers.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C["muted"],
        )
        ax.axis("off")
        _apply_figure_style(fig)
        return fig

    ax.set_title(
        f"{title_verb} digit count (mean ± 1.96·SE) — all layers except Head",
        fontsize=13,
        color=C["fg"],
        pad=_AX_TITLE_PAD,
    )
    ax.set_xlabel("Iteration" if use_iter else "Checkpoint")
    ax.set_ylabel(f"# {status} digits")
    ax.set_ylim(-0.5, _NETWORK_N_DIGITS + 0.5)
    add_trial_boundaries_mpl(ax, metrics, x_mode)
    ax.legend(loc="best", fontsize=8, framealpha=0.92, edgecolor=C["border"], ncol=2)
    fig.tight_layout()
    _apply_figure_style(fig)
    return fig


def mpl_recovery_cumsum(
    df_dead: pd.DataFrame,
    metrics: dict[str, np.ndarray],
    layer_order: list[str],
) -> Figure:
    """Cumulative dead→active recoveries per hidden layer (one trace per layer)."""
    hidden = [ln for ln in layer_order if ln != "head"]
    all_cp_idxs, by_layer = compute_recovery_cumsum_by_layer(df_dead, layer_order)
    if not hidden or not all_cp_idxs:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        ax.text(
            0.5,
            0.5,
            "No recovery data (need dead-neuron table and hidden layers).",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C["muted"],
        )
        ax.axis("off")
        _apply_figure_style(fig)
        return fig

    xs, use_iter, x_mode = _x_values_for_checkpoints(all_cp_idxs, metrics)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=100)
    n_plotted = 0
    for li, layer_name in enumerate(hidden):
        cum = by_layer.get(layer_name)
        if cum is None or len(cum) != len(all_cp_idxs):
            continue
        color = _LAYER_COMBINED_COLORS[li % len(_LAYER_COMBINED_COLORS)]
        display_name = layer_name.replace("hidden.", "H").replace("head", "Head")
        ax.plot(
            xs,
            cum,
            marker=".",
            ms=4,
            lw=1.5,
            color=color,
            label=display_name,
            zorder=li,
        )
        n_plotted += 1

    if n_plotted == 0:
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        ax.text(
            0.5,
            0.5,
            "No recovery traces to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C["muted"],
        )
        ax.axis("off")
        _apply_figure_style(fig)
        return fig

    ax.set_title(
        "Cumulative recovery count (dead → active) — hidden layers",
        fontsize=13,
        color=C["fg"],
        pad=_AX_TITLE_PAD,
    )
    ax.set_xlabel("Iteration" if use_iter else "Checkpoint")
    ax.set_ylabel("Cumulative recoveries")
    add_trial_boundaries_mpl(ax, metrics, x_mode)
    ax.legend(loc="best", fontsize=8, framealpha=0.92, edgecolor=C["border"], ncol=2)
    fig.tight_layout()
    _apply_figure_style(fig)
    return fig


def split_ever_dead_by_final_checkpoint(
    df_dead: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    """Split neurons that were ever ``is_dead`` into two sets.

    Returns ``(dead_at_last_checkpoint, recovered)`` where *recovered* were dead
    at some earlier checkpoint but alive (not dead) at the last checkpoint.
    """
    if df_dead.empty:
        return set(), set()
    last_cp = int(df_dead["checkpoint_idx"].max())
    final = df_dead[df_dead["checkpoint_idx"] == last_cp]
    dm = pd.Series(final["is_dead"]).fillna(False)
    if dm.dtype == object:
        dead_mask = dm.astype(str).str.lower().isin(("true", "1", "t"))
    else:
        dead_mask = dm.astype(bool)
    dead_at_end = {str(x) for x in final.loc[dead_mask, "neuron_id"].tolist()}
    em = pd.Series(df_dead["is_dead"]).fillna(False)
    if em.dtype == object:
        ever_mask = em.astype(str).str.lower().isin(("true", "1", "t"))
    else:
        ever_mask = em.astype(bool)
    ever_dead = {str(x) for x in df_dead.loc[ever_mask, "neuron_id"].tolist()}
    recovered = ever_dead - dead_at_end
    return dead_at_end, recovered


def mpl_dead_layer_traces(
    layer_name: str,
    df_nd: pd.DataFrame,
    df_dead: pd.DataFrame,
    metrics: dict[str, np.ndarray],
    *,
    neuron_ids: Iterable[str] | None = None,
    title: str | None = None,
    empty_message: str | None = None,
) -> Figure:
    layer_dead = df_dead[(df_dead["layer_name"] == layer_name) & df_dead["is_dead"]]
    raw = layer_dead["neuron_id"].unique()
    if neuron_ids is not None:
        allow = {str(x) for x in neuron_ids}
        ever_dead_nids = np.array([n for n in raw if str(n) in allow], dtype=object)
    else:
        ever_dead_nids = raw

    empty_default = "No dead neurons in this layer."
    if len(ever_dead_nids) == 0:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        ax.text(
            0.5,
            0.5,
            empty_message if empty_message is not None else empty_default,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C["muted"],
            fontsize=14,
        )
        ax.axis("off")
        _apply_figure_style(fig)
        return fig

    layer_nd = df_nd[df_nd["layer_name"] == layer_name]
    inactive_counts = (
        layer_nd[layer_nd["status"] == "inactive"]
        .groupby(["neuron_id", "checkpoint_idx"])
        .size()
        .reset_index(name="n_inactive")
    )
    all_cp_idxs = sorted(layer_nd["checkpoint_idx"].unique())
    xs, use_iter, x_mode = _x_values_for_checkpoints(all_cp_idxs, metrics)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    cmap = plt.get_cmap("tab10")
    for i, nid in enumerate(ever_dead_nids):
        nc = inactive_counts[inactive_counts["neuron_id"] == nid].set_index("checkpoint_idx")["n_inactive"]
        ys = [int(nc.get(ci, 0)) for ci in all_cp_idxs]
        ax.plot(
            xs,
            ys,
            marker=".",
            ms=3,
            lw=1.5,
            color=cmap(i % 10),
            label=_compact_neuron_label(nid),
        )

    display_name = layer_name.replace("hidden.", "H").replace("head", "Head")
    head = title if title is not None else "Dead neurons"
    ax.set_title(
        f"{head} — inactive digit count — {display_name}",
        fontsize=13,
        color=C["fg"],
        pad=_AX_TITLE_PAD,
    )
    ax.set_xlabel("Iteration" if use_iter else "Checkpoint")
    ax.set_ylabel("# inactive digits")
    ax.set_ylim(-0.5, _NETWORK_N_DIGITS + 0.5)
    add_trial_boundaries_mpl(ax, metrics, x_mode)
    ax.legend(loc="best", fontsize=9, framealpha=0.92, edgecolor=C["border"])
    fig.tight_layout()
    _apply_figure_style(fig)
    return fig
