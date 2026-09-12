"""Matplotlib variants of Network Analysis panel plots (see network_analysis.ipynb)."""

from collections.abc import Sequence
from typing import Iterable

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from models.unit_node_id import parse_unit_node_id

from visualize_webapp.app import (
    C,
    _activation_sample_matrix,
    _DIGIT_COLORS,
    _NETWORK_EVAL_K_LINE_COLORS,
)
from visualize_webapp.common import _compact_neuron_label
from visualize_webapp.constants import (
    _NETWORK_EXPECTED_BATCH,
    _NETWORK_N_DIGITS,
    _NETWORK_SAMPLES_PER_DIGIT,
)
from visualize_webapp.plot_helpers import add_trial_boundaries_mpl, _trial_boundaries_from_metrics
from visualize_webapp.post_processing import (
    compute_recovery_cumsum_by_layer,
    dead_layer_inactive_trace_matrix,
    layer_inactive_count_lookup,
    layer_status_count_matrix,
    neuron_assigned_count_series,
)


def _rgba_tuple(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    """Matplotlib-compatible RGBA; avoid CSS `rgba(...)` strings for facecolor."""
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


def parse_neuron_id_mpl(neuron_id):
    parsed = parse_unit_node_id(neuron_id)
    if neuron_id is None or parsed is None:
        return ""
    layer_idx = int(parsed["layer_name"].split(".")[-1]) // 2
    return r"$n^{("+str(layer_idx)+r")}_{"+str(parsed["unit_index"])+r"}$"


def _apply_figure_style(fig: Figure) -> None:
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.grid(True, color=C["grid"], linestyle="-", linewidth=0.8)
        ax.tick_params(colors=C["fg"])
        for spine in ax.spines.values():
            spine.set_color(C["border"])


def _eval_act_x_aligned(
    neuron_id: str,
    nts: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[float], str] | None:
    """Align `neuron_timeseries` activation rows with metrics x-axis, or return None."""
    A = _activation_sample_matrix(nts, str(neuron_id))
    if A is None or A.ndim != 2:
        return None
    tags = [str(t) for t in nts.get("checkpoint_tags", [])]
    cp_iters = metrics.get("checkpoint_iterations")
    n = int(A.shape[0])
    if not tags or len(tags) != n:
        return None
    if cp_iters is None or len(cp_iters) != len(tags):
        return None
    all_cp_idxs = list(range(n))
    xs, _, x_mode = _x_values_for_checkpoints(all_cp_idxs, metrics)
    return A, xs, x_mode


def mpl_neuron_assigned_digit_count_over_time(
    neuron_id: str,
    df_nd: pd.DataFrame,
    metrics: dict[str, np.ndarray],
) -> Figure:
    """Line chart: number of assigned digits per checkpoint for one neuron."""
    layer_name, all_cp_idxs, y = neuron_assigned_count_series(df_nd, neuron_id)
    if layer_name is None:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        ax.text(
            0.5,
            0.5,
            "No neuron-digit rows for this ID.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C["muted"],
            fontsize=12,
        )
        ax.axis("off")
        _apply_figure_style(fig)
        return fig
    xs, _, x_mode = _x_values_for_checkpoints(all_cp_idxs, metrics)

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)
    ax.plot(xs, y, marker="o", markersize=4, color=C["blue"], linewidth=1.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Iteration" if x_mode == "iteration" else "Checkpoint")
    ax.set_ylabel("# assigned digits")
    disp = layer_name.replace("hidden.", "H").replace("head", "Head")
    ax.set_title(
        f"Assigned digit count over time — {parse_neuron_id_mpl(neuron_id)} ({disp})",
        fontsize=12,
        color=C["fg"],
        pad=_AX_TITLE_PAD,
    )
    add_trial_boundaries_mpl(ax, metrics, x_mode)
    fig.tight_layout()
    _apply_figure_style(fig)
    return fig


def mpl_neuron_eval_activation_mean_over_time(
    neuron_id: str,
    nts: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    *,
    min_idx: int = 0,
    max_idx: int | None = None,
    cp=None,
) -> Figure:
    """Mean pre-nonlinearity eval-batch activation per checkpoint (`neuron_timeseries.npz`).

    `min_idx` / optional `max_idx` (exclusive, like Python slicing) restrict the
    plotted checkpoints to `xs[min_idx:max_idx]`. Trial/stage separators follow
    the same x-window; optional `cp` draws an extra dashed vertical highlight when
    not `None`.
    """
    aligned = _eval_act_x_aligned(neuron_id, nts, metrics)
    if aligned is None:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        msg = (
            "No eval-batch activation timeseries for this neuron "
            "(need neuron_timeseries.npz with aligned checkpoint_tags / iterations)."
        )
        if not nts:
            msg = "No neuron_timeseries.npz data loaded."
        ax.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C["muted"],
            fontsize=11,
        )
        ax.axis("off")
        _apply_figure_style(fig)
        return fig

    A, xs, x_mode = aligned
    mean_act = np.nanmean(A, axis=1)
    n_s = int(A.shape[1])
    if n_s > 1:
        std_act = np.nanstd(A, axis=1, ddof=1)
        sem = std_act / np.sqrt(n_s)
        lo = mean_act - 1.96 * sem
        hi = mean_act + 1.96 * sem
    else:
        lo = hi = mean_act

    n_cp = int(len(xs))
    mni = int(min_idx)
    if mni < 0:
        mni = 0
    if mni > n_cp:
        mni = n_cp
    if max_idx is None:
        mxi = n_cp
    else:
        mxi = min(int(max_idx), n_cp)
        if mxi < mni:
            mxi = mni

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)
    ax.fill_between(
        xs[mni:mxi],
        lo[mni:mxi],
        hi[mni:mxi],
        color=C["blue"],
        alpha=0.22,
        linewidth=0,
        label="95% CI (mean)",
    )
    ax.plot(
        xs[mni:mxi],
        mean_act[mni:mxi],
        color=C["blue"],
        linewidth=1.5,
        marker="o",
        markersize=3,
        label="mean (eval batch)",
    )
    ax.set_xlabel("Iteration" if x_mode == "iteration" else "Checkpoint")
    ax.set_ylabel("Pre-ReLU activation")
    ax.set_title(
        f"Mean eval-batch activation — {parse_neuron_id_mpl(neuron_id)}",
        fontsize=12,
        color=C["fg"],
        pad=_AX_TITLE_PAD,
    )
    ax.axhline(0.0, color=C["red"], linestyle="-")
    if cp is not None:
        ax.axvline(float(cp), color="k", linestyle="--", linewidth=1, zorder=2)
    add_trial_boundaries_mpl(
        ax,
        metrics,
        x_mode,
        min_idx=mni,
        max_idx=mxi if max_idx is not None else None,
        plot_x_per_segment=xs,
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.92, edgecolor=C["border"])
    fig.tight_layout()
    _apply_figure_style(fig)
    return fig


def mpl_neuron_eval_activation_angle_over_time(
    neuron_id: str,
    nts: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    *,
    signed_angle_y_deg_limit: float = 90.0,
    min_idx: int = 0,
    max_idx: int | None = None,
    cp = None,
    report = False,
    only_digits=None,
    inset_center: float | None = None,
) -> Figure:
    """Per-digit signed slope angle in the (iteration, activation) plane.

    For each eval sample (K=5 per digit), consecutive checkpoints define a segment
    from `(x_t, a_t)` to `(x_{t+1}, a_{t+1})`. The signed angle is
    `atan2(Δa, Δx)` in degrees (positive when activation rises with x).
    Values are clipped to `±signed_angle_y_deg_limit` for display (default 90°).

    Layout: 10 subplots in a 5x2 grid (two digits per row), matching the MNIST
    eval batch order `5*d + k`.

    Trial/stage separators: black dashed vertical lines from `metrics` (same keys
    as Plotly: `trial_names` / `trial_end_*` or `stage_*`). `min_idx` and
    optional `max_idx` (exclusive, like Python slice) select the same segment
    window as the curves (`xs[1:][min_idx:max_idx]`); joins and trial labels
    outside that x-range are omitted. Optional `cp` draws an extra dashed vertical
    highlight when not `None`.

    When `only_digits` and `inset_center` are both set, each digit panel gets a
    bottom-right inset zooming the x-axis to
    `[inset_center - 5, inset_center + 5]` (same angles as the main trace for that
    digit). On each inset, the x-axis tick at `inset_center` is labeled
    `$t_\\text{start}^{(b_i)}$` instead of the numeric value (no extra vline or
    inset title on the inset). Inset x-ticks are spaced by 2 in the zoom window;
    inset grid uses dashed lines. `mark_inset` draws a rectangle on the main axes
    for the zoomed region and connector lines from that rectangle’s lower-left and
    upper-right to the inset’s matching corners.
    """
    aligned = _eval_act_x_aligned(neuron_id, nts, metrics)
    if aligned is None:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        msg = (
            "No eval-batch activation timeseries for this neuron "
            "(need neuron_timeseries.npz with aligned checkpoint_tags / iterations)."
        )
        if not nts:
            msg = "No neuron_timeseries.npz data loaded."
        ax.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C["muted"],
            fontsize=11,
        )
        ax.axis("off")
        _apply_figure_style(fig)
        return fig

    A, xs, x_mode = aligned
    if A.shape[0] < 2:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        ax.text(
            0.5,
            0.5,
            "Need at least two checkpoints for angle plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C["muted"],
            fontsize=12,
        )
        ax.axis("off")
        _apply_figure_style(fig)
        return fig

    if int(A.shape[1]) < _NETWORK_EXPECTED_BATCH:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        ax.text(
            0.5,
            0.5,
            f"Expected {_NETWORK_EXPECTED_BATCH} eval samples; got {int(A.shape[1])}.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C["muted"],
            fontsize=11,
        )
        ax.axis("off")
        _apply_figure_style(fig)
        return fig

    xs_arr = np.asarray(xs, dtype=np.float64)
    dx = np.diff(xs_arr)
    xs_seg = xs_arr[1:]
    n_seg = int(xs_seg.shape[0])
    mni = int(min_idx)
    if mni < 0:
        mni = 0
    if mni > n_seg:
        mni = n_seg
    if max_idx is None:
        mxi = n_seg
    else:
        mxi = min(int(max_idx), n_seg)
        if mxi < mni:
            mxi = mni

    if only_digits is not None:
        n_rows, n_cols = 1,len(only_digits)
        digits = only_digits
        size = (6 * len(only_digits),4)
    else:
        digits = range(_NETWORK_N_DIGITS)
        n_rows, n_cols = 5, 2
        size = (10, 14)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=size,
        dpi=100,
        sharey=True,
    )
    lim = float(signed_angle_y_deg_limit)

    handles, labels = None, None  # Will store handles for the legend (from first axes)

    for i,d in enumerate(digits):
        if only_digits is not None:
            ax = axes[i]
        else:
            ax = axes[i // n_cols, i % n_cols]
        for k in range(_NETWORK_SAMPLES_PER_DIGIT):
            col = _NETWORK_SAMPLES_PER_DIGIT * d + k
            dy = np.diff(A[:, col].astype(np.float64, copy=False))
            ang_deg = np.degrees(np.arctan2(dy, dx))
            ax.plot(
                xs_seg[mni:mxi],
                ang_deg[mni:mxi],
                color=_NETWORK_EVAL_K_LINE_COLORS[k % len(_NETWORK_EVAL_K_LINE_COLORS)],
                linewidth=1.25,
                marker=".",
                markersize=2.5,
                label=f"K={k + 1}",
                alpha=0.5,
            )
        ax.axhline(0.0, color=C["border"], linewidth=0.8, linestyle="-", zorder=0)
        ax.set_ylim(-lim, lim)
        ax.set_title(
            f"Digit {d}",
            fontsize=10,
            color=C["fg"],
            pad=_AX_TITLE_PAD,
        )
        if cp is not None:
            ax.axvline(float(cp), color=C["red"], linestyle="--", linewidth=1, zorder=2)
        add_trial_boundaries_mpl(
            ax,
            metrics,
            x_mode,
            min_idx=mni,
            max_idx=mxi if max_idx is not None else None,
            plot_x_per_segment=xs_seg,
            report=report,
        )
        if d == 0:
            # Store handles and labels for the legend from the first axes
            handles, labels = ax.get_legend_handles_labels()

    angle_inset_axes: list[Axes] = []
    if only_digits is not None and inset_center is not None and len(digits) > 0:
        xc = float(inset_center)
        x_lo = xc - 5.0
        x_hi = xc + 5.0
        sb_ins = _trial_boundaries_from_metrics(metrics, x_mode)
        flat_axes = np.atleast_1d(axes).flatten()
        for i, d in enumerate(digits):
            ax_parent = flat_axes[i]
            d_i = int(d)
            axins = ax_parent.inset_axes(
                (0.52, 0.12, 0.44, 0.36),
                transform=ax_parent.transAxes,
            )
            for k in range(_NETWORK_SAMPLES_PER_DIGIT):
                col = _NETWORK_SAMPLES_PER_DIGIT * d_i + k
                dy = np.diff(A[:, col].astype(np.float64, copy=False))
                ang_deg = np.degrees(np.arctan2(dy, dx))
                axins.plot(
                    xs_seg[mni:mxi],
                    ang_deg[mni:mxi],
                    color=_NETWORK_EVAL_K_LINE_COLORS[k % len(_NETWORK_EVAL_K_LINE_COLORS)],
                    linewidth=1.0,
                    marker=".",
                    markersize=2.0,
                    alpha=0.5,
                )
            axins.axhline(0.0, color=C["border"], linewidth=0.8, linestyle="-", zorder=0)
            axins.set_xlim(x_lo, x_hi)
            axins.set_ylim(-lim, lim)
            first_x = float(np.ceil(x_lo / 2.0) * 2.0)
            last_x = float(np.floor(x_hi / 2.0) * 2.0)
            if last_x >= first_x:
                n_steps = int(round((last_x - first_x) / 2.0)) + 1
                by2 = first_x + 2.0 * np.arange(n_steps, dtype=np.float64)
            else:
                by2 = np.array([], dtype=np.float64)
            xticks = np.unique(np.concatenate([by2, np.array([xc], dtype=np.float64)]))
            xticks = np.sort(xticks[(xticks >= x_lo - 1e-12) & (xticks <= x_hi + 1e-12)])
            atol = max(1e-9, 1e-7 * (abs(xc) + 1.0))
            xtick_labels: list[str] = []
            for t in xticks:
                if np.isclose(t, xc, rtol=0.0, atol=atol):
                    xtick_labels.append(r"$t_\text{start}^{(b_i)}$")
                else:
                    xtick_labels.append(f"{float(t):g}")
            axins.set_xticks(xticks)
            axins.set_xticklabels(xtick_labels, fontsize=7)
            axins.tick_params(axis="y", labelsize=7)
            if sb_ins is not None:
                _, boundary_xs_ins, _ = sb_ins
                for xv in boundary_xs_ins:
                    xb = float(xv)
                    if xb < x_lo or xb > x_hi or abs(xb - xc) < 1e-9:
                        continue
                    axins.axvline(xb, color="#000000", linestyle="--", linewidth=1.0, zorder=1)
            # Zoom bracket: parent rectangle + lines from its lower-left / upper-right
            # to the inset’s matching corners (matplotlib loc 3 and 1).
            mark_inset(
                ax_parent,
                axins,
                loc1=1,
                loc2=3,
                fc="none",
                ec=C["border"],
                linewidth=0.9,
                zorder=5,
            )
            angle_inset_axes.append(axins)

    # Digit grid: no per-axes x title (matches Plotly neuron detail digit rows:
    # title_text="" on those x-axes so trial/stage annotations stay readable).
    x_label = "Iteration" if x_mode == "iteration" else "Checkpoint"
    if only_digits is not None:
        for ax in axes:
            ax.set_ylabel("Signed angle (°)", fontsize=10, color=C["fg"])
    else:
        for ax in axes[:, 0]:
            ax.set_ylabel("Signed angle (°)", fontsize=10, color=C["fg"])

    fig.suptitle(
        r"$\arctan(\Delta \mathbf{A}(t))$ per eval sample over time $-$" + parse_neuron_id_mpl(neuron_id),
        fontsize=12,
        color=C["fg"],
        y=0.995,
    )
    if handles and labels:
        # Place the legend at the top center below the title
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=_NETWORK_SAMPLES_PER_DIGIT,
            fontsize=9,
            framealpha=0.92,
            edgecolor=C["border"],
        )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.supxlabel(x_label, fontsize=10, color=C["fg"],y=-0.02)
    _apply_figure_style(fig)
    for axins in angle_inset_axes:
        axins.grid(True, color=C["grid"], linestyle="--", linewidth=0.8)
    return fig


def mpl_neuron_eval_activation_per_sample_over_time(
    neuron_id: str,
    nts: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    *,
    min_idx: int = 0,
    max_idx: int | None = None,
    cp=None,
    report = False,
    only_digits=None,
    inset_center = None,
) -> Figure:
    """Pre-nonlinearity activation at each checkpoint for each eval sample (K per digit).

    Layout: 10 subplots in a 5x2 grid (MNIST eval batch order `5*d + k`), matching
    `mpl_neuron_eval_activation_angle_over_time` but y is activation vs checkpoint
    x-axis (one point per checkpoint, not per segment angle).

    `min_idx` / optional `max_idx` (exclusive) select `xs[min_idx:max_idx]`.
    Trial/stage separators use the same window via `metrics`. Optional `cp`:
    dashed vertical highlight when not `None`.
    """
    aligned = _eval_act_x_aligned(neuron_id, nts, metrics)
    if aligned is None:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        msg = (
            "No eval-batch activation timeseries for this neuron "
            "(need neuron_timeseries.npz with aligned checkpoint_tags / iterations)."
        )
        if not nts:
            msg = "No neuron_timeseries.npz data loaded."
        ax.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C["muted"],
            fontsize=11,
        )
        ax.axis("off")
        _apply_figure_style(fig)
        return fig

    A, xs, x_mode = aligned
    if int(A.shape[0]) < 1:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        ax.text(
            0.5,
            0.5,
            "No checkpoints in activation timeseries.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C["muted"],
            fontsize=12,
        )
        ax.axis("off")
        _apply_figure_style(fig)
        return fig

    if int(A.shape[1]) < _NETWORK_EXPECTED_BATCH:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        ax.text(
            0.5,
            0.5,
            f"Expected {_NETWORK_EXPECTED_BATCH} eval samples; got {int(A.shape[1])}.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C["muted"],
            fontsize=11,
        )
        ax.axis("off")
        _apply_figure_style(fig)
        return fig

    n_cp = int(len(xs))
    mni = int(min_idx)
    if mni < 0:
        mni = 0
    if mni > n_cp:
        mni = n_cp
    if max_idx is None:
        mxi = n_cp
    else:
        mxi = min(int(max_idx), n_cp)
        if mxi < mni:
            mxi = mni

    if only_digits is not None:
        n_rows, n_cols = 1,len(only_digits)
        size = (6 * len(only_digits),4)
        digits = only_digits
    else:
        digits = range(_NETWORK_N_DIGITS)
        n_rows, n_cols = 5, 2
        size = (10, 14)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=size,
        dpi=100,
        sharey=True,
    )

    handles, labels = None, None

    for i,d in enumerate(digits):
        if only_digits is not None:
            ax = axes[i]
        else:
            ax = axes[i // n_cols, i % n_cols]
        for k in range(_NETWORK_SAMPLES_PER_DIGIT):
            col = _NETWORK_SAMPLES_PER_DIGIT * d + k
            yv = A[mni:mxi, col].astype(np.float64, copy=False)
            ax.plot(
                xs[mni:mxi],
                yv,
                color=_NETWORK_EVAL_K_LINE_COLORS[k % len(_NETWORK_EVAL_K_LINE_COLORS)],
                linewidth=1.25,
                marker=".",
                markersize=2.5,
                label=f"K={k + 1}",
                alpha=0.5,
            )
        ax.axhline(0.0, color=C["red"], linewidth=0.8, linestyle="-", zorder=3)
        ax.set_title(
            f"Digit {d}",
            fontsize=10,
            color=C["fg"],
            pad=_AX_TITLE_PAD,
        )
        if cp is not None:
            ax.axvline(float(cp), color=C["red"], linestyle="--", linewidth=1, zorder=2)
        add_trial_boundaries_mpl(
            ax,
            metrics,
            x_mode,
            min_idx=mni,
            max_idx=mxi if max_idx is not None else None,
            plot_x_per_segment=xs,
            report=report,
        )
        if d == 0:
            handles, labels = ax.get_legend_handles_labels()

    x_label = "Iteration" if x_mode == "iteration" else "Checkpoint"
    if only_digits is not None:
        for ax in axes:
            ax.set_ylabel("Pre-ReLU activation", fontsize=10, color=C["fg"])
    else:
        for ax in axes[:, 0]:
            ax.set_ylabel("Pre-ReLU activation", fontsize=10, color=C["fg"])

    fig.suptitle(
        r"Activation $\mathbf{A}(t)$ per digit over time $-$"+parse_neuron_id_mpl(neuron_id),
        fontsize=12,
        color=C["fg"],
        y=0.995,
    )
    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=_NETWORK_SAMPLES_PER_DIGIT,
            fontsize=9,
            framealpha=0.92,
            edgecolor=C["border"],
        )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.supxlabel(x_label, fontsize=10, color=C["fg"], y=-0.02)
    _apply_figure_style(fig)
    return fig


def mpl_layer_inactive_count(
    layer_name: str,
    df_nd: pd.DataFrame,
    metrics: dict[str, np.ndarray],
) -> Figure:
    all_cp_idxs, lookup = layer_inactive_count_lookup(df_nd, layer_name)

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
    all_cp_idxs, _all_neurons, mat = layer_status_count_matrix(df_nd, layer_name, status)

    display_name = layer_name.replace("hidden.", "H").replace("head", "Head")
    base_color = C["blue"] if status == "assigned" else C["red"]

    use_iter, cp_iters = _x_mode_from_metrics(metrics)

    if boxplot:
        tick_labels: list[str] = []
        data: list[np.ndarray] = []
        for cp_i, ci in enumerate(all_cp_idxs):
            label = (
                str(int(cp_iters[ci])) if use_iter and cp_iters is not None else str(ci)
            )
            tick_labels.append(label)
            data.append(mat[:, cp_i].astype(float))

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
        for cp_i, ci in enumerate(all_cp_idxs):
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
        all_cp_idxs, _all_neurons, mat = layer_status_count_matrix(df_nd, layer_name, status)
        if not len(all_cp_idxs):
            continue

        xs, _, _ = _x_values_for_checkpoints(all_cp_idxs, metrics)
        means: list[float] = []
        ses: list[float] = []
        for cp_i, ci in enumerate(all_cp_idxs):
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
    all_cp_idxs, selected_nids, traces = dead_layer_inactive_trace_matrix(
        layer_name,
        df_nd,
        df_dead,
        neuron_ids=neuron_ids,
    )

    empty_default = "No dead neurons in this layer."
    if len(selected_nids) == 0:
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
    xs, use_iter, x_mode = _x_values_for_checkpoints(all_cp_idxs, metrics)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    cmap = plt.get_cmap("tab10")
    for i, nid in enumerate(selected_nids):
        ys = traces[i, :].astype(int).tolist()
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
