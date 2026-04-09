"""Cross-experiment plots for reassigned-neuron summaries (see cross_task_aggregate.ipynb)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from visualize_webapp.app import (
    _compact_neuron_label,
    _is_dead_column_to_bool,
    _pp_dead,
    _pp_neuron_digit,
)

_REASSIGNED_SUMMARY_COLUMNS: list[str] = [
    "compact",
    "neuron_id",
    "layer_name",
    "first_reassign_checkpoint",
    "digits_at_first_reassign",
    "first_dead_checkpoint",
]

_REACTIVATED_SUMMARY_COLUMNS: list[str] = [
    "compact",
    "neuron_id",
    "layer_name",
    "first_recovery_checkpoint",
]


def _empty_reassigned_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=_REASSIGNED_SUMMARY_COLUMNS)


def _normalize_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty and not df.columns.tolist():
        return _empty_reassigned_summary()
    missing = [c for c in _REASSIGNED_SUMMARY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"summary DataFrame missing columns: {missing}")
    return cast(pd.DataFrame, df[_REASSIGNED_SUMMARY_COLUMNS].copy())


def _empty_reactivated_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=_REACTIVATED_SUMMARY_COLUMNS)


def _normalize_reactivated_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty and not df.columns.tolist():
        return _empty_reactivated_summary()
    missing = [c for c in _REACTIVATED_SUMMARY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"reactivated summary DataFrame missing columns: {missing}")
    return cast(pd.DataFrame, df[_REACTIVATED_SUMMARY_COLUMNS].copy())


def compute_reassigned_summary(
    df_nd: pd.DataFrame | None,
    df_dead: pd.DataFrame | None,
) -> tuple[pd.DataFrame, int]:
    """Same logic as the *Reassigned neurons* cell in network_analysis*.ipynb.

    Returns ``(df_reassigned_summary, n_reassigned_neurons)``. Empty summaries use
    the standard column schema for safe concatenation.
    """
    if df_nd is None or df_nd.empty or df_dead is None or df_dead.empty:
        return _empty_reassigned_summary(), 0

    first_dead = (
        df_dead.loc[df_dead["is_dead"], ["neuron_id", "checkpoint_idx"]]
        .groupby("neuron_id", sort=False)["checkpoint_idx"]
        .min()
        .rename("first_dead_checkpoint")
    )
    assigned = df_nd.loc[
        df_nd["status"] == "assigned",
        ["neuron_id", "layer_name", "checkpoint_idx", "digit"],
    ]
    merged = assigned.merge(first_dead.reset_index(), on="neuron_id", how="inner")
    after_dead = merged[merged["checkpoint_idx"] > merged["first_dead_checkpoint"]]

    reassigned_neuron_ids = sorted(after_dead["neuron_id"].unique().tolist())
    n = len(reassigned_neuron_ids)

    if after_dead.empty:
        return _empty_reassigned_summary(), 0

    first_reassign_cp = after_dead.groupby("neuron_id", sort=False)["checkpoint_idx"].min()
    fr = first_reassign_cp.rename("first_reassign_checkpoint").reset_index()
    at_first = after_dead.merge(
        fr,
        left_on=["neuron_id", "checkpoint_idx"],
        right_on=["neuron_id", "first_reassign_checkpoint"],
    )
    digits_col = (
        at_first.groupby(["neuron_id", "layer_name", "first_reassign_checkpoint"], sort=False)[
            "digit"
        ]
        .apply(lambda s: ",".join(map(str, sorted(s.unique()))))
        .reset_index(name="digits_at_first_reassign")
    )
    df_reassigned_summary = digits_col.merge(
        first_dead.reset_index(), on="neuron_id", how="left"
    )
    df_reassigned_summary.insert(
        0,
        "compact",
        df_reassigned_summary["neuron_id"].map(_compact_neuron_label),
    )
    df_reassigned_summary = df_reassigned_summary.sort_values(
        ["layer_name", "first_reassign_checkpoint", "compact"],
        kind="stable",
    )
    return _normalize_summary_columns(df_reassigned_summary), n


def compute_reactivated_summary(
    df_dead: pd.DataFrame | None,
    *,
    exclude_head: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Neurons with at least one dead→alive step between consecutive checkpoints.

    Matches the *recovery* definition in ``compute_recovery_cumsum_by_layer`` /
    ``per_neuron_recovery_counts`` (hidden layers only when *exclude_head* is True).
    Returns ``(one row per such neuron with first recovery checkpoint, count)``.
    """
    if df_dead is None or df_dead.empty:
        return _empty_reactivated_summary(), 0

    sub = df_dead
    if exclude_head:
        sub = sub[sub["layer_name"].astype(str) != "head"]
    if sub.empty:
        return _empty_reactivated_summary(), 0

    rows: list[dict[str, str | int]] = []
    for ln in sorted(pd.Series(sub["layer_name"]).unique().tolist(), key=str):
        layer_df = sub[sub["layer_name"] == ln]
        all_cp_idxs = sorted(
            int(x) for x in pd.Series(layer_df["checkpoint_idx"]).unique().tolist()
        )
        n_cp = len(all_cp_idxs)
        if n_cp < 2:
            continue
        all_neurons = pd.Series(layer_df["neuron_id"]).unique().tolist()
        cols = pd.DataFrame(
            layer_df[["neuron_id", "checkpoint_idx", "is_dead"]],
            copy=True,
        )
        full_idx = pd.DataFrame(
            [(n, c) for n in all_neurons for c in all_cp_idxs],
            columns=["neuron_id", "checkpoint_idx"],
        )
        merged = full_idx.merge(cols, on=["neuron_id", "checkpoint_idx"], how="left")
        dead = _is_dead_column_to_bool(
            pd.Series(merged["is_dead"]).fillna(False)
        )
        n_per = n_cp
        for i, nid in enumerate(all_neurons):
            d = dead[i * n_per : (i + 1) * n_per]
            first_cp: int | None = None
            for j in range(n_cp - 1):
                if d[j] and not d[j + 1]:
                    first_cp = all_cp_idxs[j + 1]
                    break
            if first_cp is not None:
                nid_str = str(nid)
                rows.append(
                    {
                        "neuron_id": nid_str,
                        "layer_name": str(ln),
                        "first_recovery_checkpoint": first_cp,
                    }
                )

    if not rows:
        return _empty_reactivated_summary(), 0

    df_out = pd.DataFrame(rows)
    df_out.insert(0, "compact", df_out["neuron_id"].map(_compact_neuron_label))
    df_out = df_out.sort_values(
        ["layer_name", "first_recovery_checkpoint", "compact"],
        kind="stable",
    )
    return _normalize_reactivated_columns(df_out), len(df_out)


def concat_reassigned_summaries(
    rows: Iterable[tuple[str, str, pd.DataFrame]],
) -> pd.DataFrame:
    """Concatenate per-run summaries with a two-level row MultiIndex.

    We avoid ``pd.concat(..., keys=...)`` on frames with a default RangeIndex: that
    would create an extra unnamed index level (row position within each piece).
    """
    pieces: list[pd.DataFrame] = []
    for experiment_id, optimizer_id, df in rows:
        sdf = _normalize_summary_columns(df).reset_index(drop=True)
        pieces.append(
            sdf.assign(experiment_id=experiment_id, optimizer_id=optimizer_id)
        )

    if not pieces:
        return pd.DataFrame(
            columns=_REASSIGNED_SUMMARY_COLUMNS,
            index=pd.MultiIndex.from_tuples([], names=["experiment_id", "optimizer_id"]),
        )

    out = pd.concat(pieces, ignore_index=True)
    return out.set_index(["experiment_id", "optimizer_id"])


def concat_reactivated_summaries(
    rows: Iterable[tuple[str, str, pd.DataFrame]],
) -> pd.DataFrame:
    """Same row MultiIndex convention as :func:`concat_reassigned_summaries`."""
    pieces: list[pd.DataFrame] = []
    for experiment_id, optimizer_id, df in rows:
        sdf = _normalize_reactivated_columns(df).reset_index(drop=True)
        pieces.append(
            sdf.assign(experiment_id=experiment_id, optimizer_id=optimizer_id)
        )

    if not pieces:
        return pd.DataFrame(
            columns=_REACTIVATED_SUMMARY_COLUMNS,
            index=pd.MultiIndex.from_tuples([], names=["experiment_id", "optimizer_id"]),
        )

    out = pd.concat(pieces, ignore_index=True)
    return out.set_index(["experiment_id", "optimizer_id"])


def optimizer_ids_for_experiment_runs(
    tree: dict[str, dict[str, dict[str, list[str]]]],
    experiment_to_run: dict[str, str],
    model_id: str,
) -> list[str]:
    """All ``optimizer_id`` strings under the selected runs (same order as discovery).

    Use this when plots or tables should include optimizers that have **zero**
    reassigned neurons: those optimizers do not appear in
    :func:`load_summaries_for_experiment_runs` because empty summaries add no rows.
    """
    seen: dict[str, None] = {}
    for eid, rid in experiment_to_run.items():
        models = tree.get(eid, {})
        runs = models.get(model_id, {})
        for oid in runs.get(rid, []):
            seen.setdefault(oid, None)
    return sorted(seen.keys())


def load_summaries_for_experiment_runs(
    tree: dict[str, dict[str, dict[str, list[str]]]],
    experiment_to_run: dict[str, str],
    model_id: str,
) -> pd.DataFrame:
    """Load post-processing CSVs for each (experiment, run, optimizer) and concatenate summaries.

    Optimizers with no reassigned neurons produce an empty summary and contribute **no
    rows**, so they are absent from the returned index. See
    :func:`optimizer_ids_for_experiment_runs` for the full list from discovery.
    """
    pieces: list[tuple[str, str, pd.DataFrame]] = []
    for eid, rid in experiment_to_run.items():
        models = tree.get(eid, {})
        runs = models.get(model_id, {})
        oids = runs.get(rid, [])
        for oid in oids:
            df_nd = _pp_neuron_digit(eid, model_id, oid, rid=rid)
            df_dead = _pp_dead(eid, model_id, oid, rid=rid)
            summary, _ = compute_reassigned_summary(df_nd, df_dead)
            pieces.append((eid, oid, summary))
    return concat_reassigned_summaries(pieces)


def load_reactivated_summaries_for_experiment_runs(
    tree: dict[str, dict[str, dict[str, list[str]]]],
    experiment_to_run: dict[str, str],
    model_id: str,
) -> pd.DataFrame:
    """Load dead-neuron CSVs only; one summary row per neuron with ≥1 recovery transition."""
    pieces: list[tuple[str, str, pd.DataFrame]] = []
    for eid, rid in experiment_to_run.items():
        models = tree.get(eid, {})
        runs = models.get(model_id, {})
        oids = runs.get(rid, [])
        for oid in oids:
            df_dead = _pp_dead(eid, model_id, oid, rid=rid)
            summary, _ = compute_reactivated_summary(df_dead)
            pieces.append((eid, oid, summary))
    return concat_reactivated_summaries(pieces)


def _ensure_two_level_experiment_optimizer_index(df_multi: pd.DataFrame) -> pd.DataFrame:
    idx = df_multi.index
    if idx.nlevels == 3 and list(idx.names[:2]) == [
        "experiment_id",
        "optimizer_id",
    ]:
        return df_multi.droplevel(-1)
    return df_multi


def _plot_grouped_bar_counts_by_experiment_optimizer(
    df_multi: pd.DataFrame,
    *,
    ax: Axes | None,
    experiment_order: list[str] | None,
    optimizer_order: list[str] | None,
    colors: list[Any] | None,
    bar_width: float,
    ylabel: str,
    title: str,
) -> Axes:
    df_multi = _ensure_two_level_experiment_optimizer_index(df_multi)
    idx = df_multi.index

    if idx.nlevels != 2 or list(idx.names) != ["experiment_id", "optimizer_id"]:
        raise ValueError(
            "df_multi must have a two-level index (experiment_id, optimizer_id)"
        )

    counts = df_multi.groupby(level=["experiment_id", "optimizer_id"], sort=False).size()
    experiments = (
        experiment_order
        if experiment_order is not None
        else list(dict.fromkeys(counts.index.get_level_values(0).tolist()))
    )
    optimizers = (
        optimizer_order
        if optimizer_order is not None
        else sorted(set(counts.index.get_level_values(1).tolist()))
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6.0, 0.45 * len(optimizers)), 4.0), dpi=100)

    n_exp = len(experiments)
    if n_exp == 0:
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return ax

    x = np.arange(len(optimizers), dtype=float)
    offsets = (np.arange(n_exp, dtype=float) - (n_exp - 1) / 2.0) * bar_width
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(n_exp)]

    for i, eid in enumerate(experiments):
        heights = [
            int(counts.loc[(eid, oid)]) if (eid, oid) in counts.index else 0
            for oid in optimizers
        ]
        ax.bar(
            x + offsets[i],
            heights,
            width=bar_width,
            label=eid,
            color=colors[i % len(colors)],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(optimizers, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Optimizer")
    ax.set_title(title)
    ax.legend(title="Experiment", loc="best")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.8, alpha=0.35)
    return ax


def plot_reassigned_neuron_counts_grouped(
    df_multi: pd.DataFrame,
    *,
    ax: Axes | None = None,
    experiment_order: list[str] | None = None,
    optimizer_order: list[str] | None = None,
    colors: list[Any] | None = None,
    bar_width: float = 0.38,
) -> Axes:
    """Grouped bars: x = optimizer, one bar per experiment, y = reassigned neuron count."""
    return _plot_grouped_bar_counts_by_experiment_optimizer(
        df_multi,
        ax=ax,
        experiment_order=experiment_order,
        optimizer_order=optimizer_order,
        colors=colors,
        bar_width=bar_width,
        ylabel="Reassigned neurons (count)",
        title="Reassigned neurons after first death",
    )


def plot_reactivated_neuron_counts_grouped(
    df_multi: pd.DataFrame,
    *,
    ax: Axes | None = None,
    experiment_order: list[str] | None = None,
    optimizer_order: list[str] | None = None,
    colors: list[Any] | None = None,
    bar_width: float = 0.38,
) -> Axes:
    """Grouped bars: x = optimizer, one bar per experiment, y = re-activated neuron count.

    Re-activated means at least one consecutive-checkpoint transition from dead to not dead
    (hidden layers only; same as cumulative recovery plots in the network analysis notebooks).
    """
    return _plot_grouped_bar_counts_by_experiment_optimizer(
        df_multi,
        ax=ax,
        experiment_order=experiment_order,
        optimizer_order=optimizer_order,
        colors=colors,
        bar_width=bar_width,
        ylabel="Re-activated neurons (count)",
        title="Re-activated neurons (dead → alive, hidden layers)",
    )


__all__ = [
    "compute_reassigned_summary",
    "compute_reactivated_summary",
    "concat_reassigned_summaries",
    "concat_reactivated_summaries",
    "optimizer_ids_for_experiment_runs",
    "load_summaries_for_experiment_runs",
    "load_reactivated_summaries_for_experiment_runs",
    "plot_reassigned_neuron_counts_grouped",
    "plot_reactivated_neuron_counts_grouped",
]
