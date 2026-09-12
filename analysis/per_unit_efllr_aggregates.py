"""Per-weight effective learning rate and |Δw| aggregates (Sections B/C of per_unit_efllr.ipynb)."""

import gc
from collections.abc import Callable, Iterator
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from compute_results.defaults import DEFAULT_ADAM_EPS
from models.dnn_head import HIDDEN_SIZE
from models.unit_node_id import parse_unit_node_id
from analysis.common import _opt_color, _opt_display, _opt_type, _sort_optimizer_ids
from analysis.eff_lr import _sem_stderr, _welch_pairwise_holm_sig_matrix
from analysis.eff_lr import _parse_lr_from_optimizer_id, _signal_row_for_checkpoint
from analysis.final.final_additional_plots_helpers import (
    _btsp_segment_spec_for_experiment,
    _sem_across_values,
    _spearman_annotation,
)
from analysis.final.final_compare_all_optimizers_helpers import (
    _is_category_experiment,
    _sort_layer_display_names,
)
from analysis.final.final_experiment_display import (
    ExperimentDisplayLabels,
    two_row_cat12_from_sorted,
)
from analysis.io import _metrics, _nts, _sigs
from analysis.scoring_constants import N_SAMPLES_PER_TRIAL
from analysis.scoring_helpers import _unit_key, is_pretrain_shuffle_mislabel_experiment

AggregateQuantity = Literal["efflr", "dw"]

N_START_ITS = 10
WINDOW_SIZE = 100
N_SURROUNDING_PERIODSTART_ITS = 3
SEM_Z = 1.96

DEFAULT_LAYER_NAMES = _sort_layer_display_names([f"hidden.{i * 2}" for i in range(5)])

_PREPOST_BAR_W = 0.34
_PREPOST_GROUP_GAP = 0.22


# Core per-weight effective LR helpers (from per_unit_efllr.ipynb)


def _h_inv_block_keys(layer_name: str) -> tuple[str, str]:
    safe = str(layer_name).replace(".", "__")
    return f"h_inv__{safe}__weight__L", f"h_inv__{safe}__weight__R"


def _require_pure_shampoo_blocks(
    sigs: dict,
    oid: str,
    run_key: tuple[str, str, str, str],
    *,
    layer_names: list[str] | None = None,
) -> None:
    if _opt_type(oid) != "pure_shampoo":
        return
    layers = layer_names or DEFAULT_LAYER_NAMES
    missing: list[str] = []
    for layer_name in layers:
        l_key, r_key = _h_inv_block_keys(layer_name)
        if l_key not in sigs:
            missing.append(l_key)
        if r_key not in sigs:
            missing.append(r_key)
    if missing:
        eid, mid, oid_s, rid = run_key
        avail = sorted(k for k in sigs if str(k).startswith("h_inv__"))
        raise FileNotFoundError(
            f"Pure Shampoo signals.npz missing required block matrices for "
            f"{eid}/{mid}/{oid_s}/{rid}. Missing keys: {missing}. "
            f"Available h_inv__ keys: {avail or '(none)'}"
        )


def _load_run_sigs_metrics_fresh(
    eid: str,
    mid: str,
    oid: str,
    rid: str,
    *,
    layer_names: list[str] | None = None,
) -> tuple[dict, dict]:
    """Load signals/metrics from disk without retaining them in a module cache."""
    run_key = (str(eid), str(mid), str(oid), str(rid))
    sigs = _sigs(eid, mid, oid, rid=rid, w_cache=False)
    metrics = _metrics(eid, mid, oid, rid=rid, w_cache=False)
    _require_pure_shampoo_blocks(sigs, oid, run_key, layer_names=layer_names)
    return sigs, metrics


def _unload_run_sigs_metrics(sigs: dict, metrics: dict) -> None:
    sigs.clear()
    metrics.clear()
    gc.collect()


def load_run_sigs_metrics(
    row: pd.Series | dict[str, Any],
    *,
    layer_names: list[str] | None = None,
) -> tuple[dict, dict]:
    """Load signals/metrics for one run (no RAM cache — caller owns the objects)."""
    if isinstance(row, pd.Series):
        eid = str(row.experiment_id)
        mid = str(row.model_id)
        oid = str(row.optimizer_id)
        rid = str(row.run_id)
    else:
        eid = str(row["experiment_id"])
        mid = str(row["model_id"])
        oid = str(row["optimizer_id"])
        rid = str(row["run_id"])
    return _load_run_sigs_metrics_fresh(
        eid, mid, oid, rid, layer_names=layer_names
    )


def load_run_sigs_metrics_key(
    eid: str, mid: str, oid: str, rid: str,
) -> tuple[dict, dict]:
    return _load_run_sigs_metrics_fresh(eid, mid, oid, rid)


def n_incoming_weights(layer_name: str) -> int:
    return 784 if str(layer_name) == "hidden.0" else HIDDEN_SIZE


def _safe_neuron_key(neuron_id: str) -> str:
    return str(neuron_id).replace(":", "__")


def gamma_eff_weights_at_cp(
    sigs: dict,
    oid: str,
    neuron_id: str,
    cp_row_idx: int | None,
) -> np.ndarray:
    if cp_row_idx is None or cp_row_idx < 0:
        parsed = parse_unit_node_id(str(neuron_id))
        n_in = n_incoming_weights(parsed["layer_name"]) if parsed else HIDDEN_SIZE
        return np.full(n_in, np.nan)

    otype = _opt_type(oid)
    lr = _parse_lr_from_optimizer_id(oid)
    safe = _safe_neuron_key(neuron_id)
    parsed = parse_unit_node_id(str(neuron_id))
    if parsed is None:
        return np.array([float("nan")])
    unit_idx = int(parsed["unit_index"])
    layer_name = str(parsed["layer_name"])
    n_in = n_incoming_weights(layer_name)

    if otype == "sgd":
        return np.full(n_in, lr, dtype=np.float64)

    if otype in ("adagrad", "grafted_shampoo"):
        key = f"effective_lr__{safe}"
        if key not in sigs:
            return np.full(n_in, np.nan)
        arr = sigs[key]
        if cp_row_idx >= len(arr):
            return np.full(n_in, np.nan)
        val = arr[cp_row_idx]
        out = np.asarray(val, dtype=np.float64)
        return out if out.ndim else np.array([float(out)])

    if otype == "adam":
        key = f"exp_avg_sq__{safe}"
        if key not in sigs:
            return np.full(n_in, np.nan)
        arr = sigs[key]
        if cp_row_idx >= len(arr):
            return np.full(n_in, np.nan)
        val = np.asarray(arr[cp_row_idx], dtype=np.float64)
        v = np.sqrt(val)
        return lr / (v + DEFAULT_ADAM_EPS)

    if otype == "pure_shampoo":
        l_key, r_key = _h_inv_block_keys(layer_name)
        if l_key not in sigs or r_key not in sigs:
            raise FileNotFoundError(
                f"Pure Shampoo missing block matrices for layer {layer_name!r}: "
                f"expected {l_key!r} and {r_key!r} in signals.npz"
            )
        l_arr, r_arr = sigs[l_key], sigs[r_key]
        if cp_row_idx >= len(l_arr) or cp_row_idx >= len(r_arr):
            raise IndexError(
                f"Pure Shampoo checkpoint row {cp_row_idx} out of range for "
                f"{l_key} (len={len(l_arr)}) / {r_key} (len={len(r_arr)})"
            )
        L = np.asarray(l_arr[cp_row_idx], dtype=np.float64)
        R = np.asarray(r_arr[cp_row_idx], dtype=np.float64)
        row_norm = float(np.linalg.norm(L[unit_idx, :]))
        col_norms = np.linalg.norm(R, axis=0)
        return row_norm * col_norms

    return np.full(n_in, np.nan)


def neuron_ids_in_layer(sigs: dict, layer_name: str) -> list[str]:
    out: list[str] = []
    for nid in sigs.get("unit_node_ids", []):
        parsed = parse_unit_node_id(str(nid))
        if parsed is None:
            continue
        if str(parsed["layer_name"]) == str(layer_name):
            out.append(str(nid))
    return sorted(out)


def all_hidden_neuron_ids(sigs: dict) -> list[tuple[str, str]]:
    """Return (neuron_id, layer_name) for all hidden-layer units."""
    out: list[tuple[str, str]] = []
    for layer_name in DEFAULT_LAYER_NAMES:
        for nid in neuron_ids_in_layer(sigs, layer_name):
            out.append((nid, layer_name))
    return out


def _cp_index_for_iter(metrics: dict, target_iter: int) -> int | None:
    cp_iters = metrics.get("checkpoint_iterations")
    if cp_iters is None:
        return None
    arr = np.asarray(cp_iters, dtype=np.int64)
    matches = np.where(arr == int(target_iter))[0]
    if len(matches):
        return int(matches[0])
    ge = np.where(arr >= int(target_iter))[0]
    return int(ge[0]) if len(ge) else None


def unit_mean_efflr_at_t(
    sigs: dict,
    metrics: dict,
    oid: str,
    neuron_id: str,
    trial_start_cp: int,
    t: int,
) -> float:
    cp_idx = int(trial_start_cp) + int(t)
    cp_row = _signal_row_for_checkpoint(sigs, metrics, cp_idx)
    weights = gamma_eff_weights_at_cp(sigs, oid, neuron_id, cp_row)
    return float(np.nanmean(weights))


def unit_weight_std_at_t(
    sigs: dict,
    metrics: dict,
    oid: str,
    neuron_id: str,
    trial_start_cp: int,
    t: int,
) -> float:
    cp_idx = int(trial_start_cp) + int(t)
    cp_row = _signal_row_for_checkpoint(sigs, metrics, cp_idx)
    weights = gamma_eff_weights_at_cp(sigs, oid, neuron_id, cp_row)
    return float(np.nanstd(weights))


def layer_mean_efflr_at_t(
    sigs: dict,
    metrics: dict,
    oid: str,
    layer_name: str,
    trial_start_cp: int,
    t: int,
) -> float:
    vals: list[float] = []
    for nid in neuron_ids_in_layer(sigs, layer_name):
        w = gamma_eff_weights_at_cp(
            sigs,
            oid,
            nid,
            _signal_row_for_checkpoint(sigs, metrics, int(trial_start_cp) + int(t)),
        )
        vals.extend(w[np.isfinite(w)].tolist())
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def trial_weight_trajectories(
    sigs: dict,
    metrics: dict,
    row: pd.Series,
    t_end: int,
    *,
    neuron_id: str | None = None,
    trial_start_cp: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tsc = int(trial_start_cp if trial_start_cp is not None else row.trial_start_cp)
    nid = str(neuron_id if neuron_id is not None else row.neuron_id)
    x = np.arange(int(t_end) + 1, dtype=np.int64)
    rows: list[np.ndarray] = []
    for t in x:
        cp_idx = tsc + int(t)
        cp_row = _signal_row_for_checkpoint(sigs, metrics, cp_idx)
        rows.append(gamma_eff_weights_at_cp(sigs, str(row.optimizer_id), nid, cp_row))
    Y = np.stack(rows, axis=0) if rows else np.empty((0, 0))
    return x, Y


# Core per-weight |Δw| helpers (Section C)


def _load_run_nts_metrics_fresh(
    eid: str,
    mid: str,
    oid: str,
    rid: str,
) -> tuple[dict, dict]:
    """Load neuron_timeseries + training_metrics from disk (no RAM cache)."""
    nts = _nts(eid, mid, oid, rid=rid, w_cache=False)
    metrics = _metrics(eid, mid, oid, rid=rid, w_cache=False)
    return nts, metrics


def delta_w_weights_at_cp(
    nts: dict,
    neuron_id: str,
    cp_idx: int,
) -> np.ndarray:
    """Per-synapse |Δw| at checkpoint `cp_idx` (consecutive checkpoint difference)."""
    parsed = parse_unit_node_id(str(neuron_id))
    n_in = n_incoming_weights(parsed["layer_name"]) if parsed else HIDDEN_SIZE
    key = _unit_key("weights", neuron_id)
    if key not in nts:
        return np.full(n_in, np.nan)
    w_arr = np.asarray(nts[key], dtype=np.float64)
    if cp_idx <= 0 or cp_idx >= len(w_arr):
        return np.full(n_in, np.nan)
    delta = w_arr[cp_idx] - w_arr[cp_idx - 1]
    return np.abs(delta)


def unit_mean_dw_at_t(
    nts: dict,
    neuron_id: str,
    trial_start_cp: int,
    t: int,
) -> float:
    cp_idx = int(trial_start_cp) + int(t)
    weights = delta_w_weights_at_cp(nts, neuron_id, cp_idx)
    return float(np.nanmean(weights))


def unit_weight_std_dw_at_t(
    nts: dict,
    neuron_id: str,
    trial_start_cp: int,
    t: int,
) -> float:
    cp_idx = int(trial_start_cp) + int(t)
    weights = delta_w_weights_at_cp(nts, neuron_id, cp_idx)
    return float(np.nanstd(weights))


def layer_mean_dw_at_t(
    nts: dict,
    layer_name: str,
    trial_start_cp: int,
    t: int,
) -> float:
    cp_idx = int(trial_start_cp) + int(t)
    vals: list[float] = []
    for nid in neuron_ids_in_layer(nts, layer_name):
        w = delta_w_weights_at_cp(nts, nid, cp_idx)
        vals.extend(w[np.isfinite(w)].tolist())
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def trial_delta_w_trajectories(
    nts: dict,
    row: pd.Series,
    t_end: int,
    *,
    neuron_id: str | None = None,
    trial_start_cp: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tsc = int(trial_start_cp if trial_start_cp is not None else row.trial_start_cp)
    nid = str(neuron_id if neuron_id is not None else row.neuron_id)
    x = np.arange(int(t_end) + 1, dtype=np.int64)
    rows: list[np.ndarray] = []
    for t in x:
        cp_idx = tsc + int(t)
        rows.append(delta_w_weights_at_cp(nts, nid, cp_idx))
    Y = np.stack(rows, axis=0) if rows else np.empty((0, 0))
    return x, Y


def _unit_window_means_dw(
    nts: dict,
    neuron_id: str,
    trial_start_cp: int,
    *,
    n_start: int,
    window: int,
) -> tuple[float, float, float]:
    unit_means: list[float] = []
    for t in range(window):
        unit_means.append(unit_mean_dw_at_t(nts, neuron_id, trial_start_cp, t))
    arr = np.asarray(unit_means, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return float("nan"), float("nan"), float("nan")
    mean_first = float(np.nanmean(arr[:n_start]))
    mean_last = float(np.nanmean(arr[n_start:]))
    return mean_first, mean_last, mean_first - mean_last


def _layer_window_means_dw(
    nts: dict,
    layer_name: str,
    trial_start_cp: int,
    *,
    n_start: int,
    window: int,
) -> tuple[float, float, float]:
    layer_means: list[float] = []
    for t in range(window):
        layer_means.append(layer_mean_dw_at_t(nts, layer_name, trial_start_cp, t))
    arr = np.asarray(layer_means, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return float("nan"), float("nan"), float("nan")
    mean_first = float(np.nanmean(arr[:n_start]))
    mean_last = float(np.nanmean(arr[n_start:]))
    return mean_first, mean_last, mean_first - mean_last


def _unit_window_means(
    sigs: dict,
    metrics: dict,
    oid: str,
    neuron_id: str,
    trial_start_cp: int,
    *,
    n_start: int,
    window: int,
) -> tuple[float, float, float]:
    unit_means: list[float] = []
    for t in range(window):
        unit_means.append(
            unit_mean_efflr_at_t(sigs, metrics, oid, neuron_id, trial_start_cp, t)
        )
    arr = np.asarray(unit_means, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return float("nan"), float("nan"), float("nan")
    mean_first = float(np.nanmean(arr[:n_start]))
    mean_last = float(np.nanmean(arr[n_start:]))
    return mean_first, mean_last, mean_first - mean_last


def _layer_window_means(
    sigs: dict,
    metrics: dict,
    oid: str,
    layer_name: str,
    trial_start_cp: int,
    *,
    n_start: int,
    window: int,
) -> tuple[float, float, float]:
    layer_means: list[float] = []
    for t in range(window):
        layer_means.append(
            layer_mean_efflr_at_t(sigs, metrics, oid, layer_name, trial_start_cp, t)
        )
    arr = np.asarray(layer_means, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return float("nan"), float("nan"), float("nan")
    mean_first = float(np.nanmean(arr[:n_start]))
    mean_last = float(np.nanmean(arr[n_start:]))
    return mean_first, mean_last, mean_first - mean_last


def _relative_increase_pct(value: float, baseline_mean: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline_mean) or abs(baseline_mean) <= 1e-15:
        return float("nan")
    return float((value / baseline_mean - 1.0) * 100.0)


# Time-segment enumeration


def iter_run_time_segments(
    eid: str,
    experiment: str,
    metrics: dict,
    *,
    window: int = WINDOW_SIZE,
    labels: ExperimentDisplayLabels | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield (time_idx, trial_start_iter) for cat1 windows or cat2 trials."""
    cp_iters = metrics.get("checkpoint_iterations")
    if cp_iters is None or len(cp_iters) == 0:
        return
    max_iter = int(np.max(cp_iters))

    if is_pretrain_shuffle_mislabel_experiment(str(eid)):
        n_windows = max_iter // window
        for w in range(n_windows):
            yield w, w * window
        return

    spec = _btsp_segment_spec_for_experiment(experiment, eid, labels) if labels else None
    if spec is not None:
        skip = int(spec["skip_pretrain_iters"])
        n_trials_per_seg = int(spec["segment_iters"]) // window
        n_segments = int(spec["n_segments"])
        time_idx = 0
        for seg in range(n_segments):
            seg_start = skip + seg * int(spec["segment_iters"])
            for t_in_seg in range(n_trials_per_seg):
                trial_start = seg_start + t_in_seg * window
                if trial_start + window <= max_iter + 1:
                    yield time_idx, trial_start
                time_idx += 1
        return

    n_trials = max_iter // window
    for t in range(n_trials):
        yield t, t * window


def _category_runs(
    all_results: dict[tuple[str, str, str, str], dict],
    rename_fn: Callable[[str], str],
) -> list[tuple[str, str, str, str, str]]:
    runs: list[tuple[str, str, str, str, str]] = []
    for (eid, mid, oid, rid), payload in all_results.items():
        if not _is_category_experiment(eid):
            continue
        if payload.get("error"):
            continue
        if str(oid).lower().startswith("sgd"):
            continue
        runs.append((str(eid), str(mid), str(oid), str(rid), rename_fn(str(eid))))
    return runs



def _run_key_from_parts(eid: str, mid: str, oid: str, rid: str) -> tuple[str, str, str, str]:
    return str(eid), str(mid), str(oid), str(rid)


def _index_periods_by_run(
    df_cat_periods: pd.DataFrame,
) -> dict[tuple[str, str, str, str], pd.DataFrame]:
    if df_cat_periods.empty:
        return {}
    group_cols = ["experiment_id", "model_id", "optimizer_id", "run_id"]
    out: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    for run_key, group in df_cat_periods.groupby(group_cols, sort=False):
        key = (str(run_key[0]), str(run_key[1]), str(run_key[2]), str(run_key[3]))
        out[key] = group
    return out


def _compute_window_rows_for_run(
    sigs: dict,
    metrics: dict,
    *,
    eid: str,
    mid: str,
    oid: str,
    rid: str,
    experiment: str,
    n_start: int,
    window: int,
    labels: ExperimentDisplayLabels | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unit_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    units = all_hidden_neuron_ids(sigs)
    n_cp = len(metrics.get("checkpoint_iterations", []))

    for time_idx, trial_start_iter in iter_run_time_segments(
        eid, experiment, metrics, window=window, labels=labels
    ):
        trial_start_cp = _cp_index_for_iter(metrics, trial_start_iter)
        if trial_start_cp is None or trial_start_cp + window > n_cp:
            continue

        for neuron_id, layer_name in units:
            mf, ml, delta = _unit_window_means(
                sigs,
                metrics,
                oid,
                neuron_id,
                trial_start_cp,
                n_start=n_start,
                window=window,
            )
            if not np.isfinite(mf):
                continue
            unit_rows.append(
                {
                    "experiment_id": eid,
                    "experiment": experiment,
                    "model_id": mid,
                    "optimizer_id": oid,
                    "run_id": rid,
                    "time_idx": int(time_idx),
                    "trial_start_iter": int(trial_start_iter),
                    "neuron_id": neuron_id,
                    "layer_name": layer_name,
                    "mean_first": mf,
                    "mean_last": ml,
                    "delta": delta,
                }
            )

        for layer_name in DEFAULT_LAYER_NAMES:
            if not neuron_ids_in_layer(sigs, layer_name):
                continue
            mf, ml, delta = _layer_window_means(
                sigs,
                metrics,
                oid,
                layer_name,
                trial_start_cp,
                n_start=n_start,
                window=window,
            )
            if not np.isfinite(mf):
                continue
            layer_rows.append(
                {
                    "experiment_id": eid,
                    "experiment": experiment,
                    "model_id": mid,
                    "optimizer_id": oid,
                    "run_id": rid,
                    "time_idx": int(time_idx),
                    "trial_start_iter": int(trial_start_iter),
                    "layer_name": layer_name,
                    "mean_first": mf,
                    "mean_last": ml,
                    "delta": delta,
                }
            )

    return unit_rows, layer_rows


def _compute_btsp_rows_for_run(
    sigs: dict,
    metrics: dict,
    periods: pd.DataFrame,
    *,
    eid: str,
    mid: str,
    oid: str,
    rid: str,
    n_surround: int,
) -> list[dict[str, Any]]:
    if str(oid).lower().startswith("sgd"):
        return []

    n = int(n_surround)
    rows: list[dict[str, Any]] = []

    for _, row in periods.iterrows():
        t_start = int(row.point_of_max_acceleration)
        t_assign = int(row.assignment_within_trial)
        trial_start_cp = int(row.trial_start_cp)
        nid = str(row.neuron_id)
        layer_name = _layer_name_from_row(row)
        if layer_name is None:
            continue

        need_lo = t_start - n
        need_hi = t_start + n
        if need_lo < 0 or need_hi >= N_SAMPLES_PER_TRIAL:
            continue
        if t_assign < 0 or t_assign >= N_SAMPLES_PER_TRIAL:
            continue

        u_trace = _unit_mean_trace(
            sigs, metrics, str(oid), nid, trial_start_cp, t_assign
        )
        l_trace = _layer_mean_trace(
            sigs, metrics, str(oid), layer_name, trial_start_cp, t_assign
        )

        if not np.any(np.isfinite(u_trace[: t_assign + 1])):
            continue

        peak_u = int(np.nanargmax(u_trace[: t_assign + 1]))
        peak_l = int(np.nanargmax(l_trace[: t_assign + 1]))

        u_at_start = float(u_trace[t_start])
        l_at_start = float(l_trace[t_start])
        pre_u_base = float(np.nanmean(u_trace[t_start - n : t_start]))
        post_u_base = float(np.nanmean(u_trace[t_start + 1 : t_start + n + 1]))
        pre_l_base = float(np.nanmean(l_trace[t_start - n : t_start]))
        post_l_base = float(np.nanmean(l_trace[t_start + 1 : t_start + n + 1]))

        std_tstart = unit_weight_std_at_t(
            sigs, metrics, str(oid), nid, trial_start_cp, t_start
        )
        pre_stds = [
            unit_weight_std_at_t(
                sigs, metrics, str(oid), nid, trial_start_cp, t_start - k
            )
            for k in range(n, 0, -1)
        ]
        post_stds = [
            unit_weight_std_at_t(
                sigs, metrics, str(oid), nid, trial_start_cp, t_start + k
            )
            for k in range(1, n + 1)
        ]

        rows.append(
            {
                "experiment_id": eid,
                "experiment": row.experiment,
                "model_id": mid,
                "optimizer_id": oid,
                "run_id": rid,
                "neuron_id": nid,
                "period_id": int(row.period_id),
                "layer_name": layer_name,
                "point_highest_unit_efflr": peak_u,
                "point_highest_layer_efflr": peak_l,
                "pre_unit_relative_increase_pct": _relative_increase_pct(
                    u_at_start, pre_u_base
                ),
                "post_unit_relative_increase_pct": _relative_increase_pct(
                    u_at_start, post_u_base
                ),
                "pre_layer_relative_increase_pct": _relative_increase_pct(
                    l_at_start, pre_l_base
                ),
                "post_layer_relative_increase_pct": _relative_increase_pct(
                    l_at_start, post_l_base
                ),
                "std_weight_efflr_tstart": std_tstart,
                "mean_pre_std_weight_efflr": float(np.nanmean(pre_stds)),
                "mean_post_std_weight_efflr": float(np.nanmean(post_stds)),
            }
        )

    return rows


def compute_all_efflr_aggregates(
    all_results: dict[tuple[str, str, str, str], dict],
    df_cat_periods: pd.DataFrame,
    rename_fn: Callable[[str], str],
    *,
    n_start: int = N_START_ITS,
    window: int = WINDOW_SIZE,
    n_surround: int = N_SURROUNDING_PERIODSTART_ITS,
    labels: ExperimentDisplayLabels | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One disk load per run: unit/layer window stats + BTSP period stats, then unload."""
    unit_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    btsp_rows: list[dict[str, Any]] = []
    periods_by_run = _index_periods_by_run(df_cat_periods)
    runs = _category_runs(all_results, rename_fn)

    for eid, mid, oid, rid, experiment in tqdm(runs, desc="efflr aggregates"):
        try:
            sigs, metrics = _load_run_sigs_metrics_fresh(eid, mid, oid, rid)
        except Exception:
            continue

        u_rows, l_rows = _compute_window_rows_for_run(
            sigs,
            metrics,
            eid=eid,
            mid=mid,
            oid=oid,
            rid=rid,
            experiment=experiment,
            n_start=n_start,
            window=window,
            labels=labels,
        )
        unit_rows.extend(u_rows)
        layer_rows.extend(l_rows)

        run_key = _run_key_from_parts(eid, mid, oid, rid)
        periods = periods_by_run.get(run_key)
        if periods is not None and not periods.empty:
            btsp_rows.extend(
                _compute_btsp_rows_for_run(
                    sigs,
                    metrics,
                    periods,
                    eid=eid,
                    mid=mid,
                    oid=oid,
                    rid=rid,
                    n_surround=n_surround,
                )
            )

        _unload_run_sigs_metrics(sigs, metrics)

    return pd.DataFrame(unit_rows), pd.DataFrame(layer_rows), pd.DataFrame(btsp_rows)


def _compute_window_rows_for_run_dw(
    nts: dict,
    metrics: dict,
    *,
    eid: str,
    mid: str,
    oid: str,
    rid: str,
    experiment: str,
    n_start: int,
    window: int,
    labels: ExperimentDisplayLabels | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unit_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    units = all_hidden_neuron_ids(nts)
    n_cp = len(metrics.get("checkpoint_iterations", []))

    for time_idx, trial_start_iter in iter_run_time_segments(
        eid, experiment, metrics, window=window, labels=labels
    ):
        trial_start_cp = _cp_index_for_iter(metrics, trial_start_iter)
        if trial_start_cp is None or trial_start_cp + window > n_cp:
            continue

        for neuron_id, layer_name in units:
            mf, ml, delta = _unit_window_means_dw(
                nts,
                neuron_id,
                trial_start_cp,
                n_start=n_start,
                window=window,
            )
            if not np.isfinite(mf):
                continue
            unit_rows.append(
                {
                    "experiment_id": eid,
                    "experiment": experiment,
                    "model_id": mid,
                    "optimizer_id": oid,
                    "run_id": rid,
                    "time_idx": int(time_idx),
                    "trial_start_iter": int(trial_start_iter),
                    "neuron_id": neuron_id,
                    "layer_name": layer_name,
                    "mean_first": mf,
                    "mean_last": ml,
                    "delta": delta,
                }
            )

        for layer_name in DEFAULT_LAYER_NAMES:
            if not neuron_ids_in_layer(nts, layer_name):
                continue
            mf, ml, delta = _layer_window_means_dw(
                nts,
                layer_name,
                trial_start_cp,
                n_start=n_start,
                window=window,
            )
            if not np.isfinite(mf):
                continue
            layer_rows.append(
                {
                    "experiment_id": eid,
                    "experiment": experiment,
                    "model_id": mid,
                    "optimizer_id": oid,
                    "run_id": rid,
                    "time_idx": int(time_idx),
                    "trial_start_iter": int(trial_start_iter),
                    "layer_name": layer_name,
                    "mean_first": mf,
                    "mean_last": ml,
                    "delta": delta,
                }
            )

    return unit_rows, layer_rows


def _unit_mean_trace_dw(
    nts: dict,
    neuron_id: str,
    trial_start_cp: int,
    t_end: int,
) -> np.ndarray:
    out = np.empty(int(t_end) + 1, dtype=np.float64)
    for t in range(int(t_end) + 1):
        out[t] = unit_mean_dw_at_t(nts, neuron_id, trial_start_cp, t)
    return out


def _layer_mean_trace_dw(
    nts: dict,
    layer_name: str,
    trial_start_cp: int,
    t_end: int,
) -> np.ndarray:
    out = np.empty(int(t_end) + 1, dtype=np.float64)
    for t in range(int(t_end) + 1):
        out[t] = layer_mean_dw_at_t(nts, layer_name, trial_start_cp, t)
    return out


def _compute_btsp_rows_for_run_dw(
    nts: dict,
    periods: pd.DataFrame,
    *,
    eid: str,
    mid: str,
    oid: str,
    rid: str,
    n_surround: int,
) -> list[dict[str, Any]]:
    if str(oid).lower().startswith("sgd"):
        return []

    n = int(n_surround)
    rows: list[dict[str, Any]] = []

    for _, row in periods.iterrows():
        t_start = int(row.point_of_max_acceleration)
        t_assign = int(row.assignment_within_trial)
        trial_start_cp = int(row.trial_start_cp)
        nid = str(row.neuron_id)
        layer_name = _layer_name_from_row(row)
        if layer_name is None:
            continue

        need_lo = t_start - n
        need_hi = t_start + n
        if need_lo < 0 or need_hi >= N_SAMPLES_PER_TRIAL:
            continue
        if t_assign < 0 or t_assign >= N_SAMPLES_PER_TRIAL:
            continue

        u_trace = _unit_mean_trace_dw(nts, nid, trial_start_cp, t_assign)
        l_trace = _layer_mean_trace_dw(nts, layer_name, trial_start_cp, t_assign)

        if not np.any(np.isfinite(u_trace[: t_assign + 1])):
            continue

        peak_u = int(np.nanargmax(u_trace[: t_assign + 1]))
        peak_l = int(np.nanargmax(l_trace[: t_assign + 1]))

        u_at_start = float(u_trace[t_start])
        l_at_start = float(l_trace[t_start])
        pre_u_base = float(np.nanmean(u_trace[t_start - n : t_start]))
        post_u_base = float(np.nanmean(u_trace[t_start + 1 : t_start + n + 1]))
        pre_l_base = float(np.nanmean(l_trace[t_start - n : t_start]))
        post_l_base = float(np.nanmean(l_trace[t_start + 1 : t_start + n + 1]))

        std_tstart = unit_weight_std_dw_at_t(nts, nid, trial_start_cp, t_start)
        pre_stds = [
            unit_weight_std_dw_at_t(nts, nid, trial_start_cp, t_start - k)
            for k in range(n, 0, -1)
        ]
        post_stds = [
            unit_weight_std_dw_at_t(nts, nid, trial_start_cp, t_start + k)
            for k in range(1, n + 1)
        ]

        rows.append(
            {
                "experiment_id": eid,
                "experiment": row.experiment,
                "model_id": mid,
                "optimizer_id": oid,
                "run_id": rid,
                "neuron_id": nid,
                "period_id": int(row.period_id),
                "layer_name": layer_name,
                "point_highest_unit_efflr": peak_u,
                "point_highest_layer_efflr": peak_l,
                "pre_unit_relative_increase_pct": _relative_increase_pct(
                    u_at_start, pre_u_base
                ),
                "post_unit_relative_increase_pct": _relative_increase_pct(
                    u_at_start, post_u_base
                ),
                "pre_layer_relative_increase_pct": _relative_increase_pct(
                    l_at_start, pre_l_base
                ),
                "post_layer_relative_increase_pct": _relative_increase_pct(
                    l_at_start, post_l_base
                ),
                "std_weight_efflr_tstart": std_tstart,
                "mean_pre_std_weight_efflr": float(np.nanmean(pre_stds)),
                "mean_post_std_weight_efflr": float(np.nanmean(post_stds)),
            }
        )

    return rows


def compute_all_dw_aggregates(
    all_results: dict[tuple[str, str, str, str], dict],
    df_cat_periods: pd.DataFrame,
    rename_fn: Callable[[str], str],
    *,
    n_start: int = N_START_ITS,
    window: int = WINDOW_SIZE,
    n_surround: int = N_SURROUNDING_PERIODSTART_ITS,
    labels: ExperimentDisplayLabels | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One disk load per run: unit/layer |Δw| window stats + BTSP period stats, then unload."""
    unit_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    btsp_rows: list[dict[str, Any]] = []
    periods_by_run = _index_periods_by_run(df_cat_periods)
    runs = _category_runs(all_results, rename_fn)

    for eid, mid, oid, rid, experiment in tqdm(runs, desc="|Δw| aggregates"):
        try:
            nts, metrics = _load_run_nts_metrics_fresh(eid, mid, oid, rid)
        except Exception:
            continue

        u_rows, l_rows = _compute_window_rows_for_run_dw(
            nts,
            metrics,
            eid=eid,
            mid=mid,
            oid=oid,
            rid=rid,
            experiment=experiment,
            n_start=n_start,
            window=window,
            labels=labels,
        )
        unit_rows.extend(u_rows)
        layer_rows.extend(l_rows)

        run_key = _run_key_from_parts(eid, mid, oid, rid)
        periods = periods_by_run.get(run_key)
        if periods is not None and not periods.empty:
            btsp_rows.extend(
                _compute_btsp_rows_for_run_dw(
                    nts,
                    periods,
                    eid=eid,
                    mid=mid,
                    oid=oid,
                    rid=rid,
                    n_surround=n_surround,
                )
            )

        _unload_run_sigs_metrics(nts, metrics)

    return pd.DataFrame(unit_rows), pd.DataFrame(layer_rows), pd.DataFrame(btsp_rows)


def compute_unit_window_stats(
    all_results: dict[tuple[str, str, str, str], dict],
    rename_fn: Callable[[str], str],
    *,
    n_start: int = N_START_ITS,
    window: int = WINDOW_SIZE,
    labels: ExperimentDisplayLabels | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    runs = _category_runs(all_results, rename_fn)
    for eid, mid, oid, rid, experiment in tqdm(runs, desc="unit window stats"):
        try:
            sigs, metrics = _load_run_sigs_metrics_fresh(eid, mid, oid, rid)
        except Exception:
            continue
        u_rows, _ = _compute_window_rows_for_run(
            sigs,
            metrics,
            eid=eid,
            mid=mid,
            oid=oid,
            rid=rid,
            experiment=experiment,
            n_start=n_start,
            window=window,
            labels=labels,
        )
        rows.extend(u_rows)
        _unload_run_sigs_metrics(sigs, metrics)
    return pd.DataFrame(rows)


def compute_layer_window_stats(
    all_results: dict[tuple[str, str, str, str], dict],
    rename_fn: Callable[[str], str],
    *,
    n_start: int = N_START_ITS,
    window: int = WINDOW_SIZE,
    labels: ExperimentDisplayLabels | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    runs = _category_runs(all_results, rename_fn)
    for eid, mid, oid, rid, experiment in tqdm(runs, desc="layer window stats"):
        try:
            sigs, metrics = _load_run_sigs_metrics_fresh(eid, mid, oid, rid)
        except Exception:
            continue
        _, l_rows = _compute_window_rows_for_run(
            sigs,
            metrics,
            eid=eid,
            mid=mid,
            oid=oid,
            rid=rid,
            experiment=experiment,
            n_start=n_start,
            window=window,
            labels=labels,
        )
        rows.extend(l_rows)
        _unload_run_sigs_metrics(sigs, metrics)
    return pd.DataFrame(rows)


def _unit_mean_trace(
    sigs: dict,
    metrics: dict,
    oid: str,
    neuron_id: str,
    trial_start_cp: int,
    t_end: int,
) -> np.ndarray:
    out = np.empty(int(t_end) + 1, dtype=np.float64)
    for t in range(int(t_end) + 1):
        out[t] = unit_mean_efflr_at_t(
            sigs, metrics, oid, neuron_id, trial_start_cp, t
        )
    return out


def _layer_mean_trace(
    sigs: dict,
    metrics: dict,
    oid: str,
    layer_name: str,
    trial_start_cp: int,
    t_end: int,
) -> np.ndarray:
    out = np.empty(int(t_end) + 1, dtype=np.float64)
    for t in range(int(t_end) + 1):
        out[t] = layer_mean_efflr_at_t(
            sigs, metrics, oid, layer_name, trial_start_cp, t
        )
    return out


def _layer_name_from_row(row: pd.Series) -> str | None:
    if pd.notna(row.get("layer_name")):
        return str(row["layer_name"])
    parsed = parse_unit_node_id(str(row["neuron_id"]))
    if parsed is None:
        return None
    return str(parsed["layer_name"])


def compute_btsp_period_stats(
    df_cat_periods: pd.DataFrame,
    load_run_fn: Callable[[pd.Series], tuple[dict, dict]] | None = None,
    *,
    n_surround: int = N_SURROUNDING_PERIODSTART_ITS,
) -> pd.DataFrame:
    if df_cat_periods.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["experiment_id", "model_id", "optimizer_id", "run_id"]
    use_fresh = load_run_fn is None

    for run_key, group in tqdm(
        df_cat_periods.groupby(group_cols, sort=False),
        desc="BTSP period efflr",
    ):
        eid, mid, oid, rid = (str(x) for x in run_key)
        if str(oid).lower().startswith("sgd"):
            continue
        try:
            if use_fresh:
                sigs, metrics = _load_run_sigs_metrics_fresh(eid, mid, oid, rid)
            else:
                sigs, metrics = load_run_fn(group.iloc[0])  # type: ignore[misc]
        except Exception:
            continue

        rows.extend(
            _compute_btsp_rows_for_run(
                sigs,
                metrics,
                group,
                eid=eid,
                mid=mid,
                oid=oid,
                rid=rid,
                n_surround=n_surround,
            )
        )
        if use_fresh:
            _unload_run_sigs_metrics(sigs, metrics)

    return pd.DataFrame(rows)


# Plot helpers


def _quantity_text(quantity: AggregateQuantity, *, efflr: str, dw: str) -> str:
    return dw if quantity == "dw" else efflr


def _opt_order_without_sgd(opt_ids: list[str]) -> list[str]:
    return [oid for oid in _sort_optimizer_ids(opt_ids) if not str(oid).lower().startswith("sgd")]


def _xtick_label(exp: str) -> str:
    return "\n".join(str(exp).split(" "))


def _pre_post_panel_x_centers(n: int) -> np.ndarray:
    return np.arange(n, dtype=float) * (2 * _PREPOST_BAR_W + _PREPOST_GROUP_GAP)


def _pair_bracket_label(sig_pre: bool, sig_post: bool) -> str:
    parts: list[str] = []
    if sig_pre:
        parts.append(r"$*_{\mathrm{pre}}$")
    if sig_post:
        parts.append(r"$*_{\mathrm{post}}$")
    return " ".join(parts)


def _draw_pre_post_brackets(
    ax,
    x_centers: np.ndarray,
    sig_pre: np.ndarray,
    sig_post: np.ndarray,
    *,
    y_base: float,
    y_step: float,
) -> None:
    k = len(x_centers)
    stack = 0
    pitch = max(y_step * 1.15, y_step)
    riser = y_step * 0.25
    for i in range(k):
        for j in range(i + 1, k):
            pre_ij = bool(sig_pre[i, j])
            post_ij = bool(sig_post[i, j])
            if not pre_ij and not post_ij:
                continue
            y = y_base + stack * pitch
            stack += 1
            x1, x2 = float(x_centers[i]), float(x_centers[j])
            y_riser = y + riser
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
                va="bottom",
                fontsize=8,
                color="0.1",
                clip_on=False,
            )


def _plot_pre_post_optimizer_panel(
    ax,
    opt_order: list[str],
    pre_samples: dict[str, np.ndarray],
    post_samples: dict[str, np.ndarray],
    *,
    ylim: tuple[float, float] | None = None,
    show_xticklabels: bool = True,
    show_ylabel: bool = True,
) -> None:
    n = len(opt_order)
    if n == 0:
        ax.set_visible(False)
        return

    x_centers = _pre_post_panel_x_centers(n)
    x_pre = x_centers - _PREPOST_BAR_W / 2.0
    x_post = x_centers + _PREPOST_BAR_W / 2.0

    pre_means: list[float] = []
    post_means: list[float] = []
    pre_sems: list[float] = []
    post_sems: list[float] = []

    for oid in opt_order:
        p_arr = pre_samples.get(oid, np.array([], dtype=np.float64))
        q_arr = post_samples.get(oid, np.array([], dtype=np.float64))
        p_arr = p_arr[np.isfinite(p_arr)]
        q_arr = q_arr[np.isfinite(q_arr)]
        pre_means.append(float(np.mean(p_arr)) if p_arr.size else float("nan"))
        post_means.append(float(np.mean(q_arr)) if q_arr.size else float("nan"))
        pre_sems.append(_sem_stderr(p_arr))
        post_sems.append(_sem_stderr(q_arr))

    for i, oid in enumerate(opt_order):
        color = _opt_color(oid)
        if np.isfinite(pre_means[i]):
            ax.bar(
                x_pre[i],
                pre_means[i],
                width=_PREPOST_BAR_W,
                color=color,
                hatch="///",
                edgecolor="0.2",
                linewidth=0.5,
                zorder=2,
            )
            ax.errorbar(
                x_pre[i],
                pre_means[i],
                yerr=SEM_Z * pre_sems[i],
                fmt="none",
                ecolor="0.15",
                capsize=2,
                linewidth=0.9,
                zorder=3,
            )
        if np.isfinite(post_means[i]):
            ax.bar(
                x_post[i],
                post_means[i],
                width=_PREPOST_BAR_W,
                color=color,
                edgecolor="0.2",
                linewidth=0.5,
                zorder=2,
            )
            ax.errorbar(
                x_post[i],
                post_means[i],
                yerr=SEM_Z * post_sems[i],
                fmt="none",
                ecolor="0.15",
                capsize=2,
                linewidth=0.9,
                zorder=3,
            )

    sig_pre = _welch_pairwise_holm_sig_matrix(pre_samples, opt_order)
    sig_post = _welch_pairwise_holm_sig_matrix(post_samples, opt_order)

    if ylim is None:
        hi = 0.0
        lo = 0.0
        for m, s in zip(pre_means + post_means, pre_sems + post_sems, strict=True):
            if np.isfinite(m):
                err = SEM_Z * s if np.isfinite(s) else 0.0
                hi = max(hi, m + err)
                lo = min(lo, m - err)
        span = max(hi - lo, 1e-6)
        pad = 0.15 * span
        n_br = int(np.sum(np.triu(sig_pre, k=1))) + int(np.sum(np.triu(sig_post, k=1)))
        pad += n_br * 0.08 * span
        ylim = (lo - pad, hi + pad)

    ax.axhline(0.0, color="0.35", lw=0.8, zorder=1)
    ax.set_ylim(ylim)
    ax.set_xticks(x_centers)
    if show_xticklabels:
        ax.set_xticklabels([_opt_display(o) for o in opt_order], rotation=25, ha="right", fontsize=7)
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel(r"relative increase (\%)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)

    span = ylim[1] - ylim[0]
    y_base = ylim[1] - 0.18 * span
    y_step = 0.1 * span
    _draw_pre_post_brackets(ax, x_centers, sig_pre, sig_post, y_base=y_base, y_step=y_step)


# UNIT plots


def plot_unit_start_mean_boxplot(
    df: pd.DataFrame,
    experiment: str,
    *,
    opt_order: list[str] | None = None,
    quantity: AggregateQuantity = "efflr",
) -> plt.Figure | None:
    sub = df[df["experiment"] == experiment]
    if sub.empty:
        return None
    opts = opt_order or _opt_order_without_sgd(sub["optimizer_id"].unique().tolist())
    time_idxs = sorted(sub["time_idx"].unique())
    if not time_idxs or not opts:
        return None

    n_t = len(time_idxs)
    n_o = len(opts)
    width = 0.8 / max(n_o, 1)
    fig, ax = plt.subplots(figsize=(max(8, n_t * 1.2), 5.5))

    for oi, oid in enumerate(opts):
        positions: list[float] = []
        data: list[np.ndarray] = []
        for ti, tidx in enumerate(time_idxs):
            vals = sub[(sub["time_idx"] == tidx) & (sub["optimizer_id"] == oid)][
                "mean_first"
            ].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            positions.append(float(ti) + (oi - (n_o - 1) / 2.0) * width)
            data.append(vals)
        if data:
            ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.85,
                patch_artist=True,
                manage_ticks=False,
                boxprops={"facecolor": _opt_color(oid), "alpha": 0.55, "linewidth": 0.8},
                medianprops={"color": "0.15", "linewidth": 1.0},
                whiskerprops={"linewidth": 0.8},
                capprops={"linewidth": 0.8},
            )

    ax.set_xticks(range(n_t))
    ax.set_xticklabels([str(t) for t in time_idxs])
    ax.set_xlabel("time index (trial / 100-iter window)")
    ax.set_ylabel(
        _quantity_text(
            quantity,
            efflr=r"unit mean $\gamma_{\mathrm{eff}}$ (first $N$ iters)",
            dw=r"unit mean $|\Delta w|$ (first $N$ iters)",
        )
    )
    ax.set_title(
        f"{experiment} — unit start-window "
        + _quantity_text(quantity, efflr="eff LR", dw=r"$|\Delta w|$")
    )
    handles = [
        plt.Line2D([0], [0], color=_opt_color(o), lw=6, alpha=0.55, label=_opt_display(o))
        for o in opts
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_unit_start_mean_boxplot_pooled(
    df: pd.DataFrame,
    pool_experiments: list[str],
    *,
    opt_order: list[str] | None = None,
    quantity: AggregateQuantity = "efflr",
) -> plt.Figure | None:
    sub = df[df["experiment"].isin(pool_experiments)].copy()
    if sub.empty:
        return None
    opts = opt_order or _opt_order_without_sgd(sub["optimizer_id"].unique().tolist())
    time_idxs = sorted(sub["time_idx"].unique())
    if not time_idxs or not opts:
        return None

    n_t = len(time_idxs)
    n_o = len(opts)
    width = 0.8 / max(n_o, 1)
    fig, ax = plt.subplots(figsize=(max(8, n_t * 1.2), 5.5))

    for oi, oid in enumerate(opts):
        positions: list[float] = []
        data: list[np.ndarray] = []
        for ti, tidx in enumerate(time_idxs):
            vals = sub[(sub["time_idx"] == tidx) & (sub["optimizer_id"] == oid)][
                "mean_first"
            ].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            positions.append(float(ti) + (oi - (n_o - 1) / 2.0) * width)
            data.append(vals)
        if data:
            ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.85,
                patch_artist=True,
                manage_ticks=False,
                boxprops={"facecolor": _opt_color(oid), "alpha": 0.55, "linewidth": 0.8},
                medianprops={"color": "0.15", "linewidth": 1.0},
            )

    ax.set_xticks(range(n_t))
    ax.set_xticklabels([str(t) for t in time_idxs])
    ax.set_xlabel("time index (aligned trial / window)")
    ax.set_ylabel(
        _quantity_text(
            quantity,
            efflr=r"unit mean $\gamma_{\mathrm{eff}}$ (first $N$ iters)",
            dw=r"unit mean $|\Delta w|$ (first $N$ iters)",
        )
    )
    pool_label = ", ".join(pool_experiments)
    ax.set_title(
        f"Pooled Cat2 ({pool_label}) — unit start-window "
        + _quantity_text(quantity, efflr="eff LR", dw=r"$|\Delta w|$")
    )
    handles = [
        plt.Line2D([0], [0], color=_opt_color(o), lw=6, alpha=0.55, label=_opt_display(o))
        for o in opts
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_unit_delta_bars(
    df: pd.DataFrame,
    experiment: str,
    *,
    opt_order: list[str] | None = None,
    quantity: AggregateQuantity = "efflr",
) -> plt.Figure | None:
    sub = df[df["experiment"] == experiment]
    if sub.empty:
        return None
    opts = opt_order or _opt_order_without_sgd(sub["optimizer_id"].unique().tolist())
    time_idxs = sorted(sub["time_idx"].unique())
    if not time_idxs or not opts:
        return None

    agg = (
        sub.groupby(["time_idx", "optimizer_id"], observed=False)["delta"]
        .agg(mean="mean", sem=_sem_across_values)
        .reset_index()
    )

    n_t = len(time_idxs)
    n_o = len(opts)
    width = 0.8 / max(n_o, 1)
    fig, ax = plt.subplots(figsize=(max(8, n_t * 1.2), 5.5))
    x = np.arange(n_t)

    for oi, oid in enumerate(opts):
        means = []
        yerr = []
        for tidx in time_idxs:
            row = agg[(agg["time_idx"] == tidx) & (agg["optimizer_id"] == oid)]
            if row.empty:
                means.append(float("nan"))
                yerr.append(0.0)
            else:
                means.append(float(row.iloc[0]["mean"]))
                yerr.append(SEM_Z * float(row.iloc[0]["sem"]))
        ax.bar(
            x + (oi - (n_o - 1) / 2.0) * width,
            means,
            width=width * 0.95,
            color=_opt_color(oid),
            label=_opt_display(oid),
            yerr=yerr,
            capsize=2,
            error_kw={"elinewidth": 0.9, "ecolor": "0.15"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in time_idxs])
    ax.set_xlabel("time index (trial / 100-iter window)")
    ax.set_ylabel(
        _quantity_text(
            quantity,
            efflr=r"mean unit $\Delta\gamma_{\mathrm{eff}}$ (first $N$ − last)",
            dw=r"mean unit $\Delta|\Delta w|$ (first $N$ − last)",
        )
    )
    ax.set_title(
        f"{experiment} — unit start vs end window "
        + _quantity_text(quantity, efflr="eff LR", dw=r"$|\Delta w|$")
    )
    ax.legend(fontsize=8)
    ax.axhline(0.0, color="0.35", lw=0.8)
    fig.tight_layout()
    return fig


# LAYER plots


def plot_layer_mean_bars(
    df: pd.DataFrame,
    experiment: str,
    *,
    opt_order: list[str] | None = None,
    layer_names: list[str] | None = None,
    quantity: AggregateQuantity = "efflr",
) -> plt.Figure | None:
    sub = df[df["experiment"] == experiment]
    if sub.empty:
        return None
    opts = opt_order or _opt_order_without_sgd(sub["optimizer_id"].unique().tolist())
    layers = layer_names or DEFAULT_LAYER_NAMES
    if not opts:
        return None

    agg = (
        sub.groupby(["optimizer_id", "layer_name"], observed=False)["mean_first"]
        .agg(mean="mean", sem=_sem_across_values)
        .reset_index()
    )

    n_o = len(opts)
    n_l = len(layers)
    width = 0.8 / max(n_l, 1)
    fig, ax = plt.subplots(figsize=(max(7, n_o * 1.5), 5.5))
    x = np.arange(n_o)

    for li, layer in enumerate(layers):
        means = []
        yerr = []
        for oid in opts:
            row = agg[(agg["optimizer_id"] == oid) & (agg["layer_name"] == layer)]
            if row.empty:
                means.append(float("nan"))
                yerr.append(0.0)
            else:
                means.append(float(row.iloc[0]["mean"]))
                yerr.append(SEM_Z * float(row.iloc[0]["sem"]))
        ax.bar(
            x + (li - (n_l - 1) / 2.0) * width,
            means,
            width=width * 0.95,
            label=layer,
            yerr=yerr,
            capsize=2,
            error_kw={"elinewidth": 0.9, "ecolor": "0.15"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_opt_display(o) for o in opts], rotation=20, ha="right")
    ax.set_ylabel(
        _quantity_text(
            quantity,
            efflr=r"layer mean $\gamma_{\mathrm{eff}}$ (first $N$ iters)",
            dw=r"layer mean $|\Delta w|$ (first $N$ iters)",
        )
    )
    ax.set_title(
        f"{experiment} — layer start-window "
        + _quantity_text(quantity, efflr="eff LR", dw=r"$|\Delta w|$")
    )
    ax.legend(title="layer", fontsize=8)
    fig.tight_layout()
    return fig


def plot_layer_time_traces(
    df: pd.DataFrame,
    experiment: str,
    optimizer_id: str,
    *,
    layer_names: list[str] | None = None,
    quantity: AggregateQuantity = "efflr",
) -> plt.Figure | None:
    sub = df[(df["experiment"] == experiment) & (df["optimizer_id"] == optimizer_id)]
    if sub.empty:
        return None
    layers = layer_names or DEFAULT_LAYER_NAMES
    time_idxs = sorted(sub["time_idx"].unique())
    if not time_idxs:
        return None

    agg = (
        sub.groupby(["time_idx", "layer_name"], observed=False)["mean_first"]
        .agg(mean="mean", sem=_sem_across_values)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(max(7, len(time_idxs) * 0.35), 5.0))
    x = np.asarray(time_idxs, dtype=float)

    for layer in layers:
        g = agg[agg["layer_name"] == layer].set_index("time_idx").reindex(time_idxs)
        mu = g["mean"].to_numpy(dtype=float)
        sem = g["sem"].to_numpy(dtype=float)
        valid = np.isfinite(mu)
        if not np.any(valid):
            continue
        ax.plot(x, mu, lw=1.4, label=layer)
        ax.fill_between(
            x,
            mu - SEM_Z * sem,
            mu + SEM_Z * sem,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("time index (trial / 100-iter window)")
    ax.set_ylabel(
        _quantity_text(
            quantity,
            efflr=r"layer mean $\gamma_{\mathrm{eff}}$ (first $N$ iters)",
            dw=r"layer mean $|\Delta w|$ (first $N$ iters)",
        )
    )
    ax.set_title(
        f"{experiment} — {_opt_display(optimizer_id)} — layer "
        + _quantity_text(quantity, efflr="eff LR over time", dw=r"$|\Delta w|$ over time")
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# BTSP plots


def plot_btsp_peak_offset_bars(
    df: pd.DataFrame,
    value_col: str,
    *,
    exp_order: list[str],
    opt_order: list[str] | None = None,
    labels: ExperimentDisplayLabels | None = None,
    title: str = "",
    quantity: AggregateQuantity = "efflr",
) -> plt.Figure | None:
    if df.empty or value_col not in df.columns:
        return None
    opts = opt_order or _opt_order_without_sgd(df["optimizer_id"].unique().tolist())
    if not exp_order or not opts:
        return None

    agg = (
        df.groupby(["experiment", "optimizer_id"], observed=False)[value_col]
        .agg(mean="mean", sem=_sem_across_values)
        .reset_index()
    )
    agg["ci_lo"] = agg["mean"] - SEM_Z * agg["sem"]
    agg["ci_hi"] = agg["mean"] + SEM_Z * agg["sem"]

    n_exp = len(exp_order)
    width = 0.8 / max(len(opts), 1)
    fig, ax = plt.subplots(figsize=(max(10, n_exp * 1.75), 5.8))
    x = np.arange(n_exp)

    for i, oid in enumerate(opts):
        sub = agg[agg["optimizer_id"] == oid].set_index("experiment").reindex(exp_order)
        means = sub["mean"].to_numpy(dtype=float)
        ci_lo = sub["ci_lo"].to_numpy(dtype=float)
        ci_hi = sub["ci_hi"].to_numpy(dtype=float)
        yerr_lo = means - ci_lo
        yerr_hi = ci_hi - means
        ax.bar(
            x + (i - (len(opts) - 1) / 2.0) * width,
            means,
            width=width * 0.95,
            color=_opt_color(oid),
            label=_opt_display(oid),
            yerr=[yerr_lo, yerr_hi],
            capsize=2,
            error_kw={"elinewidth": 0.9, "ecolor": "0.15"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_xtick_label(e) for e in exp_order], fontsize=9)
    cat1_row: list[str] = []
    if labels is not None:
        cat1_row, _, _ = two_row_cat12_from_sorted(exp_order, labels)
    if cat1_row:
        div = len(cat1_row) - 0.5
        ax.axvline(div, color="0.4", ls="--", lw=1.0)
    ax.set_ylabel(
        _quantity_text(
            quantity,
            efflr=f"mean peak offset ± {SEM_Z:.2f}·SEM",
            dw=f"mean $|\\Delta w|$ peak offset ± {SEM_Z:.2f}·SEM",
        )
    )
    ax.set_title(title or f"BTSP peak offset — {value_col}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_btsp_pre_post_relative_facets(
    df: pd.DataFrame,
    pre_col: str,
    post_col: str,
    *,
    exp_order: list[str],
    opt_order: list[str] | None = None,
    labels: ExperimentDisplayLabels | None = None,
    quantity_label: str = "",
    quantity: AggregateQuantity = "efflr",
) -> plt.Figure | None:
    if df.empty or pre_col not in df.columns or post_col not in df.columns:
        return None
    opts = opt_order or _opt_order_without_sgd(df["optimizer_id"].unique().tolist())
    if not exp_order or not opts:
        return None

    labs = labels
    cat1_row, cat2_row = [], []
    if labs is not None:
        cat1_row, cat2_row, _ = two_row_cat12_from_sorted(exp_order, labs)
    row_specs = [("Cat1", cat1_row), ("Cat2", cat2_row)]
    row_specs = [(lab, exps) for lab, exps in row_specs if exps]
    if not row_specs:
        return None

    n_cols = max(len(exps) for _, exps in row_specs)
    fig, axes = plt.subplots(
        len(row_specs),
        n_cols,
        figsize=(3.2 * n_cols, 2.9 * len(row_specs)),
        squeeze=False,
    )

    all_vals: list[float] = []
    for col_name in (pre_col, post_col):
        v = pd.to_numeric(df[col_name], errors="coerce").dropna()
        all_vals.extend(v.tolist())
    if all_vals:
        span = max(abs(min(all_vals)), abs(max(all_vals)), 1e-6)
        ylim = (-1.15 * span, 1.15 * span)
    else:
        ylim = (-1.0, 1.0)

    for ri, (cat_lab, exps) in enumerate(row_specs):
        for ci in range(n_cols):
            ax = axes[ri, ci]
            if ci >= len(exps):
                ax.set_visible(False)
                continue
            exp = exps[ci]
            sub = df[df["experiment"] == exp]
            pre_samples: dict[str, np.ndarray] = {}
            post_samples: dict[str, np.ndarray] = {}
            for oid in opts:
                s = sub[sub["optimizer_id"] == oid]
                pre_samples[oid] = pd.to_numeric(s[pre_col], errors="coerce").to_numpy(
                    dtype=float
                )
                post_samples[oid] = pd.to_numeric(s[post_col], errors="coerce").to_numpy(
                    dtype=float
                )

            _plot_pre_post_optimizer_panel(
                ax,
                opts,
                pre_samples,
                post_samples,
                ylim=ylim,
                show_ylabel=(ci == 0),
                show_xticklabels=(ri == len(row_specs) - 1),
            )
            if ri == 0:
                ax.set_title(exp, fontsize=9)
            if ci == 0:
                ax.text(
                    -0.42,
                    0.5,
                    cat_lab,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9,
                )

    q_label = quantity_label or ("unit" if "unit" in pre_col else "layer")
    fig.suptitle(
        _quantity_text(
            quantity,
            efflr=f"BTSP pre/post relative eff-LR increase ({q_label}) ± {SEM_Z:.2f}·SEM",
            dw=f"BTSP pre/post relative $|\\Delta w|$ increase ({q_label}) ± {SEM_Z:.2f}·SEM",
        ),
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_btsp_relative_scatter(
    df: pd.DataFrame,
    *,
    side: Literal["pre", "post"],
    exp_order: list[str],
    opt_order: list[str] | None = None,
    quantity: AggregateQuantity = "efflr",
) -> plt.Figure | None:
    if df.empty:
        return None
    opts = opt_order or _opt_order_without_sgd(df["optimizer_id"].unique().tolist())
    if not exp_order or not opts:
        return None

    unit_col = f"{side}_unit_relative_increase_pct"
    layer_col = f"{side}_layer_relative_increase_pct"
    if unit_col not in df.columns:
        return None

    n_rows = len(exp_order)
    n_cols = len(opts)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.8 * n_cols, 2.4 * n_rows),
        squeeze=False,
    )

    for ri, exp in enumerate(exp_order):
        for ci, oid in enumerate(opts):
            ax = axes[ri, ci]
            sub = df[(df["experiment"] == exp) & (df["optimizer_id"] == oid)]
            x = pd.to_numeric(sub[unit_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sub[layer_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if x.size == 0:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.scatter(x, y, s=12, alpha=0.65, color=_opt_color(oid), edgecolors="none")
                if x.size >= 2:
                    try:
                        coef = np.polyfit(x, y, 1)
                        xs = np.linspace(float(np.min(x)), float(np.max(x)), 50)
                        ax.plot(xs, coef[0] * xs + coef[1], color="0.25", lw=1.0)
                    except np.linalg.LinAlgError:
                        pass
                ax.text(
                    0.03,
                    0.97,
                    _spearman_annotation(x, y),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                )
            if ri == 0:
                ax.set_title(_opt_display(oid), fontsize=8)
            if ci == 0:
                ax.text(
                    -0.38,
                    0.5,
                    _xtick_label(exp),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=7,
                )
                ax.set_ylabel("layer (%)", fontsize=7)
            if ri == n_rows - 1:
                ax.set_xlabel("unit (%)", fontsize=7)

    fig.suptitle(
        _quantity_text(
            quantity,
            efflr=f"BTSP {side}-period unit vs layer relative increase",
            dw=f"BTSP {side}-period unit vs layer relative $|\\Delta w|$ increase",
        ),
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_btsp_weight_std_bars(
    df: pd.DataFrame,
    *,
    exp_order: list[str],
    opt_order: list[str] | None = None,
    quantity: AggregateQuantity = "efflr",
) -> plt.Figure | None:
    if df.empty:
        return None
    opts = opt_order or _opt_order_without_sgd(df["optimizer_id"].unique().tolist())
    if not exp_order or not opts:
        return None

    value_cols = [
        "mean_pre_std_weight_efflr",
        "std_weight_efflr_tstart",
        "mean_post_std_weight_efflr",
    ]
    bar_labels = ["pre", r"$t_{\mathrm{start}}$", "post"]

    n_exp = len(exp_order)
    fig, axes = plt.subplots(n_exp, 1, figsize=(max(8, len(opts) * 0.9), 3.2 * n_exp), squeeze=False)

    for ei, exp in enumerate(exp_order):
        ax = axes[ei, 0]
        sub = df[df["experiment"] == exp]
        n_o = len(opts)
        n_b = len(value_cols)
        group_w = 0.8
        bar_w = group_w / n_b
        x = np.arange(n_o)

        for bi, (col, blab) in enumerate(zip(value_cols, bar_labels, strict=True)):
            means = []
            yerr = []
            for oid in opts:
                vals = pd.to_numeric(
                    sub[sub["optimizer_id"] == oid][col], errors="coerce"
                ).dropna()
                if vals.empty:
                    means.append(float("nan"))
                    yerr.append(0.0)
                else:
                    means.append(float(vals.mean()))
                    yerr.append(SEM_Z * _sem_across_values(vals))
            offset = (bi - (n_b - 1) / 2.0) * bar_w
            ax.bar(
                x + offset,
                means,
                width=bar_w * 0.95,
                label=blab,
                yerr=yerr,
                capsize=2,
                error_kw={"elinewidth": 0.9, "ecolor": "0.15"},
            )

        ax.set_xticks(x)
        ax.set_xticklabels([_opt_display(o) for o in opts], rotation=25, ha="right", fontsize=7)
        ax.set_title(_xtick_label(exp), fontsize=9, loc="left")
        ax.set_ylabel(
            _quantity_text(
                quantity,
                efflr=r"weight $\gamma_{\mathrm{eff}}$ std ± CI",
                dw=r"weight $|\Delta w|$ std ± CI",
            )
        )
        if ei == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        _quantity_text(
            quantity,
            efflr=f"BTSP per-weight eff-LR std around $t_{{\\mathrm{{start}}}}$ ± {SEM_Z:.2f}·SEM",
            dw=f"BTSP per-weight $|\\Delta w|$ std around $t_{{\\mathrm{{start}}}}$ ± {SEM_Z:.2f}·SEM",
        ),
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    return fig
