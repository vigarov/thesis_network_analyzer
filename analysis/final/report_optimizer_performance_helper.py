"""Optimizer performance metrics for REPORT_plots (peak ratio, TTC).

Logic aligned with `local/time_to_stabilization.ipynb` (Section B/C):
- Peak magnitude = `loss_at_peak / mean(training loss)` per detected peak
- TTC: Cat1 → `first_most`; Cat2 → `each_most` (`MOST_BIN_SIZE`-checkpoint bins)
"""
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from analysis.dataset_filter import INCEPTION_MODEL_ID, MNIST_MODEL_ID, RESNET_MODEL_ID
from analysis.final.final_additional_plots_helpers import (
    _clean_find_peaks_kwargs,
    is_cat2_sequence_experiment,
    is_category_experiment,
    metrics_x_axis,
    peak_ratios_per_peak,
)
from analysis.io import _metrics

MOST_BIN_SIZE = 5

REPORT_FIND_PEAKS_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    MNIST_MODEL_ID: {
        "cat1_sample_shuffle_control_tr100": {
            "prominence": 3.0,
            "distance": 10,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat1_global_pretrain_control_K1000_tr100": {
            "prominence": 3.0,
            "distance": 10,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat1_sample_shuffle_constrained_digits0-1_tr100": {
            "prominence": None,
            "distance": 10,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat1_sample_shuffle_interleaved_digits0-1_tr100": {
            "prominence": 0.5,
            "distance": 8,
            "height": 1.5,
            "width": None,
            "threshold": 1.5,
        },
        "cat1_sample_shuffle_finetune_K1000_tr100": {
            "prominence": 3.0,
            "distance": 10,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat2_sequence_control_tr100": {
            "prominence": 0.5,
            "distance": 10,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat2_sequence_pretrain_control_K1000_tr100": {
            "prominence": 0.5,
            "distance": 12,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat2_sequence_labelperm_K1000_tr100": {
            "prominence": 0.5,
            "distance": 12,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat2_sequence_recover_K1000_tr100_a0": {
            "prominence": 1.0,
            "distance": 12,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat2_sequence_reinforce_K1000_tr100_a0": {
            "prominence": 1.0,
            "distance": 20,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
    },
    INCEPTION_MODEL_ID: {
        "cat1_sample_shuffle_control_tr100": {
            "prominence": 3.0,
            "distance": 10,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat1_global_pretrain_control_K6000_tr100": {
            "prominence": 3.0,
            "distance": 10,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat1_sample_shuffle_constrained_digits0-1_tr100": {
            "prominence": 1.0,
            "distance": 15,
            "height": 1.5,
            "width": None,
            "threshold": 2.0,
        },
        "cat1_sample_shuffle_interleaved_digits0-1_tr100": {
            "prominence": None,
            "distance": 15,
            "height": 1.5,
            "width": None,
            "threshold": 3.0,
        },
        "cat1_sample_shuffle_finetune_K6000_tr100": {
            "prominence": None,
            "distance": 20,
            "height": 0.5,
            "width": 0.5,
            "threshold": 1.5,
        },
        "cat2_sequence_control_tr100": {
            "prominence": 0.5,
            "distance": 12,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat2_sequence_pretrain_control_K6000_tr100": {
            "prominence": 0.5,
            "distance": 15,
            "height": 1.5,
            "width": None,
            "threshold": 1.5,
        },
        "cat2_sequence_labelperm_K6000_tr100": {
            "prominence": 0.5,
            "distance": 12,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat2_sequence_recover_K6000_tr100_a0": {
            "prominence": 3.0,
            "distance": 8,
            "height": 1.5,
            "width": 1.0,
            "threshold": 2.0,
        },
        "cat2_sequence_reinforce_K6000_tr100_a0": {
            "prominence": 2.0,
            "distance": 20,
            "height": None,
            "width": 1.0,
            "threshold": 1.5,
        },
    },
    RESNET_MODEL_ID: {
        "cat1_sample_shuffle_control_tr100": {
            "prominence": None,
            "distance": 10,
            "height": None,
            "width": None,
            "threshold": 0.1,
        },
        "cat1_global_pretrain_control_K15000_tr100": {
            "prominence": None,
            "distance": 10,
            "height": None,
            "width": None,
            "threshold": 0.1,
        },
        "cat1_sample_shuffle_constrained_digits0-1_tr100": {
            "prominence": 1.5,
            "distance": 15,
            "height": 1.5,
            "width": None,
            "threshold": 1.5,
        },
        "cat1_sample_shuffle_interleaved_digits0-1_tr100": {
            "prominence": 4.5,
            "distance": 10,
            "height": None,
            "width": None,
            "threshold": 2.6,
        },
        "cat1_sample_shuffle_finetune_K15000_tr100": {
            "prominence": 3.5,
            "distance": 10,
            "height": 1.5,
            "width": None,
            "threshold": 1.5,
        },
        "cat2_sequence_control_tr100": {
            "prominence": 0.5,
            "distance": 50,
            "height": None,
            "width": None,
            "threshold": None,
        },
        "cat2_sequence_pretrain_control_K15000_tr100": {
            "prominence": 2.5,
            "distance": 10,
            "height": 1.5,
            "width": None,
            "threshold": 2.0,
        },
        "cat2_sequence_labelperm_K15000_tr100": {
            "prominence": 2.5,
            "distance": 30,
            "height": 1.5,
            "width": None,
            "threshold": None,
        },
        "cat2_sequence_recover_K15000_tr100_a0": {
            "prominence": 5.0,
            "distance": 20,
            "height": 1.5,
            "width": None,
            "threshold": 0.3,
        },
        "cat2_sequence_reinforce_K15000_tr100_a0": {
            "prominence": 3.0,
            "distance": 30,
            "height": None,
            "width": None,
            "threshold": 2.0,
        },
    },
}


def _find_peaks_kwargs(model_id: str, eid: str) -> dict[str, Any]:
    specs = REPORT_FIND_PEAKS_SPECS.get(str(model_id), {})
    if eid not in specs:
        raise KeyError(f"No find_peaks params for model_id={model_id!r}, eid={eid!r}")
    return dict(specs[eid])


def _trial_segment_ranges(metrics: dict) -> list[tuple[int, int]]:
    cp = metrics.get("trial_end_checkpoint_idxs", metrics.get("stage_end_checkpoint_idxs"))
    if cp is None:
        return []
    cp = [int(i) for i in cp]
    starts = [0] + [i + 1 for i in cp[:-1]]
    ends = [i + 1 for i in cp]
    return list(zip(starts, ends, strict=True))


def cat1_late_phase_threshold(y: np.ndarray) -> float:
    half = np.asarray(y, dtype=float)[len(y) // 2 :]
    half = half[np.isfinite(half)]
    if half.size == 0:
        raise ValueError("No finite loss values in Cat1 late-phase half")
    return float(np.median(half))


def cat2_per_trial_thresholds(
    y: np.ndarray,
    metrics: dict,
) -> list[tuple[int, int, float]]:
    y = np.asarray(y, dtype=float)
    segments = _trial_segment_ranges(metrics)
    if not segments:
        raise ValueError("Cat2 metrics missing trial boundary indices")
    thresholds: list[tuple[int, int, float]] = []
    for start, end in segments:
        seg = y[start:end]
        if seg.size == 0:
            continue
        half = seg[seg.size // 2 :]
        half = half[np.isfinite(half)]
        if half.size == 0:
            continue
        thresholds.append((start, end, float(np.median(half))))
    if not thresholds:
        raise ValueError("No finite loss values in Cat2 late-phase halves")
    return thresholds


def _peak_indices(y: np.ndarray, fp_kwargs: dict) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return np.array([], dtype=int)
    idx_map = np.where(finite)[0]
    y_f = y[finite]
    peaks, _ = find_peaks(y_f, **_clean_find_peaks_kwargs(fp_kwargs))
    return idx_map[peaks]


def _first_raw_crossing(y: np.ndarray, start: int, end: int, threshold: float) -> int | None:
    for i in range(start + 1, end):
        if np.isfinite(y[i]) and y[i] < threshold:
            return i
    return None


def _first_binned_crossing(
    y: np.ndarray,
    start: int,
    end: int,
    threshold: float,
    *,
    bin_size: int = MOST_BIN_SIZE,
) -> int | None:
    i = start + 1
    while i < end:
        j = min(i + bin_size, end)
        chunk = y[i:j]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size and float(np.mean(chunk)) < threshold:
            return i
        i = j
    return None


def _ttc_after_peak(
    y: np.ndarray,
    xs: np.ndarray,
    peak_idx: int,
    end_idx: int,
    threshold: float,
    *,
    use_bins: bool,
    most_bin_size: int = MOST_BIN_SIZE,
) -> float | None:
    if use_bins:
        cross = _first_binned_crossing(
            y, peak_idx, end_idx, threshold, bin_size=most_bin_size
        )
    else:
        cross = _first_raw_crossing(y, peak_idx, end_idx, threshold)
    if cross is None:
        return None
    return float(xs[cross] - xs[peak_idx])


def compute_ttc_first(
    y: np.ndarray,
    xs: np.ndarray,
    threshold: float,
    fp_kwargs: dict,
    *,
    use_bins: bool,
    most_bin_size: int = MOST_BIN_SIZE,
) -> float:
    peaks = _peak_indices(y, fp_kwargs)
    if peaks.size == 0:
        return np.nan
    peak_idx = int(peaks[0])
    ttc = _ttc_after_peak(
        y, xs, peak_idx, len(y), threshold, use_bins=use_bins, most_bin_size=most_bin_size
    )
    return np.nan if ttc is None else ttc


def compute_ttc_each(
    y: np.ndarray,
    xs: np.ndarray,
    threshold: float,
    fp_kwargs: dict,
    *,
    use_bins: bool,
    most_bin_size: int = MOST_BIN_SIZE,
) -> list[float]:
    peaks = _peak_indices(y, fp_kwargs)
    if peaks.size == 0:
        return []
    out: list[float] = []
    for i, peak_idx in enumerate(peaks):
        peak_idx = int(peak_idx)
        end_idx = int(peaks[i + 1]) if i + 1 < peaks.size else len(y)
        ttc = _ttc_after_peak(
            y, xs, peak_idx, end_idx, threshold, use_bins=use_bins, most_bin_size=most_bin_size
        )
        if ttc is None:
            if end_idx <= peak_idx + 1:
                continue
            ttc = float(xs[end_idx - 1] - xs[peak_idx])
        out.append(ttc)
    return out


def _theta_values_for_run(
    y: np.ndarray,
    metrics: dict,
    eid: str,
) -> list[float]:
    """Late-phase loss thresholds used in TTC (one value for Cat1; one per trial for Cat2)."""
    if is_cat2_sequence_experiment(eid):
        return [thr for _, _, thr in cat2_per_trial_thresholds(y, metrics)]
    return [cat1_late_phase_threshold(y)]


def _ttc_values_for_run(
    y: np.ndarray,
    xs: np.ndarray,
    metrics: dict,
    fp_kwargs: dict,
    eid: str,
    *,
    most_bin_size: int = MOST_BIN_SIZE,
) -> list[float]:
    if is_cat2_sequence_experiment(eid):
        technique = "each_most"
    else:
        technique = "first_most"

    if technique == "first_most":
        thr = cat1_late_phase_threshold(y)
        val = compute_ttc_first(
            y, xs, thr, fp_kwargs, use_bins=True, most_bin_size=most_bin_size
        )
        return [float(val)] if np.isfinite(val) else []

    thr_trials = cat2_per_trial_thresholds(y, metrics)
    out: list[float] = []
    for start, end, thr in thr_trials:
        seg_y = y[start:end]
        seg_x = xs[start:end]
        out.extend(
            compute_ttc_each(
                seg_y, seg_x, thr, fp_kwargs, use_bins=True, most_bin_size=most_bin_size
            )
        )
    return [float(v) for v in out if np.isfinite(v)]


def build_report_peak_ratio_long(
    all_results: dict[tuple[str, str, str, str], dict],
    rename_fn: Callable[[str], str],
) -> pd.DataFrame:
    """One row per detected loss peak (peak magnitude = loss_at_peak / mean loss)."""
    rows: list[dict] = []
    for (eid, mid, oid, rid), out in all_results.items():
        if out.get("error") or not is_category_experiment(eid):
            continue
        try:
            fp_kw = _find_peaks_kwargs(mid, eid)
        except KeyError:
            continue
        m = _metrics(eid, mid, oid, rid=rid, w_cache=False)
        if not m or "loss_all_train_mean" not in m:
            continue
        y = np.asarray(m["loss_all_train_mean"], dtype=float)
        for peak_i, ratio in enumerate(peak_ratios_per_peak(y, fp_kw)):
            if not np.isfinite(ratio):
                continue
            rows.append(
                {
                    "experiment_id": eid,
                    "model_id": mid,
                    "experiment": rename_fn(eid),
                    "optimizer_id": oid,
                    "run_id": rid,
                    "peak_index": int(peak_i),
                    "value": float(ratio),
                    "metric": "peak_ratio",
                }
            )
    return pd.DataFrame(rows)


def build_report_ttc_long(
    all_results: dict[tuple[str, str, str, str], dict],
    rename_fn: Callable[[str], str],
    *,
    most_bin_size: int = MOST_BIN_SIZE,
) -> pd.DataFrame:
    """TTC rows: Cat1 → `first_most`; Cat2 → `each_most` (per trial / peak)."""
    rows: list[dict] = []
    for (eid, mid, oid, rid), out in all_results.items():
        if out.get("error") or not is_category_experiment(eid):
            continue
        try:
            fp_kw = _find_peaks_kwargs(mid, eid)
        except KeyError:
            continue
        m = _metrics(eid, mid, oid, rid=rid, w_cache=False)
        if not m or "loss_all_train_mean" not in m:
            continue
        y = np.asarray(m["loss_all_train_mean"], dtype=float)
        xs, _x_mode = metrics_x_axis(m)
        try:
            values = _ttc_values_for_run(
                y, xs, m, fp_kw, eid, most_bin_size=most_bin_size
            )
        except ValueError:
            continue
        for val in values:
            rows.append(
                {
                    "experiment_id": eid,
                    "model_id": mid,
                    "experiment": rename_fn(eid),
                    "optimizer_id": oid,
                    "run_id": rid,
                    "value": float(val),
                    "metric": "ttc",
                }
            )
    return pd.DataFrame(rows)


def build_report_theta_long(
    all_results: dict[tuple[str, str, str, str], dict],
    rename_fn: Callable[[str], str],
) -> pd.DataFrame:
    """One row per TTC threshold sample (Cat1: one per run; Cat2: one per trial)."""
    rows: list[dict] = []
    for (eid, mid, oid, rid), out in all_results.items():
        if out.get("error") or not is_category_experiment(eid):
            continue
        m = _metrics(eid, mid, oid, rid=rid, w_cache=False)
        if not m or "loss_all_train_mean" not in m:
            continue
        y = np.asarray(m["loss_all_train_mean"], dtype=float)
        try:
            values = _theta_values_for_run(y, m, eid)
        except ValueError:
            continue
        for val in values:
            if not np.isfinite(val):
                continue
            rows.append(
                {
                    "experiment_id": eid,
                    "model_id": mid,
                    "experiment": rename_fn(eid),
                    "optimizer_id": oid,
                    "run_id": rid,
                    "value": float(val),
                    "metric": "theta",
                }
            )
    return pd.DataFrame(rows)
