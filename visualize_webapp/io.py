from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from visualize_webapp.common import _sort_optimizer_ids
from visualize_webapp.constants import RESULTS


def _optimizer_npz_path(eid: str, rid: str, mid: str, oid: str, fname: str) -> Path:
    return RESULTS / eid / rid / mid / "optimizers" / oid / fname


def _npz(
    eid: str,
    mid: str,
    oid: str,
    fname: str,
    *,
    rid: str,
    w_cache: bool = True,
    cache: dict[tuple, Any] | None = None,
) -> dict[str, np.ndarray]:
    key = (fname, eid, rid, mid, oid)
    if cache is not None and key in cache:
        return cache[key]
    path = _optimizer_npz_path(eid, rid, mid, oid, fname)
    if not path.exists():
        return {}
    data = dict(np.load(str(path), allow_pickle=True))
    if w_cache and cache is not None:
        cache[key] = data
    return data

# `pp` stands for post-processing
# See compute_results/post_processing.py

def _pp_csv(
    eid: str,
    mid: str,
    oid: str,
    fname: str,
    *,
    rid: str,
    w_cache: bool = True,
    cache: dict[tuple, Any] | None = None,
) -> pd.DataFrame | None:
    key = ("__pp_csv__", fname, eid, rid, mid, oid)
    if cache is not None and key in cache:
        return cache[key]
    path = _optimizer_npz_path(eid, rid, mid, oid, fname)
    if not path.exists():
        if cache is not None:
            cache[key] = None
        return None
    df = pd.read_csv(str(path))
    if w_cache and cache is not None:
        cache[key] = df
    return df


def _pp_neuron_digit(
    e: str,
    m: str,
    o: str,
    *,
    rid: str,
    w_cache: bool = True,
    cache: dict[tuple, Any] | None = None,
) -> pd.DataFrame | None:
    return _pp_csv(
        e, m, o, "post_processing_neuron_digit.csv", rid=rid, w_cache=w_cache, cache=cache
    )


def _pp_dead(
    e: str, m: str, o: str, *, rid: str, w_cache: bool = True, cache: dict[tuple, Any] | None = None
) -> pd.DataFrame | None:
    return _pp_csv(e, m, o, "post_processing_dead.csv", rid=rid, w_cache=w_cache, cache=cache)


def _metrics(
    e: str,
    m: str,
    o: str,
    *,
    rid: str,
    w_cache: bool = True,
    cache: dict[tuple, Any] | None = None,
) -> dict[str, np.ndarray]:
    return _npz(e, m, o, "training_metrics.npz", rid=rid, w_cache=w_cache, cache=cache)


def _nts(
    e: str,
    m: str,
    o: str,
    *,
    rid: str,
    w_cache: bool = True,
    cache: dict[tuple, Any] | None = None,
) -> dict[str, np.ndarray]:
    return _npz(e, m, o, "neuron_timeseries.npz", rid=rid, w_cache=w_cache, cache=cache)


def _sigs(
    e: str,
    m: str,
    o: str,
    *,
    rid: str,
    w_cache: bool = True,
    cache: dict[tuple, Any] | None = None,
) -> dict[str, np.ndarray]:
    return _npz(e, m, o, "signals.npz", rid=rid, w_cache=w_cache, cache=cache)


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


def scan_results(
    *,
    cache: dict[tuple, Any] | None = None,
    w_cache: bool = True,
    refresh: bool = False,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Return `{experiment_id: {model_id: {run_id: [optimizer_id, …]}}}`.

    Layout: `results/<experiment_id>/<run_id>/<model_id>/optimizers/<optimizer_id>/`.
    """
    key = ("__scan__",)
    if not refresh and w_cache and cache is not None and key in cache:
        return cache[key]

    tree: dict[str, dict[str, dict[str, list[str]]]] = {}
    if not RESULTS.exists():
        if w_cache and cache is not None:
            cache[key] = tree
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

    if w_cache and cache is not None:
        cache[key] = tree
    return tree
