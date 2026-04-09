#!/usr/bin/env python3
"""One-shot regeneration of post-processing CSVs from saved ``*.npz`` (no retraining).

Overwrites, in each given optimizer directory:

- ``post_processing_neuron_digit.csv``
- ``post_processing_dead.csv``

All semantics (*dead*, assigned/inactive/partial) live in
``compute_results.post_processing`` — this script only loads files and calls
:func:`~compute_results.post_processing.post_processing_dataframes_from_nts`.

Does **not** rebuild ``post_processing_train_act.npz`` (needs the trained model).

``--ppd`` only adds/updates the ``is_ppd`` column on existing
``post_processing_dead.csv`` files (fast; no ``neuron_timeseries`` load).

Usage (from the repo root)::

    uv run python visualize_webapp/notebook/convert_post_processing.py path/to/optimizer_dir
    uv run python visualize_webapp/notebook/convert_post_processing.py --results-dir path/to/results
    uv run python visualize_webapp/notebook/convert_post_processing.py -r results/ exp1/opt_a
    uv run python visualize_webapp/notebook/convert_post_processing.py --ppd -r path/to/results

``--results-dir`` walks the tree and converts every directory that contains
``neuron_timeseries.npz`` (e.g. under ``.../optimizers/<id>/``).

If ``units`` cannot be inferred, you need an existing
``post_processing_neuron_digit.csv`` for layer names (same as before).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from compute_results.post_processing import (
    add_is_ppd_column,
    dead_label_mask_for_optimizer_dir,
    post_processing_dataframes_from_nts,
)
from tqdm import tqdm


def optimizer_dirs_under_results_root(results_root: Path) -> list[Path]:
    """Directories that contain ``neuron_timeseries.npz`` under ``results_root``."""
    root = results_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted({p.parent for p in root.rglob("neuron_timeseries.npz")})


def recompute_post_processing_csvs(
    optimizer_dir: Path,
    units: list[dict[str, Any]] | None = None,
    *,
    nts: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load saved tensors and rebuild the two CSV tables (see module docstring)."""
    nts_path = optimizer_dir / "neuron_timeseries.npz"
    if nts is None:
        if not nts_path.exists():
            raise FileNotFoundError(nts_path)
        nts = dict(np.load(str(nts_path), allow_pickle=True))

    checkpoint_tags = [str(t) for t in nts.get("checkpoint_tags", [])]
    n_checkpoints = len(checkpoint_tags)
    if n_checkpoints == 0:
        return pd.DataFrame(), pd.DataFrame()

    if units is None:
        nd_path = optimizer_dir / "post_processing_neuron_digit.csv"
        if not nd_path.exists():
            raise FileNotFoundError(
                f"{nd_path} is required when units=None to recover layer_name per neuron",
            )
        df_prev = pd.read_csv(nd_path)
        layer_by_nid = df_prev.drop_duplicates("neuron_id").set_index("neuron_id")[
            "layer_name"
        ].to_dict()
        raw_ids = nts.get("unit_node_ids")
        if raw_ids is None:
            raise ValueError("neuron_timeseries.npz missing unit_node_ids")
        units = [
            {"node_id": str(nid), "layer_name": layer_by_nid.get(str(nid), "")}
            for nid in raw_ids
        ]

    cp_mask = dead_label_mask_for_optimizer_dir(optimizer_dir, n_checkpoints)
    return post_processing_dataframes_from_nts(nts, units, cp_mask)


def _run_one(od: Path) -> None:
    df_nd, df_dead = recompute_post_processing_csvs(od, units=None)
    out_nd = od / "post_processing_neuron_digit.csv"
    out_dead = od / "post_processing_dead.csv"
    df_nd.to_csv(str(out_nd), index=False)
    df_dead.to_csv(str(out_dead), index=False)
    print(f"Wrote {out_nd} and {out_dead} ({len(df_dead)} rows in dead CSV).")


def _run_ppd_only(od: Path) -> None:
    """Refresh only ``is_ppd`` on ``post_processing_dead.csv``."""
    dead_path = od / "post_processing_dead.csv"
    if not dead_path.exists():
        raise FileNotFoundError(dead_path)
    df = pd.read_csv(dead_path)
    if "is_dead" not in df.columns:
        raise ValueError(f"{dead_path} has no is_dead column")
    df = add_is_ppd_column(df)
    df.to_csv(str(dead_path), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute post_processing_*.csv from neuron_timeseries + metrics.",
    )
    parser.add_argument(
        "optimizer_dirs",
        nargs="*",
        type=Path,
        default=[],
        help="Optimizer output dirs (each with neuron_timeseries.npz). Optional if "
        "--results-dir is set.",
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        action="append",
        default=[],
        type=Path,
        metavar="DIR",
        help="Traverse each DIR: full mode uses parents of neuron_timeseries.npz; "
        "with --ppd, parents of post_processing_dead.csv.",
    )
    parser.add_argument(
        "--ppd",
        action="store_true",
        help="Only add/update is_ppd on post_processing_dead.csv (no full recompute).",
    )
    args = parser.parse_args()

    to_run: list[Path] = []
    for rd in args.results_dir:
        try:
            if args.ppd:
                # Any directory with post_processing_dead.csv (same walk pattern as full mode).
                root = rd.resolve()
                if root.is_dir():
                    to_run.extend(
                        sorted({p.parent for p in root.rglob("post_processing_dead.csv")}),
                    )
            else:
                to_run.extend(optimizer_dirs_under_results_root(rd))
        except OSError as e:
            print(f"Skip --results-dir {rd}: {e}", file=sys.stderr)
    for raw in args.optimizer_dirs:
        to_run.append(raw.resolve())

    uniq: list[Path] = []
    seen_paths: set[Path] = set()
    for p in to_run:
        r = p.resolve()
        if r not in seen_paths:
            seen_paths.add(r)
            uniq.append(r)

    runner = _run_ppd_only if args.ppd else _run_one
    for od in tqdm(uniq, desc="post_processing", unit="dir"):
        try:
            runner(od)
        except (OSError, ValueError, FileNotFoundError) as e:
            print(f"Skip {od}: {e}", file=sys.stderr)

    if not uniq:
        parser.error(
            "Pass at least one optimizer directory or use --results-dir / -r.",
        )


if __name__ == "__main__":
    main()
