"""Min-max normalize robustness scores across all cached analysis checkpoints.

Loads every run checkpoint under `--output-dir` (matching the simulation tree
in `--input-dir`), applies global robustness min-max normalization, rescoring,
and saves updated checkpoints.
"""
import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.constants import set_results_root
from analysis.final.final_compare_all_optimizers_helpers import (
    _load_run_checkpoint,
    minmax_normalize_robustness_all_results,
)
from analysis.io import scan_results
from scripts.analyse_sim_results import DEFAULT_SCORE_FACTORS, _parse_score_factors


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory of simulation results (used to discover all runs).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory of gzip-pickled analysis checkpoints to update.",
    )
    p.add_argument(
        "--score-factors",
        type=_parse_score_factors,
        default=None,
        help=f"JSON object of score weights (default: {json.dumps(DEFAULT_SCORE_FACTORS)}).",
    )
    p.add_argument(
        "--require-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any simulation run is missing an analysis checkpoint.",
    )
    return p


def load_all_checkpoints(
    input_dir: Path,
    output_dir: Path,
) -> tuple[dict[tuple[str, str, str, str], dict], list[tuple[str, str, str, str]]]:
    set_results_root(input_dir)
    tree = scan_results(refresh=True, w_cache=False)

    all_results: dict[tuple[str, str, str, str], dict] = {}
    missing: list[tuple[str, str, str, str]] = []
    for eid, models in tree.items():
        for mid, runs in models.items():
            for rid, oids in runs.items():
                for oid in oids:
                    key = (eid, mid, oid, rid)
                    loaded = _load_run_checkpoint(str(output_dir), eid, mid, oid, rid)
                    if loaded is None:
                        missing.append(key)
                        continue
                    if loaded.get("error"):
                        continue
                    all_results[key] = loaded
    return all_results, missing


def run_minmax(
    *,
    input_dir: Path,
    output_dir: Path,
    score_factors: dict[str, float],
    require_all: bool = True,
) -> int:
    all_results, missing = load_all_checkpoints(input_dir, output_dir)
    if missing:
        msg = (
            f"{len(missing)} run(s) missing analysis checkpoints under {output_dir} "
            f"(first: eid={missing[0][0]!r} mid={missing[0][1]!r} "
            f"oid={missing[0][2]!r} rid={missing[0][3]!r})"
        )
        if require_all:
            print(f"ERROR: {msg}", file=sys.stderr)
            return 1
        print(f"WARNING: {msg}", file=sys.stderr)

    if not all_results:
        print("No analysis checkpoints found to min-max normalize.")
        return 1

    n_saved = minmax_normalize_robustness_all_results(
        all_results,
        score_factors,
        save_dir=str(output_dir),
    )
    if n_saved:
        print(
            f"Min-max normalized robustness across {len(all_results)} cached run(s) "
            f"and saved {n_saved} checkpoint(s)"
        )
    else:
        print(
            f"Robustness already min-max normalized for all {len(all_results)} "
            "cached run(s); nothing to update"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    score_factors = args.score_factors if args.score_factors is not None else dict(DEFAULT_SCORE_FACTORS)
    return run_minmax(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        score_factors=score_factors,
        require_all=args.require_all,
    )


if __name__ == "__main__":
    raise SystemExit(main())
