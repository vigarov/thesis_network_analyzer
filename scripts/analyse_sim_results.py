"""Compute FINAL optimizer-comparison metrics from simulation results.

Reads run_simulation.py artifacts from `--input-dir` (same layout as
`results/<experiment_id>/<run_id>/<model_id>/optimizers/<optimizer_id>/`),
writes gzip-pickled checkpoints under `--output-dir`.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.final.final_analysis_config import (
    DEFAULT_SCORE_FACTORS,
    EXPERT_MODEL_DIR_BY_DATASET,
    dataset_key,
)
from analysis.final.load_final_analysis import run_final_analysis
from scripts.utils.cluster_utils import add_output_dir_argument

# Backward-compatible aliases for Slurm scripts and external callers.
DEFAULT_EXPERT_MODEL_DIR_MNIST = EXPERT_MODEL_DIR_BY_DATASET["mnist"]
DEFAULT_EXPERT_MODEL_DIR_CIFAR = EXPERT_MODEL_DIR_BY_DATASET["cifar10"]


def _parse_score_factors(raw: str) -> dict[str, float]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON for --score-factors: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--score-factors must be a JSON object")
    out: dict[str, float] = {}
    for key, value in parsed.items():
        out[str(key)] = float(value)
    return out


def run_analysis(**kwargs):
    """Legacy wrapper: returns `(all_results, errors)` like the pre-refactor API."""
    all_results, _tree, errors, _ds_key, _model_ids = run_final_analysis(**kwargs)
    return all_results, errors


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory of simulation results (experiment/run/model tree).",
    )
    add_output_dir_argument(
        p,
        help="Directory for gzip-pickled analysis checkpoints.",
        required=True,
    )
    p.add_argument(
        "--dataset",
        default="mnist",
        help="Dataset family (default: mnist). Accepts cifar / cifar10.",
    )
    p.add_argument(
        "--expert-model-dir",
        type=str,
        default=None,
        help=(
            "Expert saliency checkpoints relative to project root "
            "(!OPT → optimizer id; !ARCH → inception/resnet for CIFAR; "
            "!SD → config.json seed, appended if omitted). "
            f"Default: MNIST {EXPERT_MODEL_DIR_BY_DATASET['mnist']!r}, "
            f"CIFAR {EXPERT_MODEL_DIR_BY_DATASET['cifar10']!r}."
        ),
    )
    p.add_argument(
        "--n-checkpoint-samples",
        type=int,
        default=20,
        help="Uniform checkpoint samples per recovery period (default: 20).",
    )
    p.add_argument(
        "--score-factors",
        type=_parse_score_factors,
        default=None,
        help=f"JSON object of score weights (default: {json.dumps(DEFAULT_SCORE_FACTORS)}).",
    )
    p.add_argument(
        "--rescore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute total_score from cached component columns when loading checkpoints.",
    )
    p.add_argument(
        "--minmax-normalize-robustness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Min-max normalize robustness_score per model_id after compute.",
    )
    p.add_argument(
        "--reparse-digit-a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refresh digitA on recover/reinforce cached runs from config.json.",
    )
    p.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Strict dead-neuron filtering when building recovery periods.",
    )
    p.add_argument(
        "--btsp-start-strategy",
        type=str,
        default="acceleration",
        choices=("acceleration", "trial_start"),
        help="BTSP start point for angle / checkpoint sampling.",
    )
    p.add_argument(
        "--use-real-progression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use real progression for activation-angle computation.",
    )
    p.add_argument(
        "--verbose-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print full tracebacks to stderr when a run fails.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ds = dataset_key(args.dataset)
    score_factors = args.score_factors if args.score_factors is not None else dict(DEFAULT_SCORE_FACTORS)
    if args.expert_model_dir is None:
        expert_model_dir = (
            EXPERT_MODEL_DIR_BY_DATASET["cifar10"]
            if ds == "cifar"
            else EXPERT_MODEL_DIR_BY_DATASET["mnist"]
        )
    else:
        expert_model_dir = args.expert_model_dir

    all_results, _, errors, _, _ = run_final_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        project_root=_PROJECT_ROOT,
        expert_model_dir=expert_model_dir,
        n_checkpoint_samples=args.n_checkpoint_samples,
        score_factors=score_factors,
        rescore=args.rescore,
        minmax_normalize_robustness=args.minmax_normalize_robustness,
        reparse_digit_a=args.reparse_digit_a,
        strict=args.strict,
        btsp_start_strategy=args.btsp_start_strategy,
        use_real_progression=args.use_real_progression,
        dataset=args.dataset,
        verbose_errors=args.verbose_errors,
    )

    print(f"Loaded/saved {len(all_results)} runs; {len(errors)} errors")
    if errors:
        print(
            pd.DataFrame(errors)
            .drop(columns=["traceback"], errors="ignore")
            .to_string(index=False)
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
