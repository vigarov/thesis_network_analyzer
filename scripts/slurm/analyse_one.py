"""Run FINAL analysis for one (experiment_id, optimizer_id) shard.

Each shard is independent (all models/runs for that experiment and optimizer).
Robustness min-max normalization is skipped because it must run over all runs;
`submit_analyse_jobs.py` applies `minmax_normalize_robustness_all_results`
across all cached checkpoints after shard jobs finish.
"""
import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.analyse_sim_results import (
	DEFAULT_SCORE_FACTORS,
	_parse_score_factors,
	run_analysis,
)


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument(
		"--experiment-id",
		type=str,
		required=True,
		help="Experiment directory name under --input-dir.",
	)
	p.add_argument(
		"--optimizer-id",
		type=str,
		required=True,
		help="Optimizer directory name under .../optimizers/.",
	)
	p.add_argument(
		"--input-dir",
		type=Path,
		required=True,
		help="Root directory of simulation results (experiment/run/model tree).",
	)
	p.add_argument(
		"--output-dir",
		type=Path,
		required=True,
		help="Directory for gzip-pickled analysis checkpoints.",
	)
	p.add_argument(
		"--expert-model-dir",
		type=str,
		default="pretrained_models/expert/!OPT/",
		help=(
			"Expert saliency checkpoints relative to project root "
			"(!OPT → optimizer id; !SD → config.json seed, appended if omitted)."
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
		default=False,
		help=(
			"Min-max normalize robustness_score across all runs after compute "
			"(default: off for shard jobs; run full analyse once after all shards)."
		),
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
	score_factors = args.score_factors if args.score_factors is not None else dict(DEFAULT_SCORE_FACTORS)

	all_results, errors = run_analysis(
		input_dir=args.input_dir,
		output_dir=args.output_dir,
		project_root=_PROJECT_ROOT,
		expert_model_dir=args.expert_model_dir,
		n_checkpoint_samples=args.n_checkpoint_samples,
		score_factors=score_factors,
		rescore=args.rescore,
		minmax_normalize_robustness=args.minmax_normalize_robustness,
		reparse_digit_a=args.reparse_digit_a,
		strict=args.strict,
		btsp_start_strategy=args.btsp_start_strategy,
		use_real_progression=args.use_real_progression,
		verbose_errors=args.verbose_errors,
		experiment_id=args.experiment_id,
		optimizer_id=args.optimizer_id,
	)

	print(
		f"Shard eid={args.experiment_id!r} oid={args.optimizer_id!r}: "
		f"loaded/saved {len(all_results)} runs; {len(errors)} errors"
	)
	if errors:
		import pandas as pd

		print(
			pd.DataFrame(errors)
			.drop(columns=["traceback"], errors="ignore")
			.to_string(index=False)
		)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
