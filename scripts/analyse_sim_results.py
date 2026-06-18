"""Compute FINAL optimizer-comparison metrics from simulation results.

Reads run_simulation.py artifacts from `--input-dir` (same layout as
`results/<experiment_id>/<run_id>/<model_id>/optimizers/<optimizer_id>/`),
writes gzip-pickled checkpoints under `--output-dir`.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.constants import set_results_root
from analysis.final.final_compare_all_optimizers_helpers import (
	_load_run_checkpoint,
	_save_run_checkpoint,
	final_run_one,
	minmax_normalize_robustness_all_results,
	reparse_digit_a_final_run,
	rescore_final_run,
)
from analysis.io import scan_results
from scripts.utils.cluster_utils import add_output_dir_argument

DEFAULT_SCORE_FACTORS: dict[str, float] = {
	"angle_score": 1.0,
	"length_score": 1.0,
	"n_partial_decay_score": 0.0,
	"n_active_decay_score": 0.0,
	"robustness_score": 1.0,
}


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
		"--expert-model-dir",
		type=str,
		default="local/save/expert_optimizer_pretrained/!OPT/",
		help="Expert saliency models relative to project root (!OPT → optimizer id).",
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
		help="Min-max normalize robustness_score across all runs after compute.",
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


def run_analysis(
	*,
	input_dir: Path,
	output_dir: Path,
	project_root: Path,
	expert_model_dir: str,
	n_checkpoint_samples: int,
	score_factors: dict[str, float],
	rescore: bool,
	minmax_normalize_robustness: bool,
	reparse_digit_a: bool,
	strict: bool,
	btsp_start_strategy: str,
	use_real_progression: bool,
	verbose_errors: bool,
) -> tuple[dict[tuple[str, str, str, str], dict], list[dict[str, Any]]]:
	set_results_root(input_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	tree = scan_results()
	prebuilt_pyramids: dict = {}
	expert_saliency_by_oid: dict[str, dict] | None = {}
	all_results: dict[tuple[str, str, str, str], dict] = {}
	errors: list[dict[str, Any]] = []

	for eid, models in tqdm(sorted(tree.items()), desc="Experiments"):
		for mid, runs in sorted(models.items()):
			for rid, oids in sorted(runs.items()):
				for oid in oids:
					key = (eid, mid, oid, rid)
					loaded = _load_run_checkpoint(str(output_dir), eid, mid, oid, rid)
					if loaded is not None:
						out = loaded
						needs_save = False
						if reparse_digit_a:
							out, changed = reparse_digit_a_final_run(out, eid, mid, rid)
							needs_save = needs_save or changed
						if rescore:
							out = rescore_final_run(
								out,
								score_factors,
								use_minmax_normalized_robustness=minmax_normalize_robustness,
							)
							needs_save = True
						if needs_save:
							_save_run_checkpoint(str(output_dir), eid, mid, oid, rid, out)
						all_results[key] = out
						continue
					try:
						out = final_run_one(
							eid,
							mid,
							oid,
							rid,
							prebuilt_pyramids,
							expert_saliency_by_oid,
							project_root=project_root,
							save_dir=str(output_dir),
							n_checkpoint_samples=n_checkpoint_samples,
							strict=strict,
							btsp_start_strategy=btsp_start_strategy,
							use_real_progression=use_real_progression,
							expert_model_dir=expert_model_dir,
							score_factors=score_factors,
						)
					except Exception as exc:
						tb = traceback.format_exc()
						if verbose_errors:
							print(
								f"\n--- analyse failed: eid={eid!r} mid={mid!r} "
								f"rid={rid!r} oid={oid!r} ---",
								file=sys.stderr,
							)
							print(tb, file=sys.stderr)
						errors.append(
							{
								"eid": eid,
								"mid": mid,
								"rid": rid,
								"oid": oid,
								"error": repr(exc),
								"traceback": tb,
							}
						)
						continue
					if out.get("error"):
						errors.append(
							{"eid": eid, "mid": mid, "rid": rid, "oid": oid, "error": out["error"]}
						)
						continue
					_save_run_checkpoint(str(output_dir), eid, mid, oid, rid, out)
					all_results[key] = out

	if minmax_normalize_robustness:
		n_saved = minmax_normalize_robustness_all_results(
			all_results,
			score_factors,
			save_dir=str(output_dir),
		)
		if n_saved:
			print(
				f"Min-max normalized robustness across all runs and saved {n_saved} checkpoints "
				"(minmax_normalized_robustness_scores = (r - min) / (max - min))"
			)
		else:
			print(
				"Robustness already min-max normalized in cached runs; skipped min-max normalization"
			)

	return all_results, errors


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
