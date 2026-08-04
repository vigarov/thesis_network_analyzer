"""Load or compute FINAL optimizer-comparison analysis checkpoints."""
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from analysis.constants import set_results_root
from analysis.dataset_filter import prepare_results_tree
from analysis.final.final_analysis_config import dataset_key, model_ids_for_dataset
from analysis.final.final_compare_all_optimizers_helpers import (
	_load_run_checkpoint,
	_load_run_config,
	_parse_seed_from_config,
	_save_run_checkpoint,
	final_run_one,
	get_cifar_eval_inputs,
	minmax_normalize_robustness_all_results,
	reparse_digit_a_final_run,
	rescore_final_run,
)
from analysis.grad_nam import build_pre_gap_conv_cache_entry
from analysis.io import scan_results

ResultsTree = dict[str, dict[str, dict[str, list[str]]]]


def get_pre_gap_conv_acts(
	tree: ResultsTree,
	project_root: Path,
	*,
	experiment_id: str | None = None,
	optimizer_id: str | None = None,
) -> dict[tuple[str, str, int], Any]:
	"""Warm up pre-GAP conv activation cache for unique (mid, oid, seed) keys."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cache: dict[tuple[str, str, int], Any] = {}
	seen: set[tuple[str, str, int]] = set()

	for eid, models in tqdm(sorted(tree.items()), desc="Pre-GAP conv cache"):
		if experiment_id is not None and eid != experiment_id:
			continue
		for mid, runs in sorted(models.items()):
			for rid, oids in sorted(runs.items()):
				for oid in oids:
					if optimizer_id is not None and oid != optimizer_id:
						continue
					seed = _parse_seed_from_config(eid, mid, rid)
					key = (mid, oid, seed)
					if key in seen:
						continue
					seen.add(key)
					config = _load_run_config(eid, mid, rid)
					eval_inputs = get_cifar_eval_inputs(mid, device)
					cache[key] = build_pre_gap_conv_cache_entry(
						use_initial_model=config["use_initial_model"],
						model_class=config["model_class"],
						model_config=config["model_config"],
						mid=mid,
						oid=oid,
						seed=seed,
						eval_inputs=eval_inputs,
						device=device,
					)
	return cache


def run_final_analysis(
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
	dataset: str,
	experiment_id: str | None = None,
	optimizer_id: str | None = None,
	verbose_errors: bool = True,
) -> tuple[
	dict[tuple[str, str, str, str], dict[str, Any]],
	ResultsTree,
	list[dict[str, Any]],
	str,
	list[str],
]:
	ds_key = dataset_key(dataset)
	set_results_root(input_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	tree = scan_results()
	tree, skipped = prepare_results_tree(tree, ds_key)
	for msg in skipped:
		print(f"Skipping CIFAR experiment: {msg}")

	prebuilt_pyramids: dict = {}
	pre_gap_conv_cache: dict[tuple[str, str, int], Any] | None = None
	if ds_key == "cifar":
		pre_gap_conv_cache = get_pre_gap_conv_acts(
			tree,
			project_root,
			experiment_id=experiment_id,
			optimizer_id=optimizer_id,
		)

	expert_saliency_by_oid: dict[str, dict] | None = {}
	all_results: dict[tuple[str, str, str, str], dict] = {}
	errors: list[dict[str, Any]] = []

	for eid, models in tqdm(sorted(tree.items()), desc="Experiments"):
		if experiment_id is not None and eid != experiment_id:
			continue
		for mid, runs in sorted(models.items()):
			for rid, oids in sorted(runs.items()):
				for oid in oids:
					if optimizer_id is not None and oid != optimizer_id:
						continue
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
							dataset=ds_key,
							pre_gap_conv_cache=pre_gap_conv_cache,
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
				f"Min-max normalized robustness (per model_id) across runs and saved "
				f"{n_saved} checkpoints "
				"(minmax_normalized_robustness_scores = (r - min) / (max - min))"
			)
		else:
			print(
				"Robustness already min-max normalized in cached runs; skipped min-max normalization"
			)

	model_ids = sorted(
		{mid for (_, mid, _, _) in all_results} or model_ids_for_dataset(dataset)
	)
	return all_results, tree, errors, ds_key, model_ids
