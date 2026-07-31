import re
from typing import Any

MNIST_MODEL_ID = "dnn_5x64"
INCEPTION_MODEL_ID = "inception_small_dnn_5x64"
RESNET_MODEL_ID = "resnet32_dnn_5x64"
CIFAR_MODEL_IDS = frozenset({INCEPTION_MODEL_ID, RESNET_MODEL_ID})

# CIFAR cat1/cat2 `_K` experiments.
CIFAR_K_BY_MID = {
	INCEPTION_MODEL_ID: 6000,
	RESNET_MODEL_ID: 15000,
}

_SPLIT_K_EID_RE = re.compile(r"_K(\d+)_")

ResultsTree = dict[str, dict[str, dict[str, list[str]]]]


def mid_to_model_class(mid: str) -> str:
	if mid == INCEPTION_MODEL_ID:
		return "InceptionCIFAR"
	if mid == RESNET_MODEL_ID:
		return "ResNetCIFAR"
	if mid == MNIST_MODEL_ID:
		return "DNN5Hidden64"
	raise ValueError(f"Unknown model_id for analysis: {mid!r}")


def mid_to_arch_slug(mid: str) -> str:
	if mid.startswith("inception"):
		return "inception"
	if mid.startswith("resnet"):
		return "resnet"
	raise ValueError(f"Cannot derive architecture slug from model_id: {mid!r}")


def is_cifar_mid(mid: str) -> bool:
	return mid in CIFAR_MODEL_IDS


def k_value_from_eid(eid: str) -> int | None:
	"""Extract `K` from `..._K<number>_...` experiment ids, else `None`."""
	match = _SPLIT_K_EID_RE.search(eid)
	if match is None:
		return None
	return int(match.group(1))


def logical_cifar_split_k_eid(eid: str) -> str | None:
	"""Logical experiment key for split-K CIFAR runs (strip `_K<number>_` once)."""
	if k_value_from_eid(eid) is None:
		return None
	return _SPLIT_K_EID_RE.sub("_", eid, count=1)


def filter_tree_by_dataset(tree: ResultsTree, dataset: str) -> ResultsTree:
	"""Keep only model folders matching `dataset` (`mnist` or `cifar`)."""
	out: ResultsTree = {}
	for eid, models in tree.items():
		filtered_models: dict[str, dict[str, list[str]]] = {}
		for mid, runs in models.items():
			if dataset == "mnist":
				if mid != MNIST_MODEL_ID:
					continue
			elif dataset == "cifar":
				if mid not in CIFAR_MODEL_IDS:
					continue
			else:
				raise ValueError(f"Unsupported dataset: {dataset!r}")
			filtered_models[mid] = runs
		if filtered_models:
			out[eid] = filtered_models
	return out


def _oids_for_mid(models: dict[str, dict[str, list[str]]], mid: str) -> set[str]:
	oid_set: set[str] = set()
	for oids in models[mid].values():
		oid_set.update(oids)
	return oid_set


def _optimizer_sets_match(
	models_a: dict[str, dict[str, list[str]]],
	mid_a: str,
	models_b: dict[str, dict[str, list[str]]],
	mid_b: str,
) -> tuple[bool, set[str], set[str]]:
	oids_a = _oids_for_mid(models_a, mid_a)
	oids_b = _oids_for_mid(models_b, mid_b)
	return bool(oids_a) and oids_a == oids_b, oids_a, oids_b


def require_cifar_experiment_complete(tree: ResultsTree) -> tuple[ResultsTree, list[str]]:
	"""Drop experiments missing either CIFAR architecture or mismatched optimizer sets.

	Unified layout: both architectures under one `eid` (e.g. `cat2_sequence_control_tr100`).

	Split-K layout: Inception lives under `..._K6000_...` and ResNet under `..._K15000_...`
	with the same logical name after stripping the K segment.
	"""
	skipped: list[str] = []
	out: ResultsTree = {}
	split_groups: dict[str, dict[str, tuple[str, dict[str, dict[str, list[str]]]]]] = {}

	for eid, models in tree.items():
		if INCEPTION_MODEL_ID in models and RESNET_MODEL_ID in models:
			match, inception_oids, resnet_oids = _optimizer_sets_match(
				models, INCEPTION_MODEL_ID, models, RESNET_MODEL_ID
			)
			if not match:
				skipped.append(
					f"{eid!r}: optimizer mismatch "
					f"(inception={sorted(inception_oids)!r}, resnet={sorted(resnet_oids)!r})"
				)
				continue
			out[eid] = models
			continue

		logical = logical_cifar_split_k_eid(eid)
		if logical is None:
			skipped.append(
				f"{eid!r}: missing one of {INCEPTION_MODEL_ID!r} / {RESNET_MODEL_ID!r}"
			)
			continue

		if len(models) != 1:
			skipped.append(
				f"{eid!r}: split-K CIFAR eid must contain exactly one model, "
				f"got {sorted(models)!r}"
			)
			continue

		mid = next(iter(models))
		expected_k = CIFAR_K_BY_MID.get(mid)
		actual_k = k_value_from_eid(eid)
		if expected_k is not None and actual_k != expected_k:
			skipped.append(
				f"{eid!r}: expected K{expected_k} for {mid!r}, found K{actual_k}"
			)
			continue

		group = split_groups.setdefault(logical, {})
		if mid in group:
			prev_eid, _ = group[mid]
			skipped.append(
				f"{logical!r}: duplicate {mid!r} entries ({prev_eid!r} and {eid!r})"
			)
			continue
		group[mid] = (eid, models)

	for logical, members in split_groups.items():
		if INCEPTION_MODEL_ID not in members or RESNET_MODEL_ID not in members:
			physical = {mid: physical_eid for mid, (physical_eid, _) in members.items()}
			skipped.append(
				f"{logical!r}: incomplete split-K CIFAR pair "
				f"(physical eids={physical!r})"
			)
			continue

		inception_eid, inception_models = members[INCEPTION_MODEL_ID]
		resnet_eid, resnet_models = members[RESNET_MODEL_ID]
		match, inception_oids, resnet_oids = _optimizer_sets_match(
			inception_models,
			INCEPTION_MODEL_ID,
			resnet_models,
			RESNET_MODEL_ID,
		)
		if not match:
			skipped.append(
				f"{logical!r}: optimizer mismatch across split-K pair "
				f"({inception_eid!r} inception={sorted(inception_oids)!r}, "
				f"{resnet_eid!r} resnet={sorted(resnet_oids)!r})"
			)
			continue

		out[inception_eid] = inception_models
		out[resnet_eid] = resnet_models

	return out, skipped


def prepare_results_tree(tree: ResultsTree, dataset: str) -> tuple[ResultsTree, list[str]]:
	"""Filter by dataset; for CIFAR apply experiment completeness gate."""
	filtered = filter_tree_by_dataset(tree, dataset)
	if dataset != "cifar":
		return filtered, []
	return require_cifar_experiment_complete(filtered)
