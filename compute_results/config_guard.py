"""Config schema, fingerprinting, and the compare-existing-config guard."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

import experiments  # noqa: F401 – side-effect: registers all experiment classes
from compute_results.constants import (
	ALL_AVAILABLE_SELECTION,
	FINGERPRINT_EXCLUDE_KEYS,
	INITIAL_MODEL_CHECKPOINT_FILENAMES,
	INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS,
	INITIAL_MODEL_OPTIMIZER_STATE_FILENAME,
)
from experiments.base import get_registered_experiment_class
from experiments_new.mnist.common.label_perm import parse_restrain_digits


def parse_checkpoint_cadence(cadence: str) -> tuple[str, int | None]:
	"""Parse checkpoint_cadence into a (mode, interval) pair.

	Returns:
		- ("every_epoch", None) for "every_epoch"
		- ("every_k_its", K) for "K its" or "K iterations" where K is a positive int

	Raises:
		ValueError: if cadence is not recognized.
	"""
	s = cadence.strip()
	if s.lower() == "every_epoch":
		return ("every_epoch", None)
	m = re.match(r"^(\d+)\s+(?:its|iterations?)\s*$", s, re.IGNORECASE)
	if m:
		k = int(m.group(1))
		if k <= 0:
			raise ValueError(f"checkpoint_cadence interval must be positive, got {k}")
		return ("every_k_its", k)
	raise ValueError(
		f"Invalid checkpoint_cadence: {cadence!r}. "
		"Use 'every_epoch' or 'K its' / 'K iterations' (e.g. '100 its')."
	)


def normalize_loss(loss: str) -> str:
	"""Map a user-facing loss name to a canonical key: ``ce`` or ``mse``."""
	s = loss.strip().lower().replace("_", " ")
	s = re.sub(r"\s+", " ", s)
	if s in ("ce", "cross entropy", "crossentropy"):
		return "ce"
	if s == "mse":
		return "mse"
	raise ValueError(
		f"Invalid loss: {loss!r}. Use 'ce', 'cross entropy', or 'mse'."
	)


def expand_registry_selection(
	names: list[str],
	*,
	available: list[str],
	selection: str = ALL_AVAILABLE_SELECTION,
) -> list[str]:
	"""Simply expand the "all" option to the full available set.
	"""
	if not names:
		raise ValueError("expand_registry_selection: names must be non-empty")
	selection_l = selection.lower()
	has_sentinel = any(n.strip().lower() == selection_l for n in names)
	if not has_sentinel:
		return names
	explicit = [
		n.strip()
		for n in names
		if n.strip() and n.strip().lower() != selection_l
	]
	avail_set = set(available)
	merged = set(explicit) | avail_set
	return sorted(merged)


def _apply_previous_training_config_defaults(stored: dict[str, Any]) -> None:
	"""Fill missing keys so older config.json matches new default fingerprint fields.
	Needed because the config changes over time -> do not require re-training for functionnally equivalent configs."""
	stored.setdefault("save_model_cp", False)
	stored.setdefault("use_initial_model", "")
	stored.setdefault("initial_model_mode", "init")
	stored.setdefault("initial_model_sha256", "")


_DEPRECATED_ALL_KEYS: frozenset[str] = frozenset({
	"samples_per_digit",
})

_DEPRECATED_ALL_HINTS: dict[str, str] = {
	"samples_per_digit": (
		"'samples_per_digit' has been removed. Control training sample counts via "
		"'pretrain_on_k_samples' and 'num_trial_samples' instead."
	),
}

_DEPRECATED_PRETRAIN_KEYS: frozenset[str] = frozenset({
	"base_ratio",
	"num_pretrain_samples",
	"num_stage_samples",
})

_DEPRECATED_PRETRAIN_HINTS: dict[str, str] = {
	"base_ratio": (
		"Use 'pretrain_on_k_samples' (total balanced K, divisible by 10) instead."
	),
	"num_pretrain_samples": (
		"Use 'pretrain_on_k_samples' instead "
		"(pretrain_on_k_samples = 10 * old_num_pretrain_samples)."
	),
	"num_stage_samples": (
		"Use 'num_trial_samples' instead."
	),
}


def validate_experiment_config_constraints(
	experiment_class: str,
	experiment_config: dict[str, Any],
	*,
	experiment_runs: int | None = None,
) -> None:
	"""Validate experiment-specific config constraints.

	For experiments with ``has_pretrain=True``:
	- Rejects deprecated keys: ``base_ratio``, ``num_pretrain_samples``, ``num_stage_samples``.
	- Requires ``pretrain_on_k_samples``: ``int``, ``>= 1``, divisible by 10.
	- Requires ``num_trial_samples``: ``int``, ``>= 1``.
	- For ``PretrainThenShuffleMislabel``, when *experiment_runs* is provided,
	  requires ``(num_trial_samples * experiment_runs) % N == 0`` where ``N`` is 10
	  or ``len(restrain_digits)`` when ``restrain_digits`` is set.
	- For ``PretrainControlBase``, requires ``num_trial_samples % 10 == 0`` (balanced remainder trials).

	For ``ControlBase`` (no pretrain flag): requires ``pretrain_on_k_samples`` (K split),
	``num_trial_samples``, and ``num_trial_samples % 10 == 0``.

	Raises:
		KeyError: if ``experiment_class`` is not registered.
		ValueError / TypeError: on constraint violations.
	"""
	# --- reject globally deprecated keys (all experiment types, even unknown ones) ---
	for key in _DEPRECATED_ALL_KEYS:
		if key in experiment_config:
			hint = _DEPRECATED_ALL_HINTS.get(key, "")
			raise ValueError(
				f"experiment_config for {experiment_class}: '{key}' is no longer supported. "
				+ hint
			)

	try:
		cls = get_registered_experiment_class(experiment_class)
	except KeyError:
		# Unknown class — let the caller (run_simulation) produce the appropriate error.
		return

	if "restrain_digits" in experiment_config:
		if cls.__name__ not in ("PretrainThenShuffleMislabel", "Cat1SampleShuffleConstrained"):
			raise ValueError(
				f"experiment_config for {experiment_class}: 'restrain_digits' is only supported "
				"for PretrainThenShuffleMislabel and Cat1SampleShuffleConstrained."
			)
		parse_restrain_digits(experiment_config.get("restrain_digits"))

	if not getattr(cls, "has_pretrain", False):
		if cls.__name__ == "ControlBase":
			pk = experiment_config.get("pretrain_on_k_samples")
			if pk is None:
				raise ValueError(
					f"experiment_config for {experiment_class}: 'pretrain_on_k_samples' is required."
				)
			if not isinstance(pk, int):
				raise TypeError(
					f"experiment_config for {experiment_class}: 'pretrain_on_k_samples' must be int, "
					f"got {type(pk).__name__}"
				)
			if pk < 1:
				raise ValueError(
					f"experiment_config for {experiment_class}: 'pretrain_on_k_samples' must be >= 1, got {pk}"
				)
			if pk % 10 != 0:
				raise ValueError(
					f"experiment_config for {experiment_class}: 'pretrain_on_k_samples' must be divisible "
					f"by 10, got {pk}"
				)
			nt_cb = experiment_config.get("num_trial_samples")
			if nt_cb is None:
				raise ValueError(
					f"experiment_config for {experiment_class}: 'num_trial_samples' is required."
				)
			if not isinstance(nt_cb, int):
				raise TypeError(
					f"experiment_config for {experiment_class}: 'num_trial_samples' must be int, "
					f"got {type(nt_cb).__name__}"
				)
			if nt_cb < 1:
				raise ValueError(
					f"experiment_config for {experiment_class}: 'num_trial_samples' must be >= 1, got {nt_cb}"
				)
			if nt_cb % 10 != 0:
				raise ValueError(
					f"experiment_config for {experiment_class}: 'num_trial_samples' must be divisible by 10 "
					f"(balanced remainder trials); got {nt_cb}."
				)
		elif cls.__name__ == "Cat1SampleShuffleControl":
			nt_cb = experiment_config.get("num_trial_samples")
			if nt_cb is None:
				raise ValueError(
					f"experiment_config for {experiment_class}: 'num_trial_samples' is required."
				)
			if not isinstance(nt_cb, int):
				raise TypeError(
					f"experiment_config for {experiment_class}: 'num_trial_samples' must be int, "
					f"got {type(nt_cb).__name__}"
				)
			if nt_cb < 1:
				raise ValueError(
					f"experiment_config for {experiment_class}: 'num_trial_samples' must be >= 1, got {nt_cb}"
				)
			if nt_cb % 10 != 0:
				raise ValueError(
					f"experiment_config for {experiment_class}: 'num_trial_samples' must be divisible by 10; "
					f"got {nt_cb}."
				)
		elif cls.__name__ == "Cat2SequenceControl":
			nt_cb = experiment_config.get("num_trial_samples")
			if nt_cb is None:
				raise ValueError(
					f"experiment_config for {experiment_class}: 'num_trial_samples' is required."
				)
			if not isinstance(nt_cb, int):
				raise TypeError(
					f"experiment_config for {experiment_class}: 'num_trial_samples' must be int, "
					f"got {type(nt_cb).__name__}"
				)
			if nt_cb < 1:
				raise ValueError(
					f"experiment_config for {experiment_class}: 'num_trial_samples' must be >= 1, got {nt_cb}"
				)
		return

	# --- reject deprecated keys (has_pretrain experiments) ---
	for key in _DEPRECATED_PRETRAIN_KEYS:
		if key in experiment_config:
			hint = _DEPRECATED_PRETRAIN_HINTS.get(key, "")
			raise ValueError(
				f"experiment_config for {experiment_class}: '{key}' is no longer supported. "
				+ hint
			)

	# --- require pretrain_on_k_samples ---
	pk = experiment_config.get("pretrain_on_k_samples")
	if pk is None:
		raise ValueError(
			f"experiment_config for {experiment_class}: 'pretrain_on_k_samples' is required."
		)
	if not isinstance(pk, int):
		raise TypeError(
			f"experiment_config for {experiment_class}: 'pretrain_on_k_samples' must be int, "
			f"got {type(pk).__name__}"
		)
	if pk < 1:
		raise ValueError(
			f"experiment_config for {experiment_class}: 'pretrain_on_k_samples' must be >= 1, got {pk}"
		)
	n_div_pretrain = 10
	if cls.__name__ == "PretrainThenShuffleMislabel":
		rd = parse_restrain_digits(experiment_config.get("restrain_digits"))
		n_div_pretrain = len(rd) if rd is not None else 10
	elif cls.__name__ == "Cat1SampleShuffleConstrained":
		rd = parse_restrain_digits(experiment_config.get("restrain_digits"))
		if rd is None:
			raise ValueError(
				f"experiment_config for {experiment_class}: "
				f"'restrain_digits' is required for Cat1SampleShuffleConstrained."
			)
		n_div_pretrain = len(rd)
	if pk % n_div_pretrain != 0:
		raise ValueError(
			f"experiment_config for {experiment_class}: 'pretrain_on_k_samples' must be divisible "
			f"by {n_div_pretrain} (balanced K over each training digit), got {pk}"
		)

	# --- require num_trial_samples ---
	nt = experiment_config.get("num_trial_samples")
	if nt is None:
		raise ValueError(
			f"experiment_config for {experiment_class}: 'num_trial_samples' is required."
		)
	if not isinstance(nt, int):
		raise TypeError(
			f"experiment_config for {experiment_class}: 'num_trial_samples' must be int, "
			f"got {type(nt).__name__}"
		)
	if nt < 1:
		raise ValueError(
			f"experiment_config for {experiment_class}: 'num_trial_samples' must be >= 1, got {nt}"
		)

	if cls.__name__ in ("Cat1SampleShuffleFinetune", "Cat1SampleShuffleConstrained"):
		if int(nt) % n_div_pretrain != 0:
			raise ValueError(
				f"experiment_config for {experiment_class}: 'num_trial_samples' must be divisible "
				f"by {n_div_pretrain}; got {nt}."
			)

	if cls.__name__ in (
		"PretrainThenShuffleMislabel",
		"Cat1SampleShuffleFinetune",
		"Cat1SampleShuffleConstrained",
	) and experiment_runs is not None:
		prod = int(nt) * int(experiment_runs)
		if prod % n_div_pretrain != 0:
			raise ValueError(
				f"experiment_config for {experiment_class}: "
				f"num_trial_samples * experiment_runs must be divisible by {n_div_pretrain} "
				f"(balanced post-pretrain pool); got {nt} * {experiment_runs} = {prod}."
			)

	if cls.__name__ == "PretrainControlBase" and int(nt) % 10 != 0:
		raise ValueError(
			f"experiment_config for {experiment_class}: 'num_trial_samples' must be divisible by 10 "
			f"(balanced remainder trials); got {nt}."
		)


def normalize_initial_model_mode(mode: str) -> str:
	"""Return ``init`` or ``pretrain``."""
	m = mode.strip().lower()
	if m in ("init", "pretrain"):
		return m
	raise ValueError(
		f"initial_model_mode must be 'init' or 'pretrain', got {mode!r}."
	)


def _resolve_use_initial_model_base(path_str: str) -> Path:
	raw = path_str.strip()
	if not raw:
		raise ValueError("use_initial_model path is empty.")
	path = Path(raw)
	if not path.is_absolute():
		path = Path.cwd() / path
	return path.resolve()


def _resolve_initial_model_file(path_str: str) -> Path:
	path = _resolve_use_initial_model_base(path_str)
	if not path.is_file():
		raise ValueError(f"use_initial_model path is not a file: {path}")
	if path.suffix.lower() not in (".pt", ".pth"):
		raise ValueError(
			f"use_initial_model must point to a .pt or .pth file, got suffix {path.suffix!r}."
		)
	return path


def _bundle_subdir_has_checkpoint(sub: Path) -> bool:
	return any((sub / fn).is_file() for fn in INITIAL_MODEL_CHECKPOINT_FILENAMES)


def _initial_bundle_probe_order(registry_class_name: str) -> list[tuple[str, bool]]:
	"""(probe_name, is_exact_registry_class). Registry class is matched by exact dirname only;
	shorthands match any immediate child directory whose name *starts with* the shorthand."""
	rc = registry_class_name.strip()
	if not rc:
		return []
	out: list[tuple[str, bool]] = [(rc, True)]
	seen_shorthand: set[str] = set()
	for shorthand, cls in sorted(INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS.items()):
		if cls == rc and shorthand not in seen_shorthand:
			seen_shorthand.add(shorthand)
			out.append((shorthand, False))
	return out


def _bundle_subdirs_for_probe(
	bundle_root: Path, probe_name: str, *, exact: bool
) -> list[Path]:
	"""Resolve to zero or one subfolder path(s) for this probe (see ``_initial_bundle_probe_order``)."""
	if exact:
		sub = bundle_root / probe_name
		return [sub] if sub.is_dir() else []
	matches = sorted(
		p
		for p in bundle_root.iterdir()
		if p.is_dir() and p.name.startswith(probe_name)
	)
	if len(matches) > 1:
		names = [m.name for m in matches]
		raise ValueError(
			f"use_initial_model directory {bundle_root}: multiple subfolders start with "
			f"shorthand {probe_name!r}: {names!r}; use a single directory per optimizer."
		)
	return matches


def _checkpoint_in_bundle_subdir(bundle_root: Path, registry_class_name: str) -> Path:
	"""Resolve ``bundle_root / <subfolder> / model.pt`` (or ``model.pth``).

	The registry class directory name is matched **exactly**. Each CLI shorthand from
	``INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS`` matches a child directory whose name
	**starts with** that shorthand. Probes run in order: class name first, then shorthands
	(sorted by shorthand key).
	"""
	rc = registry_class_name.strip()
	if not rc:
		raise ValueError("empty extractor class name for use_initial_model bundle.")
	probe_order = _initial_bundle_probe_order(rc)
	last_nonempty_dir: str | None = None
	for probe_name, exact in probe_order:
		subs = _bundle_subdirs_for_probe(bundle_root, probe_name, exact=exact)
		if not subs:
			continue
		sub = subs[0]
		last_nonempty_dir = sub.name
		present = [
			sub / fn for fn in INITIAL_MODEL_CHECKPOINT_FILENAMES if (sub / fn).is_file()
		]
		if len(present) > 1:
			raise ValueError(
				f"use_initial_model directory {bundle_root}: subfolder {sub.name!r} must contain "
				f"at most one of {INITIAL_MODEL_CHECKPOINT_FILENAMES}, found multiple."
			)
		if len(present) == 1:
			for other_name, other_exact in probe_order:
				for opath in _bundle_subdirs_for_probe(
					bundle_root, other_name, exact=other_exact
				):
					if opath.resolve() == sub.resolve():
						continue
					if _bundle_subdir_has_checkpoint(opath):
						raise ValueError(
							f"use_initial_model directory {bundle_root}: ambiguous layout for "
							f"optimizer {rc!r}: both {sub.name!r} and {opath.name!r} contain one of "
							f"{INITIAL_MODEL_CHECKPOINT_FILENAMES}; keep a single subfolder."
						)
			return present[0]
	if last_nonempty_dir is None:
		probes = [p[0] for p in probe_order]
		raise FileNotFoundError(
			f"use_initial_model directory {bundle_root}: no subfolder for {rc!r}; "
			f"tried exact class name and shorthand prefixes {probes!r}, each with one of "
			f"{INITIAL_MODEL_CHECKPOINT_FILENAMES} inside."
		)
	raise ValueError(
		f"use_initial_model directory {bundle_root}: subfolder {last_nonempty_dir!r} exists but "
		f"must contain exactly one of {INITIAL_MODEL_CHECKPOINT_FILENAMES}."
	)


def _validate_initial_model_directory(
	root: Path,
	registry_classes: list[str],
) -> str:
	if not registry_classes:
		raise ValueError(
			"use_initial_model is a directory; provide optimizer_registry_classes_for_initial_model "
			"(one registry class per requested optimizer, same order as --optimizer)."
		)
	seen_registry_classes: set[str] = set()
	used_disk_subfolders: set[str] = set()
	resolved_disk_folder: dict[str, str] = {}
	parts: list[tuple[str, str, str]] = []
	for registry_class in registry_classes:
		rc = registry_class.strip()
		if rc in seen_registry_classes:
			raise ValueError(
				f"use_initial_model directory {root}: duplicate optimizer {rc!r} in the run."
			)
		seen_registry_classes.add(rc)
		ckpt = _checkpoint_in_bundle_subdir(root, rc)
		disk_folder = ckpt.parent.name
		if disk_folder in used_disk_subfolders:
			raise ValueError(
				f"use_initial_model directory {root}: two optimizers resolved to the same "
				f"subfolder {disk_folder!r}."
			)
		used_disk_subfolders.add(disk_folder)
		resolved_disk_folder[rc] = disk_folder
		model_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
		torch.load(ckpt, map_location="cpu", weights_only=True)
		opt_ckpt = ckpt.parent / INITIAL_MODEL_OPTIMIZER_STATE_FILENAME
		if not opt_ckpt.is_file():
			raise ValueError(
				f"use_initial_model directory {root}: subfolder {disk_folder!r} must contain "
				f"{INITIAL_MODEL_OPTIMIZER_STATE_FILENAME!r} (optimizer state dict), "
				f"but it was not found."
			)
		opt_sha = hashlib.sha256(opt_ckpt.read_bytes()).hexdigest()
		torch.load(opt_ckpt, map_location="cpu", weights_only=False)
		parts.extend([(rc, "model", model_sha), (rc, "optimizer", opt_sha)])

	canonical = json.dumps(sorted(parts), sort_keys=True)
	return hashlib.sha256(canonical.encode()).hexdigest()


def validate_initial_model_path(
	path_str: str,
	*,
	optimizer_registry_classes: list[str] | None = None,
) -> str:
	"""Verify *path_str* and return a SHA-256 fingerprint for the initial weights.

	- Single file: must be a readable ``.pt`` or ``.pth`` checkpoint
	  (``torch.save(model.state_dict(), ...)``); returns SHA-256 of file bytes.
	- Directory: for each entry in *optimizer_registry_classes* (resolved extractor
	  class per ``--optimizer`` entry, e.g. ``AdamExtractor``), there must be a
	  subfolder whose name equals that class, or (if no class-named folder matches)
	  whose name **starts with** a CLI shorthand key from
	  ``INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS``. Each subfolder must contain:
		- exactly one of ``model.pt`` or ``model.pth`` (model weights),
		- ``optimizer.pt`` (optimizer state dict, ``torch.save(opt.state_dict(), ...)``).
	  Class name is tried before shorthand prefixes. Extra subfolders are ignored.
	  Returns a deterministic hash of sorted
	  ``(registry_class, "model"|"optimizer", sha256)`` triples.
	"""
	path = _resolve_use_initial_model_base(path_str)
	if path.is_file():
		path = _resolve_initial_model_file(path_str)
		digest = hashlib.sha256(path.read_bytes()).hexdigest()
		torch.load(path, map_location="cpu", weights_only=True)
		return digest
	if path.is_dir():
		return _validate_initial_model_directory(path, list(optimizer_registry_classes or ()))
	raise ValueError(
		f"use_initial_model must be a .pt/.pth file or a directory of per-optimizer checkpoints, "
		f"got: {path}"
	)


def _require_bundle_registry_class(
	where: str, registry_class_name: str
) -> str:
	rc = registry_class_name.strip()
	if not rc:
		raise ValueError(
			f"{where}: directory bundle requires registry_class_name "
			"(extractor class, e.g. AdamExtractor)."
		)
	return rc


def load_initial_model_state_dict(
	path_str: str,
	*,
	registry_class_name: str = "",
) -> dict[str, Any]:
	"""Load model ``state_dict`` from *path_str* (relative to cwd if not absolute).

	If *path_str* is a directory, *registry_class_name* selects the optimizer; the
	subdirectory is resolved with the same rules as validation (exact class dirname,
	else a child directory whose name starts with a configured shorthand).
	"""
	base = _resolve_use_initial_model_base(path_str)
	if base.is_file():
		path = _resolve_initial_model_file(path_str)
		return torch.load(path, map_location="cpu", weights_only=True)
	if base.is_dir():
		rc = _require_bundle_registry_class("load_initial_model_state_dict", registry_class_name)
		ckpt = _checkpoint_in_bundle_subdir(base, rc)
		return torch.load(ckpt, map_location="cpu", weights_only=True)
	raise ValueError(f"use_initial_model path is not a file or directory: {base}")


def load_initial_optimizer_state_dict(
	path_str: str,
	*,
	registry_class_name: str,
	map_location: Any = "cpu",
) -> dict[str, Any]:
	"""Load optimizer ``state_dict`` from a bundle directory.

	*path_str* must be a directory (only meaningful for bundle layouts). The subfolder
	is resolved by the same rules as ``load_initial_model_state_dict``; ``optimizer.pt``
	inside that subfolder is loaded with ``weights_only=False`` (optimizer state dicts
	include non-tensor ``param_groups`` entries).
	"""
	base = _resolve_use_initial_model_base(path_str)
	if not base.is_dir():
		raise ValueError(
			f"load_initial_optimizer_state_dict: expected a bundle directory, got: {base}"
		)
	rc = _require_bundle_registry_class("load_initial_optimizer_state_dict", registry_class_name)
	ckpt = _checkpoint_in_bundle_subdir(base, rc)
	opt_path = ckpt.parent / INITIAL_MODEL_OPTIMIZER_STATE_FILENAME
	if not opt_path.is_file():
		raise ValueError(
			f"load_initial_optimizer_state_dict: {INITIAL_MODEL_OPTIMIZER_STATE_FILENAME!r} "
			f"not found in {ckpt.parent}"
		)
	return torch.load(opt_path, map_location=map_location, weights_only=False)


def fingerprint_payload(config: dict[str, Any]) -> dict[str, Any]:
	"""Training-relevant subset of *config* used for hashing and equality checks."""
	return {k: v for k, v in config.items() if k not in FINGERPRINT_EXCLUDE_KEYS}


def compute_fingerprint(config: dict[str, Any]) -> str:
	"""Full SHA-256 hex digest of the fingerprint payload (deterministic)."""
	payload = fingerprint_payload(config)
	canonical = json.dumps(payload, sort_keys=True, default=str)
	return hashlib.sha256(canonical.encode()).hexdigest()


def result_id_for_config(
	config: dict[str, Any],
	*,
	experiment_dir: Path,
	model_id: str,
) -> str:
	"""Directory name under *experiment_dir* for this training config (first 6 hex of digest).

	If ``experiment_dir / <id> / model_id`` already exists with a different training
	payload, tries ``<prefix>_2``, ``<prefix>_3``, ...
	"""
	prefix = compute_fingerprint(config)[:6]
	for i in range(1, 10_000):
		rid = prefix if i == 1 else f"{prefix}_{i}"
		results_dir = experiment_dir / rid / model_id
		config_path = results_dir / "config.json"
		if not config_path.exists():
			return rid
		stored = dict(json.loads(config_path.read_text()))
		_apply_previous_training_config_defaults(stored)
		if fingerprint_payload(config) == fingerprint_payload(stored):
			return rid
	raise ConfigConflictError(
		f"Could not allocate a result_id under {experiment_dir} for model {model_id!r} "
		f"(too many fingerprint collisions for prefix {prefix!r})."
	)


def build_training_config(
	*,
	experiment_class: str,
	experiment_config: dict[str, Any],
	model_class: str,
	model_config: dict[str, Any],
	trial_epochs: int = 1,
	base_lr: float = 1e-3,
	batch_size: int = 1,
	seed: int = 3003,
	activation: str = "relu",
	checkpoint_cadence: str = "every_epoch",
	save_model_cp: bool = False,
	he_init: int | str | float = 3,
	init_epsilon: float = 1e-10,
	internal_keep_tensors: bool = False,
	experiment_runs: int = 3,
	experiment_variability: str = "",
	loss: str = "ce",
	use_initial_model: str = "",
	initial_model_mode: str = "init",
	save_model: str = "",
	optimizer_registry_classes_for_initial_model: list[str] | None = None,
) -> dict[str, Any]:
	parse_checkpoint_cadence(checkpoint_cadence)
	if experiment_runs < 1:
		raise ValueError(f"experiment_runs must be a positive integer, got {experiment_runs}")
	loss_key = normalize_loss(loss)
	mode_key = normalize_initial_model_mode(initial_model_mode)
	path_for_io = str(use_initial_model).strip()
	sha = ""
	if path_for_io:
		sha = validate_initial_model_path(
			path_for_io,
			optimizer_registry_classes=optimizer_registry_classes_for_initial_model,
		)
	config: dict[str, Any] = {
		"experiment_class": experiment_class,
		"experiment_config": experiment_config,
		"model_class": model_class,
		"model_config": model_config,
		"trial_epochs": trial_epochs,
		"base_lr": base_lr,
		"batch_size": batch_size,
		"seed": seed,
		"activation": activation,
		"checkpoint_cadence": checkpoint_cadence,
		"save_model_cp": save_model_cp,
		"he_init": he_init,
		"init_epsilon": init_epsilon,
		"internal_keep_tensors": internal_keep_tensors,
		"experiment_runs": experiment_runs,
		"experiment_variability": experiment_variability,
		"loss": loss_key,
		"use_initial_model": use_initial_model,
		"initial_model_mode": mode_key,
		"initial_model_sha256": sha,
		"save_model": str(save_model).strip(),
	}
	config["training_config_fingerprint"] = compute_fingerprint(config)
	return config


class ConfigConflictError(Exception):
	"""Raised when an incoming config conflicts with a stored one."""


def _diff_fingerprint_payloads(
	incoming: dict[str, Any],
	stored: dict[str, Any],
) -> list[str]:
	pin = fingerprint_payload(incoming)
	pst = fingerprint_payload(stored)
	diffs: list[str] = []
	for k in sorted(set(pin) | set(pst)):
		v_in = pin.get(k)
		v_st = pst.get(k)
		if v_in != v_st:
			diffs.append(f"  {k}: incoming={v_in!r}  stored={v_st!r}")
	return diffs


def guard_training_config(results_dir: Path, config: dict[str, Any]) -> None:
	"""Check that *config* matches any previously stored training config on disk.

	Always raises ConfigConflictError on fingerprint-payload mismatch (``--force`` does
	not bypass this; it only affects optimizer reruns and optimizer config guards).
	"""
	config_path = results_dir / "config.json"
	if not config_path.exists():
		return

	stored = dict(json.loads(config_path.read_text()))
	_apply_previous_training_config_defaults(stored)
	if fingerprint_payload(config) == fingerprint_payload(stored):
		return

	diffs = _diff_fingerprint_payloads(config, stored)
	msg = (
		f"Incoming training config conflicts with stored config at "
		f"{config_path}:\n" + "\n".join(diffs) + "\n"
		"Training config mismatches cannot be overridden with --force."
	)
	raise ConfigConflictError(msg)


def guard_optimizer_config(
	optimizer_dir: Path,
	optimizer_config: dict[str, Any],
	*,
	force: bool = False,
) -> None:
	"""Check optimizer-specific hyperparams against stored version."""
	opt_config_path = optimizer_dir / "optimizer_config.json"
	if not opt_config_path.exists():
		return

	stored = json.loads(opt_config_path.read_text())
	diffs: list[str] = []
	all_keys = sorted(set(optimizer_config) | set(stored))
	for k in all_keys:
		if optimizer_config.get(k) != stored.get(k):
			diffs.append(
				f"  {k}: incoming={optimizer_config.get(k)!r}  "
				f"stored={stored.get(k)!r}"
			)

	if diffs and not force:
		msg = (
			f"Incoming optimizer config conflicts with stored config at "
			f"{opt_config_path}:\n" + "\n".join(diffs) + "\n"
			"Use --force to overwrite."
		)
		raise ConfigConflictError(msg)


def save_config(results_dir: Path, config: dict[str, Any]) -> None:
	results_dir.mkdir(parents=True, exist_ok=True)
	(results_dir / "config.json").write_text(
		json.dumps(config, indent=2, default=str) + "\n"
	)


def optimizer_has_results(optimizer_dir: Path) -> bool:
	"""Return True if this optimizer run has completed (signals.npz exists)."""
	return (optimizer_dir / "signals.npz").exists()


def load_existing_optimizer_ids(results_dir: Path) -> list[str]:
	"""Load optimizer_ids from metadata.json if it exists."""
	meta_path = results_dir / "metadata.json"
	if not meta_path.exists():
		return []
	meta = json.loads(meta_path.read_text())
	return meta.get("optimizer_ids", [])


def save_metadata(
	results_dir: Path,
	*,
	experiment_class: str,
	model_class: str,
	optimizer_ids: list[str],
	merge_optimizer_ids: bool = False,
) -> None:
	"""Save metadata. If merge_optimizer_ids, merge with existing optimizer_ids."""
	if merge_optimizer_ids:
		existing = load_existing_optimizer_ids(results_dir)
		optimizer_ids = sorted(set(existing) | set(optimizer_ids))
	meta = {
		"experiment_class": experiment_class,
		"model_class": model_class,
		"optimizer_ids": optimizer_ids,
		"checkpoints_dir": "checkpoints/",
		"optimizers_dir": "optimizers/",
	}
	results_dir.mkdir(parents=True, exist_ok=True)
	(results_dir / "metadata.json").write_text(
		json.dumps(meta, indent=2) + "\n"
	)


def save_optimizer_config(
	optimizer_dir: Path,
	optimizer_config: dict[str, Any],
) -> None:
	optimizer_dir.mkdir(parents=True, exist_ok=True)
	(optimizer_dir / "optimizer_config.json").write_text(
		json.dumps(optimizer_config, indent=2, default=str) + "\n"
	)
