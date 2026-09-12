"""Config schema, fingerprinting, and the compare-existing-config guard."""
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

import experiments
from compute_results.constants import (
    ALL_AVAILABLE_SELECTION,
    DEFAULT_DATASET,
    FINGERPRINT_EXCLUDE_KEYS,
    INITIAL_MODEL_CHECKPOINT_FILENAMES,
    INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS,
    INITIAL_MODEL_OPTIMIZER_STATE_FILENAME,
    PRETRAIN_REUSE_KEYS,
)
from experiments.dataset_registry import list_datasets
from experiments.base import get_registered_experiment_class
from experiments.mnist.common.label_perm import parse_restrain_digits
from compute_results.defaults import SHAMPOO_PRECONDITIONER_EPSILON_KEY
from compute_results.constants import (
        DEFAULT_THRESHOLD_ACC,
        DEFAULT_TRAIN_K_SAMPLES,
        EXPERT_THRESHOLD_ACC,
        EXPERT_THRESHOLD_ACC_CIFAR10,
        EXPERT_TRAIN_K_SAMPLES,
        EXPERT_TRAIN_K_SAMPLES_CIFAR10,
        DEFAULT_MULTI_LRS,
        DEFAULT_MULTI_SEEDS,
        MULTI_LRS_USE_DEFAULT,
        MULTI_SEEDS_USE_DEFAULT,
        PRETRAIN_FINGERPRINT_EXCLUDE_KEYS,
)

from models import parse_he_init


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
    """Map a user-facing loss name to a canonical key: `ce` or `mse`."""
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
    stored.setdefault("dataset", DEFAULT_DATASET)


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

    For experiments with `has_pretrain=True`:
    - Rejects deprecated keys: `base_ratio`, `num_pretrain_samples`, `num_stage_samples`.
    - Requires `pretrain_on_k_samples`: `int`, `>= 1`, divisible by 10.
    - Requires `num_trial_samples`: `int`, `>= 1`.
    - For `PretrainThenShuffleMislabel`, when experiment_runs is provided,
      requires `(num_trial_samples * experiment_runs) % N == 0` where `N` is 10
      or `len(restrain_digits)` when `restrain_digits` is set.
    - For `PretrainControlBase`, requires `num_trial_samples % 10 == 0` (balanced remainder trials).

    For `ControlBase` (no pretrain flag): requires `pretrain_on_k_samples` (K split),
    `num_trial_samples`, and `num_trial_samples % 10 == 0`.

    Raises:
        KeyError: if `experiment_class` is not registered.
        ValueError / TypeError: on constraint violations.
    """
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
        if cls.__name__ not in (
            "PretrainThenShuffleMislabel",
            "Cat1SampleShuffleConstrained",
            "Cat1SampleShuffleInterleaved",
        ):
            raise ValueError(
                f"experiment_config for {experiment_class}: 'restrain_digits' is only supported "
                "for PretrainThenShuffleMislabel, Cat1SampleShuffleConstrained, "
                "and Cat1SampleShuffleInterleaved."
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

    for key in _DEPRECATED_PRETRAIN_KEYS:
        if key in experiment_config:
            hint = _DEPRECATED_PRETRAIN_HINTS.get(key, "")
            raise ValueError(
                f"experiment_config for {experiment_class}: '{key}' is no longer supported. "
                + hint
            )

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
    n_div_trials = 10
    if cls.__name__ == "PretrainThenShuffleMislabel":
        rd = parse_restrain_digits(experiment_config.get("restrain_digits"))
        n_div_trials = len(rd) if rd is not None else 10
    elif cls.__name__ in ("Cat1SampleShuffleConstrained", "Cat1SampleShuffleInterleaved"):
        rd = parse_restrain_digits(experiment_config.get("restrain_digits"))
        if rd is None:
            raise ValueError(
                f"experiment_config for {experiment_class}: "
                f"'restrain_digits' is required for {cls.__name__}."
            )
        n_div_trials = len(rd)
    if pk % n_div_pretrain != 0:
        raise ValueError(
            f"experiment_config for {experiment_class}: 'pretrain_on_k_samples' must be divisible "
            f"by {n_div_pretrain} (balanced K over each training digit), got {pk}"
        )

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
        if int(nt) % n_div_trials != 0:
            raise ValueError(
                f"experiment_config for {experiment_class}: 'num_trial_samples' must be divisible "
                f"by {n_div_trials}; got {nt}."
            )

    if cls.__name__ == "Cat1SampleShuffleInterleaved":
        cycle_len = 2 * n_div_trials
        if int(nt) % 2 != 0:
            raise ValueError(
                f"experiment_config for {experiment_class}: 'num_trial_samples' must be even; "
                f"got {nt}."
            )
        if int(nt) % cycle_len != 0:
            raise ValueError(
                f"experiment_config for {experiment_class}: 'num_trial_samples' must be divisible "
                f"by 2 * len(restrain_digits)={cycle_len}; got {nt}."
            )

    if cls.__name__ in (
        "PretrainThenShuffleMislabel",
        "Cat1SampleShuffleFinetune",
        "Cat1SampleShuffleConstrained",
    ) and experiment_runs is not None:
        prod = int(nt) * int(experiment_runs)
        if prod % n_div_trials != 0:
            raise ValueError(
                f"experiment_config for {experiment_class}: "
                f"num_trial_samples * experiment_runs must be divisible by {n_div_trials} "
                f"(balanced post-pretrain pool); got {nt} * {experiment_runs} = {prod}."
            )

    if cls.__name__ == "Cat1SampleShuffleInterleaved" and experiment_runs is not None:
        prod = (int(nt) // 2) * int(experiment_runs)
        if prod % n_div_trials != 0:
            raise ValueError(
                f"experiment_config for {experiment_class}: "
                f"(num_trial_samples // 2) * experiment_runs must be divisible by "
                f"{n_div_trials} (balanced restrained post-pretrain pool); "
                f"got ({nt} // 2) * {experiment_runs} = {prod}."
            )

    if cls.__name__ == "PretrainControlBase" and int(nt) % 10 != 0:
        raise ValueError(
            f"experiment_config for {experiment_class}: 'num_trial_samples' must be divisible by 10 "
            f"(balanced remainder trials); got {nt}."
        )


def normalize_initial_model_mode(mode: str) -> str:
    """Return `init` or `pretrain`."""
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


def _resolve_seed_subdirectory(bundle_root: Path, seed: int | None) -> Path:
    """If bundle_root has a child named `str(seed)`, descend into it.

    Pretrain checkpoints from `pretrain_models.py` are written under `!SD` seed
    subdirectories (e.g. `.../adam_slug/6/model.pt`). When `use_initial_model`
    points at the parent of those seed folders, this selects the folder matching the
    training config `seed` before resolving optimizer layout or flat checkpoints.
    """
    if seed is None:
        return bundle_root
    seed_dir = bundle_root / str(seed)
    if seed_dir.is_dir():
        return seed_dir
    return bundle_root


def _checkpoint_file_in_dir(directory: Path) -> Path | None:
    """Return the single model checkpoint in directory, or `None` if absent."""
    present = [
        directory / fn
        for fn in INITIAL_MODEL_CHECKPOINT_FILENAMES
        if (directory / fn).is_file()
    ]
    if len(present) > 1:
        raise ValueError(
            f"use_initial_model directory {directory}: must contain at most one of "
            f"{INITIAL_MODEL_CHECKPOINT_FILENAMES}, found multiple."
        )
    if len(present) == 1:
        return present[0]
    return None


def _initial_bundle_probe_order(
    registry_class_name: str,
    *,
    optimizer_id: str | None = None,
) -> list[tuple[str, bool]]:
    """(probe_name, is_exact_registry_class). Registry class is matched by exact dirname only;
    shorthands match any immediate child directory whose name starts with the shorthand."""
    rc = registry_class_name.strip()
    if not rc:
        return []
    out: list[tuple[str, bool]] = []
    oid = str(optimizer_id or "").strip()
    if oid:
        out.append((oid, True))
    out.append((rc, True))
    seen_shorthand: set[str] = set()
    for shorthand, cls in sorted(INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS.items()):
        if cls == rc and shorthand not in seen_shorthand:
            seen_shorthand.add(shorthand)
            out.append((shorthand, False))
    return out


def _bundle_subdirs_for_probe(
    bundle_root: Path,
    probe_name: str,
    *,
    exact: bool,
    optimizer_id: str | None = None,
) -> list[Path]:
    """Resolve to zero or one subfolder path(s) for this probe (see `_initial_bundle_probe_order`)."""
    if exact:
        sub = bundle_root / probe_name
        return [sub] if sub.is_dir() else []
    matches = sorted(
        p
        for p in bundle_root.iterdir()
        if p.is_dir() and p.name.startswith(probe_name)
    )
    if len(matches) > 1:
        oid = str(optimizer_id or "").strip()
        if oid:
            by_id = [m for m in matches if m.name == oid]
            if len(by_id) == 1:
                return by_id
            if len(by_id) == 0:
                names = [m.name for m in matches]
                raise ValueError(
                    f"use_initial_model directory {bundle_root}: no subfolder {oid!r} among "
                    f"shorthand {probe_name!r} matches {names!r}; check the run learning rate "
                    f"matches a pretrained checkpoint."
                )
        names = [m.name for m in matches]
        raise ValueError(
            f"use_initial_model directory {bundle_root}: multiple subfolders start with "
            f"shorthand {probe_name!r}: {names!r}; use a single directory per optimizer."
        )
    return matches


def _checkpoint_in_bundle_subdir(
    bundle_root: Path,
    registry_class_name: str,
    *,
    seed: int | None = None,
    optimizer_id: str | None = None,
) -> Path:
    """Resolve a model checkpoint under bundle_root.

    Layout resolution order:

    1. If seed is set and `bundle_root / str(seed)` exists, use that directory.
    2. If `model.pt` / `model.pth` lies directly in that directory (pretrain
       `!OPT/!SD` layout after `!OPT` expansion), return it.
    3. Otherwise probe per-optimizer subfolders (exact `optimizer_id` when given,
       then exact registry class name, then CLI shorthand prefix). After each match,
       apply step 1 again on that subfolder.
    """
    rc = registry_class_name.strip()
    if not rc:
        raise ValueError("empty extractor class name for use_initial_model bundle.")
    root = _resolve_seed_subdirectory(bundle_root, seed)
    flat = _checkpoint_file_in_dir(root)
    if flat is not None:
        return flat

    probe_order = _initial_bundle_probe_order(rc, optimizer_id=optimizer_id)
    last_nonempty_dir: str | None = None
    for probe_name, exact in probe_order:
        subs = _bundle_subdirs_for_probe(
            root, probe_name, exact=exact, optimizer_id=optimizer_id
        )
        if not subs:
            continue
        sub = _resolve_seed_subdirectory(subs[0], seed)
        last_nonempty_dir = sub.name
        ckpt = _checkpoint_file_in_dir(sub)
        if ckpt is not None:
            for other_name, other_exact in probe_order:
                for opath in _bundle_subdirs_for_probe(
                    root,
                    other_name,
                    exact=other_exact,
                    optimizer_id=optimizer_id,
                ):
                    opath_resolved = _resolve_seed_subdirectory(opath, seed)
                    if opath_resolved.resolve() == sub.resolve():
                        continue
                    if _bundle_subdir_has_checkpoint(opath_resolved):
                        raise ValueError(
                            f"use_initial_model directory {bundle_root}: ambiguous layout for "
                            f"optimizer {rc!r}: both {sub.name!r} and {opath_resolved.name!r} "
                            f"contain one of {INITIAL_MODEL_CHECKPOINT_FILENAMES}; keep a single "
                            f"subfolder."
                        )
            return ckpt
    if last_nonempty_dir is None:
        probes = [p[0] for p in probe_order]
        seed_hint = f" (seed={seed})" if seed is not None else ""
        raise FileNotFoundError(
            f"use_initial_model directory {bundle_root}{seed_hint}: no subfolder for {rc!r}; "
            f"tried exact class name and shorthand prefixes {probes!r}, each with one of "
            f"{INITIAL_MODEL_CHECKPOINT_FILENAMES} inside."
        )
    raise ValueError(
        f"use_initial_model directory {bundle_root}: subfolder {last_nonempty_dir!r} exists but "
        f"must contain exactly one of {INITIAL_MODEL_CHECKPOINT_FILENAMES}."
    )


def _bundle_location_key(bundle_root: Path, checkpoint: Path) -> str:
    """Stable subpath under bundle_root for deduplicating optimizer resolutions."""
    root = bundle_root.resolve()
    location = checkpoint.parent.resolve()
    try:
        return str(location.relative_to(root))
    except ValueError:
        return location.name


def _validate_initial_model_directory(
    root: Path,
    registry_classes: list[str],
    *,
    optimizer_ids: list[str] | None = None,
    seed: int | None = None,
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
    oids = list(optimizer_ids or ())
    for i, registry_class in enumerate(registry_classes):
        rc = registry_class.strip()
        if rc in seen_registry_classes:
            raise ValueError(
                f"use_initial_model directory {root}: duplicate optimizer {rc!r} in the run."
            )
        seen_registry_classes.add(rc)
        oid = oids[i].strip() if i < len(oids) else None
        ckpt = _checkpoint_in_bundle_subdir(
            root, rc, seed=seed, optimizer_id=oid or None
        )
        disk_folder = _bundle_location_key(root, ckpt)
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
    optimizer_ids: list[str] | None = None,
    seed: int | None = None,
) -> str:
    """Verify path_str and return a SHA-256 fingerprint for the initial weights.

    - Single file: must be a readable `.pt` or `.pth` checkpoint
      (`torch.save(model.state_dict(), ...)`); returns SHA-256 of file bytes.
    - Directory: for each entry in optimizer_registry_classes (resolved extractor
      class per `--optimizer` entry, e.g. `AdamExtractor`), there must be a
      subfolder whose name equals that class, or (if no class-named folder matches)
      whose name starts with a CLI shorthand key from
      `INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS`. Each subfolder must contain:
        - exactly one of `model.pt` or `model.pth` (model weights),
        - `optimizer.pt` (optimizer state dict, `torch.save(opt.state_dict(), ...)`).
      Class name is tried before shorthand prefixes. Extra subfolders are ignored.
      When seed is set, a child directory named `str(seed)` is selected first
      (pretrain `!SD` layout); checkpoints may then lie directly in that folder
      or in per-optimizer subfolders beneath it.
      Returns a deterministic hash of sorted
      `(registry_class, "model"|"optimizer", sha256)` triples.
    """
    path = _resolve_use_initial_model_base(path_str)
    if path.is_file():
        path = _resolve_initial_model_file(path_str)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        torch.load(path, map_location="cpu", weights_only=True)
        return digest
    if path.is_dir():
        return _validate_initial_model_directory(
            path,
            list(optimizer_registry_classes or ()),
            optimizer_ids=optimizer_ids,
            seed=seed,
        )
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
    optimizer_id: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Load model `state_dict` from path_str (relative to cwd if not absolute).

    If path_str is a directory, registry_class_name selects the optimizer; the
    subdirectory is resolved with the same rules as validation (seed subdirectory
    first, then exact class dirname or shorthand prefix, with optional seed under
    each optimizer folder).
    """
    base = _resolve_use_initial_model_base(path_str)
    if base.is_file():
        path = _resolve_initial_model_file(path_str)
        return torch.load(path, map_location="cpu", weights_only=True)
    if base.is_dir():
        rc = _require_bundle_registry_class("load_initial_model_state_dict", registry_class_name)
        ckpt = _checkpoint_in_bundle_subdir(
            base, rc, seed=seed, optimizer_id=optimizer_id
        )
        return torch.load(ckpt, map_location="cpu", weights_only=True)
    raise ValueError(f"use_initial_model path is not a file or directory: {base}")


def load_initial_optimizer_state_dict(
    path_str: str,
    *,
    registry_class_name: str,
    optimizer_id: str | None = None,
    map_location: Any = "cpu",
    seed: int | None = None,
) -> dict[str, Any]:
    """Load optimizer `state_dict` from a bundle directory.

    path_str must be a directory (only meaningful for bundle layouts). The subfolder
    is resolved by the same rules as `load_initial_model_state_dict`; `optimizer.pt`
    inside that subfolder is loaded with `weights_only=False` (optimizer state dicts
    include non-tensor `param_groups` entries).
    """
    base = _resolve_use_initial_model_base(path_str)
    if not base.is_dir():
        raise ValueError(
            f"load_initial_optimizer_state_dict: expected a bundle directory, got: {base}"
        )
    rc = _require_bundle_registry_class("load_initial_optimizer_state_dict", registry_class_name)
    ckpt = _checkpoint_in_bundle_subdir(
        base, rc, seed=seed, optimizer_id=optimizer_id
    )
    opt_path = ckpt.parent / INITIAL_MODEL_OPTIMIZER_STATE_FILENAME
    if not opt_path.is_file():
        raise ValueError(
            f"load_initial_optimizer_state_dict: {INITIAL_MODEL_OPTIMIZER_STATE_FILENAME!r} "
            f"not found in {ckpt.parent}"
        )
    return torch.load(opt_path, map_location=map_location, weights_only=False)


def fingerprint_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Training-relevant subset of config used for hashing and equality checks."""
    payload = {k: v for k, v in config.items() if k not in FINGERPRINT_EXCLUDE_KEYS}
    # Drop the default dataset so historical (pre-`dataset`) MNIST configs keep the
    # same fingerprint / result directory; non-default datasets stay in the payload.
    if payload.get("dataset") == DEFAULT_DATASET:
        payload.pop("dataset", None)
    return payload


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
    """Directory name under experiment_dir for this training config (first 6 hex of digest).

    If `experiment_dir / <id> / model_id` already exists with a different training
    payload, tries `<prefix>_2`, `<prefix>_3`, ...
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
    dataset: str = DEFAULT_DATASET,
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
    optimizer_ids_for_initial_model: list[str] | None = None,
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
            optimizer_ids=optimizer_ids_for_initial_model,
            seed=seed,
        )
    config: dict[str, Any] = {
        "experiment_class": experiment_class,
        "experiment_config": experiment_config,
        "model_class": model_class,
        "model_config": model_config,
        "dataset": dataset,
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
    """Check that config matches any previously stored training config on disk.

    Always raises ConfigConflictError on fingerprint-payload mismatch (`--force` does
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


# Pretrain config (threshold-based MNIST pretraining)

def _parse_csv_or_list(value: str | list | float | int | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    return [str(value).strip()]


def _parse_int_list(value: str | list | int | None) -> list[int]:
    parts = _parse_csv_or_list(value)
    return [int(p) for p in parts]


def _parse_float_list(value: str | list | float | int | None) -> list[float]:
    parts = _parse_csv_or_list(value)
    return [float(p) for p in parts]


def _resolve_pretrain_model_seeds(
    raw: dict[str, Any],
    *,
    cli_multi_seeds: str | None,
) -> list[int]:
    if cli_multi_seeds is not None:
        if cli_multi_seeds == MULTI_SEEDS_USE_DEFAULT:
            return list(DEFAULT_MULTI_SEEDS)
        return _parse_int_list(cli_multi_seeds)

    for key in ("multi_seeds", "model_seeds"):
        if key in raw and raw[key] is not None:
            val = raw[key]
            if isinstance(val, int):
                return [int(val)]
            return [int(x) for x in val]

    seed = raw.get("seed")
    if seed is None:
        seed = raw.get("dataset_seed")
    if seed is not None:
        return [int(seed)]

    return list(DEFAULT_MULTI_SEEDS)


def _resolve_pretrain_base_lrs(
    raw: dict[str, Any],
    *,
    cli_multi_lr: str | None,
    resolved_base_lr: float,
) -> list[float]:
    """Base learning rates to pretrain over.

    Precedence mirrors `_resolve_pretrain_model_seeds`: explicit CLI `--multi_lr`
    (sentinel -> default sweep list, else CSV) wins, then a `multi_lr` /
    `multi_lrs` key in the JSON config, else the single resolved_base_lr.
    """
    if cli_multi_lr is not None:
        if cli_multi_lr == MULTI_LRS_USE_DEFAULT:
            return list(DEFAULT_MULTI_LRS)
        return _parse_float_list(cli_multi_lr)

    for key in ("multi_lr", "multi_lrs"):
        if key in raw and raw[key] is not None:
            return _parse_float_list(raw[key])

    return [float(resolved_base_lr)]


def parse_pretrain_inputs(
    raw: dict[str, Any],
    *,
    cli: Any,
) -> dict[str, Any]:
    """Merge JSON config and CLI overrides into normalized pretrain parameters."""

    expert = bool(raw.get("expert", False)) or bool(getattr(cli, "expert", False))

    exp_cfg = dict(raw.get("experiment_config") or {})

    model_class = getattr(cli, "model", None) or raw.get("model_class")
    if not model_class:
        raise ValueError("model_class is required (JSON model_class or --model).")

    optimizer_raw = getattr(cli, "optimizer", None) or raw.get("optimizer")
    if not optimizer_raw:
        raise ValueError("optimizer is required (JSON optimizer or --optimizer).")
    optimizer_names = _parse_csv_or_list(optimizer_raw)

    model_config = dict(raw.get("model_config") or {})
    activation = getattr(cli, "activation", None) or raw.get("activation", "relu")
    if not model_config:
        model_config = {"activation": activation}
    elif "activation" not in model_config:
        model_config["activation"] = activation

    base_lr = raw.get("base_lr", 1e-3)
    if getattr(cli, "base_lr", None) is not None:
        base_lr = float(cli.base_lr)

    cli_multi_lr = getattr(cli, "multi_lr", None)
    base_lrs = _resolve_pretrain_base_lrs(
        raw, cli_multi_lr=cli_multi_lr, resolved_base_lr=float(base_lr)
    )

    batch_size = int(raw.get("batch_size", 1))
    if getattr(cli, "batch_size", None) is not None:
        batch_size = int(cli.batch_size)

    he_init = raw.get("he_init", 3)
    if getattr(cli, "he_init", None) is not None:
        he_init = cli.he_init

    init_epsilon = float(raw.get("init_epsilon", 1e-8))
    if getattr(cli, "init_epsilon", None) is not None:
        init_epsilon = float(cli.init_epsilon)

    dataset_seed = int(raw.get("dataset_seed", raw.get("seed", 3003)))
    if getattr(cli, "dataset_seed", None) is not None:
        dataset_seed = int(cli.dataset_seed)

    cli_multi = getattr(cli, "multi_seeds", None)
    model_seeds = _resolve_pretrain_model_seeds(raw, cli_multi_seeds=cli_multi)

    train_k_samples = raw.get("train_k_samples")
    if train_k_samples is None:
        train_k_samples = exp_cfg.get("pretrain_on_k_samples")
    if getattr(cli, "train_k_samples", None) is not None:
        train_k_samples = int(cli.train_k_samples)
    elif train_k_samples is None:
        train_k_samples = DEFAULT_TRAIN_K_SAMPLES
    else:
        train_k_samples = int(train_k_samples)
    dataset = str(raw.get("dataset", DEFAULT_DATASET))
    if getattr(cli, "dataset", None) is not None:
        dataset = str(cli.dataset)

    if expert:
        train_k_samples = (
            EXPERT_TRAIN_K_SAMPLES_CIFAR10
            if dataset == "cifar10"
            else EXPERT_TRAIN_K_SAMPLES
        )

    if expert:
        if getattr(cli, "threshold_acc", None) is not None:
            threshold_acc = float(cli.threshold_acc)
        else:
            threshold_acc = (
                EXPERT_THRESHOLD_ACC_CIFAR10
                if dataset == "cifar10"
                else EXPERT_THRESHOLD_ACC
            )
    else:
        threshold_acc = raw.get("threshold_acc")
        if getattr(cli, "threshold_acc", None) is not None:
            threshold_acc = float(cli.threshold_acc)
        elif threshold_acc is None:
            threshold_acc = DEFAULT_THRESHOLD_ACC
        else:
            threshold_acc = float(threshold_acc)

    max_epochs = int(raw.get("max_epochs", 200))
    if getattr(cli, "max_epochs", None) is not None:
        max_epochs = int(cli.max_epochs)

    data_root = str(raw.get("data_root", "./data"))
    if getattr(cli, "data_root", None) is not None:
        data_root = str(cli.data_root)

    if dataset not in list_datasets():
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {list_datasets()}"
        )

    save = bool(raw.get("save", True))
    if getattr(cli, "no_save", False):
        save = False
    elif getattr(cli, "save", False):
        save = True

    force = bool(raw.get("force", False)) or bool(getattr(cli, "force", False))

    cli_device = getattr(cli, "device", None)
    if cli_device is not None:
        device = str(cli_device)
    elif raw.get("device") is not None:
        device = str(raw["device"])
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prefix = "expert_" if expert else ""
    default_out = f"save/{prefix}optimizer_pretrained/!OPT/!SD/"
    out_dir = str(raw.get("out_dir", default_out))
    if getattr(cli, "out_dir", None) is not None:
        out_dir = str(cli.out_dir)
    if getattr(cli, "output_dir", None) is not None:
        out_dir = str(cli.output_dir)

    opt_extra_kwargs: dict[str, Any] = {}
    eps_from_json = raw.get(SHAMPOO_PRECONDITIONER_EPSILON_KEY)
    if eps_from_json is not None:
        opt_extra_kwargs[SHAMPOO_PRECONDITIONER_EPSILON_KEY] = float(eps_from_json)
    cli_eps = getattr(cli, "shampoo_preconditioner_epsilon", None)
    if cli_eps is not None:
        opt_extra_kwargs[SHAMPOO_PRECONDITIONER_EPSILON_KEY] = float(cli_eps)

    return {
        "model_class": str(model_class).strip(),
        "model_config": model_config,
        "activation": activation,
        "optimizer_names": optimizer_names,
        "base_lr": float(base_lr),
        "base_lrs": base_lrs,
        "batch_size": batch_size,
        "he_init": he_init,
        "init_epsilon": init_epsilon,
        "dataset_seed": dataset_seed,
        "model_seeds": model_seeds,
        "train_k_samples": int(train_k_samples),
        "threshold_acc": float(threshold_acc),
        "max_epochs": max_epochs,
        "data_root": data_root,
        "dataset": dataset,
        "save": save,
        "force": force,
        "device": device,
        "out_dir": out_dir,
        "expert": expert,
        "opt_extra_kwargs": opt_extra_kwargs,
    }


def pretrain_fingerprint_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        k: v for k, v in config.items() if k not in PRETRAIN_FINGERPRINT_EXCLUDE_KEYS
    }
    if payload.get("dataset") == DEFAULT_DATASET:
        payload.pop("dataset", None)
    return payload


def compute_pretrain_fingerprint(config: dict[str, Any]) -> str:
    payload = pretrain_fingerprint_payload(config)
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_pretrain_config(
    *,
    model_class: str,
    model_config: dict[str, Any],
    activation: str,
    optimizer_names: list[str],
    base_lr: float,
    batch_size: int,
    he_init: int | str | float,
    init_epsilon: float,
    dataset_seed: int,
    model_seeds: list[int],
    train_k_samples: int,
    threshold_acc: float,
    max_epochs: int,
    data_root: str,
    expert: bool,
    dataset: str = DEFAULT_DATASET,
    opt_extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_class": model_class,
        "model_config": model_config,
        "activation": activation,
        "optimizer": optimizer_names,
        "base_lr": base_lr,
        "batch_size": batch_size,
        "he_init": he_init,
        "init_epsilon": init_epsilon,
        "dataset_seed": dataset_seed,
        "multi_seeds": list(model_seeds),
        "train_k_samples": train_k_samples,
        "threshold_acc": threshold_acc,
        "max_epochs": max_epochs,
        "data_root": data_root,
        "dataset": dataset,
        "expert": expert,
        "opt_extra_kwargs": dict(opt_extra_kwargs or {}),
    }
    config["pretrain_config_fingerprint"] = compute_pretrain_fingerprint(config)
    return config


def validate_pretrain_config_constraints(
    config: dict[str, Any],
    *,
    out_dir: str,
    save: bool,
    model_seeds: list[int],
) -> None:
    pk = int(config["train_k_samples"])
    if pk < 1:
        raise ValueError(f"train_k_samples must be >= 1, got {pk}")
    if pk % 10 != 0:
        raise ValueError(f"train_k_samples must be divisible by 10, got {pk}")

    ta = float(config["threshold_acc"])
    if not (0.0 < ta <= 1.0):
        raise ValueError(f"threshold_acc must be in (0, 1], got {ta}")

    if int(config["max_epochs"]) < 1:
        raise ValueError(f"max_epochs must be >= 1, got {config['max_epochs']}")

    if "!OPT" not in out_dir:
        raise ValueError("out_dir must contain the placeholder '!OPT'.")

    if config.get("expert") and "expert" not in out_dir:
        raise ValueError(
            "Expert pretraining requires 'expert' in out_dir "
            f"(e.g. expert_optimizer_pretrained/); got {out_dir!r}."
        )

    if save and len(model_seeds) > 1 and "!SD" not in out_dir:
        raise ValueError("SAVE with multiple model_seeds requires '!SD' in out_dir.")

    if float(config["init_epsilon"]) < 0:
        raise ValueError("init_epsilon must be non-negative.")

    parse_he_init(config["he_init"])


def _normalize_pretrain_reuse_value(key: str, value: Any) -> Any:
    if key == "base_lr":
        return float(value)
    if key == "train_k_samples":
        return int(value)
    if key == "dataset":
        return DEFAULT_DATASET if value is None else str(value)
    return value


def _load_stored_pretrain_metadata(save_root: Path) -> dict[str, Any] | None:
    """Load reuse-relevant metadata from a checkpoint directory, if present."""
    config_path = save_root / "pretrain_config.json"
    report_path = save_root / "report.json"

    report: dict[str, Any] | None = None
    if report_path.exists():
        report = json.loads(report_path.read_text())

    stored: dict[str, Any] | None = None
    if config_path.exists():
        stored = dict(json.loads(config_path.read_text()))
    elif report is not None:
        stored = dict(report.get("config") or {})
    else:
        return None

    if report is not None:
        opt_id = report.get("optimizer_id")
        if opt_id:
            stored["optimizer_id"] = str(opt_id)
    return stored


def _pretrain_reuse_diffs(
    incoming: dict[str, Any],
    stored: dict[str, Any],
    *,
    slug: str,
) -> list[str]:
    diffs: list[str] = []
    for key in PRETRAIN_REUSE_KEYS:
        v_in = _normalize_pretrain_reuse_value(key, incoming.get(key))
        v_st = _normalize_pretrain_reuse_value(key, stored.get(key))
        if v_in != v_st:
            diffs.append(f"  {key}: incoming={v_in!r}  stored={v_st!r}")

    stored_slug = stored.get("optimizer_id")
    if stored_slug is not None and str(stored_slug) != slug:
        diffs.append(
            f"  optimizer_id: incoming={slug!r}  stored={stored_slug!r}"
        )
    return diffs


def guard_pretrain_output(
    save_root: Path,
    config: dict[str, Any],
    *,
    slug: str,
    force: bool,
) -> bool:
    """Validate stored pretrain metadata against config.

    Returns True if the run should be skipped: `model.pt` exists and the
    checkpoint was produced with the same optimizer (`slug`), learning rate,
    and training sample count. Other pretrain settings may differ across configs.

    Raises ConfigConflictError when those reuse fields disagree (never
    overridable with `--force`).
    """
    model_path = save_root / "model.pt"
    if not model_path.exists():
        return False

    stored = _load_stored_pretrain_metadata(save_root)
    if stored is None:
        if force:
            return False
        raise ConfigConflictError(
            f"Output exists at {save_root} (model.pt) but no pretrain_config.json "
            f"or report.json; use --force to overwrite."
        )

    diffs = _pretrain_reuse_diffs(config, stored, slug=slug)
    if not diffs:
        return not force

    msg = (
        f"Existing pretrain checkpoint at {save_root} conflicts with the "
        f"requested optimizer, learning rate, or sample count:\n"
        + "\n".join(diffs)
        + "\nThese mismatches cannot be overridden with --force."
    )
    raise ConfigConflictError(msg)


def save_pretrain_config(save_root: Path, config: dict[str, Any]) -> None:
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / "pretrain_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )


def build_pretrain_report(
    *,
    pretrain_config: dict[str, Any],
    training_result: dict[str, Any],
    model_seed: int,
    optimizer_name: str,
    optimizer_class: str,
    optimizer_id: str,
) -> dict[str, Any]:
    """Build a human-readable run summary for `report.json`."""
    config = pretrain_fingerprint_payload(pretrain_config)
    return {
        "model_class": pretrain_config["model_class"],
        "model_config": dict(pretrain_config["model_config"]),
        "model_seed": model_seed,
        "optimizer_name": optimizer_name,
        "optimizer_class": optimizer_class,
        "optimizer_id": optimizer_id,
        "run": {
            "total_iterations": int(training_result["steps"]),
            "phase_a_epochs": int(training_result.get("phase_a_epochs", 0)),
            "crossing_epoch": training_result.get("crossing_epoch"),
            "step_in_crossing_epoch": training_result.get("step_in_crossing_epoch"),
            "reached_threshold": bool(training_result.get("reached", False)),
            "saved_best_at_max_epochs": bool(
                training_result.get("saved_best_at_max_epochs", False)
            ),
            "best_epoch": training_result.get("best_epoch"),
            "best_test_acc": training_result.get("best_test_acc"),
            "final_train_loss": training_result.get("final_train_loss"),
            "final_test_loss": training_result.get("final_test_loss"),
            "final_test_acc": training_result.get("final_test_acc"),
        },
        "config": config,
    }


def load_pretrain_report(save_root: Path) -> dict[str, Any]:
    """Load `report.json` from a pretrain checkpoint directory."""
    return json.loads((save_root / "report.json").read_text())


def save_pretrain_report(save_root: Path, report: dict[str, Any]) -> None:
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
