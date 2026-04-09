"""Config schema, fingerprinting, and the compare-existing-config guard."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

# Keys omitted from the training fingerprint and from training-config equality checks.
# ``framework_version`` is ignored if present in an old config.json on disk.
FINGERPRINT_EXCLUDE_KEYS = frozenset(
    {
        "internal_keep_tensors",
        "device",
        "save_model_cp",
        "save_model",
        "force",
        "training_config_fingerprint",
        "framework_version",
    }
)


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


# In JSON / CLI lists (e.g. ``model_class``, ``optimizer``), this token expands to
# every name in the corresponding registry
ALL_AVAILABLE_SELECTION = "all"


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


def normalize_initial_model_mode(mode: str) -> str:
    """Return ``init`` or ``pretrain``."""
    m = mode.strip().lower()
    if m in ("init", "pretrain"):
        return m
    raise ValueError(
        f"initial_model_mode must be 'init' or 'pretrain', got {mode!r}."
    )


def _resolve_initial_model_path(path_str: str) -> Path:
    raw = path_str.strip()
    if not raw:
        raise ValueError("use_initial_model path is empty.")
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"use_initial_model path is not a file: {path}")
    if path.suffix.lower() != ".pth":
        raise ValueError(
            f"use_initial_model must point to a .pth file, got suffix {path.suffix!r}."
        )
    return path


def validate_initial_model_path(path_str: str) -> str:
    """Verify *path_str* (relative to cwd) is a readable ``.pth``; return SHA-256 hex of file bytes.

    Expects a ``torch.save(model.state_dict(), ...)`` file (``torch>=2.6``, ``weights_only=True``).
    """
    path = _resolve_initial_model_path(path_str)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    torch.load(path, map_location="cpu", weights_only=True)
    return digest


def load_initial_model_state_dict(path_str: str) -> dict[str, Any]:
    """Load ``state_dict`` from *path_str* (relative to cwd if not absolute)."""
    path = _resolve_initial_model_path(path_str)
    return torch.load(path, map_location="cpu", weights_only=True)


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
) -> dict[str, Any]:
    parse_checkpoint_cadence(checkpoint_cadence)
    if experiment_runs < 1:
        raise ValueError(f"experiment_runs must be a positive integer, got {experiment_runs}")
    loss_key = normalize_loss(loss)
    mode_key = normalize_initial_model_mode(initial_model_mode)
    path_for_io = str(use_initial_model).strip()
    if mode_key == "pretrain" and not path_for_io:
        raise ValueError("initial_model_mode 'pretrain' requires a non-empty use_initial_model path.")
    sha = ""
    if path_for_io:
        sha = validate_initial_model_path(path_for_io)
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
