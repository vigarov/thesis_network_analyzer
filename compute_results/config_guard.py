"""Config schema, fingerprinting, and the compare-existing-config guard."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


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

FRAMEWORK_VERSION = "0.1.0"

# Fields that define the training config (optimizer-independent).
# If any of these differ between the incoming config and what is already on
# disk, the run is rejected unless --force is used.
TRAINING_CONFIG_KEYS = [
    "experiment_class",
    "experiment_config",
    "model_class",
    "model_config",
    "stage_epochs",
    "base_lr",
    "batch_size",
    "seed",
    "activation",
    "checkpoint_cadence",
    "disable_cp_turning_point",
]


def compute_fingerprint(config: dict[str, Any]) -> str:
    """Deterministic hash of the training-relevant config subset."""
    subset = {k: config[k] for k in TRAINING_CONFIG_KEYS if k in config}
    canonical = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_training_config(
    *,
    experiment_class: str,
    experiment_config: dict[str, Any],
    model_class: str,
    model_config: dict[str, Any],
    stage_epochs: int = 3,
    base_lr: float = 1e-3,
    batch_size: int = 1,
    seed: int = 3003,
    activation: str = "relu",
    checkpoint_cadence: str = "every_epoch",
    disable_cp_turning_point: bool = False,
) -> dict[str, Any]:
    parse_checkpoint_cadence(checkpoint_cadence)
    config: dict[str, Any] = {
        "experiment_class": experiment_class,
        "experiment_config": experiment_config,
        "model_class": model_class,
        "model_config": model_config,
        "stage_epochs": stage_epochs,
        "base_lr": base_lr,
        "batch_size": batch_size,
        "seed": seed,
        "activation": activation,
        "checkpoint_cadence": checkpoint_cadence,
        "disable_cp_turning_point": disable_cp_turning_point,
        "framework_version": FRAMEWORK_VERSION,
    }
    config["training_config_fingerprint"] = compute_fingerprint(config)
    return config


class ConfigConflictError(Exception):
    """Raised when an incoming config conflicts with a stored one."""


def _diff_configs(
    incoming: dict[str, Any],
    stored: dict[str, Any],
    keys: list[str],
) -> list[str]:
    diffs: list[str] = []
    for k in keys:
        v_in = incoming.get(k)
        v_st = stored.get(k)
        if v_in != v_st:
            diffs.append(f"  {k}: incoming={v_in!r}  stored={v_st!r}")
    return diffs


def guard_training_config(
    results_dir: Path,
    config: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    """Check that *config* is compatible with any previously stored config.

    Raises ConfigConflictError on mismatch unless *force* is True.
    """
    config_path = results_dir / "config.json"
    if not config_path.exists():
        return

    stored = json.loads(config_path.read_text())
    diffs = _diff_configs(config, stored, TRAINING_CONFIG_KEYS)

    if diffs and not force:
        msg = (
            f"Incoming training config conflicts with stored config at "
            f"{config_path}:\n" + "\n".join(diffs) + "\n"
            "Use --force to overwrite."
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
