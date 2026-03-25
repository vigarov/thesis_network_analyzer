#!/usr/bin/env python3
"""Main entrypoint for running training simulations.

Results are written under::

    results/<experiment_id>/<result_id>/<model_id>/

where ``result_id`` is the first six hex characters of a SHA-256 fingerprint of the
training config (with a numeric suffix if that prefix collides with a different config).

Use ``all`` as a model or optimizer token (JSON or CLI) to include every name in the
corresponding registry (sorted). You can combine with explicit names, e.g. ``adam,all``,
to union shorthands with all registered extractors.

``--force`` re-runs requested optimizers even when outputs already exist. Training
config mismatches against an existing ``config.json`` are never overridden; fix the
config or use a different run directory (different training fingerprint).

Usage examples
--------------
From a config file::

    uv run run-simulation --config path/to/config.json

With explicit arguments::

    uv run run-simulation \
        --experiment DigitAThenDigitB_75_25 \
        --digitA 1 --digitB 2 \
        --model DNN5Hidden64 \
        --optimizer sgd \
        --stage-epochs 1 --base-lr 1e-3 --batch-size 1 --seed 3003

Multiple optimizers/models/experiments (CSV lists)::

    uv run run-simulation --experiment Exp1,Exp2 --model M1,M2 --optimizer adam,adagrad
"""

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import traceback

# Ensure the project root is on sys.path so bare package imports work when
# the script is invoked directly (e.g. ``python compute_results/run_simulation.py``).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from compute_results.config_guard import (
    ConfigConflictError,
    build_training_config,
    expand_registry_selection,
    guard_optimizer_config,
    guard_training_config,
    load_existing_optimizer_ids,
    optimizer_has_results,
    result_id_for_config,
    save_config,
    save_metadata,
    save_optimizer_config,
)
from compute_results.defaults import OPTIMIZER_LR_KEY, SHAMPOO_PRECONDITIONER_EPSILON_KEY
from compute_results.training_loop import train_with_config
from experiments import get_experiment, list_experiments
from models import apply_model_weight_init, get_model, list_models, validate_he_init
from optimizers import get_extractor, list_extractors


RESULTS_ROOT = _PROJECT_ROOT / "results"

OPTIMIZER_SHORTHAND: dict[str, tuple[str, dict]] = {
    "sgd": ("SGDExtractor", {}),
    "adam": ("AdamExtractor", {}),
    "adagrad": ("AdaGradExtractor", {}),
    "pure_shampoo": ("PureShampooExtractor", {}),
    "graft_shampoo": ("GraftedShampooExtractor", {}),
}

def _parse_csv(value: str | list) -> list[str]:
    """Split a comma-separated string or list into non-empty parts."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    return [str(value).strip()]


def _resolve_optimizer(name: str, base_lr: float, **kwargs) -> tuple[str, dict]:
    """Resolve shorthand / class name to (registry class name, ``get_extractor`` kwargs).

    *base_lr* always becomes ``OPTIMIZER_LR_KEY`` (overrides the same key in
    ``**kwargs`` if present). Remaining ``**kwargs`` are forwarded
    """
    name = name.strip()
    if name in OPTIMIZER_SHORTHAND:
        cls_name, extra = OPTIMIZER_SHORTHAND[name]
        out = {**extra, **kwargs}
    else:
        cls_name = name
        out = dict(kwargs)
    out[OPTIMIZER_LR_KEY] = base_lr
    return cls_name, out


def _validate_names(
    experiments: list[str],
    models: list[str],
    optimizers: list[str],
) -> None:
    """Validate that all names exist in registries. Exit on error."""
    valid_exp = set(list_experiments())
    valid_mod = set(list_models())
    valid_opt = set(OPTIMIZER_SHORTHAND) | set(list_extractors())
    invalid_exp = [e for e in experiments if e not in valid_exp]
    invalid_mod = [m for m in models if m not in valid_mod]
    invalid_opt = [o for o in optimizers if o not in valid_opt]
    if invalid_exp or invalid_mod or invalid_opt:
        parts = []
        if invalid_exp:
            parts.append(f"experiments: {invalid_exp} (valid: {sorted(valid_exp)})")
        if invalid_mod:
            parts.append(f"models: {invalid_mod} (valid: {sorted(valid_mod)})")
        if invalid_opt:
            parts.append(f"optimizers: {invalid_opt} (valid: {sorted(valid_opt)})")
        print(f"ERROR: Unknown names:\n  " + "\n  ".join(parts), file=sys.stderr)
        sys.exit(1)


@dataclass
class _Task:
    """A single (experiment, model, optimizer) simulation task."""

    experiment_class: str
    experiment_kwargs: dict
    model_class: str
    model_kwargs: dict
    optimizer_name: str
    optimizer_class: str
    optimizer_kwargs: dict
    config: dict
    seed: int
    results_dir: Path

    @property
    def optimizer_dir(self) -> Path:
        extractor = get_extractor(self.optimizer_class, **self.optimizer_kwargs)
        return self.results_dir / "optimizers" / extractor.optimizer_id()


def _run_single_task(
    task: _Task,
    device_id: int,
) -> tuple[str, str, str, bool]:
    """Run one training task. Top-level for multiprocessing pickling.

    Returns (experiment_id, model_id, optimizer_id, success).
    """
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    t = task
    experiment = get_experiment(t.experiment_class, **t.experiment_kwargs)
    model = get_model(t.model_class, **t.model_kwargs)
    extractor = get_extractor(t.optimizer_class, **t.optimizer_kwargs)

    torch.manual_seed(t.seed)
    apply_model_weight_init(
        model,
        he_init=t.config["he_init"],
        init_epsilon=float(t.config["init_epsilon"]),
        activation=t.config.get("activation", "relu"),
    )

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        device = torch.device(f"cuda:{device_id % num_gpus}")
    else:
        device = torch.device("cpu")

    try:
        train_with_config(
            experiment=experiment,
            model=model,
            extractor=extractor,
            config=t.config,
            device=device,
            results_dir=t.results_dir,
        )
        return (
            experiment.experiment_id(),
            model.model_id(),
            extractor.optimizer_id(),
            True,
        )
    except Exception as e:
        print(f"{traceback.format_exc()}\nTask failed: {t.experiment_class}/{t.model_class}/{t.optimizer_name}: {e}")
        return (
            experiment.experiment_id(),
            model.model_id(),
            extractor.optimizer_id(),
            False,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run training simulations. Supports CSV lists for multiple "
        "experiments, models, and optimizers."
    )
    p.add_argument("--config", type=Path, help="Path to a JSON config file.")
    p.add_argument(
        "--experiment",
        help="Experiment class name(s), comma-separated (e.g. Exp1,Exp2).",
    )
    p.add_argument(
        "--model",
        help=(
            "Model class name(s), comma-separated. "
            "Use 'all' to include every registered model (optionally mix with names)."
        ),
    )
    p.add_argument(
        "--optimizer",
        help=(
            "Optimizer shorthand(s) or extractor class name(s), comma-separated. "
            "Use 'all' to include every registered extractor (optionally mix with names)."
        ),
    )
    p.add_argument("--digitA", type=int, default=1)
    p.add_argument("--digitB", type=int, default=2)
    p.add_argument(
        "--base-ratio",
        type=float,
        default=0.25,
        help="Pretrain base subset ratio (0-1). Used by Pretrain25ThenDigitAThenDigitB_25_75.",
    )
    p.add_argument("--stage-epochs", type=int, default=1)
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=3003)
    p.add_argument("--activation", default="relu")
    p.add_argument(
        "--he-init",
        type=str,
        default="3",
        metavar="K",
        help=(
            "First K Linear/Conv layers use He (Kaiming) init; 0 = all random "
            "(N(0, init_epsilon)); 'all' or a negative value = He for all layers."
        ),
    )
    p.add_argument(
        "--init-epsilon",
        type=float,
        default=1e-12,
        metavar="VAR",
        help="Variance for Gaussian init of non-He weights/biases (mean 0).",
    )
    p.add_argument("--checkpoint-cadence", default="every_epoch")
    p.add_argument(
        "--save-model-cp",
        action="store_true",
        help="Write model state_dict to checkpoints/<tag>.pt at each checkpoint.",
    )
    p.add_argument(
        "--shampoo-preconditioner-epsilon",
        type=float,
        default=None,
        metavar="EPS",
        help=(
            "DistributedShampoo preconditioner epsilon (inverse-root damping); "
            "see shampoo_preconditioner_epsilon in JSON config. "
            "Overrides the JSON value when both are set."
        ),
    )
    p.add_argument(
        "--internal-keep-tensors",
        action="store_true",
        help="Keep checkpoint activations as GPU tensors; flush to numpy only when >6 GB.",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials (full stage-sequence repetitions). Default: 3.",
    )
    p.add_argument(
        "--trial-variability",
        type=str,
        default="",
        metavar="VARIABILITY_TYPE",
        help=(
            "Variability applied between trials, interpreted per experiment. "
            "Supported: 'change_digits' (shift digit labels by +2 mod n_digits//2). "
            "Default: '' (no change)."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run all requested optimizers even if outputs already exist. "
            "Does not override a mismatched training config.json."
        ),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.config is not None:
        raw = json.loads(args.config.read_text())
        experiment_classes = _parse_csv(raw["experiment_class"])
        experiment_kwargs = raw.get("experiment_config", {})
        model_classes = _parse_csv(raw["model_class"])
        model_kwargs = raw.get("model_config", {})
        if not model_kwargs:
            model_kwargs = {"activation": raw.get("activation", "relu")}
        elif "activation" not in model_kwargs:
            model_kwargs["activation"] = raw.get("activation", "relu")
        optimizer_names = _parse_csv(raw["optimizer"])
        base_lr = raw.get("base_lr", 1e-3)
        stage_epochs = raw.get("stage_epochs", 1)
        batch_size = raw.get("batch_size", 1)
        seed = raw.get("seed", 3003)
        activation = raw.get("activation", "relu")
        checkpoint_cadence = raw.get("checkpoint_cadence", "every_epoch")
        save_model_cp = bool(raw.get("save_model_cp", False)) or args.save_model_cp
        he_init = raw.get("he_init", 3)
        init_epsilon = float(raw.get("init_epsilon", 1e-10))
        internal_keep_tensors = bool(raw.get("internal_keep_tensors", False)) or args.internal_keep_tensors
        trials = int(raw.get("trials", args.trials))
        trial_variability = str(raw.get("trial_variability", args.trial_variability))
        force = raw.get("force", False) or args.force
    else:
        if not (args.experiment and args.model and args.optimizer):
            print(
                "ERROR: Provide --config or all of --experiment, --model, --optimizer.",
                file=sys.stderr,
            )
            sys.exit(1)
        experiment_classes = _parse_csv(args.experiment)
        experiment_kwargs = {
            "digitA": args.digitA,
            "digitB": args.digitB,
            "base_ratio": args.base_ratio,
        }
        model_classes = _parse_csv(args.model)
        model_kwargs = {"activation": args.activation}
        optimizer_names = _parse_csv(args.optimizer)
        base_lr = args.base_lr
        stage_epochs = args.stage_epochs
        batch_size = args.batch_size
        seed = args.seed
        activation = args.activation
        checkpoint_cadence = args.checkpoint_cadence
        save_model_cp = args.save_model_cp
        he_init = args.he_init
        init_epsilon = args.init_epsilon
        internal_keep_tensors = args.internal_keep_tensors
        trials = args.trials
        trial_variability = args.trial_variability
        force = args.force
        raw = {}  # no JSON; CLI-only path

    opt_extra_kwargs: dict = {}
    eps_from_json = raw.get(SHAMPOO_PRECONDITIONER_EPSILON_KEY)
    if eps_from_json is not None:
        opt_extra_kwargs[SHAMPOO_PRECONDITIONER_EPSILON_KEY] = float(eps_from_json)
    if args.shampoo_preconditioner_epsilon is not None:
        opt_extra_kwargs[SHAMPOO_PRECONDITIONER_EPSILON_KEY] = float(
            args.shampoo_preconditioner_epsilon
        )

    try:
        model_classes = expand_registry_selection(
            model_classes, available=list_models()
        )
        optimizer_names = expand_registry_selection(
            optimizer_names, available=list_extractors()
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        validate_he_init(he_init)
    except (TypeError, ValueError) as e:
        print(f"ERROR: Invalid --he-init / he_init: {e}", file=sys.stderr)
        sys.exit(1)
    if init_epsilon < 0:
        print("ERROR: init_epsilon must be non-negative.", file=sys.stderr)
        sys.exit(1)

    _validate_names(experiment_classes, model_classes, optimizer_names)

    # Build cartesian product of tasks
    all_tasks: list[_Task] = []
    for exp_cls, mod_cls, opt_name in itertools.product(
        experiment_classes, model_classes, optimizer_names
    ):
        exp_kw = experiment_kwargs
        mod_kw = {**model_kwargs}
        if "activation" not in mod_kw:
            mod_kw["activation"] = activation

        experiment = get_experiment(exp_cls, **exp_kw)
        model = get_model(mod_cls, **mod_kw)
        config = build_training_config(
            experiment_class=exp_cls,
            experiment_config=exp_kw,
            model_class=mod_cls,
            model_config=mod_kw,
            stage_epochs=stage_epochs,
            base_lr=base_lr,
            batch_size=batch_size,
            seed=seed,
            activation=mod_kw.get("activation", activation),
            checkpoint_cadence=checkpoint_cadence,
            save_model_cp=save_model_cp,
            he_init=he_init,
            init_epsilon=init_epsilon,
            internal_keep_tensors=internal_keep_tensors,
            trials=trials,
            trial_variability=trial_variability,
        )
        exp_dir = RESULTS_ROOT / experiment.experiment_id()
        rid = result_id_for_config(
            config,
            experiment_dir=exp_dir,
            model_id=model.model_id(),
        )
        results_dir = exp_dir / rid / model.model_id()

        opt_class, opt_kwargs = _resolve_optimizer(
            opt_name, base_lr, **opt_extra_kwargs
        )
        all_tasks.append(
            _Task(
                experiment_class=exp_cls,
                experiment_kwargs=exp_kw,
                model_class=mod_cls,
                model_kwargs=mod_kw,
                optimizer_name=opt_name,
                optimizer_class=opt_class,
                optimizer_kwargs=opt_kwargs,
                config=config,
                seed=seed,
                results_dir=results_dir,
            )
        )

    # Filter: skip existing unless force
    to_run: list[_Task] = []
    skipped: list[tuple[str, str, str, str]] = []
    for task in all_tasks:
        opt_dir = task.optimizer_dir
        if optimizer_has_results(opt_dir):
            if force:
                to_run.append(task)
            else:
                extractor = get_extractor(
                    task.optimizer_class, **task.optimizer_kwargs
                )
                skipped.append(
                    (
                        task.results_dir.parent.parent.name,  # experiment_id
                        task.results_dir.parent.name,  # result_id
                        task.results_dir.name,  # model_id
                        extractor.optimizer_id(),
                    )
                )
        else:
            to_run.append(task)

    if skipped:
        print(
            "WARNING: Skipping simulations that already have results "
            "(use --force to overwrite):",
            file=sys.stderr,
        )
        for exp_id, rid, mod_id, opt_id in skipped:
            print(f"  - {exp_id} / {rid} / {mod_id} / {opt_id}", file=sys.stderr)

    if not to_run:
        print("Nothing to run. All requested combinations already have results.")
        return 0

    # Guards and config persistence per (experiment, model)
    exp_model_done: set[tuple[str, str]] = set()
    for task in to_run:
        key = (task.experiment_class, task.model_class)
        if key in exp_model_done:
            continue
        exp_model_done.add(key)
        results_dir = task.results_dir
        config = task.config
        try:
            guard_training_config(results_dir, config)
        except ConfigConflictError as e:
            print(f"CONFIG CONFLICT:\n{e}", file=sys.stderr)
            sys.exit(1)
        save_config(results_dir, config)

    # Per-task: optimizer guard and save
    for task in to_run:
        extractor = get_extractor(task.optimizer_class, **task.optimizer_kwargs)
        try:
            guard_optimizer_config(
                task.optimizer_dir,
                extractor.optimizer_config(),
                force=force,
            )
        except ConfigConflictError as e:
            print(f"OPTIMIZER CONFIG CONFLICT:\n{e}", file=sys.stderr)
            sys.exit(1)
        save_optimizer_config(task.optimizer_dir, extractor.optimizer_config())

    # Update metadata with all optimizer_ids per (experiment, model)
    exp_model_optimizers: dict[tuple[str, str, Path], set[str]] = {}
    for task in to_run:
        key = (task.experiment_class, task.model_class, task.results_dir)
        extractor = get_extractor(task.optimizer_class, **task.optimizer_kwargs)
        exp_model_optimizers.setdefault(key, set()).add(extractor.optimizer_id())
    for (exp_cls, mod_cls, results_dir), opt_ids in exp_model_optimizers.items():
        existing = load_existing_optimizer_ids(results_dir)
        merged = sorted(set(existing) | opt_ids)
        save_metadata(
            results_dir,
            experiment_class=exp_cls,
            model_class=mod_cls,
            optimizer_ids=merged,
        )

    # Run tasks (sequential; one GPU index per task when multiple CUDA devices exist)
    for i, task in enumerate(to_run):
        device_id = i % max(1, torch.cuda.device_count())
        print(
            f"Training: {task.experiment_class} / {task.model_class} / "
            f"{task.optimizer_name}  device={device_id}"
        )
        _run_single_task(task, device_id)

    print("Done. Results saved to:", RESULTS_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
