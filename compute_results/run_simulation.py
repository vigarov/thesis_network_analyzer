#!/usr/bin/env python3
"""Main entrypoint for running training simulations.

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
        --stage-epochs 3 --base-lr 1e-3 --batch-size 1 --seed 3003

Multiple optimizers/models/experiments (CSV lists)::

    uv run run-simulation --experiment Exp1,Exp2 --model M1,M2 --optimizer adam,adagrad

Add ``--force`` to overwrite existing results. Use ``--parallel K`` to run K simulations
in parallel.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import torch

# Ensure the project root is on sys.path so bare package imports work when
# the script is invoked directly (e.g. ``python compute_results/run_simulation.py``).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from compute_results.config_guard import (
    ConfigConflictError,
    build_training_config,
    guard_optimizer_config,
    guard_training_config,
    load_existing_optimizer_ids,
    optimizer_has_results,
    save_config,
    save_metadata,
    save_optimizer_config,
)
from compute_results.training_loop import train_with_config
from experiments import get_experiment, list_experiments
from models import apply_xavier_init, get_model, list_models
from optimizers import get_extractor, list_extractors

RESULTS_ROOT = _PROJECT_ROOT / "results"

OPTIMIZER_SHORTHAND: dict[str, tuple[str, dict]] = {
    "sgd": ("SGDExtractor", {}),
    "adam": ("AdamExtractor", {}),
    "adagrad": ("AdaGradExtractor", {}),
}


def _parse_csv(value: str | list) -> list[str]:
    """Split a comma-separated string or list into non-empty parts."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    return [str(value).strip()]


def _resolve_optimizer(name: str, base_lr: float) -> tuple[str, dict]:
    """Map short names like 'adam' to (ExtractorClass, kwargs)."""
    name = name.strip()
    if name in OPTIMIZER_SHORTHAND:
        cls_name, extra = OPTIMIZER_SHORTHAND[name]
        return cls_name, {**extra, "lr": base_lr}
    return name, {"lr": base_lr}


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
    config: dict
    seed: int
    results_dir: Path

    @property
    def optimizer_dir(self) -> Path:
        ext_cls, ext_kwargs = _resolve_optimizer(
            self.optimizer_name, self.config["base_lr"]
        )
        extractor = get_extractor(ext_cls, **ext_kwargs)
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

    import torch as _torch

    from compute_results.training_loop import train_with_config
    from experiments import get_experiment
    from models import apply_xavier_init, get_model
    from optimizers import get_extractor

    t = task
    experiment = get_experiment(t.experiment_class, **t.experiment_kwargs)
    model = get_model(t.model_class, **t.model_kwargs)
    ext_cls, ext_kwargs = _resolve_optimizer(t.optimizer_name, t.config["base_lr"])
    extractor = get_extractor(ext_cls, **ext_kwargs)

    _torch.manual_seed(t.seed)
    apply_xavier_init(model)

    num_gpus = _torch.cuda.device_count()
    if num_gpus > 0:
        device = _torch.device(f"cuda:{device_id % num_gpus}")
    else:
        device = _torch.device("cpu")

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
        print(f"Task failed: {t.experiment_class}/{t.model_class}/{t.optimizer_name}: {e}")
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
        help="Model class name(s), comma-separated (e.g. DNN5Hidden64,CNN3ConvK5Out64ThenFC64).",
    )
    p.add_argument(
        "--optimizer",
        help="Optimizer shorthand(s) or extractor class name(s), comma-separated (e.g. adam,adagrad).",
    )
    p.add_argument("--digitA", type=int, default=1)
    p.add_argument("--digitB", type=int, default=2)
    p.add_argument("--stage-epochs", type=int, default=3)
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=3003)
    p.add_argument("--activation", default="relu")
    p.add_argument("--checkpoint-cadence", default="every_epoch")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results even if they exist.",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="K",
        help="Run up to K simulations in parallel (default: 1).",
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
        stage_epochs = raw.get("stage_epochs", 3)
        batch_size = raw.get("batch_size", 1)
        seed = raw.get("seed", 3003)
        activation = raw.get("activation", "relu")
        checkpoint_cadence = raw.get("checkpoint_cadence", "every_epoch")
        force = raw.get("force", False) or args.force
        parallel = raw.get("parallel", args.parallel)
    else:
        if not (args.experiment and args.model and args.optimizer):
            print(
                "ERROR: Provide --config or all of --experiment, --model, --optimizer.",
                file=sys.stderr,
            )
            sys.exit(1)
        experiment_classes = _parse_csv(args.experiment)
        experiment_kwargs = {"digitA": args.digitA, "digitB": args.digitB}
        model_classes = _parse_csv(args.model)
        model_kwargs = {"activation": args.activation}
        optimizer_names = _parse_csv(args.optimizer)
        base_lr = args.base_lr
        stage_epochs = args.stage_epochs
        batch_size = args.batch_size
        seed = args.seed
        activation = args.activation
        checkpoint_cadence = args.checkpoint_cadence
        force = args.force
        parallel = args.parallel

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
        results_dir = RESULTS_ROOT / experiment.experiment_id() / model.model_id()

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
        )

        all_tasks.append(
            _Task(
                experiment_class=exp_cls,
                experiment_kwargs=exp_kw,
                model_class=mod_cls,
                model_kwargs=mod_kw,
                optimizer_name=opt_name,
                config=config,
                seed=seed,
                results_dir=results_dir,
            )
        )

    # Filter: skip existing unless force
    to_run: list[_Task] = []
    skipped: list[tuple[str, str, str]] = []
    for task in all_tasks:
        opt_dir = task.optimizer_dir
        if optimizer_has_results(opt_dir):
            if force:
                to_run.append(task)
            else:
                ext_cls, ext_kwargs = _resolve_optimizer(
                    task.optimizer_name, task.config["base_lr"]
                )
                extractor = get_extractor(ext_cls, **ext_kwargs)
                skipped.append(
                    (
                        task.results_dir.parent.name,  # experiment_id
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
        for exp_id, mod_id, opt_id in skipped:
            print(f"  - {exp_id} / {mod_id} / {opt_id}", file=sys.stderr)

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
            guard_training_config(results_dir, config, force=force)
        except ConfigConflictError as e:
            print(f"CONFIG CONFLICT:\n{e}", file=sys.stderr)
            sys.exit(1)
        save_config(results_dir, config)

    # Per-task: optimizer guard and save
    for task in to_run:
        ext_cls, ext_kwargs = _resolve_optimizer(
            task.optimizer_name, task.config["base_lr"]
        )
        extractor = get_extractor(ext_cls, **ext_kwargs)
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
        ext_cls, ext_kwargs = _resolve_optimizer(
            task.optimizer_name, task.config["base_lr"]
        )
        extractor = get_extractor(ext_cls, **ext_kwargs)
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

    # Run tasks
    if parallel <= 1:
        for i, task in enumerate(to_run):
            device_id = i % max(1, torch.cuda.device_count())
            print(
                f"Training: {task.experiment_class} / {task.model_class} / "
                f"{task.optimizer_name}  device={device_id}"
            )
            _run_single_task(task, device_id)
    else:
        K = min(parallel, len(to_run))
        print(f"Running {len(to_run)} simulations with {K} parallel workers")
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=K, mp_context=ctx) as pool:
            futures = {
                pool.submit(_run_single_task, task, i): task
                for i, task in enumerate(to_run)
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    exp_id, mod_id, opt_id, ok = future.result()
                    status = "OK" if ok else "FAILED"
                    print(f"  {exp_id} / {mod_id} / {opt_id}: {status}")
                except Exception as e:
                    print(f"  {task.experiment_class}/{task.model_class}/{task.optimizer_name}: {e}")

    print("Done. Results saved to:", RESULTS_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
