#!/usr/bin/env python3
"""Train experiments across seeds/optimizers and export recovery periods.

Reads base JSON configs from an input directory (non-recursively), applies
CLI seed(s) over each config's placeholder seed, trains, and writes compact
period CSVs under `results_many_seeds/<dataset>/...`.
"""
import argparse
import contextlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.constants import set_results_root
from analysis.io import _metrics, _nts, _pp_dead, _pp_neuron_digit
from analysis.scoring_helpers import (
    filter_periods,
    get_all_points_of_max_acceleration,
    get_iteration,
    get_n_assigned_digits,
    get_recovery_periods,
    index_grp,
    is_pretrain_shuffle_mislabel_experiment,
    remove_inelligible_periods,
)
from compute_results.config_guard import (
    build_training_config,
    guard_optimizer_config,
    guard_training_config,
    load_existing_optimizer_ids,
    load_initial_model_state_dict,
    normalize_loss,
    result_id_for_config,
    save_config,
    save_metadata,
    save_optimizer_config,
    validate_experiment_config_constraints,
    validate_initial_model_path,
)
from compute_results.constants import INITIAL_MODEL_OPTIMIZER_STATE_FILENAME
from compute_results.defaults import SHAMPOO_PRECONDITIONER_EPSILON_KEY
from experiments import get_experiment
from models import get_model, parse_he_init
from optimizers import get_extractor as _original_get_extractor
from optimizers.base import OptimizerSignalExtractor
from scripts.run_simulation import _Task, _resolve_optimizer, _run_single_task

DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "input_configs"
DEFAULT_PRETRAINED_ROOT: Path | None = None
DEFAULT_OUTPUT_ROOT: Path | None = None
DEFAULT_SCRATCH_ROOT: Path | None = None
SCRATCH_SUBDIR = "scratch"
DEFAULT_OPTIMIZER_NAMES = ["sgd", "adam", "adagrad", "pure_shampoo", "grafted_shampoo"]
DEFAULT_NOTEBOOK_SEEDS = [6, 7, 14, 30, 31, 35]

PER_RUN_CSV_NAME = "recovery_periods.csv"
RUN_SUMMARY_NAME = "run_summary.csv"
PERIODS_LONG_NAME = "periods_long.csv"
FAILURES_NAME = "failures.csv"

_SEED_SUFFIX_RE = re.compile(r"_seed\d+$")


_MULTI_SUFFIX_RE = re.compile(r"_multi$")


def config_slug_from_stem(stem: str) -> str:
    """Config filename stem normalized for grouping (drops `_multi` / `_seed<N>`)."""
    s = _SEED_SUFFIX_RE.sub("", stem)
    return _MULTI_SUFFIX_RE.sub("", s)


def list_config_paths(config_dir: Path) -> list[Path]:
    """Non-recursive `*.json` configs in config_dir."""
    return sorted(p for p in config_dir.glob("*.json") if p.is_file())


def infer_dataset(config_dir: Path, config_paths: list[Path]) -> str:
    if config_dir.name == "cifar10":
        return "cifar10"
    if config_paths:
        raw = json.loads(config_paths[0].read_text())
        return str(raw.get("dataset", "mnist"))
    return "mnist"


def default_output_root(config_dir: Path, config_paths: list[Path]) -> Path:
    dataset = infer_dataset(config_dir, config_paths)
    return _PROJECT_ROOT / "results_many_seeds" / dataset


def scratch_root_for(output_root: Path) -> Path:
    return output_root / SCRATCH_SUBDIR


def resolve_scratch_root(output_root: Path, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    return scratch_root_for(output_root)


def resolve_pretrained_base(raw: dict[str, Any], *, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    raw_path = str(raw.get("use_initial_model", "") or "").strip()
    if not raw_path:
        raise ValueError(
            "pretrain config missing use_initial_model; pass --pretrained-root"
        )
    base = Path(raw_path)
    if not base.is_absolute():
        base = (_PROJECT_ROOT / base).resolve()
    return base


def pretrained_dir_for_optimizer(
    raw: dict[str, Any],
    oid: str,
    *,
    override: Path | None,
) -> str:
    return str((resolve_pretrained_base(raw, override=override) / oid).resolve())


def parse_config_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    experiment_kwargs = dict(raw.get("experiment_config", {}))
    if (
        "num_trial_samples" not in experiment_kwargs
        and "num_stage_samples" in experiment_kwargs
    ):
        experiment_kwargs["num_trial_samples"] = int(experiment_kwargs["num_stage_samples"])
    model_kwargs = dict(raw.get("model_config", {}) or {})
    if not model_kwargs:
        model_kwargs = {"activation": raw.get("activation", "relu")}
    elif "activation" not in model_kwargs:
        model_kwargs["activation"] = raw.get("activation", "relu")
    return {
        "experiment_class": str(raw["experiment_class"]),
        "experiment_kwargs": experiment_kwargs,
        "model_class": str(raw["model_class"]),
        "model_kwargs": model_kwargs,
        "base_lr": float(raw.get("base_lr", 1e-3)),
        "trial_epochs": int(raw.get("trial_epochs", raw.get("stage_epochs", 1))),
        "batch_size": int(raw.get("batch_size", 1)),
        "seed": int(raw.get("seed", 3003)),
        "activation": str(raw.get("activation", "relu")),
        "checkpoint_cadence": str(raw.get("checkpoint_cadence", "every_epoch")),
        "save_model_cp": bool(raw.get("save_model_cp", False)),
        "he_init": raw.get("he_init", 3),
        "init_epsilon": float(raw.get("init_epsilon", 1e-10)),
        "internal_keep_tensors": bool(raw.get("internal_keep_tensors", False)),
        "experiment_runs": int(raw.get("experiment_runs", raw.get("trials", 3))),
        "experiment_variability": str(
            raw.get("experiment_variability", raw.get("trial_variability", ""))
        ),
        "loss": str(raw.get("loss", "ce")),
        "config_initial_model_mode": str(raw.get("initial_model_mode", "init") or "init"),
        "dataset": str(raw.get("dataset", "mnist")),
    }


def _parse_csv_ints(value: str) -> list[int]:
    return [int(p.strip()) for p in value.split(",") if p.strip()]


def _parse_csv_strs(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


def optimizer_id_for_name(
    optimizer_name: str,
    base_lr: float,
    *,
    dataset: str = "mnist",
) -> str:
    opt_class, opt_kwargs = _resolve_optimizer(optimizer_name, base_lr, dataset=dataset)
    return get_extractor(opt_class, **opt_kwargs).optimizer_id()


@contextlib.contextmanager
def silent_optimizer_signals():
    """Monkeypatch extractors to skip expensive per-step signal collection."""

    def _patch_extractor(extractor: OptimizerSignalExtractor) -> OptimizerSignalExtractor:
        extractor.signal_names = lambda: []  # type: ignore[method-assign, assignment]
        extractor.on_before_step = lambda model, optimizer: {}  # type: ignore[method-assign, assignment]
        extractor.on_after_step = lambda model, optimizer: {}  # type: ignore[method-assign, assignment]
        extractor.on_after_step_shampoo_blocks = lambda model, optimizer: {}  # type: ignore[method-assign, assignment]
        return extractor

    def patched_get_extractor(name: str, **kwargs: Any) -> OptimizerSignalExtractor:
        return _patch_extractor(_original_get_extractor(name, **kwargs))

    import optimizers
    import optimizers.base as optimizers_base

    orig_pkg = optimizers.get_extractor
    orig_base = optimizers_base.get_extractor
    optimizers.get_extractor = patched_get_extractor  # type: ignore[assignment]
    optimizers_base.get_extractor = patched_get_extractor  # type: ignore[assignment]
    try:
        yield
    finally:
        optimizers.get_extractor = orig_pkg  # type: ignore[assignment]
        optimizers_base.get_extractor = orig_base  # type: ignore[assignment]


def get_extractor(name: str, **kwargs: Any) -> OptimizerSignalExtractor:
    return _original_get_extractor(name, **kwargs)


def per_run_csv_path(
    output_root: Path,
    *,
    eid: str,
    rid: str,
    mid: str,
    oid: str,
) -> Path:
    return (
        output_root
        / eid
        / rid
        / mid
        / "optimizers"
        / oid
        / PER_RUN_CSV_NAME
    )


def build_jobs(
    config_dir: Path,
    optimizer_names: list[str],
    seeds: list[int],
    *,
    exp_slugs: list[str] | None = None,
) -> list[dict[str, Any]]:
    allowed = set(exp_slugs) if exp_slugs else None
    jobs: list[dict[str, Any]] = []
    for cfg_path in list_config_paths(config_dir):
        config_slug = config_slug_from_stem(cfg_path.stem)
        if allowed is not None and config_slug not in allowed:
            continue
        parsed = parse_config_json(cfg_path)
        base_lr = parsed["base_lr"]
        for seed in seeds:
            for opt_name in optimizer_names:
                jobs.append(
                    {
                        "config_path": cfg_path,
                        "config_stem": f"{cfg_path.stem}_seed{seed}",
                        "config_slug": config_slug,
                        "optimizer_name": opt_name,
                        "base_lr": base_lr,
                        "seed": seed,
                        "dataset": parsed["dataset"],
                        "initial_model_mode": parsed["config_initial_model_mode"],
                    }
                )
    return jobs


def available_config_slugs(config_dir: Path) -> list[str]:
    return sorted(config_slug_from_stem(p.stem) for p in list_config_paths(config_dir))


def load_completed_keys(run_summary_path: Path) -> set[tuple[str, str]]:
    if not run_summary_path.is_file():
        return set()
    df = pd.read_csv(run_summary_path)
    if df.empty:
        return set()
    return set(zip(df["config_stem"].astype(str), df["optimizer_id"].astype(str)))


def load_recovery_paths(run_summary_path: Path) -> dict[tuple[str, str], Path]:
    """Map resume keys to the per-run CSV path recorded in run_summary."""
    if not run_summary_path.is_file():
        return {}
    df = pd.read_csv(run_summary_path)
    if df.empty or "recovery_periods_path" not in df.columns:
        return {}
    out: dict[tuple[str, str], Path] = {}
    for _, row in df.iterrows():
        key = (str(row["config_stem"]), str(row["optimizer_id"]))
        out[key] = Path(str(row["recovery_periods_path"]))
    return out


def validate_pretrained_checkpoints(
    jobs: list[dict[str, Any]],
    optimizer_names: list[str],
    seeds: list[int],
    *,
    pretrained_override: Path | None,
) -> list[str]:
    """Return list of error messages; empty if all checks pass."""
    errors: list[str] = []
    checked: set[tuple[str, str, str, int]] = set()
    for job in jobs:
        if job["initial_model_mode"] != "pretrain":
            continue
        raw = json.loads(job["config_path"].read_text())
        dataset = job["dataset"]
        mod_cls = str(raw["model_class"])
        model_kwargs = dict(raw.get("model_config", {}) or {})
        if not model_kwargs:
            model_kwargs = {"activation": raw.get("activation", "relu")}
        model = get_model(mod_cls, **model_kwargs)
        for opt_name in optimizer_names:
            opt_class, opt_kwargs = _resolve_optimizer(
                opt_name, float(job["base_lr"]), dataset=dataset
            )
            extractor = get_extractor(opt_class, **opt_kwargs)
            oid = extractor.optimizer_id()
            for seed in seeds:
                cache_key = (str(job["config_path"]), oid, mod_cls, seed)
                if cache_key in checked:
                    continue
                checked.add(cache_key)
                label = f"{job['config_slug']} / {opt_name} / {oid} / seed={seed}"
                try:
                    pt_root = resolve_pretrained_base(raw, override=pretrained_override)
                    validate_initial_model_path(
                        str(pt_root),
                        optimizer_registry_classes=[opt_class],
                        optimizer_ids=[oid],
                        seed=seed,
                    )
                    state = load_initial_model_state_dict(
                        str(pt_root),
                        registry_class_name=opt_class,
                        optimizer_id=oid,
                        seed=seed,
                    )
                    model.load_state_dict(state, strict=True)
                    opt_path = (
                        pt_root / oid / str(seed) / INITIAL_MODEL_OPTIMIZER_STATE_FILENAME
                    )
                    torch.load(opt_path, map_location="cpu", weights_only=False)
                except Exception as exc:
                    errors.append(f"{label}: {exc}")
    return errors


def extract_period_metrics(
    eid: str,
    mid: str,
    oid: str,
    rid: str,
) -> tuple[int, pd.DataFrame]:
    """Extract recovery periods with t_start; output compact column names."""
    empty_cols = ["nid", "pid", "layer", "t_start", "length_iter"]
    metrics = _metrics(eid, mid, oid, rid=rid, w_cache=False)
    df_nd = cast(pd.DataFrame, _pp_neuron_digit(eid, mid, oid, rid=rid, w_cache=False))
    if df_nd is None or df_nd.empty:
        return 0, pd.DataFrame(columns=empty_cols)

    df_dead = cast(pd.DataFrame | None, _pp_dead(eid, mid, oid, rid=rid, w_cache=False))
    if df_dead is not None and not df_dead.empty:
        df_dead = df_dead.copy()
        df_dead["iteration"] = df_dead["checkpoint_idx"].map(lambda c: get_iteration(metrics, c))
    n_assigned = get_n_assigned_digits(df_nd)
    recovery_periods = get_recovery_periods(
        metrics, df_nd, n_assigned, df_dead, mode="assigned_same"
    )
    recovery_periods = remove_inelligible_periods(
        recovery_periods,
        require_until_trial_end=not is_pretrain_shuffle_mislabel_experiment(eid),
    )
    if df_dead is not None and not df_dead.empty:
        recovery_periods = filter_periods(recovery_periods, df_dead, strict=False)

    n_periods = int(len(recovery_periods))
    if n_periods == 0:
        return 0, pd.DataFrame(columns=empty_cols)

    nts = _nts(eid, mid, oid, rid=rid, w_cache=False)
    peak_agg = "max" if is_pretrain_shuffle_mislabel_experiment(eid) else "min"
    pma = get_all_points_of_max_acceleration(
        nts, recovery_periods, peak_time_aggregate=peak_agg
    ).reset_index()
    periods_df = recovery_periods[index_grp + ["layer_name", "total_length_iter"]].merge(
        pma[index_grp + ["point_of_max_acceleration"]],
        on=index_grp,
        how="left",
    )
    periods_df = periods_df.rename(
        columns={
            "neuron_id": "nid",
            "period_id": "pid",
            "layer_name": "layer",
            "total_length_iter": "length_iter",
            "point_of_max_acceleration": "t_start",
        }
    )
    return n_periods, cast(pd.DataFrame, periods_df[empty_cols])


def delete_scratch_run(results_dir: Path, scratch_root: Path) -> None:
    """Remove one transient training run and prune empty parents under scratch_root."""
    if results_dir.exists():
        shutil.rmtree(results_dir)
    parent = results_dir.parent
    while parent != scratch_root and scratch_root in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def run_one_job(
    job: dict[str, Any],
    *,
    pretrained_override: Path | None,
    scratch_root: Path,
    output_root: Path,
    device_id: int,
) -> dict[str, Any]:
    config_path = job["config_path"]
    optimizer_name = job["optimizer_name"]
    config_slug = job["config_slug"]
    dataset = job["dataset"]

    raw = json.loads(config_path.read_text())
    parsed = parse_config_json(config_path)
    exp_cls = parsed["experiment_class"]
    exp_kw = parsed["experiment_kwargs"]
    mod_cls = parsed["model_class"]
    mod_kw = dict(parsed["model_kwargs"])
    seed = int(job["seed"])
    base_lr = parsed["base_lr"]

    parse_he_init(parsed["he_init"])
    normalize_loss(parsed["loss"])
    validate_experiment_config_constraints(
        exp_cls, exp_kw, experiment_runs=parsed["experiment_runs"]
    )

    opt_extra: dict[str, Any] = {}
    if raw.get(SHAMPOO_PRECONDITIONER_EPSILON_KEY) is not None:
        opt_extra[SHAMPOO_PRECONDITIONER_EPSILON_KEY] = float(
            raw[SHAMPOO_PRECONDITIONER_EPSILON_KEY]
        )
    opt_class, opt_kwargs = _resolve_optimizer(
        optimizer_name, base_lr, dataset=dataset, **opt_extra
    )
    extractor = get_extractor(opt_class, **opt_kwargs)
    oid = extractor.optimizer_id()

    initial_model_mode = parsed["config_initial_model_mode"]
    use_initial_model = ""
    if initial_model_mode == "pretrain":
        use_initial_model = pretrained_dir_for_optimizer(
            raw, oid, override=pretrained_override
        )

    experiment = get_experiment(exp_cls, dataset=dataset, **exp_kw)
    model = get_model(mod_cls, **mod_kw)
    config = build_training_config(
        experiment_class=exp_cls,
        experiment_config=exp_kw,
        model_class=mod_cls,
        model_config=mod_kw,
        trial_epochs=parsed["trial_epochs"],
        base_lr=base_lr,
        batch_size=parsed["batch_size"],
        seed=seed,
        activation=mod_kw.get("activation", parsed["activation"]),
        checkpoint_cadence=parsed["checkpoint_cadence"],
        save_model_cp=parsed["save_model_cp"],
        he_init=parsed["he_init"],
        init_epsilon=parsed["init_epsilon"],
        internal_keep_tensors=parsed["internal_keep_tensors"],
        experiment_runs=parsed["experiment_runs"],
        experiment_variability=parsed["experiment_variability"],
        loss=parsed["loss"],
        use_initial_model=use_initial_model,
        initial_model_mode=initial_model_mode,
        optimizer_registry_classes_for_initial_model=[opt_class] if use_initial_model else None,
        optimizer_ids_for_initial_model=[oid] if use_initial_model else None,
    )

    eid = experiment.experiment_id()
    mid = model.model_id()
    rid = result_id_for_config(
        config,
        experiment_dir=scratch_root / eid,
        model_id=mid,
    )
    results_dir = scratch_root / eid / rid / mid

    delete_scratch_run(results_dir, scratch_root)
    set_results_root(scratch_root)

    try:
        guard_training_config(results_dir, config)
        save_config(results_dir, config)
        guard_optimizer_config(
            results_dir / "optimizers" / oid,
            extractor.optimizer_config(),
            force=True,
        )
        save_optimizer_config(
            results_dir / "optimizers" / oid,
            extractor.optimizer_config(),
        )
        existing = load_existing_optimizer_ids(results_dir)
        save_metadata(
            results_dir,
            experiment_class=exp_cls,
            model_class=mod_cls,
            optimizer_ids=sorted(set(existing) | {oid}),
        )

        task = _Task(
            experiment_class=exp_cls,
            experiment_kwargs=exp_kw,
            model_class=mod_cls,
            model_kwargs=mod_kw,
            optimizer_name=optimizer_name,
            optimizer_class=opt_class,
            optimizer_kwargs=opt_kwargs,
            config=config,
            seed=seed,
            results_dir=results_dir,
            dataset=dataset,
        )
        with silent_optimizer_signals():
            _, _, _, ok = _run_single_task(task, device_id=device_id)
        if not ok:
            raise RuntimeError(f"Training failed: {exp_cls} / {oid} / seed={seed}")

        n_periods, periods_df = extract_period_metrics(eid, mid, oid, rid)

        out_csv = per_run_csv_path(output_root, eid=eid, rid=rid, mid=mid, oid=oid)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        periods_df.to_csv(out_csv, index=False)

        return {
            "config_stem": job["config_stem"],
            "config_slug": config_slug,
            "experiment_class": exp_cls,
            "seed": seed,
            "optimizer_id": oid,
            "optimizer_name": optimizer_name,
            "eid": eid,
            "mid": mid,
            "rid": rid,
            "n_periods": n_periods,
            "periods_df": periods_df,
            "recovery_periods_path": str(out_csv),
        }
    finally:
        delete_scratch_run(results_dir, scratch_root)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _empty_run_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "config_stem",
            "config_slug",
            "experiment_class",
            "seed",
            "optimizer_id",
            "optimizer_name",
            "eid",
            "mid",
            "rid",
            "n_periods",
            "recovery_periods_path",
        ]
    )


def _empty_periods_long() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "config_stem",
            "config_slug",
            "seed",
            "optimizer_id",
            "eid",
            "mid",
            "rid",
            "nid",
            "pid",
            "layer",
            "t_start",
            "length_iter",
        ]
    )


def load_aggregate_tables(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_summary_path = output_root / RUN_SUMMARY_NAME
    periods_long_path = output_root / PERIODS_LONG_NAME
    if run_summary_path.is_file():
        run_summary = pd.read_csv(run_summary_path)
    else:
        run_summary = _empty_run_summary()
    if periods_long_path.is_file():
        periods_long = pd.read_csv(periods_long_path)
    else:
        periods_long = _empty_periods_long()
    return run_summary, periods_long


def append_and_save_aggregate(
    output_root: Path,
    run_summary: pd.DataFrame,
    periods_long: pd.DataFrame,
    run_row: dict[str, Any],
    periods_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_stem = str(run_row["config_stem"])
    optimizer_id = str(run_row["optimizer_id"])
    key_mask = ~(
        (run_summary["config_stem"].astype(str) == config_stem)
        & (run_summary["optimizer_id"].astype(str) == optimizer_id)
    )
    run_summary = run_summary.loc[key_mask].copy()
    if not periods_long.empty:
        periods_key_mask = ~(
            (periods_long["config_stem"].astype(str) == config_stem)
            & (periods_long["optimizer_id"].astype(str) == optimizer_id)
        )
        periods_long = periods_long.loc[periods_key_mask].copy()

    summary_row = {k: v for k, v in run_row.items() if k != "periods_df"}
    run_summary = pd.concat([run_summary, pd.DataFrame([summary_row])], ignore_index=True)

    if not periods_df.empty:
        pl = periods_df.copy()
        for col in (
            "config_stem",
            "config_slug",
            "seed",
            "optimizer_id",
            "eid",
            "mid",
            "rid",
        ):
            pl[col] = run_row[col]
        periods_long = pd.concat([periods_long, pl], ignore_index=True)

    output_root.mkdir(parents=True, exist_ok=True)
    run_summary.to_csv(output_root / RUN_SUMMARY_NAME, index=False)
    periods_long.to_csv(output_root / PERIODS_LONG_NAME, index=False)
    return run_summary, periods_long


def job_is_complete(
    job: dict[str, Any],
    completed_keys: set[tuple[str, str]],
    recovery_paths: dict[tuple[str, str], Path],
) -> bool:
    oid = optimizer_id_for_name(
        job["optimizer_name"], job["base_lr"], dataset=job["dataset"]
    )
    key = (job["config_stem"], oid)
    if key not in completed_keys:
        return False
    path = recovery_paths.get(key)
    return path is not None and path.is_file()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train many-seed many-PT runs and export compact recovery-period CSVs.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory of base experiment JSON configs (non-recursive *.json).",
    )
    parser.add_argument(
        "--pretrained-root",
        type=Path,
        default=DEFAULT_PRETRAINED_ROOT,
        help=(
            "Override pretrained checkpoint root for pretrain configs. "
            "Default: each config's use_initial_model path."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Destination root for CSV outputs (default: results_many_seeds/<dataset>). "
            "By default, transient training artifacts use <output-root>/scratch/."
        ),
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help=(
            "Transient training directory (deleted after each job). "
            "Default: <output-root>/scratch/."
        ),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_NOTEBOOK_SEEDS),
        help=f"Comma-separated seeds (default: notebook set {DEFAULT_NOTEBOOK_SEEDS}).",
    )
    parser.add_argument(
        "--optimizers",
        default=",".join(DEFAULT_OPTIMIZER_NAMES),
        help="Comma-separated optimizer shorthands.",
    )
    parser.add_argument(
        "--exp",
        default="",
        help=(
            "Comma-separated config slugs to run (default: all). "
            "Slugs match input_configs stems without the _multi suffix, e.g. "
            "cat2_sequence_recover for cat2_sequence_recover_multi.json."
        ),
    )
    parser.add_argument("--max-jobs", type=int, default=None, help="Limit pending jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs only.")
    parser.add_argument("--force", action="store_true", help="Re-run completed jobs.")
    parser.add_argument(
        "--check-pretrained-only",
        action="store_true",
        help="Validate pretrained checkpoints and exit.",
    )
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device index.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_dir = args.config_dir.resolve()
    config_paths = list_config_paths(config_dir)
    pretrained_override = args.pretrained_root.resolve() if args.pretrained_root else None
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else default_output_root(config_dir, config_paths)
    )
    scratch_root = resolve_scratch_root(
        output_root,
        args.scratch_root.resolve() if args.scratch_root else None,
    )

    seeds = _parse_csv_ints(args.seeds)
    optimizer_names = _parse_csv_strs(args.optimizers)
    exp_slugs = _parse_csv_strs(args.exp) or None
    jobs = (
        build_jobs(config_dir, optimizer_names, seeds, exp_slugs=exp_slugs)
        if config_paths
        else []
    )

    if exp_slugs and not jobs:
        print(
            f"ERROR: --exp matched no jobs. Requested: {exp_slugs}",
            file=sys.stderr,
        )
        print(f"Available slugs: {available_config_slugs(config_dir)}", file=sys.stderr)
        return 1

    if args.check_pretrained_only:
        if not jobs:
            print(f"ERROR: no *.json configs in {config_dir}", file=sys.stderr)
            return 1
        errors = validate_pretrained_checkpoints(
            jobs,
            optimizer_names,
            seeds,
            pretrained_override=pretrained_override,
        )
        total = sum(
            1
            for job in jobs
            if job["initial_model_mode"] == "pretrain"
            for _ in optimizer_names
            for _ in seeds
        )
        if errors:
            print(f"Pretrained validation failed ({len(errors)} error(s)):")
            for err in errors:
                print(f"  - {err}")
            return 1
        print(f"Pretrained validation OK ({total} checks)")
        return 0

    if not config_dir.is_dir():
        print(f"ERROR: config dir not found: {config_dir}", file=sys.stderr)
        return 1
    if not config_paths:
        print(f"ERROR: no *.json configs in {config_dir}", file=sys.stderr)
        return 1

    run_summary, periods_long = load_aggregate_tables(output_root)
    run_summary_path = output_root / RUN_SUMMARY_NAME
    completed = set() if args.force else load_completed_keys(run_summary_path)
    recovery_paths = {} if args.force else load_recovery_paths(run_summary_path)

    pending: list[dict[str, Any]] = []
    for job in jobs:
        oid = optimizer_id_for_name(
            job["optimizer_name"], job["base_lr"], dataset=job["dataset"]
        )
        key = (job["config_stem"], oid)
        if args.force or key not in completed:
            pending.append(job)
        elif not job_is_complete(job, {key}, recovery_paths):
            pending.append(job)

    if args.max_jobs is not None:
        pending = pending[: int(args.max_jobs)]

    print(f"Config dir: {config_dir} ({len(config_paths)} base configs)")
    if exp_slugs:
        print(f"Experiment filter (--exp): {exp_slugs}")
    print(f"Seeds: {seeds}")
    print(f"Total jobs: {len(jobs)}, pending: {len(pending)}")
    print(f"Output root: {output_root}")
    print(f"Scratch root: {scratch_root}")

    if args.dry_run:
        for job in pending:
            print(
                f"  {job['config_slug']} / {job['optimizer_name']} / "
                f"{job['config_stem']} / seed={job['seed']}"
            )
        return 0

    if pending and any(j["initial_model_mode"] == "pretrain" for j in pending):
        errors = validate_pretrained_checkpoints(
            pending,
            optimizer_names,
            seeds,
            pretrained_override=pretrained_override,
        )
        if errors:
            print("Pretrained validation failed; aborting.", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1

    failures: list[dict[str, Any]] = []
    for job in tqdm(pending, desc="many_seeds"):
        desc = (
            f"{job['config_slug']} / {job['optimizer_name']} / "
            f"{job['config_stem']}"
        )
        try:
            out = run_one_job(
                job,
                pretrained_override=pretrained_override,
                scratch_root=scratch_root,
                output_root=output_root,
                device_id=args.device_id,
            )
            periods_df = out.pop("periods_df")
            run_summary, periods_long = append_and_save_aggregate(
                output_root,
                run_summary,
                periods_long,
                out,
                periods_df,
            )
        except Exception as exc:
            tqdm.write(f"FAILED {desc}: {exc}")
            failures.append({**job, "error": str(exc)})

    if failures:
        failures_path = output_root / FAILURES_NAME
        output_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Failures: {len(failures)} (see {failures_path})")
        return 1

    print(f"run_summary rows: {len(run_summary)}")
    print(f"periods_long rows: {len(periods_long)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
