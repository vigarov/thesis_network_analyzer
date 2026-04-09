"""Post-training computation of neuron activity labels (assigned/inactive/dead).

After ``train_with_config`` completes, this module analyses the per-checkpoint
activations stored in ``neuron_timeseries.npz`` and writes:

- ``post_processing_neuron_digit.csv`` — per (neuron, checkpoint, digit) status
- ``post_processing_dead.csv``         — per (neuron, checkpoint) dead flag and ``is_ppd``
  (perpetually dead: dead at this checkpoint and at every later checkpoint)
- ``post_processing_train_act.npz``    — dead-neuron activations on the full training set

**Interpretation of "dead" (``is_dead`` in ``post_processing_dead.csv``)**

A neuron is *inactive on all digits* at checkpoint ``cp`` when every digit's
``K_SAMPLES`` post-nonlinearity activations are exactly zero (same as before).

- **Runs with no pretraining trial** (no training trial name contains
  ``"pretrain"``, case-insensitive): ``is_dead`` is true at ``cp`` iff the
  neuron is inactive on all digits at that checkpoint — **any** checkpoint
  may label a neuron dead.

- **Runs that include at least one pretraining trial** (some trial name contains
  ``"pretrain"``): checkpoints that fall **inside** a pretraining trial but are
  **not** that trial's **last** checkpoint never set ``is_dead``, even if the
  neuron is inactive on all digits there (early pretrain snapshots are ignored).
  At the **last** checkpoint of each pretraining trial, and at **every**
  checkpoint of every **non-pretrain** trial, ``is_dead`` is true iff inactive
  on all digits — i.e. dead is only reported from the end of pretraining
  onward (last iter of the pretrain trial or any later trial).

**``is_ppd`` (perpetually dead)**

At checkpoint ``cp``, ``is_ppd`` is true iff ``is_dead`` is true at ``cp`` and at
every later checkpoint (the neuron never recovers after becoming dead at ``cp``
through the end of training).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.base import TrialSpec
from models.base import AnalyzableModel

N_DIGITS = 10
K_SAMPLES = 5
N_EVAL = N_DIGITS * K_SAMPLES


def is_ppd_from_is_dead_chronological(is_dead: np.ndarray) -> np.ndarray:
    """Per-checkpoint *perpetually dead* flags from a chronological ``is_dead`` series.

    Checkpoint ``i`` is perpetually dead iff the neuron is dead at ``i`` and at
    every later checkpoint (``is_dead[j]`` is true for all ``j > i``).
    """
    n = len(is_dead)
    if n == 0:
        return np.array([], dtype=bool)
    d = np.asarray(is_dead, dtype=bool)
    suffix = np.ones(n, dtype=bool)
    for i in range(n - 2, -1, -1):
        suffix[i] = d[i + 1] and suffix[i + 1]
    return d & suffix


def add_is_ppd_column(df_dead: pd.DataFrame) -> pd.DataFrame:
    """Add boolean ``is_ppd`` column (perpetually dead) to ``post_processing_dead``."""
    if df_dead.empty or "is_dead" not in df_dead.columns:
        return df_dead
    df = df_dead.copy()
    df["_row_order"] = np.arange(len(df), dtype=np.int64)
    df_sorted = df.sort_values(["neuron_id", "checkpoint_idx"])
    flags: list[np.ndarray] = []
    for _, g in df_sorted.groupby("neuron_id", sort=False):
        flags.append(
            is_ppd_from_is_dead_chronological(g["is_dead"].to_numpy()),
        )
    df_sorted = df_sorted.drop(columns=["is_ppd"], errors="ignore")
    df_sorted["is_ppd"] = np.concatenate(flags)
    out = df_sorted.sort_values("_row_order").drop(columns=["_row_order"])
    return out


def dead_label_checkpoint_mask(
    n_checkpoints: int,
    trial_names: np.ndarray | list[Any],
    trial_end_checkpoint_idxs: np.ndarray | list[int],
) -> np.ndarray:
    """Where ``is_dead`` may follow the all-inactive rule (see module docstring).

    Returns a length-``n_checkpoints`` boolean array: ``True`` at ``cp`` means
    ``is_dead = all_inactive`` at that checkpoint; ``False`` means ``is_dead``
    is forced to ``False`` regardless of activations (only used for early
    checkpoints inside a pretraining trial).
    """
    if n_checkpoints <= 0:
        return np.array([], dtype=bool)

    names = [str(x) for x in trial_names]
    ends = np.asarray(trial_end_checkpoint_idxs, dtype=np.int64)
    if len(names) != len(ends):
        raise ValueError(
            "trial_names and trial_end_checkpoint_idxs must have the same length"
        )

    has_any_pretrain_trial = any("pretrain" in n.lower() for n in names)
    mask = np.ones(n_checkpoints, dtype=bool)
    if not has_any_pretrain_trial:
        return mask

    for s, name in enumerate(names):
        if "pretrain" not in name.lower():
            continue
        end_cp = int(ends[s])
        start_cp = int(ends[s - 1]) + 1 if s > 0 else 0
        if start_cp < end_cp:
            mask[start_cp:end_cp] = False
    return mask


def dead_label_mask_for_optimizer_dir(
    optimizer_dir: Path, n_checkpoints: int,
) -> np.ndarray:
    """Per-checkpoint mask for applying the *dead* rule (see module docstring).

    Uses ``training_metrics.npz`` under ``optimizer_dir``. If missing, every
    checkpoint is eligible (same as a run with no pretraining trials).
    """
    tm_path = optimizer_dir / "training_metrics.npz"
    if not tm_path.exists():
        return np.ones(n_checkpoints, dtype=bool)
    tm = dict(np.load(str(tm_path), allow_pickle=True))
    trial_names = tm.get("trial_names", tm.get("stage_names", np.array([], dtype=object)))
    trial_ends = tm.get(
        "trial_end_checkpoint_idxs",
        tm.get("stage_end_checkpoint_idxs", np.array([], dtype=np.int64)),
    )
    return dead_label_checkpoint_mask(n_checkpoints, trial_names, trial_ends)


def post_processing_dataframes_from_nts(
    nts: dict[str, Any],
    units: list[dict[str, Any]],
    cp_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build neuron-digit and dead tables from saved activations (canonical *dead* rule).

    ``cp_mask`` must have length ``n_checkpoints`` (from
    :func:`dead_label_mask_for_optimizer_dir` or :func:`dead_label_checkpoint_mask`).
    """
    checkpoint_tags = [str(t) for t in nts.get("checkpoint_tags", [])]
    n_checkpoints = len(checkpoint_tags)
    if n_checkpoints == 0 or len(cp_mask) != n_checkpoints:
        return pd.DataFrame(), pd.DataFrame()

    layer_order: list[str] = []
    for u in units:
        ln = u["layer_name"]
        if ln not in layer_order:
            layer_order.append(ln)
    head_layer = layer_order[-1] if layer_order else ""

    head_post_nl: dict[str, np.ndarray] = {}
    head_units = [u for u in units if u["layer_name"] == head_layer]
    if head_units:
        head_pre: list[np.ndarray] = []
        for u in head_units:
            safe = u["node_id"].replace(":", "__")
            key = f"act__{safe}"
            if key in nts:
                head_pre.append(np.asarray(nts[key], dtype=np.float64))
        if len(head_pre) == len(head_units):
            stacked = np.stack(head_pre, axis=-1)
            sm = torch.softmax(torch.from_numpy(stacked), dim=-1).numpy()
            for i, u in enumerate(head_units):
                head_post_nl[u["node_id"]] = sm[:, :, i]

    rows_nd: list[dict[str, Any]] = []
    rows_dead: list[dict[str, Any]] = []

    for u in units:
        nid = u["node_id"]
        ln = u["layer_name"]
        safe = nid.replace(":", "__")
        key = f"act__{safe}"
        if key not in nts:
            continue

        pre_nl = np.asarray(nts[key], dtype=np.float64)
        if pre_nl.ndim != 2 or pre_nl.shape[1] != N_EVAL:
            continue

        if nid in head_post_nl:
            post_nl = head_post_nl[nid]
        else:
            post_nl = F.relu(torch.from_numpy(pre_nl)).numpy()

        for cp in range(n_checkpoints):
            all_inactive = True
            for d in range(N_DIGITS):
                s = K_SAMPLES * d
                digit_act = post_nl[cp, s : s + K_SAMPLES]

                if np.all(digit_act > 0):
                    status = "assigned"
                    all_inactive = False
                elif np.all(digit_act == 0):
                    status = "inactive"
                else:
                    status = "partial"
                    all_inactive = False

                rows_nd.append({
                    "neuron_id": nid,
                    "layer_name": ln,
                    "checkpoint_idx": cp,
                    "checkpoint_tag": checkpoint_tags[cp],
                    "digit": d,
                    "status": status,
                })

            # ``is_dead``: all digits inactive, gated by ``cp_mask`` (module docstring).
            rows_dead.append({
                "neuron_id": nid,
                "layer_name": ln,
                "checkpoint_idx": cp,
                "checkpoint_tag": checkpoint_tags[cp],
                "is_dead": bool(all_inactive and cp_mask[cp]),
            })

    df_nd = pd.DataFrame(rows_nd)
    df_dead = add_is_ppd_column(pd.DataFrame(rows_dead))
    return df_nd, df_dead


def _collect_trial_loaders(
    trials: list[TrialSpec] | list[list[TrialSpec]],
) -> list[DataLoader]:
    """Return one DataLoader per unique training step, preserving training order.

    For flat trial lists (same trials reused across experiment runs) the loaders are
    deduplicated by object identity — non-``once_only`` trials share the same
    DataLoader object across runs, so each unique loader appears once.
    For nested trial lists (one list per run) every run's loaders are
    included, because each run builds its own independent DataLoader.
    """
    if not trials:
        return []

    if isinstance(trials[0], list):
        loaders: list[DataLoader] = []
        for run_trials in cast(list[list[TrialSpec]], trials):
            for trial in run_trials:
                loaders.append(trial.train_loader)
        return loaders

    seen: set[int] = set()
    loaders = []
    for trial in cast(list[TrialSpec], trials):
        lid = id(trial.train_loader)
        if lid not in seen:
            seen.add(lid)
            loaders.append(trial.train_loader)
    return loaders


def compute_post_processing(
    model: AnalyzableModel,
    trials: list[TrialSpec] | list[list[TrialSpec]],
    units: list[dict[str, Any]],
    optimizer_dir: Path,
    device: torch.device,
) -> None:
    nts_path = optimizer_dir / "neuron_timeseries.npz"
    if not nts_path.exists():
        return
    nts = dict(np.load(str(nts_path), allow_pickle=True))

    checkpoint_tags = [str(t) for t in nts.get("checkpoint_tags", [])]
    n_checkpoints = len(checkpoint_tags)
    if n_checkpoints == 0:
        return

    layer_order: list[str] = []
    for u in units:
        ln = u["layer_name"]
        if ln not in layer_order:
            layer_order.append(ln)
    if not layer_order:
        return

    cp_mask = dead_label_mask_for_optimizer_dir(optimizer_dir, n_checkpoints)
    rows_nd, rows_dead = post_processing_dataframes_from_nts(nts, units, cp_mask)
    rows_nd.to_csv(
        str(optimizer_dir / "post_processing_neuron_digit.csv"), index=False,
    )
    rows_dead.to_csv(
        str(optimizer_dir / "post_processing_dead.csv"), index=False,
    )

    # post_processing_train_act.npz: neurons that are dead at the *final* checkpoint
    # (inactive on all digits there — see module docstring; final cp always uses the
    # all-inactive rule, including after pretrain).
    df_dead = rows_dead
    final_dead_nids = df_dead.loc[
        (df_dead["checkpoint_idx"] == n_checkpoints - 1) & df_dead["is_dead"],
        "neuron_id",
    ].unique()

    if len(final_dead_nids) > 0:
        _capture_dead_neuron_train_activations(
            model, trials, units, list(final_dead_nids), optimizer_dir, device,
        )


def _capture_dead_neuron_train_activations(
    model: AnalyzableModel,
    trials: list[TrialSpec] | list[list[TrialSpec]],
    units: list[dict[str, Any]],
    dead_nids: list[str],
    optimizer_dir: Path,
    device: torch.device,
) -> None:
    """Forward-pass the training samples (exactly as used during training) and store
    pre-NL activations for dead neurons.

    Instead of iterating ``experiment._train_ds`` (which may contain all 10 digits
    even when training only touched a subset), we replay the actual trial
    ``train_loader``s so that the digit coverage and any label remappings
    (e.g. ``_RelabelSubset``) match the training run precisely.
    """
    dead_set = set(dead_nids)
    dead_units = [u for u in units if u["node_id"] in dead_set]
    if not dead_units:
        return

    layer_dead: dict[str, list[dict[str, Any]]] = {}
    for u in dead_units:
        layer_dead.setdefault(u["layer_name"], []).append(u)

    captured: dict[str, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    hookable = model.hookable_layers()
    for ln in layer_dead:
        if ln not in hookable:
            continue

        def _hook(mod: Any, inp: Any, out: torch.Tensor, *, name: str = ln) -> None:
            captured[name] = out.detach().cpu()

        hooks.append(hookable[ln].register_forward_hook(_hook))

    train_loaders = _collect_trial_loaders(trials)

    per_neuron: dict[str, list[np.ndarray]] = {nid: [] for nid in dead_nids}
    all_labels: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for loader in train_loaders:
            for x, y in loader:
                x = x.to(device)
                captured.clear()
                model(x)

                y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
                all_labels.append(y_np)

                for ln, dead_us in layer_dead.items():
                    act = captured.get(ln)
                    if act is None:
                        continue
                    for u in dead_us:
                        per_neuron[u["node_id"]].append(
                            act[:, u["unit_index"]].numpy().flatten()
                        )

    model.train()
    for h in hooks:
        h.remove()

    labels = np.concatenate(all_labels)
    save_dict: dict[str, np.ndarray] = {
        "neuron_ids": np.array(dead_nids),
        "digit_labels": labels,
    }
    for nid in dead_nids:
        safe = nid.replace(":", "__")
        parts = per_neuron[nid]
        if parts:
            save_dict[f"act__{safe}"] = np.concatenate(parts)

    np.savez_compressed(
        str(optimizer_dir / "post_processing_train_act.npz"), **save_dict,
    )
