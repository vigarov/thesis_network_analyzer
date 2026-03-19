"""Generic training loop that iterates through experiment stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from compute_results.checkpoint import (
    NeuronTimeseriesCollector,
    capture_activations,
    compute_weight_stats,
    extract_unit_activations,
    save_model_checkpoint,
)
from compute_results.config_guard import parse_checkpoint_cadence
from experiments.base import Experiment
from models.base import AnalyzableModel
from optimizers.base import OptimizerSignalExtractor


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Return (loss, accuracy) on *loader*."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def _run_checkpoint(
    *,
    tag: str,
    model: AnalyzableModel,
    experiment: Experiment,
    device: torch.device,
    checkpoint_dir: Path,
    neuron_collector: NeuronTimeseriesCollector,
    prev_state: dict[str, torch.Tensor] | None,
    eval_loaders: dict[str, torch.utils.data.DataLoader],
    criterion: nn.Module,
    metrics: dict[str, list],
) -> dict[str, torch.Tensor]:
    """Save checkpoint, capture activations, record neuron stats, evaluate."""
    save_model_checkpoint(model, checkpoint_dir, tag)

    eval_inputs, _ = experiment.evaluation_inputs(device)
    activations = capture_activations(model, eval_inputs, device)
    act_values = extract_unit_activations(activations, model.clickable_units())
    norm_values, cos_values = compute_weight_stats(model, prev_state)
    neuron_collector.record(tag, act_values, norm_values, cos_values)

    for loader_name, loader in eval_loaders.items():
        loss, acc = evaluate(model, loader, device, criterion)
        metrics.setdefault(f"loss_{loader_name}", []).append(loss)
        metrics.setdefault(f"acc_{loader_name}", []).append(acc)

    metrics.setdefault("checkpoint_tags", []).append(tag)

    current_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return current_state


def train_with_config(
    *,
    experiment: Experiment,
    model: AnalyzableModel,
    extractor: OptimizerSignalExtractor,
    config: dict[str, Any],
    device: torch.device,
    results_dir: Path,
) -> None:
    """Run the full multi-stage training and save all outputs."""
    model = model.to(device)
    model.train()

    units = model.clickable_units()
    node_ids = [u["node_id"] for u in units]
    extractor.set_units(units)

    optimizer = extractor.create_optimizer(model.parameters())
    criterion = nn.CrossEntropyLoss()

    stage_epochs = config["stage_epochs"]
    batch_size = config["batch_size"]
    seed = config["seed"]
    cadence_mode, cadence_k = parse_checkpoint_cadence(
        config.get("checkpoint_cadence", "every_epoch")
    )
    disable_cp_turning_point: bool = config.get("disable_cp_turning_point", False)

    checkpoint_dir = results_dir / "checkpoints"
    optimizer_dir = results_dir / "optimizers" / extractor.optimizer_id()

    stages = experiment.build_stages(batch_size=batch_size, seed=seed)

    neuron_collector = NeuronTimeseriesCollector(units)
    metrics: dict[str, list] = {}

    signal_log: dict[str, dict[str, list[float]]] = {
        s: {nid: [] for nid in node_ids}
        for s in extractor.signal_names()
    }
    iterations: list[int] = []

    all_eval_loaders: dict[str, torch.utils.data.DataLoader] = {}
    for stage in stages:
        all_eval_loaders.update(stage.eval_loaders)

    prev_state: dict[str, torch.Tensor] | None = None

    prev_state = _run_checkpoint(
        tag="init",
        model=model,
        experiment=experiment,
        device=device,
        checkpoint_dir=checkpoint_dir,
        neuron_collector=neuron_collector,
        prev_state=prev_state,
        eval_loaders=all_eval_loaders,
        criterion=criterion,
        metrics=metrics,
    )

    global_iter = 0

    for stage in stages:
        if stage.is_turning_point and not disable_cp_turning_point:
            prev_state = _run_checkpoint(
                tag=f"tp_{stage.name}_entry",
                model=model,
                experiment=experiment,
                device=device,
                checkpoint_dir=checkpoint_dir,
                neuron_collector=neuron_collector,
                prev_state=prev_state,
                eval_loaders=all_eval_loaders,
                criterion=criterion,
                metrics=metrics,
            )
            tp_iters_remaining = 10
        else:
            tp_iters_remaining = 0

        for epoch in range(stage_epochs):
            for x, y in tqdm(
                stage.train_loader,
                desc=f"{stage.name} epoch{epoch}",
            ):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()

                before_signals = extractor.on_before_step(model, optimizer)
                optimizer.step()
                after_signals = extractor.on_after_step(model, optimizer)

                all_signals = {**before_signals, **after_signals}
                for s_name in extractor.signal_names():
                    unit_vals = all_signals.get(s_name, {})
                    for nid in node_ids:
                        signal_log[s_name][nid].append(
                            unit_vals.get(nid, float("nan"))
                        )
                iterations.append(global_iter)
                global_iter += 1

                if tp_iters_remaining > 0:
                    tp_iter_idx = 11 - tp_iters_remaining
                    prev_state = _run_checkpoint(
                        tag=f"tp_{stage.name}_iter{tp_iter_idx}",
                        model=model,
                        experiment=experiment,
                        device=device,
                        checkpoint_dir=checkpoint_dir,
                        neuron_collector=neuron_collector,
                        prev_state=prev_state,
                        eval_loaders=all_eval_loaders,
                        criterion=criterion,
                        metrics=metrics,
                    )
                    tp_iters_remaining -= 1
                elif cadence_mode == "every_k_its" and cadence_k is not None and global_iter % cadence_k == 0:
                    prev_state = _run_checkpoint(
                        tag=f"iter{global_iter}",
                        model=model,
                        experiment=experiment,
                        device=device,
                        checkpoint_dir=checkpoint_dir,
                        neuron_collector=neuron_collector,
                        prev_state=prev_state,
                        eval_loaders=all_eval_loaders,
                        criterion=criterion,
                        metrics=metrics,
                    )

            if cadence_mode == "every_epoch":
                tag = f"{stage.name}_epoch{epoch}"
                prev_state = _run_checkpoint(
                    tag=tag,
                    model=model,
                    experiment=experiment,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                    neuron_collector=neuron_collector,
                    prev_state=prev_state,
                    eval_loaders=all_eval_loaders,
                    criterion=criterion,
                    metrics=metrics,
                )

    # --- Save outputs ---
    optimizer_dir.mkdir(parents=True, exist_ok=True)

    signal_arrays: dict[str, Any] = {
        "iteration": np.array(iterations),
        "unit_node_ids": np.array(node_ids),
    }
    for s_name, unit_data in signal_log.items():
        if node_ids and iterations:
            matrix = np.column_stack(
                [np.array(unit_data[nid]) for nid in node_ids]
            )
        else:
            matrix = np.array([])
        signal_arrays[s_name] = matrix
    np.savez_compressed(str(optimizer_dir / "signals.npz"), **signal_arrays)

    neuron_collector.save(optimizer_dir / "neuron_timeseries.npz")

    metric_arrays = {}
    for k, v in metrics.items():
        try:
            metric_arrays[k] = np.array(v)
        except (ValueError, TypeError):
            metric_arrays[k] = np.array(v, dtype=object)
    np.savez_compressed(str(optimizer_dir / "training_metrics.npz"), **metric_arrays)
