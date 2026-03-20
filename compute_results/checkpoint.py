"""Checkpoint saving and neuron-level metric extraction."""
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.base import AnalyzableModel


def save_model_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    tag: str,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{tag}.pt"
    torch.save(model.state_dict(), path)
    return path


@torch.no_grad()
def capture_activations(
    model: AnalyzableModel,
    inputs: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run a forward pass with hooks and return per-layer activations."""
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    for layer_name, module in model.hookable_layers().items():
        def _hook(mod, inp, out, name=layer_name):
            activations[name] = out.detach().cpu()
        hooks.append(module.register_forward_hook(_hook))

    model.eval()
    model(inputs.to(device))
    model.train()

    for h in hooks:
        h.remove()

    return activations


def extract_unit_activations(
    activations: dict[str, torch.Tensor],
    units: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """For each clickable unit, extract all activation values (no reduction)."""
    result: dict[str, np.ndarray] = {}
    for u in units:
        layer_act = activations.get(u["layer_name"])
        if layer_act is None:
            continue
        idx = u["unit_index"]
        if u["unit_type"] == "neuron":
            val = layer_act[:, idx].detach().cpu().numpy().flatten()
        elif u["unit_type"] == "channel":
            val = layer_act[:, idx].detach().cpu().numpy().flatten()
        else:
            continue
        result[u["node_id"]] = val
    return result


def compute_weight_stats(model: AnalyzableModel) -> dict[str, float]:
    """Compute per-unit weight norms."""
    norms: dict[str, float] = {}

    current_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    units = model.clickable_units()

    for u in units:
        layer_name = u["layer_name"]
        idx = u["unit_index"]
        param_key = _find_weight_key(current_state, layer_name)
        if param_key is None:
            continue

        w_current = current_state[param_key]
        if u["unit_type"] == "neuron":
            w_vec = w_current[idx].flatten().float()
        elif u["unit_type"] == "channel":
            w_vec = w_current[idx].flatten().float()
        else:
            continue

        norms[u["node_id"]] = w_vec.norm().item()

    return norms


def _find_weight_key(
    state_dict: dict[str, torch.Tensor],
    layer_name: str,
) -> str | None:
    # Gets layer weight for nn.Linear (.weight) or nn.Conv2d as first elem of a nn.Sequential (.0.weight) layers
    candidates = [f"{layer_name}.weight", f"{layer_name}.0.weight"]
    for c in candidates:
        if c in state_dict:
            return c
    for k in state_dict:
        if k.startswith(layer_name) and "weight" in k:
            return k
    return None


class NeuronTimeseriesCollector:
    """Accumulates per-checkpoint neuron-level data and saves to .npz."""

    def __init__(self, units: list[dict[str, Any]]):
        self.units = units
        self.checkpoint_tags: list[str] = []
        self.activations: dict[str, list[np.ndarray]] = {u["node_id"]: [] for u in units}
        self.weight_norms: dict[str, list[float]] = {u["node_id"]: [] for u in units}

    def record(
        self,
        tag: str,
        act_values: dict[str, np.ndarray],
        norm_values: dict[str, float],
    ) -> None:
        self.checkpoint_tags.append(tag)
        for u in self.units:
            nid = u["node_id"]
            self.activations[nid].append(
                act_values[nid] if nid in act_values else np.array([float("nan")])
            )
            self.weight_norms[nid].append(norm_values.get(nid, float("nan")))

    def save(self, path: Path) -> None:
        data: dict[str, Any] = {
            "checkpoint_tags": np.array(self.checkpoint_tags),
            "unit_node_ids": np.array([u["node_id"] for u in self.units]),
        }
        for nid in self.activations:
            safe = nid.replace(":", "__")
            data[f"act__{safe}"] = np.array(self.activations[nid])
            data[f"wnorm__{safe}"] = np.array(self.weight_norms[nid])

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **data)
