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
    *,
    keep_on_gpu: bool = False,
) -> dict[str, torch.Tensor]:
    """Run a forward pass with hooks and return per-layer activations."""
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    for layer_name, module in model.hookable_layers().items():
        def _hook(mod, inp, out, name=layer_name):
            activations[name] = out.detach() if keep_on_gpu else out.detach().cpu()
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
    *,
    keep_on_gpu: bool = False,
) -> dict[str, np.ndarray | torch.Tensor]:
    """For each clickable unit, extract all activation values (no reduction)."""
    result: dict[str, np.ndarray | torch.Tensor] = {}
    for u in units:
        layer_act = activations.get(u["layer_name"])
        if layer_act is None:
            continue
        idx = u["unit_index"]
        if u["unit_type"] in ("neuron", "channel"):
            if keep_on_gpu:
                val = layer_act[:, idx].detach().flatten()
            else:
                val = layer_act[:, idx].detach().cpu().numpy().flatten()
        else:
            continue
        result[u["node_id"]] = val
    return result


def compute_weight_stats(
    model: AnalyzableModel,
    *,
    keep_on_gpu: bool = False,
) -> dict[str, float]:
    """Compute per-unit weight norms."""
    norms: dict[str, float] = {}

    state = model.state_dict()
    if not keep_on_gpu:
        state = {k: v.detach().cpu() for k, v in state.items()}
    units = model.clickable_units()

    for u in units:
        layer_name = u["layer_name"]
        idx = u["unit_index"]
        param_key = _find_weight_key(state, layer_name)
        if param_key is None:
            continue

        w_current = state[param_key]
        if u["unit_type"] in ("neuron", "channel"):
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

    _FLUSH_THRESHOLD = 4 * 1024**3  # 6 GB

    def __init__(self, units: list[dict[str, Any]], *, keep_tensors: bool = False):
        self.units = units
        self.keep_tensors = keep_tensors
        self.checkpoint_tags: list[str] = []
        self.activations: dict[str, list[np.ndarray | torch.Tensor]] = {
            u["node_id"]: [] for u in units
        }
        self.weight_norms: dict[str, list[float]] = {u["node_id"]: [] for u in units}
        self._total_tensor_bytes: int = 0

    def record(
        self,
        tag: str,
        act_values: dict[str, np.ndarray | torch.Tensor],
        norm_values: dict[str, float],
    ) -> None:
        self.checkpoint_tags.append(tag)
        for u in self.units:
            nid = u["node_id"]
            if nid in act_values:
                val = act_values[nid]
                if isinstance(val, torch.Tensor):
                    self._total_tensor_bytes += val.nelement() * val.element_size()
            else:
                val = np.array([float("nan")])
            self.activations[nid].append(val)
            self.weight_norms[nid].append(norm_values.get(nid, float("nan")))

        if self.keep_tensors:
            self._maybe_flush()

    def _maybe_flush(self, *, force: bool = False) -> None:
        """Move GPU tensors to numpy when accumulated size exceeds threshold."""
        if not force and self._total_tensor_bytes <= self._FLUSH_THRESHOLD:
            return
        for nid, vals in self.activations.items():
            for i, v in enumerate(vals):
                if isinstance(v, torch.Tensor):
                    vals[i] = v.detach().cpu().numpy()
        self._total_tensor_bytes = 0

    def save(self, path: Path) -> None:
        self._maybe_flush(force=True)

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
