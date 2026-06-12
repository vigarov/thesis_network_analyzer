"""Checkpoint saving and neuron-level metric extraction."""
from pathlib import Path
from collections.abc import Mapping
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
			activations[name] = out.detach().clone() if keep_on_gpu else out.detach().cpu()
		hooks.append(module.register_forward_hook(_hook))

	model.eval()
	model(inputs.to(device))
	model.train()

	for h in hooks:
		h.remove()

	return activations


class PersistentActivationCapture:
	"""Forward hooks installed once for the model lifetime, gated by a flag.

	Training forward passes hit a cheap boolean check and return immediately.
	Eval captures are triggered explicitly via :meth:`capture`.
	"""

	def __init__(
		self,
		model: AnalyzableModel,
		*,
		keep_on_gpu: bool = False,
	):
		self._model = model
		self._keep_on_gpu = keep_on_gpu
		self._recording = False
		self._activations: dict[str, torch.Tensor] = {}
		self._handles: list[torch.utils.hooks.RemovableHandle] = []

	def install(self) -> None:
		"""Register hooks on all hookable layers (call once before training)."""
		for layer_name, module in self._model.hookable_layers().items():
			def _hook(mod, inp, out, *, name=layer_name):
				if not self._recording:
					return
				self._activations[name] = (
					out.detach().clone() if self._keep_on_gpu else out.detach().cpu()
				)
			self._handles.append(module.register_forward_hook(_hook))

	@torch.no_grad()
	def capture(
		self, inputs: torch.Tensor, device: torch.device
	) -> dict[str, torch.Tensor]:
		"""Run one eval forward and return the captured activations dict."""
		self._activations.clear()
		self._recording = True
		self._model.eval()
		self._model(inputs.to(device))
		self._model.train()
		self._recording = False
		result = dict(self._activations)
		self._activations.clear()
		return result

	def remove(self) -> None:
		"""Unregister all hooks (call after training is done)."""
		for h in self._handles:
			h.remove()
		self._handles.clear()


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
				val = layer_act[:, idx].detach().flatten().clone()
			else:
				val = layer_act[:, idx].detach().cpu().numpy().flatten()
		else:
			continue
		result[u["node_id"]] = val
	return result


@torch.no_grad()
def extract_unit_weights(
	model: AnalyzableModel,
	*,
	keep_on_gpu: bool = False,
) -> dict[str, np.ndarray | torch.Tensor]:
	"""Extract per-unit weight vectors.

	Uses named_parameters() for zero-copy access instead of state_dict()
	which deep-copies every tensor.
	"""
	weights: dict[str, np.ndarray | torch.Tensor] = {}

	params = dict(model.named_parameters())
	units = model.clickable_units()

	for u in units:
		layer_name = u["layer_name"]
		idx = u["unit_index"]
		param_key = _find_weight_key(params, layer_name)
		if param_key is None:
			continue

		w_current = params[param_key]
		if u["unit_type"] in ("neuron", "channel"):
			w_vec = w_current[idx].flatten().float()
		else:
			continue

		if keep_on_gpu:
			weights[u["node_id"]] = w_vec.detach().clone()
		else:
			weights[u["node_id"]] = w_vec.detach().cpu().clone().numpy()

	return weights


def _find_weight_key(
	state_dict: Mapping[str, torch.Tensor],
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
		self.weights: dict[str, list[np.ndarray | torch.Tensor]] = {
			u["node_id"]: [] for u in units
		}
		self._total_tensor_bytes: int = 0

	def record(
		self,
		tag: str,
		act_values: dict[str, np.ndarray | torch.Tensor],
		weight_values: dict[str, np.ndarray | torch.Tensor],
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

			if nid in weight_values:
				w_val = weight_values[nid]
				if isinstance(w_val, torch.Tensor):
					self._total_tensor_bytes += w_val.nelement() * w_val.element_size()
			else:
				w_val = np.array([float("nan")])
			self.weights[nid].append(w_val)

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
		for nid, vals in self.weights.items():
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
			data[f"weights__{safe}"] = np.array(self.weights[nid])

		path.parent.mkdir(parents=True, exist_ok=True)
		np.savez_compressed(str(path), **data)
