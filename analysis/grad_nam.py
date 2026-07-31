"""Grad-NAM (analytical DNN Jacobian + global-average-pool chain) for CIFAR composite models."""
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from analysis.dataset_filter import mid_to_model_class
from analysis.scoring_helpers import (
	_dnn_hidden_layer_names_from_nts,
	_layer_index_from_neuron_id,
	_unit_key,
	custom_saliency_map,
)
from compute_results.config_guard import load_initial_model_state_dict
from compute_results.constants import INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS
from models import get_model
from models.unit_node_id import parse_unit_node_id


def optimizer_registry_class_from_oid(oid: str) -> str:
	prefix = oid.split("_lr", 1)[0]
	if prefix == "graft_shampoo":
		prefix = "grafted_shampoo"
	try:
		return INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS[prefix]
	except KeyError as exc:
		raise ValueError(f"No registry class for optimizer id {oid!r}") from exc


def target_layer_for(model: nn.Module, model_class: str) -> nn.Module:
	if model_class == "ResNetCIFAR":
		return model.backbone.layer3[-1]
	if model_class == "InceptionCIFAR":
		return model.backbone.Inception_module_8
	raise ValueError(f"Unsupported model class for Grad-NAM: {model_class!r}")


def _scale_nam_image(cam: np.ndarray, target_size: tuple[int, int] | None = None) -> np.ndarray:
	"""Normalize to [0, 1] and optionally resize (Grad-NAM post-processing)."""
	cam = np.asarray(cam, dtype=np.float32)
	if cam.ndim == 2:
		cam = cam[np.newaxis, ...]
	out: list[np.ndarray] = []
	for img in cam:
		img = img - float(np.min(img))
		img = img / (float(np.max(img)) + 1e-8)
		if target_size is not None:
			img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
		out.append(img.astype(np.float32, copy=False))
	return np.stack(out, axis=0)


def _nam_from_acts_and_grads(
	activations: np.ndarray,
	grads: np.ndarray,
	target_size: tuple[int, int],
) -> np.ndarray:
	"""Combine pre-GAP conv activations and channel gradients into one Grad-NAM map."""
	weights = np.mean(grads, axis=(2, 3))
	weighted = weights[:, :, None, None] * activations
	nam = np.maximum(weighted.sum(axis=1), 0)
	nam = _scale_nam_image(nam, target_size=target_size)
	return _scale_nam_image(nam)[0]


def _conv_grads_from_jacobian_row(
	row: np.ndarray,
	n_channels: int,
	pool_h: int,
	pool_w: int,
) -> np.ndarray:
	pool_factor = float(pool_h * pool_w)
	grads = np.zeros((1, n_channels, pool_h, pool_w), dtype=np.float32)
	for channel_idx in range(n_channels):
		grads[0, channel_idx, :, :] = row[channel_idx] / pool_factor
	return grads


def grad_nam_from_tensors(
	conv_act: np.ndarray,
	jacobian_row: np.ndarray,
	*,
	target_size: tuple[int, int],
) -> np.ndarray:
	"""Build a single-sample Grad-NAM map from pre-GAP conv activations and one Jacobian row."""
	conv_act = np.asarray(conv_act, dtype=np.float32)
	if conv_act.ndim == 3:
		conv_act = conv_act[np.newaxis, ...]
	_, n_channels, pool_h, pool_w = conv_act.shape
	grads = _conv_grads_from_jacobian_row(jacobian_row, n_channels, pool_h, pool_w)
	return _nam_from_acts_and_grads(conv_act, grads, target_size)


def dnn_tensors_from_nts(nts: dict[str, Any], checkpoint_idx: int, *, device: torch.device):
	"""Stack DNN hidden weights and activations from neuron timeseries at one checkpoint."""
	layer_names = _dnn_hidden_layer_names_from_nts(nts)
	if not layer_names:
		raise ValueError("No DNN hidden layers found in neuron timeseries")

	weights = []
	activations = []
	for layer_name in layer_names:
		layer_nids_with_indices = []
		for nid in nts["unit_node_ids"]:
			nid = str(nid)
			parsed = parse_unit_node_id(nid)
			if parsed is not None and parsed["layer_name"] == layer_name:
				layer_nids_with_indices.append((parsed["unit_index"], nid))
		layer_nids = [nid for _, nid in sorted(layer_nids_with_indices)]
		weights.append(
			torch.tensor(
				np.stack([nts[_unit_key("weights", nid)][checkpoint_idx] for nid in layer_nids]),
				dtype=torch.float32,
			).to(device)
		)
		activations.append(
			torch.tensor(
				np.stack([nts[_unit_key("act", nid)][checkpoint_idx] for nid in layer_nids]),
				dtype=torch.float32,
			).to(device)
		)
	return weights, torch.stack(activations)


def capture_pre_gap_conv_acts(
	model: nn.Module,
	eval_inputs: torch.Tensor,
	model_class: str,
	*,
	device: torch.device,
) -> np.ndarray:
	"""Forward eval batch and return pre-GAP conv activations ``(n_samples, C, H, W)``."""
	conv_layer = target_layer_for(model, model_class)
	captured: list[torch.Tensor] = []
	hook = conv_layer.register_forward_hook(lambda _m, _i, output: captured.append(output.detach()))
	try:
		model.eval()
		with torch.no_grad():
			batch = eval_inputs.to(device)
			model(batch)
		acts = captured[0].detach().cpu().numpy()
	finally:
		hook.remove()
	return acts


def load_initial_backbone_model(
	*,
	use_initial_model: str,
	model_class: str,
	model_config: dict[str, Any],
	oid: str,
	seed: int,
	device: torch.device,
) -> nn.Module:
	registry_class = optimizer_registry_class_from_oid(oid)
	state = load_initial_model_state_dict(
		use_initial_model,
		registry_class_name=registry_class,
		optimizer_id=oid,
		seed=seed,
	)
	model = get_model(model_class, **model_config).to(device)
	model.load_state_dict(state)
	model.eval()
	return model


def grad_nam_maps_for_neuron_from_nts(
	conv_acts: np.ndarray,
	nts: dict[str, Any],
	checkpoint_idx: int,
	neuron_id: str,
	*,
	target_size: tuple[int, int],
	device: torch.device,
) -> list[np.ndarray]:
	"""Grad-NAM maps for one neuron at ``checkpoint_idx`` using cached conv acts + NTS."""
	layer_idx = _layer_index_from_neuron_id(neuron_id)
	parsed = parse_unit_node_id(neuron_id)
	assert parsed is not None
	unit_idx = parsed["unit_index"]

	weights, activations = dnn_tensors_from_nts(nts, checkpoint_idx, device=device)
	with torch.no_grad():
		jacobian = custom_saliency_map(weights, activations)
	row = jacobian[layer_idx, unit_idx].detach().cpu().numpy()

	maps: list[np.ndarray] = []
	for sample_idx in range(conv_acts.shape[0]):
		maps.append(
			grad_nam_from_tensors(
				conv_acts[sample_idx],
				row,
				target_size=target_size,
			)
		)
	return maps


def _get_acts_and_jac_analytical(
	model: nn.Module,
	input_tensor: torch.Tensor,
	model_class: str,
	*,
	device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, int, int, int, tuple[int, int]]:
	"""One forward pass: pre-GAP conv acts + full DNN Jacobian for one sample."""
	conv_layer = target_layer_for(model, model_class)
	layer_names = sorted(
		(name for name in model.hookable_layers() if name.startswith("hidden.")),
		key=lambda name: int(name.split(".")[-1]),
	)
	captured_conv: list[torch.Tensor] = []
	captured_dnn: dict[str, torch.Tensor] = {}
	hooks: list[torch.utils.hooks.RemovableHandle] = []

	hooks.append(conv_layer.register_forward_hook(lambda _m, _i, output: captured_conv.append(output)))
	for layer_name in layer_names:
		hooks.append(
			model.hookable_layers()[layer_name].register_forward_hook(
				lambda _m, _i, output, name=layer_name: captured_dnn.__setitem__(name, output)
			)
		)

	batch = input_tensor.unsqueeze(0).to(device)
	with torch.no_grad():
		model(batch)

	acts = captured_conv[0].detach().cpu().numpy()
	_, n_channels, pool_h, pool_w = acts.shape
	dnn_weights = [
		model.hookable_layers()[name].weight.detach().float().to(device) for name in layer_names
	]
	dnn_activations = torch.stack(
		[captured_dnn[name][0].detach().float().to(device) for name in layer_names]
	)
	jacobian = custom_saliency_map(dnn_weights, dnn_activations)
	target_size = (int(batch.shape[-2]), int(batch.shape[-1]))

	for hook in hooks:
		hook.remove()
	return acts, jacobian, n_channels, pool_h, pool_w, target_size


def grad_nam_maps_for_neuron_from_model(
	model: nn.Module,
	eval_inputs: torch.Tensor,
	model_class: str,
	neuron_id: str,
	*,
	device: torch.device,
) -> list[np.ndarray]:
	"""Grad-NAM maps for one neuron using a loaded model (e.g. expert checkpoint)."""
	layer_idx = _layer_index_from_neuron_id(neuron_id)
	parsed = parse_unit_node_id(neuron_id)
	assert parsed is not None
	unit_idx = parsed["unit_index"]

	maps: list[np.ndarray] = []
	n_samples = int(eval_inputs.shape[0])
	for sample_idx in range(n_samples):
		acts, jacobian, n_channels, pool_h, pool_w, target_size = _get_acts_and_jac_analytical(
			model,
			eval_inputs[sample_idx],
			model_class,
			device=device,
		)
		row = jacobian[layer_idx, unit_idx].cpu().numpy()
		grads = _conv_grads_from_jacobian_row(row, n_channels, pool_h, pool_w)
		maps.append(_nam_from_acts_and_grads(acts, grads, target_size))
	return maps


def grad_nam_expert_saliency_for_neurons(
	model: nn.Module,
	eval_inputs: torch.Tensor,
	model_class: str,
	neuron_ids: list[str],
	*,
	device: torch.device,
) -> dict[str, list[np.ndarray]]:
	return {
		nid: grad_nam_maps_for_neuron_from_model(
			model, eval_inputs, model_class, nid, device=device
		)
		for nid in neuron_ids
	}


def target_size_for_mid(mid: str) -> tuple[int, int]:
	if mid.startswith("inception"):
		return (28, 28)
	if mid.startswith("resnet"):
		return (32, 32)
	raise ValueError(f"Unknown target size for model_id {mid!r}")


def build_pre_gap_conv_cache_entry(
	*,
	use_initial_model: str,
	model_class: str,
	model_config: dict[str, Any],
	mid: str,
	oid: str,
	seed: int,
	eval_inputs: torch.Tensor,
	device: torch.device,
) -> np.ndarray:
	model = load_initial_backbone_model(
		use_initial_model=use_initial_model,
		model_class=model_class,
		model_config=model_config,
		oid=oid,
		seed=seed,
		device=device,
	)
	try:
		return capture_pre_gap_conv_acts(model, eval_inputs, model_class, device=device)
	finally:
		model.cpu()
		if device.type == "cuda":
			torch.cuda.empty_cache()


def model_class_for_mid(mid: str) -> str:
	return mid_to_model_class(mid)
