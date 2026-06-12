"""Base class and registry for optimizer signal extractors."""

from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn


class OptimizerSignalExtractor(abc.ABC):
	"""Interface for extracting optimizer-specific signals during training.

	Each concrete extractor wraps a specific optimizer type and knows how to
	pull relevant internal state (moments, accumulators, etc.) out of it.

	All signals are **per-unit, per-iteration**: every signal maps `{node_id: scalar_value}`
	"""

	def __init__(self) -> None:
		self._units: list[dict[str, Any]] = []
		self._prev_grads: dict[str, torch.Tensor] = {}

	def set_units(self, units: list[dict[str, Any]]) -> None:
		"""Register the clickable units from the model (call before training)."""
		self._units = units

	@staticmethod
	def _find_weight_param(
		model: nn.Module, layer_name: str,
	) -> nn.Parameter | None:
		"""Find the weight parameter for a named layer."""
		params = dict(model.named_parameters())
		for candidate in (f"{layer_name}.weight", f"{layer_name}.0.weight"):
			if candidate in params:
				return params[candidate]
		for name, param in params.items():
			if name.startswith(layer_name) and "weight" in name:
				return param
		return None

	def _compute_grad_signals(
		self, model: nn.Module,
	) -> dict[str, dict[str, float]]:
		"""Per-unit gradient norm and cosine similarity vs previous iteration.

		Shared by all extractors - call from `on_before_step`.
		"""
		norms: dict[str, float] = {}
		cosines: dict[str, float] = {}

		for u in self._units:
			nid = u["node_id"]
			param = self._find_weight_param(model, u["layer_name"])
			if param is None or param.grad is None:
				norms[nid] = float("nan")
				cosines[nid] = float("nan")
				continue

			grad = param.grad[u["unit_index"]].detach().flatten().float()
			norms[nid] = grad.norm().item()

			prev = self._prev_grads.get(nid)
			if prev is not None and prev.shape == grad.shape:
				cosines[nid] = torch.nn.functional.cosine_similarity(
					grad.unsqueeze(0), prev.unsqueeze(0),
				).item()
			else:
				cosines[nid] = 0.0

			self._prev_grads[nid] = grad.clone()

		return {"grad_norm": norms, "grad_cosine_sim": cosines}

	# ------------------------------------------------------------------
	# Abstract interface
	# ------------------------------------------------------------------

	@abc.abstractmethod
	def optimizer_id(self) -> str:
		"""Unique identifier (e.g. 'adam_lr1e-3')."""

	@abc.abstractmethod
	def optimizer_config(self) -> dict[str, Any]:
		"""Optimizer-specific hyperparams to store / compare."""

	@abc.abstractmethod
	def create_optimizer(self, params) -> torch.optim.Optimizer:
		"""Instantiate the underlying optimizer."""

	@abc.abstractmethod
	def signal_names(self) -> list[str]:
		"""Names of per-unit, per-iteration scalar signals this extractor logs."""

	@abc.abstractmethod
	def on_before_step(
		self,
		model: nn.Module,
		optimizer: torch.optim.Optimizer,
	) -> dict[str, dict[str, float]]:
		"""Called after `loss.backward()` but before `optimizer.step()`.

		Return `{signal_name: {node_id: value}}`.
		"""

	@abc.abstractmethod
	def on_after_step(
		self,
		model: nn.Module,
		optimizer: torch.optim.Optimizer,
	) -> dict[str, dict[str, float]]:
		"""Called after `optimizer.step()`.

		Return `{signal_name: {node_id: value}}`.
		"""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_EXTRACTOR_REGISTRY: dict[str, type[OptimizerSignalExtractor]] = {}


def register_extractor(cls: type[OptimizerSignalExtractor]) -> type[OptimizerSignalExtractor]:
	key = cls.__name__
	_EXTRACTOR_REGISTRY[key] = cls
	return cls


def get_extractor(name: str, **kwargs: Any) -> OptimizerSignalExtractor:
	if name not in _EXTRACTOR_REGISTRY:
		raise KeyError(
			f"Unknown extractor '{name}'. "
			f"Available: {sorted(_EXTRACTOR_REGISTRY)}"
		)
	return _EXTRACTOR_REGISTRY[name](**kwargs)


def list_extractors() -> list[str]:
	return sorted(_EXTRACTOR_REGISTRY)
