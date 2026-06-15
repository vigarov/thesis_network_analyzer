"""PureShampoo optimizer signal extractor (no grafting).

Per-unit signals (gradient signals inherited from base):
- grad_norm, grad_cosine_sim
- h_inv_norm: ||L^{-1/4} ⊗ R^{-1/4}||_F derived from saved inverse-factor blocks

Layer-wise block signals (via on_after_step_shampoo_blocks):
- h_inv__{param}__L / __R: inverse Kronecker factor matrices (paper notation)
  for each preconditioned parameter (weights and biases).
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from compute_results.defaults import (
	DEFAULT_LR,
	DEFAULT_SHAMPOO_BETAS,
	DEFAULT_SHAMPOO_PRECONDITIONER_EPSILON,
	OPTIMIZER_BETAS_KEY,
	OPTIMIZER_LR_KEY,
	OPTIMIZER_TYPE_KEY,
	SHAMPOO_PRECONDITIONER_EPSILON_KEY,
)
from distributed_shampoo import DistributedShampoo, RootInvShampooPreconditionerConfig
from distributed_shampoo.shampoo_types import SHAMPOO_PRECONDITIONER_LIST, SingleDeviceDistributedConfig

from optimizers.base import OptimizerSignalExtractor, register_extractor

_SHAMPOO_FACTOR_SUFFIXES = ("L", "R")


def _param_names_in_optimizer_order(
	model: nn.Module, optimizer: torch.optim.Optimizer,
) -> list[str]:
	param_to_name = {id(p): name for name, p in model.named_parameters()}
	names: list[str] = []
	for group in optimizer.param_groups:
		for param in group["params"]:
			names.append(param_to_name.get(id(param), f"param_{len(names)}"))
	return names


def _h_inv_norm_from_blocks(blocks: dict[str, np.ndarray]) -> float:
	"""Global Frobenius norm from per-parameter block products.

	Keys are `h_inv__{param}__L` / `__R`; factors sharing a parameter prefix
	belong to one Kronecker block.  Because ||A ⊗ B||_F = ||A||_F · ||B||_F, the
	per-parameter norm is the product of factor Frobenius norms; the global
	norm is the vector-2-norm of those scalars.
	"""
	per_param_norms: list[float] = []
	prefixes: set[str] = set()
	for key in blocks:
		for suffix in _SHAMPOO_FACTOR_SUFFIXES:
			if key.endswith(f"__{suffix}"):
				prefixes.add(key[: -len(suffix) - 2])
				break
		else:
			prefixes.add(key)
	for prefix in sorted(prefixes):
		block_norm = 1.0
		for suffix in _SHAMPOO_FACTOR_SUFFIXES:
			mat = blocks.get(f"{prefix}__{suffix}")
			if mat is not None:
				block_norm *= float(np.linalg.norm(mat, ord="fro"))
		per_param_norms.append(block_norm)
	if not per_param_norms:
		return float("nan")
	return float(np.linalg.norm(np.asarray(per_param_norms, dtype=np.float32)))


def _extract_h_inv_blocks(
	optimizer: DistributedShampoo, model: nn.Module,
) -> tuple[dict[str, np.ndarray], float]:
	"""Extract inverse-factor matrices and derive the global h_inv norm.

	Returns `(blocks, h_inv_norm)` where each block is float32 on CPU with
	shape `(d, d)` and keys `h_inv__{safe_param}__L` / `__R`.
	"""
	blocks: dict[str, np.ndarray] = {}
	param_names = _param_names_in_optimizer_order(model, optimizer)
	name_idx = 0

	for state_lists in optimizer._per_group_state_lists:
		shampoo_list = state_lists[SHAMPOO_PRECONDITIONER_LIST]
		kf_list = shampoo_list._local_kronecker_factors_unwrapped
		for kf in kf_list:
			param_name = (
				param_names[name_idx]
				if name_idx < len(param_names)
				else f"param_{name_idx}"
			)
			name_idx += 1
			safe = param_name.replace(".", "__")
			inv_mats = getattr(kf, "inv_factor_matrices", None)
			if not inv_mats:
				continue
			for fi, mat in enumerate(inv_mats):
				suffix = (
					_SHAMPOO_FACTOR_SUFFIXES[fi]
					if fi < len(_SHAMPOO_FACTOR_SUFFIXES)
					else f"f{fi}"
				)
				blocks[f"h_inv__{safe}__{suffix}"] = (
					mat.detach().float().cpu().numpy()
				)

	h_inv_norm = _h_inv_norm_from_blocks(blocks)
	return blocks, h_inv_norm


@register_extractor
class PureShampooExtractor(OptimizerSignalExtractor):

	def __init__(
		self,
		*,
		lr: float = DEFAULT_LR,
		betas: tuple[float, float] = DEFAULT_SHAMPOO_BETAS,
		shampoo_preconditioner_epsilon: float = DEFAULT_SHAMPOO_PRECONDITIONER_EPSILON,
		**kwargs: Any,
	) -> None:
		super().__init__()
		self.lr = lr
		self.betas = betas
		self.shampoo_preconditioner_epsilon = shampoo_preconditioner_epsilon
		self._pending_block_signals: dict[str, np.ndarray] = {}

	def optimizer_id(self) -> str:
		return f"pure_shampoo_lr{self.lr}"

	def optimizer_config(self) -> dict[str, Any]:
		return {
			OPTIMIZER_TYPE_KEY: "pure_shampoo",
			OPTIMIZER_LR_KEY: self.lr,
			OPTIMIZER_BETAS_KEY: list(self.betas),
			SHAMPOO_PRECONDITIONER_EPSILON_KEY: self.shampoo_preconditioner_epsilon,
		}

	def create_optimizer(self, params) -> torch.optim.Optimizer:
		return DistributedShampoo(
			params,
			lr=self.lr,
			betas=self.betas,
			weight_decay=0.0,
			epsilon=self.shampoo_preconditioner_epsilon,
			grafting_config=None,
			preconditioner_config=RootInvShampooPreconditionerConfig(
				inv_factor_matrix_dtype=torch.float64,
			),
			max_preconditioner_dim = 1024,
			distributed_config=SingleDeviceDistributedConfig(
				target_parameter_dimensionality=2,
			),
		)

	def signal_names(self) -> list[str]:
		return ["grad", "grad_norm", "grad_cosine_sim", "h_inv_norm"]

	def on_before_step(
		self, model: nn.Module, optimizer: torch.optim.Optimizer,
	) -> dict[str, dict[str, float]]:
		return self._compute_grad_signals(model)

	def on_after_step(
		self, model: nn.Module, optimizer: torch.optim.Optimizer,
	) -> dict[str, dict[str, float]]:
		assert isinstance(optimizer, DistributedShampoo)
		blocks, h_inv = _extract_h_inv_blocks(optimizer, model)
		self._pending_block_signals = blocks
		return {
			"h_inv_norm": {u["node_id"]: h_inv for u in self._units},
		}

	def on_after_step_blocks(
		self, model: nn.Module, optimizer: torch.optim.Optimizer,
	) -> dict[str, np.ndarray]:
		return dict(self._pending_block_signals)
