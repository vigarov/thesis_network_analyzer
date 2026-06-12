"""Adam optimizer signal extractor.

Per-unit signals (gradient signals inherited from base):
- grad_norm, grad_cosine_sim
- exp_avg_norm: L2 norm of the first-moment slice for each unit.
- exp_avg_sq_norm: L2 norm of the second-moment slice for each unit.
- moment_cosine_sim: cosine similarity between consecutive first-moment
  snapshots for each unit.
"""
from typing import Any

import torch

from compute_results.defaults import (
	DEFAULT_ADAM_BETAS,
	DEFAULT_ADAM_EPS,
	DEFAULT_LR,
	OPTIMIZER_BETAS_KEY,
	OPTIMIZER_EPS_KEY,
	OPTIMIZER_LR_KEY,
	OPTIMIZER_TYPE_KEY,
)
from optimizers.base import OptimizerSignalExtractor, register_extractor


@register_extractor
class AdamExtractor(OptimizerSignalExtractor):

	def __init__(
		self,
		*,
		lr: float = DEFAULT_LR,
		betas: tuple[float, float] = DEFAULT_ADAM_BETAS,
		eps: float = DEFAULT_ADAM_EPS,
		**kwargs: Any,
	) -> None:
		super().__init__()
		self.lr = lr
		self.betas = betas
		self.eps = eps
		self._prev_exp_avg: dict[str, torch.Tensor] = {}

	def optimizer_id(self) -> str:
		return f"adam_lr{self.lr}"

	def optimizer_config(self) -> dict[str, Any]:
		return {
			OPTIMIZER_TYPE_KEY: "adam",
			OPTIMIZER_LR_KEY: self.lr,
			OPTIMIZER_BETAS_KEY: list(self.betas),
			OPTIMIZER_EPS_KEY: self.eps,
		}

	def create_optimizer(self, params) -> torch.optim.Optimizer:
		return torch.optim.Adam(params, lr=self.lr, betas=self.betas, eps=self.eps)

	def signal_names(self) -> list[str]:
		return [
			"grad_norm",
			"grad_cosine_sim",
			"exp_avg_norm",
			"exp_avg_sq_norm",
			"moment_cosine_sim",
		]

	def on_before_step(
		self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
	) -> dict[str, dict[str, float]]:
		return self._compute_grad_signals(model)

	def on_after_step(
		self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
	) -> dict[str, dict[str, float]]:
		avg_norms: dict[str, float] = {}
		sq_norms: dict[str, float] = {}
		moment_cos: dict[str, float] = {}

		for u in self._units:
			nid = u["node_id"]
			param = self._find_weight_param(model, u["layer_name"])
			if param is None:
				avg_norms[nid] = float("nan")
				sq_norms[nid] = float("nan")
				moment_cos[nid] = float("nan")
				continue

			state = optimizer.state.get(param)
			if not state or "exp_avg" not in state:
				avg_norms[nid] = float("nan")
				sq_norms[nid] = float("nan")
				moment_cos[nid] = float("nan")
				continue

			idx = u["unit_index"]
			ea = state["exp_avg"][idx].detach().flatten().float()
			eas = state["exp_avg_sq"][idx].detach().flatten().float()

			avg_norms[nid] = ea.norm().item()
			sq_norms[nid] = eas.norm().item()

			prev = self._prev_exp_avg.get(nid)
			if prev is not None and prev.shape == ea.shape:
				moment_cos[nid] = torch.nn.functional.cosine_similarity(
					ea.unsqueeze(0), prev.unsqueeze(0),
				).item()
			else:
				moment_cos[nid] = 0.0
			self._prev_exp_avg[nid] = ea.clone()

		return {
			"exp_avg_norm": avg_norms,
			"exp_avg_sq_norm": sq_norms,
			"moment_cosine_sim": moment_cos,
		}
