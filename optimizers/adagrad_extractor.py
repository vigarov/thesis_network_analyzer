"""AdaGrad optimizer signal extractor.

Per-unit signals (gradient signals inherited from base):
- grad_norm, grad_cosine_sim
- effective_lr_mean: mean effective learning rate for each unit's incoming
  weights, computed as lr / sqrt(accumulator + eps).
"""
from typing import Any

import torch
from optimizers.base import OptimizerSignalExtractor, register_extractor


@register_extractor
class AdaGradExtractor(OptimizerSignalExtractor):

    def __init__(self, lr: float = 1e-3, eps: float = 1e-10):
        super().__init__()
        self.lr = lr
        self.eps = eps

    def optimizer_id(self) -> str:
        return f"adagrad_lr{self.lr}"

    def optimizer_config(self) -> dict[str, Any]:
        return {"type": "adagrad", "lr": self.lr, "eps": self.eps}

    def create_optimizer(self, params) -> torch.optim.Optimizer:
        return torch.optim.Adagrad(params, lr=self.lr, eps=self.eps)

    def signal_names(self) -> list[str]:
        return ["grad_norm", "grad_cosine_sim", "effective_lr_mean"]

    def on_before_step(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        return self._compute_grad_signals(model)

    def on_after_step(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        eff_lrs: dict[str, float] = {}

        for u in self._units:
            nid = u["node_id"]
            param = self._find_weight_param(model, u["layer_name"])
            if param is None:
                eff_lrs[nid] = float("nan")
                continue

            state = optimizer.state.get(param)
            if not state or "sum" not in state:
                eff_lrs[nid] = self.lr
                continue

            idx = u["unit_index"]
            accum = state["sum"][idx].detach().flatten().float()
            eff = self.lr / (accum.sqrt() + self.eps)
            eff_lrs[nid] = eff.mean().item()

        return {"effective_lr_mean": eff_lrs}
