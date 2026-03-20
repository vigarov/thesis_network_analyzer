"""PureShampoo optimizer signal extractor (no grafting).

Per-unit signals (gradient signals inherited from base):
- grad_norm, grad_cosine_sim
- h_inv_norm: global ||L^{-1/4} ⊗ R^{-1/4}||_F computed from
  per-block Kronecker factor inverse matrices.  Because
  ||A ⊗ B||_F = ||A||_F · ||B||_F, the overall norm is the
  vector norm of per-block products.
"""
from typing import Any

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
from distributed_shampoo.shampoo_types import SHAMPOO_PRECONDITIONER_LIST

from optimizers.base import OptimizerSignalExtractor, register_extractor


def _compute_h_inv_norm(optimizer: DistributedShampoo) -> float:
    """||L^{-1/4} ⊗ R^{-1/4}||_F across all parameter blocks.

    For each block the Kronecker-product Frobenius norm factors as
    the product of per-factor Frobenius norms.  The global norm is
    the vector-2-norm of these per-block scalars.
    """
    per_block_norms = []
    for state_lists in optimizer._per_group_state_lists:
        shampoo_list = state_lists[SHAMPOO_PRECONDITIONER_LIST]
        for kf in shampoo_list._local_kronecker_factors_unwrapped:
            inv_mats = getattr(kf, "inv_factor_matrices", None)
            if inv_mats:
                block_norm = 1.0
                for mat in inv_mats:
                    block_norm *= torch.linalg.matrix_norm(
                        mat.detach().float(), ord="fro"
                    ).item()
                per_block_norms.append(block_norm)
    if not per_block_norms:
        return float("nan")
    return torch.linalg.vector_norm(
        torch.tensor(per_block_norms, dtype=torch.float32)
    ).item()


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
        )

    def signal_names(self) -> list[str]:
        return ["grad_norm", "grad_cosine_sim", "h_inv_norm"]

    def on_before_step(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        return self._compute_grad_signals(model)

    def on_after_step(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        assert isinstance(optimizer, DistributedShampoo)
        h_inv = _compute_h_inv_norm(optimizer)
        return {
            "h_inv_norm": {u["node_id"]: h_inv for u in self._units},
        }
