"""Experiment: Pre-train on 25 % base, then digitA, then digitB (within 75 % remaining).

Uses the official MNIST train/test split:
- Training stages draw from ``train=True`` filtered to the relevant digits.
- Evaluation uses ``train=False`` filtered to the relevant digits.

Within the train set, 25 % goes to a "base" subset and 75 % to a "remaining" subset.

Stage A: train on all digits in the base subset (shuffled).
Stage B: train on digitA within the remaining subset.
Stage C: train on digitB within the remaining subset.
"""

from typing import Any

import torch

from experiments.base import StageSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader


def _split_base_remaining(
    indices_by_digit: dict[int, list[int]],
    base_ratio: float,
    seed: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Split per-digit index lists into base / remaining."""
    rng = torch.Generator().manual_seed(seed)
    base: dict[int, list[int]] = {}
    remaining: dict[int, list[int]] = {}
    for d, idxs in indices_by_digit.items():
        perm = torch.randperm(len(idxs), generator=rng).tolist()
        n_base = int(len(idxs) * base_ratio)
        base[d] = [idxs[i] for i in perm[:n_base]]
        remaining[d] = [idxs[i] for i in perm[n_base:]]
    return base, remaining


@register_experiment
class Pretrain25ThenDigitAThenDigitB_25_75(MNISTWrapper):
    """Three-stage continual learning with a pre-training base."""

    def experiment_id(self) -> str:
        return f"pretrain25_then_{self.digitA}_then_{self.digitB}_25_75"

    def config_fields(self) -> dict[str, Any]:
        return {
            "digitA": self.digitA,
            "digitB": self.digitB,
            "base_ratio": 0.25,
        }

    def build_stages(self, batch_size: int, seed: int) -> list[StageSpec]:
        train_by_digit = digit_indices(self._train_ds, self.digits)
        test_by_digit = digit_indices(self._test_ds, self.digits)

        base_idx, remaining_idx = _split_base_remaining(
            train_by_digit, base_ratio=self.config_fields()["base_ratio"], seed=seed,
        )

        all_test = test_by_digit[self.digitA] + test_by_digit[self.digitB]

        eval_loaders = {
            "all_test": make_loader(self._test_ds, all_test, batch_size=256, shuffle=False),
            f"digitA_{self.digitA}_test": make_loader(
                self._test_ds, test_by_digit[self.digitA], batch_size=256, shuffle=False,
            ),
            f"digitB_{self.digitB}_test": make_loader(
                self._test_ds, test_by_digit[self.digitB], batch_size=256, shuffle=False,
            ),
        }

        all_base = base_idx[self.digitA] + base_idx[self.digitB]

        stage_a = StageSpec(
            name="stageA_base_pretrain",
            train_loader=make_loader(self._train_ds, all_base, batch_size),
            eval_loaders=eval_loaders,
        )
        stage_b = StageSpec(
            name=f"stageB_digit{self.digitA}",
            train_loader=make_loader(self._train_ds, remaining_idx[self.digitA], batch_size),
            eval_loaders=eval_loaders,
        )
        stage_c = StageSpec(
            name=f"stageC_digit{self.digitB}",
            train_loader=make_loader(self._train_ds, remaining_idx[self.digitB], batch_size),
            eval_loaders=eval_loaders,
            is_turning_point=True,
        )
        return [stage_a, stage_b, stage_c]
