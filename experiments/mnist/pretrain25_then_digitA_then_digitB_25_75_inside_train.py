"""Experiment: Pre-train on 25 % base, then digitA, then digitB (within 75 % remaining).

Uses the official MNIST train/test split:
- Training stages draw from ``train=True`` filtered to the relevant digits.
- Evaluation uses ``train=False`` filtered to the relevant digits.

Within the train set, base_ratio (e.g.: 25 %) goes to a "base" subset that the model will be pre-trained on
 and the rest (e.g.: 75 %) to a "remaining" subset used for per-digit training.

Stage A (pretrain): train on all digits in the base subset (shuffled).  [once_only=True]
Stage B (digitA): train on digitA within the remaining subset.
Stage C (digitB): train on digitB within the remaining subset.

Trials / variability
--------------------
- Empty variability: flat list; pretrain runs only in trial 0 via ``once_only=True``.
- ``change_digits``: nested list of length ``num_trials``; trial 0 includes pretrain +
  original digits; further trials shift the digit pairs like in digitA_then_digitB_75_25.py
"""

from typing import Any

import torch
from torch.utils.data import Subset

from experiments.base import StageSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader


def _split_base_remaining(
    base_ratio: float,
    indices_by_digit: dict[int, list[int]],
    seed: int,
    shuffle: bool = True,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Split per-digit index lists into base / remaining.
    Args:
        base_ratio: ratio of indices to put in the base subset (for pretraining)
        indices_by_digit: dict of {digit: [indices]} for all digits to take into account
        seed: random seed
    Returns:
        tuple of (base_indices_by_digit, remaining_indices_by_digit)
    """
    rng = torch.Generator().manual_seed(seed)
    base: dict[int, list[int]] = {}
    remaining: dict[int, list[int]] = {}
    for d, idxs in indices_by_digit.items():
        if shuffle:
            perm = torch.randperm(len(idxs), generator=rng).tolist()
        else:
            perm = list(range(len(idxs)))
        n_base = int(len(idxs) * base_ratio)
        base[d] = [idxs[i] for i in perm[:n_base]]
        remaining[d] = [idxs[i] for i in perm[n_base:]]
    return base, remaining


@register_experiment
class Pretrain25ThenDigitAThenDigitB_25_75(MNISTWrapper):
    """Three-stage continual learning with a pre-training base."""

    def __init__(
        self,
        digitA: int = 1,
        digitB: int = 2,
        base_ratio: float = 0.25,
        **kwargs,
    ):
        if not (0 < base_ratio < 1):
            raise ValueError(f"base_ratio must be in (0, 1), got {base_ratio}")
        super().__init__(digitA=digitA, digitB=digitB, **kwargs)
        self.base_ratio = base_ratio
        self.all_digits_test_indices = tuple(range(self.N_DIGITS))

    def experiment_id(self) -> str:
        pct = int(self.base_ratio * 100)
        return f"pretrain{pct}_then_{self.digitA}_then_{self.digitB}_25_75"

    def config_fields(self) -> dict[str, Any]:
        return {
            "digitA": self.digitA,
            "digitB": self.digitB,
            "base_ratio": self.base_ratio,
        }

    def _digit_stages(
        self,
        pretrain_indices: list[int] | None, # None if no pretraining at that trial
        remaining_indices_by_digit: dict[int, list[int]],
        digitA: int,
        digitB: int,
        batch_size: int,
    ) -> tuple[StageSpec, ...]:
        """Build digit stages B and C for a given pair; also returns pretrain stage if applicable."""
        
        both_digits_test = self._test_by_digit_indices[digitA] + self._test_by_digit_indices[digitB]
        eval_loaders = {
            "all_test": make_loader(self._test_ds, sum(self._test_by_digit_indices.values(), []), batch_size=256, shuffle=False),
            }
        
        extra_eval_loaders = {
            "both_test": make_loader(self._test_ds, both_digits_test, batch_size=256, shuffle=False),
            f"digitA_{digitA}_test": make_loader(
                self._test_ds, self._test_by_digit_indices[digitA], batch_size=256, shuffle=False,
            ),
            f"digitB_{digitB}_test": make_loader(
                self._test_ds, self._test_by_digit_indices[digitB], batch_size=256, shuffle=False,
            ),
        }


        stage_b = StageSpec(
            name=f"stageB_digit{digitA}",
            train_loader=make_loader(self._train_ds, remaining_indices_by_digit[digitA], batch_size),
            eval_loaders=eval_loaders | extra_eval_loaders,
        )
        stage_c = StageSpec(
            name=f"stageC_digit{digitB}",
            train_loader=make_loader(self._train_ds, remaining_indices_by_digit[digitB], batch_size),
            eval_loaders=eval_loaders | extra_eval_loaders,
        )

        if pretrain_indices is not None:
            stage_a = StageSpec(
                name="stageA_base_pretrain",
                train_loader=make_loader(self._train_ds, pretrain_indices, batch_size=batch_size),
                eval_loaders=eval_loaders,
                once_only=True,
            )
            return stage_a, stage_b, stage_c
        return stage_b, stage_c

    def _build_stages(
        self, batch_size: int, seed: int, *, num_trials: int
    ) -> list[StageSpec] | list[list[StageSpec]]:
        variability = self._trial_variability.strip().lower()
        if variability not in ("", "change_digits"):
            raise ValueError(
                f"Pretrain25ThenDigitAThenDigitB_25_75 does not support "
                f"trial_variability={self._trial_variability!r}. "
                "Supported values: '' (none), 'change_digits'."
            )

        pretrain_indices_by_digit, remaining_indices_by_digit = _split_base_remaining(
            self.base_ratio, 
            digit_indices(self._train_ds,tuple(range(self.N_DIGITS))),
            seed=seed,
            shuffle=True,
        )

        # trial 0: pretrain + original digits
        t0_pretrain, t0_digitA, t0_digitB = self._digit_stages(
            sum(pretrain_indices_by_digit.values(), []), \
            remaining_indices_by_digit, \
            self.digitA, self.digitB, batch_size)
        
        t0_list = [t0_pretrain, t0_digitA, t0_digitB]
        if variability == "":
            return t0_list
        else:
            per_trial: list[list[StageSpec]] = [t0_list]
            if variability == "change_digits":
                for t in range(num_trials-1):
                    dA = (self.digitA + 2 * t) % self.N_DIGIT_MOD
                    dB = (self.digitB + 2 * t) % self.N_DIGIT_MOD
                    stage_b, stage_c = self._digit_stages(None,remaining_indices_by_digit, dA, dB, batch_size)
                    per_trial.append([stage_b, stage_c])
        return per_trial
