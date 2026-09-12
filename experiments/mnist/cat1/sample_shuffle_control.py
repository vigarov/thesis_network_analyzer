"""Category 1 — individual samples: permuted supervision via global `LABEL_PERM`.

Pool construction matches the intent of `control_base` / `pretrain_then_shuffle_mislabel`:
after applying `LABEL_PERM` to true labels, take a balanced slice
`num_trial_samples // N_DIGITS` per true digit, merge, and train with
`DataLoader(..., shuffle=True)`. One trial per experiment run; total supervised
steps scale with `experiment_runs * num_trial_samples` (e.g. 10×100 = 1000 samples).
"""
from typing import Any, ClassVar

import torch
from torch.utils.data import DataLoader, Subset

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader
from experiments.mnist.relabel_ds import PermutedLabelMapDataset
from experiments.mnist.common.label_perm import LABEL_PERM


@register_experiment
class Cat1SampleShuffleControl(MNISTWrapper):
    """Control (no pretrain). Train from scratch on shuffled-digit supervision only.

    Each experiment run is a single trial of `num_trial_samples` images: relabel with
    the fixed global `LABEL_PERM` (same map as `pretrain_then_shuffle_mislabel`),
    then take the first `num_trial_samples // N_DIGITS` indices per true digit for
    uniformity, concatenate, and feed one shuffled `DataLoader`. Digit order across
    runs cycles through remainder slices (run `r` uses slice `r` per digit). No
    pretrain trial.
    """

    has_pretrain: ClassVar[bool] = False

    def __init__(
        self,
        *,
        num_trial_samples: int,
        **kwargs: Any,
    ) -> None:
        if not isinstance(num_trial_samples, int):
            raise TypeError(
                f"num_trial_samples must be int, got {type(num_trial_samples).__name__}"
            )
        if num_trial_samples < 1:
            raise ValueError(f"num_trial_samples must be >= 1, got {num_trial_samples}")
        if num_trial_samples % self.N_DIGITS != 0:
            raise ValueError(
                f"num_trial_samples must be divisible by N_DIGITS={self.N_DIGITS}, "
                f"got {num_trial_samples}."
            )
        super().__init__(**kwargs)
        self.num_trial_samples = num_trial_samples

    @property
    def first_digits(self) -> tuple[int, ...]:
        return tuple(range(self.N_DIGITS))

    def experiment_id(self) -> str:
        return f"cat1_sample_shuffle_control_tr{self.num_trial_samples}"

    def config_fields(self) -> dict[str, Any]:
        return {"num_trial_samples": self.num_trial_samples}

    def _eval_loaders(self) -> dict[str, DataLoader]:
        eval_bs = 256
        all_test_idx: list[int] = []
        for d in range(self.N_DIGITS):
            all_test_idx.extend(self._test_by_digit_indices[d])
        test_perm = PermutedLabelMapDataset(
            Subset(self._test_ds, all_test_idx),
            LABEL_PERM,
        )
        return {
            "all_test_perm_supervision": DataLoader(
                test_perm, batch_size=eval_bs, shuffle=False
            ),
            "all_test": make_loader(
                self._test_ds, all_test_idx, batch_size=eval_bs, shuffle=False
            ),
        }

    def _build_trials(
        self, batch_size: int, seed: int, *, num_experiment_runs: int
    ) -> list[list[TrialSpec]]:
        variability = self._experiment_variability.strip().lower()
        if variability != "":
            raise ValueError(
                f"{type(self).__name__} does not support experiment_variability="
                f"{self._experiment_variability!r}. Supported: ''."
            )
        per_digit = self.num_trial_samples // self.N_DIGITS
        indices_by_digit = digit_indices(self._train_ds, tuple(range(self.N_DIGITS)))
        for d in range(self.N_DIGITS):
            need = per_digit * num_experiment_runs
            if len(indices_by_digit[d]) < need:
                raise ValueError(
                    f"Not enough train indices for digit {d}: need {need} "
                    f"({per_digit} per run * {num_experiment_runs} runs), got {len(indices_by_digit[d])}."
                )

        eval_loaders = self._eval_loaders()
        out: list[list[TrialSpec]] = []
        for run_idx in range(num_experiment_runs):
            flat_indices: list[int] = []
            for d in range(self.N_DIGITS):
                idcs = indices_by_digit[d]
                lo = run_idx * per_digit
                hi = (run_idx + 1) * per_digit
                flat_indices.extend(idcs[lo:hi])
            post_ds = PermutedLabelMapDataset(
                Subset(self._train_ds, flat_indices),
                LABEL_PERM,
            )
            gen = torch.Generator()
            gen.manual_seed(seed + run_idx)
            loader = DataLoader(
                post_ds,
                batch_size=batch_size,
                shuffle=True,
                generator=gen,
            )
            out.append(
                [
                    TrialSpec(
                        name="trial_shuffle_labelperm_pool_balanced_global",
                        train_loader=loader,
                        eval_loaders=eval_loaders,
                    ),
                ]
            )
        return out
