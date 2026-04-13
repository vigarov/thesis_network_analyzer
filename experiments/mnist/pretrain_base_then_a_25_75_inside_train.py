"""Experiment: Pre-train on a base fraction of each digit, then one digit trial (remaining pool).

Uses the official MNIST train/test split:
- Training trials draw from ``train=True`` filtered to the relevant digits.
- Evaluation uses ``train=False`` filtered to the relevant digits.

Within the train set, ``base_ratio`` (e.g. 25%) goes to a "base" subset used for pretraining;
the rest goes to a "remaining" subset used for the per-digit trial.

Trial A (pretrain): train on all digits in the base subset (shuffled).  [``once_only=True``]
Trial B: train on ``digitA`` within the remaining subset.

Experiment runs / variability
-----------------------------
- Empty variability: flat list; pretrain runs only in run 0 via ``once_only=True``.
- ``change_digits``: nested list of length ``num_experiment_runs``; run 0 is pretrain + ``digitA``;
  later runs omit pretrain and use ``(digitA + r) % 10`` for run index ``r >= 1``.
"""

from typing import Any

import torch

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader


def _split_base_remaining(
    base_ratio: float,
    indices_by_digit: dict[int, list[int]],
    seed: int,
    shuffle: bool = True,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Split per-digit index lists into base / remaining."""
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
class PretrainBase_thenDigitA(MNISTWrapper):
    """Pretrain on a base subset, then a single digit trial in the remainder."""

    def __init__(
        self,
        digitA: int = 0,
        base_ratio: float = 0.25,
        **kwargs: Any,
    ):
        if not (0 < base_ratio < 1):
            raise ValueError(f"base_ratio must be in (0, 1), got {base_ratio}")
        super().__init__(digitA=digitA, digitB=digitA, **kwargs)
        self.base_ratio = base_ratio
        self.all_digits_test_indices = tuple(range(self.N_DIGITS))

    @property
    def first_digits(self) -> tuple[int, ...]:
        return (self.digitA,)

    def experiment_id(self) -> str:
        pct = int(self.base_ratio * 100)
        return f"pretrain{pct}_then_{self.digitA}"

    def config_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {"digitA": self.digitA, "base_ratio": self.base_ratio}
        if self.samples_per_digit is not None:
            fields["samples_per_digit"] = self.samples_per_digit
        return fields

    def _digit_trials(
        self,
        pretrain_indices: list[int] | None,
        remaining_indices_by_digit: dict[int, list[int]],
        digit: int,
        batch_size: int,
    ) -> tuple[TrialSpec, ...]:
        eval_loaders = {
            "all_test": make_loader(
                self._test_ds,
                sum(self._test_by_digit_indices.values(), []),
                batch_size=256,
                shuffle=False,
            ),
        }
        extra_eval_loaders = {
            f"digit_{digit}_test": make_loader(
                self._test_ds,
                self._test_by_digit_indices[digit],
                batch_size=256,
                shuffle=False,
            ),
        }

        trial_digit = TrialSpec(
            name=f"trial_digit{digit}",
            train_loader=make_loader(
                self._train_ds,
                remaining_indices_by_digit[digit],
                batch_size,
                shuffle=False,
            ),
            eval_loaders=eval_loaders | extra_eval_loaders,
        )

        if pretrain_indices is not None:
            trial_pre = TrialSpec(
                name="trial_base_pretrain",
                train_loader=make_loader(
                    self._train_ds, pretrain_indices, batch_size=batch_size, shuffle=True
                ),
                eval_loaders=eval_loaders,
                once_only=True,
            )
            return trial_pre, trial_digit
        return (trial_digit,)

    def _build_trials(
        self, batch_size: int, seed: int, *, num_experiment_runs: int
    ) -> list[TrialSpec] | list[list[TrialSpec]]:
        variability = self._experiment_variability.strip().lower()
        if variability not in ("", "change_digits"):
            raise ValueError(
                f"PretrainBase_thenDigitA does not support "
                f"experiment_variability={self._experiment_variability!r}. "
                "Supported values: '' (none), 'change_digits'."
            )

        pretrain_indices_by_digit, remaining_indices_by_digit = _split_base_remaining(
            self.base_ratio,
            digit_indices(self._train_ds, tuple(range(self.N_DIGITS))),
            seed=seed,
            shuffle=True,
        )

        t0_pretrain, t0_digit = self._digit_trials(
            sum(pretrain_indices_by_digit.values(), []),
            remaining_indices_by_digit,
            self.digitA,
            batch_size,
        )
        t0_list = [t0_pretrain, t0_digit]
        if variability == "":
            return t0_list

        per_run: list[list[TrialSpec]] = [t0_list]
        for t in range(num_experiment_runs - 1):
            d = (self.digitA + t + 1) % self.N_DIGITS
            (trial_digit,) = self._digit_trials(
                None, remaining_indices_by_digit, d, batch_size
            )
            per_run.append([trial_digit])
        return per_run
