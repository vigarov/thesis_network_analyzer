"""Experiment: Pretrain on all digits (first K per digit), then a self-contained adjacent-digit schedule.

Train pool: official MNIST train minus pinned eval samples (``MNISTWrapper``), no global
``samples_per_digit`` cap.

- **Pretrain** (``once_only=True``): first ``num_pretrain_samples`` indices per digit for
  digits 0--9, concatenated, shuffled.
- **Trial 1:** first ``num_trial_samples`` from remainder of digit ``a`` (correct label ``a``).
- **Trial 2:** same images as trial 1, training targets are ``b = (a + 1) % 10``.
- **Trial 3:** first ``num_trial_samples`` from remainder of digit ``b`` (correct label ``b``).

Evaluation loaders use true test labels. Only the mislabel trial uses replaced targets.

``experiment_variability``: ``""`` (flat list) or ``change_digits`` (run 0 includes pretrain with
configured anchor ``digitA``; later runs omit pretrain and use anchor
``(digitA + 2 * (t + 1)) % 10`` for ``t = 0 .. num_experiment_runs - 2``).
"""

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader


class _RelabelSubset(Dataset):
    """Subset of *base_ds* by *indices* with a fixed integer label for every sample."""

    def __init__(
        self,
        base_ds: torch.utils.data.Dataset,
        indices: list[int],
        label: int,
    ) -> None:
        self._base = base_ds
        self._indices = indices
        self._label = label

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self._base[self._indices[i]]
        y = torch.tensor(self._label, dtype=torch.long)
        return x, y


def _pretrain_and_remainder(
    indices_by_digit: dict[int, list[int]],
    num_pretrain: int,
) -> tuple[list[int], dict[int, list[int]]]:
    """First *num_pretrain* per digit 0..9 → pretrain flat list; rest → remainder per digit."""
    pretrain_parts: list[list[int]] = []
    remainder: dict[int, list[int]] = {}
    for d in range(10):
        idxs = indices_by_digit[d]
        if len(idxs) < num_pretrain:
            raise ValueError(
                f"Not enough train indices for digit {d}: need at least "
                f"{num_pretrain} for pretrain, got {len(idxs)}"
            )
        pretrain_parts.append(idxs[:num_pretrain])
        remainder[d] = idxs[num_pretrain:]
    pretrain_flat: list[int] = []
    for part in pretrain_parts:
        pretrain_flat.extend(part)
    return pretrain_flat, remainder


@register_experiment
class PretrainThenDigitAThenMislabelAsBThenDigitB(MNISTWrapper):
    """Pretrain, then *a* correct, same images mislabeled as *b*, then *b* correct (*b*=*a*+1 mod 10)."""

    def __init__(
        self,
        digitA: int = 0,
        num_pretrain_samples: int = 500,
        num_trial_samples: int = 100,
        **kwargs: Any,
    ) -> None:
        if not 0 <= digitA < self.N_DIGITS:
            raise ValueError(f"digitA must be in [0, {self.N_DIGITS}), got {digitA}")
        if num_pretrain_samples < 1:
            raise ValueError(f"num_pretrain_samples must be >= 1, got {num_pretrain_samples}")
        if num_trial_samples < 1:
            raise ValueError(f"num_trial_samples must be >= 1, got {num_trial_samples}")
        digit_b = (digitA + 1) % self.N_DIGITS
        super().__init__(digitA=digitA, digitB=digit_b, samples_per_digit=None, **kwargs)
        self.num_pretrain_samples = num_pretrain_samples
        self.num_trial_samples = num_trial_samples

    @property
    def first_digits(self) -> tuple[int, ...]:
        return (self.digitA, (self.digitA + 1) % self.N_DIGITS)

    def experiment_id(self) -> str:
        b = (self.digitA + 1) % self.N_DIGITS
        return (
            f"pretrain_{self.digitA}_mislabel{b}_{b}_"
            f"pt{self.num_pretrain_samples}_tr{self.num_trial_samples}"
        )

    def config_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "digitA": self.digitA,
            "num_pretrain_samples": self.num_pretrain_samples,
            "num_trial_samples": self.num_trial_samples,
        }
        if self.samples_per_digit is not None:
            fields["samples_per_digit"] = self.samples_per_digit
        return fields

    def _eval_loaders_for_pair(
        self,
        digit_a: int,
        digit_b: int,
    ) -> dict[str, DataLoader]:
        all_test_idx = sum(self._test_by_digit_indices.values(), [])
        ab_test = (
            self._test_by_digit_indices[digit_a] + self._test_by_digit_indices[digit_b]
        )
        return {
            "all_test": make_loader(
                self._test_ds, all_test_idx, batch_size=256, shuffle=False
            ),
            "ab_test": make_loader(self._test_ds, ab_test, batch_size=256, shuffle=False),
            f"digitA_{digit_a}_test": make_loader(
                self._test_ds,
                self._test_by_digit_indices[digit_a],
                batch_size=256,
                shuffle=False,
            ),
            f"digitB_{digit_b}_test": make_loader(
                self._test_ds,
                self._test_by_digit_indices[digit_b],
                batch_size=256,
                shuffle=False,
            ),
        }

    def _digit_trials(
        self,
        pretrain_indices: list[int] | None,
        remaining_by_digit: dict[int, list[int]],
        digit_a: int,
        batch_size: int,
    ) -> tuple[TrialSpec, ...]:
        digit_b = (digit_a + 1) % self.N_DIGITS
        eval_loaders = self._eval_loaders_for_pair(digit_a, digit_b)

        for d, name in ((digit_a, "digit_a"), (digit_b, "digit_b")):
            rem = remaining_by_digit[d]
            if len(rem) < self.num_trial_samples:
                raise ValueError(
                    f"Not enough remaining train indices for {name} (digit {d}): "
                    f"need {self.num_trial_samples}, got {len(rem)}"
                )

        idx_a = remaining_by_digit[digit_a][: self.num_trial_samples]
        idx_b = remaining_by_digit[digit_b][: self.num_trial_samples]

        trial_a_correct = TrialSpec(
            name=f"trial_digit{digit_a}",
            train_loader=make_loader(
                self._train_ds, idx_a, batch_size, shuffle=False
            ),
            eval_loaders=eval_loaders,
        )
        relabel_ds = _RelabelSubset(self._train_ds, idx_a, digit_b)
        train_mis = DataLoader(
            relabel_ds,
            batch_size=batch_size,
            shuffle=False,
        )
        trial_a_mislabel = TrialSpec(
            name=f"trial_digit{digit_a}_as_label{digit_b}",
            train_loader=train_mis,
            eval_loaders=eval_loaders,
        )
        trial_b_correct = TrialSpec(
            name=f"trial_digit{digit_b}",
            train_loader=make_loader(
                self._train_ds, idx_b, batch_size, shuffle=False
            ),
            eval_loaders=eval_loaders,
        )

        if pretrain_indices is not None:
            trial_pre = TrialSpec(
                name="trial_pretrain_all_digits",
                train_loader=make_loader(
                    self._train_ds,
                    pretrain_indices,
                    batch_size=batch_size,
                    shuffle=True,
                ),
                eval_loaders=eval_loaders,
                once_only=True,
            )
            return trial_pre, trial_a_correct, trial_a_mislabel, trial_b_correct
        return trial_a_correct, trial_a_mislabel, trial_b_correct

    def _build_trials(
        self, batch_size: int, seed: int, *, num_experiment_runs: int
    ) -> list[TrialSpec] | list[list[TrialSpec]]:
        variability = self._experiment_variability.strip().lower()
        if variability not in ("", "change_digits"):
            raise ValueError(
                f"PretrainThenDigitAThenMislabelAsBThenDigitB does not support "
                f"experiment_variability={self._experiment_variability!r}. "
                "Supported: '', 'change_digits'."
            )

        indices_by_digit = digit_indices(
            self._train_ds, tuple(range(self.N_DIGITS))
        )
        pretrain_flat, remaining_by_digit = _pretrain_and_remainder(
            indices_by_digit, self.num_pretrain_samples
        )

        s_pre, s_a, s_mis, s_b = self._digit_trials(
            pretrain_flat,
            remaining_by_digit,
            self.digitA,
            batch_size,
        )
        t0_list = [s_pre, s_a, s_mis, s_b]

        if variability == "":
            return t0_list

        per_run: list[list[TrialSpec]] = [t0_list]
        for t in range(num_experiment_runs - 1):
            anchor = (self.digitA + 2 * (t + 1)) % self.N_DIGITS
            trials = self._digit_trials(
                None, remaining_by_digit, anchor, batch_size
            )
            per_run.append(list(trials))
        return per_run
