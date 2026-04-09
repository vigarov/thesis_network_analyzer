"""Experiment: Pretrain on all digits (first K per digit), then digitA, digitB, digitC with C relabeled as B.

Train pool: official MNIST train minus pinned eval samples (``MNISTWrapper``), no global ``samples_per_digit`` cap.

- **Pretrain** (``once_only=True``): first ``num_pretrain_samples`` indices per digit for digits 0--9, concatenated, shuffled.
- **digitA / digitB**: first ``num_trial_samples`` from each digit's remainder after the pretrain prefix.
- **digitC**: same images as digitC remainder, but training targets are ``digitB`` (mislabeling).

Evaluation loaders use true test labels. Only the digitC *training* trial uses replaced labels.

``experiment_variability``: ``""`` (flat list) or ``change_digits`` (run 0 includes pretrain with configured
``digitA`` / ``digitB`` / ``digitC``; each later run omits pretrain and sets
``digitA <- (digitC + 2t) % 10``, ``digitB <- (digitC + 2t + 1) % 10``,
``digitC <- (digitC + 2t + 2) % 10`` (images of the new C, labels of the new B), for
``t = 0 .. num_experiment_runs-2`` in the follow-up runs).
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
class PretrainRelabelC(MNISTWrapper):
    """Pretrain on all digits, then A, B, C with C trained as label B."""

    def __init__(
        self,
        digitA: int = 0,
        digitB: int = 1,
        digitC: int = 2,
        num_pretrain_samples: int = 500,
        num_trial_samples: int = 100,
        **kwargs: Any,
    ) -> None:
        if digitA == digitB or digitA == digitC or digitB == digitC:
            raise ValueError(
                f"digitA, digitB, digitC must be pairwise distinct; got "
                f"{digitA}, {digitB}, {digitC}"
            )
        if num_pretrain_samples < 1:
            raise ValueError(f"num_pretrain_samples must be >= 1, got {num_pretrain_samples}")
        if num_trial_samples < 1:
            raise ValueError(f"num_trial_samples must be >= 1, got {num_trial_samples}")
        super().__init__(digitA=digitA, digitB=digitB, samples_per_digit=None, **kwargs)
        self.digitC = digitC
        self.num_pretrain_samples = num_pretrain_samples
        self.num_trial_samples = num_trial_samples

    @property
    def first_digits(self) -> tuple[int, ...]:
        return (self.digitA, self.digitB, self.digitC)

    def experiment_id(self) -> str:
        return (
            f"pretrain_relabelC_{self.digitA}_{self.digitB}_{self.digitC}_"
            f"pt{self.num_pretrain_samples}_tr{self.num_trial_samples}"
        )

    def config_fields(self) -> dict[str, Any]:
        return {
            **super().config_fields(),
            "digitC": self.digitC,
            "num_pretrain_samples": self.num_pretrain_samples,
            "num_trial_samples": self.num_trial_samples,
        }

    def _eval_loaders_for_digits(
        self,
        digitA: int,
        digitB: int,
        digitC: int,
    ) -> dict[str, DataLoader]:
        all_test_idx = sum(self._test_by_digit_indices.values(), [])
        abc_test = (
            self._test_by_digit_indices[digitA]
            + self._test_by_digit_indices[digitB]
            + self._test_by_digit_indices[digitC]
        )
        return {
            "all_test": make_loader(
                self._test_ds, all_test_idx, batch_size=256, shuffle=False
            ),
            "abc_test": make_loader(self._test_ds, abc_test, batch_size=256, shuffle=False),
            f"digitA_{digitA}_test": make_loader(
                self._test_ds,
                self._test_by_digit_indices[digitA],
                batch_size=256,
                shuffle=False,
            ),
            f"digitB_{digitB}_test": make_loader(
                self._test_ds,
                self._test_by_digit_indices[digitB],
                batch_size=256,
                shuffle=False,
            ),
            f"digitC_{digitC}_test": make_loader(
                self._test_ds,
                self._test_by_digit_indices[digitC],
                batch_size=256,
                shuffle=False,
            ),
        }

    def _digit_trials(
        self,
        pretrain_indices: list[int] | None,
        remaining_by_digit: dict[int, list[int]],
        digitA: int,
        digitB: int,
        digitC: int,
        batch_size: int,
    ) -> tuple[TrialSpec, ...]:
        eval_loaders = self._eval_loaders_for_digits(digitA, digitB, digitC)

        for d, name in (
            (digitA, "digitA"),
            (digitB, "digitB"),
            (digitC, "digitC"),
        ):
            rem = remaining_by_digit[d]
            if len(rem) < self.num_trial_samples:
                raise ValueError(
                    f"Not enough remaining train indices for {name} (digit {d}): "
                    f"need {self.num_trial_samples}, got {len(rem)}"
                )

        idx_a = remaining_by_digit[digitA][: self.num_trial_samples]
        idx_b = remaining_by_digit[digitB][: self.num_trial_samples]
        idx_c = remaining_by_digit[digitC][: self.num_trial_samples]

        trial_a = TrialSpec(
            name=f"trial_digit{digitA}",
            train_loader=make_loader(
                self._train_ds, idx_a, batch_size, shuffle=False
            ),
            eval_loaders=eval_loaders,
        )
        trial_b = TrialSpec(
            name=f"trial_digit{digitB}",
            train_loader=make_loader(
                self._train_ds, idx_b, batch_size, shuffle=False
            ),
            eval_loaders=eval_loaders,
        )
        relabel_ds = _RelabelSubset(self._train_ds, idx_c, digitB)
        train_c = DataLoader(
            relabel_ds,
            batch_size=batch_size,
            shuffle=False,
        )
        trial_c = TrialSpec(
            name=f"trial_digit{digitC}_as_label{digitB}",
            train_loader=train_c,
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
            return trial_pre, trial_a, trial_b, trial_c
        return trial_a, trial_b, trial_c

    def _build_trials(
        self, batch_size: int, seed: int, *, num_experiment_runs: int
    ) -> list[TrialSpec] | list[list[TrialSpec]]:
        variability = self._experiment_variability.strip().lower()
        if variability not in ("", "change_digits"):
            raise ValueError(
                f"PretrainRelabelC does not support experiment_variability="
                f"{self._experiment_variability!r}. Supported: '', 'change_digits'."
            )

        indices_by_digit = digit_indices(
            self._train_ds, tuple(range(self.N_DIGITS))
        )
        pretrain_flat, remaining_by_digit = _pretrain_and_remainder(
            indices_by_digit, self.num_pretrain_samples
        )

        s_pre, s_a, s_b, s_c = self._digit_trials(
            pretrain_flat,
            remaining_by_digit,
            self.digitA,
            self.digitB,
            self.digitC,
            batch_size,
        )
        t0_list = [s_pre, s_a, s_b, s_c]

        if variability == "":
            return t0_list

        per_run: list[list[TrialSpec]] = [t0_list]
        c0 = self.digitC
        for t in range(num_experiment_runs - 1):
            # After run 0: A<-C, B<-C+1, C<-C+2 (C mislabeled as B); each further run shifts by +2 on the chain.
            d_a = (c0 + 2 * t) % self.N_DIGITS
            d_b = (c0 + 2 * t + 1) % self.N_DIGITS
            d_c = (c0 + 2 * t + 2) % self.N_DIGITS
            trials = self._digit_trials(
                None, remaining_by_digit, d_a, d_b, d_c, batch_size
            )
            per_run.append(list(trials))
        return per_run
