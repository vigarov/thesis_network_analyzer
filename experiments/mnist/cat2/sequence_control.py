"""Category 2 — digit sequences: Control (no pretrain).

One trial per run; each trial trains `num_trial_samples` images of a single true
digit with correct labels. Runs advance 0,1,…,9,0,… by slice index (same spirit as
`digitA_then_digitB_75_25` but one digit per run instead of multi-trial blocks).
"""
from typing import Any, ClassVar

from torch.utils.data import DataLoader

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader


@register_experiment
class Cat2SequenceControl(MNISTWrapper):
    """Control (no pretrain). Sequential single-digit trials with true labels only.

    Run index `r` trains digit `d = r % N_DIGITS` using the `(r // N_DIGITS)`-th
    block of `num_trial_samples` consecutive train indices for that digit (no
    shuffling within the trial). The sequence always starts from digit 0 for
    `r = 0`; the user does not pick the starting digit.
    """

    has_pretrain: ClassVar[bool] = False

    def __init__(
        self,
        *,
        num_trial_samples: int,
        **kwargs: Any,
    ) -> None:
        if not isinstance(num_trial_samples, int) or num_trial_samples < 1:
            raise ValueError(f"num_trial_samples must be int >= 1, got {num_trial_samples!r}")
        super().__init__(**kwargs)
        self.num_trial_samples = num_trial_samples

    @property
    def first_digits(self) -> tuple[int, ...]:
        return tuple(range(self.N_DIGITS))

    def experiment_id(self) -> str:
        return f"cat2_sequence_control_tr{self.num_trial_samples}"

    def config_fields(self) -> dict[str, Any]:
        return {"num_trial_samples": self.num_trial_samples}

    def _eval_loaders(self) -> dict[str, DataLoader]:
        return {
            "all_test": make_loader(
                self._test_ds,
                sum(self._test_by_digit_indices.values(), []),
                batch_size=256,
                shuffle=False,
            ),
        }

    def _build_trials(
        self, batch_size: int, seed: int, *, num_experiment_runs: int
    ) -> list[list[TrialSpec]]:
        if self._experiment_variability.strip() != "":
            raise ValueError(
                f"{type(self).__name__} does not support experiment_variability."
            )
        train_by = digit_indices(self._train_ds, tuple(range(self.N_DIGITS)))
        eval_loaders = self._eval_loaders()
        out: list[list[TrialSpec]] = []
        for run_idx in range(num_experiment_runs):
            d = run_idx % self.N_DIGITS
            k = run_idx // self.N_DIGITS
            lo = k * self.num_trial_samples
            hi = (k + 1) * self.num_trial_samples
            idcs = train_by[d]
            if len(idcs) < hi:
                raise ValueError(
                    f"Not enough train samples for digit {d}: need {hi}, got {len(idcs)} "
                    f"(run_idx={run_idx})."
                )
            slice_idx = idcs[lo:hi]
            name = f"trial_digit_{d}_train_correct_images"
            out.append(
                [
                    TrialSpec(
                        name=name,
                        train_loader=make_loader(
                            self._train_ds, slice_idx, batch_size, shuffle=False
                        ),
                        eval_loaders=eval_loaders,
                    ),
                ]
            )
        return out
