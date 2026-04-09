"""Experiment: Train on digitA then digitB.

Uses the official MNIST train/test split:
- Training trials draw from ``train=True`` filtered to the relevant digits.
- Evaluation uses ``train=False`` filtered to the relevant digits.

Trial A: train only on digitA samples; evaluate accuracy on {digitA, digitB}.
Trial B: train only on digitB samples; evaluate accuracy on {digitA, digitB}.

Experiment runs / variability
-----------------------------
- Empty variability (default): flat list returned; same two trials repeat each run.
- ``change_digits``: nested list of length ``num_experiment_runs``; run ``t`` uses
  ``digitA' = (2*digitA + t) % 10`` and ``digitB' = (2*digitB + t) % 10``.
"""

from typing import Any

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader


def _trials_for_digits(
    train_ds,
    test_ds,
    test_by_digit_indices: dict[int, list[int]],
    digitA: int,
    digitB: int,
    batch_size: int,
    *,
    include_all_test: bool = False
) -> list[TrialSpec]:
    """Build [trialA, trialB] for the given digit pair."""
    digits = (digitA, digitB)
    train_by_digit = digit_indices(train_ds, digits)

    both_test = test_by_digit_indices[digitA] + test_by_digit_indices[digitB]
    eval_loaders = {
        "eval_both_digits_test": make_loader(test_ds, both_test, batch_size=256, shuffle=False),
        f"digitA_{digitA}_test": make_loader(
            test_ds, test_by_digit_indices[digitA], batch_size=256, shuffle=False,
        ),
        f"digitB_{digitB}_test": make_loader(
            test_ds, test_by_digit_indices[digitB], batch_size=256, shuffle=False,
        ),
    }
    if include_all_test:
        eval_loaders["all_test"] = make_loader(test_ds, sum(test_by_digit_indices.values(),[]), batch_size=256, shuffle=False)
    trial_a = TrialSpec(
        name=f"trialA_digit{digitA}",
        train_loader=make_loader(train_ds, train_by_digit[digitA], batch_size,shuffle=False),
        eval_loaders=eval_loaders,
    )
    trial_b = TrialSpec(
        name=f"trialB_digit{digitB}",
        train_loader=make_loader(train_ds, train_by_digit[digitB], batch_size,shuffle=False),
        eval_loaders=eval_loaders,
    )
    return [trial_a, trial_b]


@register_experiment
class DigitAThenDigitB_75_25(MNISTWrapper):
    """Two-trial continual learning: digitA then digitB."""

    def experiment_id(self) -> str:
        return f"digit_ordered_{self.digitA}_{self.digitB}_75_25"

    def config_fields(self) -> dict[str, Any]:
        return super().config_fields()

    def _build_trials(
        self, batch_size: int, seed: int, *, num_experiment_runs: int
    ) -> list[TrialSpec] | list[list[TrialSpec]]:
        variability = self._experiment_variability.strip().lower()
        if variability not in ("", "change_digits"):
            raise ValueError(
                f"DigitAThenDigitB_75_25 does not support experiment_variability={self._experiment_variability!r}. "
                "Supported values: '' (none), 'change_digits'."
            )

        if variability == "":
            return _trials_for_digits(
                self._train_ds, self._test_ds, self._test_by_digit_indices, self.digitA, self.digitB, batch_size
            )
        else:
            per_run: list[list[TrialSpec]] = []
            if variability == "change_digits":
                for t in range(num_experiment_runs):
                    dA = (self.digitA + 2*t) % self.N_DIGITS
                    dB = (self.digitB + 2*t) % self.N_DIGITS
                    per_run.append(
                        _trials_for_digits(self._train_ds, self._test_ds, self._test_by_digit_indices, dA, dB, batch_size, include_all_test=True)
                    )
            return per_run
