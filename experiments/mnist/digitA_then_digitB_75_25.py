"""Experiment: Train on digitA then digitB.

Uses the official MNIST train/test split:
- Training stages draw from ``train=True`` filtered to the relevant digits.
- Evaluation uses ``train=False`` filtered to the relevant digits.

Stage A: train only on digitA samples; evaluate accuracy on {digitA, digitB}.
Stage B: train only on digitB samples; evaluate accuracy on {digitA, digitB}.

Trials / variability
--------------------
- Empty variability (default): flat list returned; same two stages repeat each trial.
- ``change_digits``: nested list of length ``num_trials``; trial ``t`` uses
  ``digitA' = (2*digitA + t) % 10`` and ``digitB' = (2*digitB + t) % 10``.
"""

from typing import Any

from experiments.base import StageSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader


def _stages_for_digits(
    train_ds,
    test_ds,
    test_by_digit_indices: dict[int, list[int]],
    digitA: int,
    digitB: int,
    batch_size: int,
    *,
    include_all_test: bool = False
) -> list[StageSpec]:
    """Build [stageA, stageB] for the given digit pair."""
    digits = (digitA, digitB)
    train_by_digit = digit_indices(train_ds, digits)

    both_test = test_by_digit_indices[digitA] + test_by_digit_indices[digitB]
    eval_loaders = {
        "trial_both_test": make_loader(test_ds, both_test, batch_size=256, shuffle=False),
        f"digitA_{digitA}_test": make_loader(
            test_ds, test_by_digit_indices[digitA], batch_size=256, shuffle=False,
        ),
        f"digitB_{digitB}_test": make_loader(
            test_ds, test_by_digit_indices[digitB], batch_size=256, shuffle=False,
        ),
    }
    if include_all_test:
        eval_loaders["all_test"] = make_loader(test_ds, sum(test_by_digit_indices.values(),[]), batch_size=256, shuffle=False)
    stage_a = StageSpec(
        name=f"stageA_digit{digitA}",
        train_loader=make_loader(train_ds, train_by_digit[digitA], batch_size),
        eval_loaders=eval_loaders,
    )
    stage_b = StageSpec(
        name=f"stageB_digit{digitB}",
        train_loader=make_loader(train_ds, train_by_digit[digitB], batch_size),
        eval_loaders=eval_loaders,
    )
    return [stage_a, stage_b]


@register_experiment
class DigitAThenDigitB_75_25(MNISTWrapper):
    """Two-stage continual learning: digitA then digitB."""

    def experiment_id(self) -> str:
        return f"digit_ordered_{self.digitA}_{self.digitB}_75_25"

    def config_fields(self) -> dict[str, Any]:
        return {"digitA": self.digitA, "digitB": self.digitB}

    def _build_stages(
        self, batch_size: int, seed: int, *, num_trials: int
    ) -> list[StageSpec] | list[list[StageSpec]]:
        variability = self._trial_variability.strip().lower()
        if variability not in ("", "change_digits"):
            raise ValueError(
                f"DigitAThenDigitB_75_25 does not support trial_variability={self._trial_variability!r}. "
                "Supported values: '' (none), 'change_digits'."
            )

        if variability == "":
            return _stages_for_digits(
                self._train_ds, self._test_ds, self._test_by_digit_indices, self.digitA, self.digitB, batch_size
            )
        else:
            per_trial: list[list[StageSpec]] = []
            if variability == "change_digits":
                for t in range(num_trials):
                    dA = (2*self.digitA + t) % self.N_DIGITS
                    dB = (2*self.digitB + t) % self.N_DIGITS
                    per_trial.append(
                        _stages_for_digits(self._train_ds, self._test_ds, self._test_by_digit_indices, dA, dB, batch_size, include_all_test=True)
                    )
            return per_trial
