"""Experiment: Train on digitA then digitB.

Uses the official MNIST train/test split:
- Training stages draw from ``train=True`` filtered to the relevant digits.
- Evaluation uses ``train=False`` filtered to the relevant digits.

Stage A: train only on digitA samples; evaluate accuracy on {digitA, digitB}.
Stage B: train only on digitB samples; evaluate accuracy on {digitA, digitB}.
"""

from typing import Any

from experiments.base import StageSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader


@register_experiment
class DigitAThenDigitB_75_25(MNISTWrapper):
    """Two-stage continual learning: digitA then digitB."""

    def experiment_id(self) -> str:
        return f"digit_ordered_{self.digitA}_{self.digitB}_75_25"

    def config_fields(self) -> dict[str, Any]:
        return {"digitA": self.digitA, "digitB": self.digitB}

    def build_stages(self, batch_size: int, seed: int) -> list[StageSpec]:
        train_by_digit = digit_indices(self._train_ds, self.digits)
        test_by_digit = digit_indices(self._test_ds, self.digits)

        all_train = train_by_digit[self.digitA] + train_by_digit[self.digitB]
        all_test = test_by_digit[self.digitA] + test_by_digit[self.digitB]

        eval_loaders = {
            "all_train": make_loader(self._train_ds, all_train, batch_size=256, shuffle=False),
            "all_test": make_loader(self._test_ds, all_test, batch_size=256, shuffle=False),
            f"digitA_{self.digitA}_test": make_loader(
                self._test_ds, test_by_digit[self.digitA], batch_size=256, shuffle=False,
            ),
            f"digitB_{self.digitB}_test": make_loader(
                self._test_ds, test_by_digit[self.digitB], batch_size=256, shuffle=False,
            ),
        }

        stage_a = StageSpec(
            name=f"stageA_digit{self.digitA}",
            train_loader=make_loader(self._train_ds, train_by_digit[self.digitA], batch_size),
            eval_loaders=eval_loaders,
        )
        stage_b = StageSpec(
            name=f"stageB_digit{self.digitB}",
            train_loader=make_loader(self._train_ds, train_by_digit[self.digitB], batch_size),
            eval_loaders=eval_loaders,
            is_turning_point=True,
        )
        return [stage_a, stage_b]
