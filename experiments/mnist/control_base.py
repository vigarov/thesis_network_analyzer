"""Experiment: single pooled trial on the post-pretrain remainder only (shuffled, correct labels).

Uses the same `digit_indices` + `split_uniform_k_pretrain_remaining` split as other
pretrain MNIST experiments with the same `pretrain_on_k_samples`, but **never** trains on
the K-sample pretrain pool — only the concatenated remainder (digits 0..9), `shuffle=True`.

`num_trial_samples` and other kwargs may be injected by the runner; they are ignored.
"""

from typing import Any

from torch.utils.data import DataLoader

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader
from experiments.mnist.pretrain_split import split_uniform_k_pretrain_remaining


def _flatten_remainder(remainder_by_digit: dict[int, list[int]], n_digits: int) -> list[int]:
	out: list[int] = []
	for d in range(n_digits):
		out.extend(remainder_by_digit[d])
	return out


@register_experiment
class ControlBase(MNISTWrapper):
	"""Shuffled pooled remainder only; same K/remainder partition as pretrain controls."""

	def __init__(
		self,
		digitA: int = 0,
		*,
		pretrain_on_k_samples: int,
		**kwargs: Any,
	) -> None:
		if not isinstance(pretrain_on_k_samples, int):
			raise TypeError(
				f"pretrain_on_k_samples must be int, got {type(pretrain_on_k_samples).__name__}"
			)

		mnist_kw = dict(kwargs)
		mnist_kw.pop("digitB", None)
		mnist_kw.pop("digitA", None)
		mnist_kw.pop("samples_per_digit", None)
		mnist_kw.pop("num_trial_samples", None)
		mnist_kw.pop("pretrain_on_k_samples", None)
		super().__init__(digitA=digitA, digitB=digitA, **mnist_kw)

		if pretrain_on_k_samples < 1:
			raise ValueError(f"pretrain_on_k_samples must be >= 1, got {pretrain_on_k_samples}")
		if pretrain_on_k_samples % self.N_DIGITS != 0:
			raise ValueError(
				f"pretrain_on_k_samples must be divisible by N_DIGITS={self.N_DIGITS}, "
				f"got {pretrain_on_k_samples}."
			)
		self.pretrain_on_k_samples = pretrain_on_k_samples

	@property
	def first_digits(self) -> tuple[int, ...]:
		return tuple(range(self.N_DIGITS))

	def experiment_id(self) -> str:
		return f"control_base_K{self.pretrain_on_k_samples}_remainder_shuffle"

	def config_fields(self) -> dict[str, Any]:
		return {
			"digitA": self.digitA,
			"pretrain_on_k_samples": self.pretrain_on_k_samples,
		}

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
	) -> list[TrialSpec] | list[list[TrialSpec]]:
		variability = self._experiment_variability.strip().lower()
		if variability != "":
			raise ValueError(
				f"ControlBase does not support experiment_variability={self._experiment_variability!r}. "
				"Supported: ''."
			)

		indices_by_digit = digit_indices(self._train_ds, tuple(range(self.N_DIGITS)))
		_, remainder_by_digit = split_uniform_k_pretrain_remaining(
			self.pretrain_on_k_samples, indices_by_digit, self.N_DIGITS
		)
		remainder_flat = _flatten_remainder(remainder_by_digit, self.N_DIGITS)
		if not remainder_flat:
			raise ValueError("Remainder pool is empty after pretrain split.")

		eval_loaders = self._eval_loaders()
		return [
			TrialSpec(
				name="trial_remaindershuffle",
				train_loader=make_loader(
					self._train_ds, remainder_flat, batch_size=batch_size, shuffle=True
				),
				eval_loaders=eval_loaders,
			),
		]
