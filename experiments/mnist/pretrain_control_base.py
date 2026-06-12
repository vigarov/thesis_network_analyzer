"""Experiment: balanced-K pretrain pool then pooled remainder, both shuffled (correct labels).

Uses the same `digit_indices` + `split_uniform_k_pretrain_remaining` partition as other
MNIST pretrain experiments, but only to split the train set into pretrain vs remainder.
Both phases train on the **concatenated** index list with `shuffle=True` (no per-digit trials).

- **Trial 1:** `pretrain_flat`, `once_only=True`.
- **Trial 2:** all remainder indices (digits 0..9 concatenated), `once_only=False`.

`num_trial_samples` is accepted for `config_guard` / JSON parity with sibling configs and
is not used to build loaders.
"""

from typing import Any, ClassVar

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
class PretrainControlBase(MNISTWrapper):
	"""Balanced-K pretrain (shuffled) then shuffled pooled remainder; correct labels throughout."""

	has_pretrain: ClassVar[bool] = True

	def __init__(
		self,
		digitA: int = 0,
		*,
		pretrain_on_k_samples: int,
		num_trial_samples: int,
		**kwargs: Any,
	) -> None:
		if not isinstance(pretrain_on_k_samples, int):
			raise TypeError(
				f"pretrain_on_k_samples must be int, got {type(pretrain_on_k_samples).__name__}"
			)
		if not isinstance(num_trial_samples, int):
			raise TypeError(
				f"num_trial_samples must be int, got {type(num_trial_samples).__name__}"
			)
		if num_trial_samples < 1:
			raise ValueError(f"num_trial_samples must be >= 1, got {num_trial_samples}")

		mnist_kw = dict(kwargs)
		mnist_kw.pop("digitB", None)
		mnist_kw.pop("digitA", None)
		mnist_kw.pop("samples_per_digit", None)
		super().__init__(digitA=digitA, digitB=digitA, **mnist_kw)

		if pretrain_on_k_samples < 1:
			raise ValueError(f"pretrain_on_k_samples must be >= 1, got {pretrain_on_k_samples}")
		if pretrain_on_k_samples % self.N_DIGITS != 0:
			raise ValueError(
				f"pretrain_on_k_samples must be divisible by N_DIGITS={self.N_DIGITS}, "
				f"got {pretrain_on_k_samples}."
			)
		self.pretrain_on_k_samples = pretrain_on_k_samples
		self.num_trial_samples = num_trial_samples

	@property
	def first_digits(self) -> tuple[int, ...]:
		return tuple(range(self.N_DIGITS))

	def pretrain_sample_count(self) -> int:
		return self.pretrain_on_k_samples

	def experiment_id(self) -> str:
		return f"pretrain_control_base_K{self.pretrain_on_k_samples}_shuffle_remainder"

	def config_fields(self) -> dict[str, Any]:
		return {
			"digitA": self.digitA,
			"pretrain_on_k_samples": self.pretrain_on_k_samples,
			"num_trial_samples": self.num_trial_samples,
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
				f"PretrainControlBase does not support experiment_variability={self._experiment_variability!r}. "
				"Supported: ''."
			)

		indices_by_digit = digit_indices(self._train_ds, tuple(range(self.N_DIGITS)))
		pretrain_flat, remainder_by_digit = split_uniform_k_pretrain_remaining(
			self.pretrain_on_k_samples, indices_by_digit, self.N_DIGITS
		)
		remainder_flat = _flatten_remainder(remainder_by_digit, self.N_DIGITS)
		if not remainder_flat:
			raise ValueError("Remainder pool is empty after pretrain split.")

		eval_loaders = self._eval_loaders()
		trial_pre = TrialSpec(
			name="trial_base_pretrain",
			train_loader=make_loader(
				self._train_ds, pretrain_flat, batch_size=batch_size, shuffle=True
			),
			eval_loaders=eval_loaders,
			once_only=True,
		)
		trial_post = TrialSpec(
			name="trial_remainder_shuffled",
			train_loader=make_loader(
				self._train_ds, remainder_flat, batch_size=batch_size, shuffle=True
			),
			eval_loaders=eval_loaders,
		)
		return [trial_pre, trial_post]
