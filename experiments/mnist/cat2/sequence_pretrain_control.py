"""Category 2 — *digit sequences*: **Control (pretrain)**.

Balanced pretrain on all digits, then the same per-run single-digit schedule as
:class:`Cat2SequenceControl`, but each trial draws its `num_trial_samples` from the
*post-pretrain remainder* per digit (same idea as `pretrain_base_then_a_25_75_inside_train`,
without extra intra-run trials).
"""
from typing import Any, ClassVar

from torch.utils.data import DataLoader

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader
from experiments.mnist.pretrain_split import split_uniform_k_pretrain_remaining


@register_experiment
class Cat2SequencePretrainControl(MNISTWrapper):
	"""**Control (pretrain).** Correct labels; one digit per trial on remainder slices.

	Run `0` runs one balanced pretrain trial (`pretrain_on_k_samples`, `K % 10 == 0`)
	then one trial on digit `0` from remainders. Later runs omit pretrain and cycle
	digits `0..9` on successive remainder blocks of size `num_trial_samples` each.
	"""

	has_pretrain: ClassVar[bool] = True

	def __init__(
		self,
		*,
		pretrain_on_k_samples: int,
		num_trial_samples: int,
		**kwargs: Any,
	) -> None:
		if not isinstance(pretrain_on_k_samples, int) or pretrain_on_k_samples < 1:
			raise ValueError(f"pretrain_on_k_samples must be int >= 1, got {pretrain_on_k_samples!r}")
		if pretrain_on_k_samples % MNISTWrapper.N_DIGITS != 0:
			raise ValueError(
				f"pretrain_on_k_samples must be divisible by {MNISTWrapper.N_DIGITS}, got {pretrain_on_k_samples}"
			)
		if not isinstance(num_trial_samples, int) or num_trial_samples < 1:
			raise ValueError(f"num_trial_samples must be int >= 1, got {num_trial_samples!r}")
		super().__init__(**kwargs)
		self.pretrain_on_k_samples = pretrain_on_k_samples
		self.num_trial_samples = num_trial_samples

		idx_by_digit = digit_indices(self._train_ds, tuple(range(self.N_DIGITS)))
		_, remaining = split_uniform_k_pretrain_remaining(
			pretrain_on_k_samples, idx_by_digit, self.N_DIGITS
		)
		for d in range(self.N_DIGITS):
			if len(remaining[d]) < num_trial_samples:
				raise ValueError(
					f"pretrain_on_k_samples={pretrain_on_k_samples} leaves only "
					f"{len(remaining[d])} remainder samples for digit {d}; "
					f"need at least {num_trial_samples}."
				)

	def pretrain_sample_count(self) -> int:
		return self.pretrain_on_k_samples

	@property
	def first_digits(self) -> tuple[int, ...]:
		return tuple(range(self.N_DIGITS))

	def experiment_id(self) -> str:
		return (
			f"cat2_sequence_pretrain_control_K{self.pretrain_on_k_samples}_"
			f"tr{self.num_trial_samples}"
		)

	def config_fields(self) -> dict[str, Any]:
		return {
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
	) -> list[list[TrialSpec]]:
		if self._experiment_variability.strip() != "":
			raise ValueError(
				f"{type(self).__name__} does not support experiment_variability."
			)
		indices_by_digit = digit_indices(self._train_ds, tuple(range(self.N_DIGITS)))
		pretrain_flat, remaining = split_uniform_k_pretrain_remaining(
			self.pretrain_on_k_samples,
			indices_by_digit,
			self.N_DIGITS,
		)
		eval_loaders = self._eval_loaders()
		trial_pre = TrialSpec(
			name="trial_pretrain_pool_balanced_all_digits",
			train_loader=make_loader(
				self._train_ds, pretrain_flat, batch_size, shuffle=True
			),
			eval_loaders=eval_loaders,
			once_only=True,
		)

		def digit_trial(run_idx: int) -> TrialSpec:
			d = run_idx % self.N_DIGITS
			k = run_idx // self.N_DIGITS
			lo = k * self.num_trial_samples
			hi = (k + 1) * self.num_trial_samples
			rem = remaining[d]
			if len(rem) < hi:
				raise ValueError(
					f"Not enough remainder for digit {d}: need {hi}, got {len(rem)} "
					f"(run_idx={run_idx})."
				)
			slice_idx = rem[lo:hi]
			return TrialSpec(
				name=f"trial_digit_{d}_train_correct_images_remainder",
				train_loader=make_loader(
					self._train_ds, slice_idx, batch_size, shuffle=False
				),
				eval_loaders=eval_loaders,
			)

		out: list[list[TrialSpec]] = []
		for run_idx in range(num_experiment_runs):
			if run_idx == 0:
				out.append([trial_pre, digit_trial(0)])
			else:
				out.append([digit_trial(run_idx)])
		return out
