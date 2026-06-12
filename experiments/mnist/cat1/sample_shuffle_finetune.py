"""Category 1 — *individual samples*: **Shuffled (all)** after balanced pretrain.

Same balanced permuted-label pool as :class:`Cat1SampleShuffleControl`, but the model
is first pretrained on `pretrain_on_k_samples` images (balanced across all digits,
`K % 10 == 0`). Run 0 = one pretrain trial + one shuffled trial; later runs are
shuffled trials only (remainder slices, analogous to `pretrain_control_base` /
unconstrained `pretrain_then_shuffle_mislabel`, without that experiment's fixed
three-trial layout).
"""
from typing import Any, ClassVar

import torch
from torch.utils.data import DataLoader, Subset

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader
from experiments.mnist.pretrain_split import split_uniform_k_pretrain_remaining
from experiments.mnist.relabel_ds import PermutedLabelMapDataset
from experiments.mnist.common.label_perm import LABEL_PERM


@register_experiment
class Cat1SampleShuffleFinetune(MNISTWrapper):
	"""**Shuffled (all digits).** Finetune a pretrained network on global `LABEL_PERM` pools.

	After uniform pretrain on all digits, each run uses one trial: build
	`num_trial_samples` points by taking `num_trial_samples // N_DIGITS` remainder
	indices per true digit, relabel with `LABEL_PERM`, merge, and train with a
	shuffled loader. One trial per run after pretrain; run 0 additionally runs the
	pretrain trial once (`once_only`).
	"""

	has_pretrain: ClassVar[bool] = True

	def __init__(
		self,
		*,
		pretrain_on_k_samples: int,
		num_trial_samples: int,
		**kwargs: Any,
	) -> None:
		if not isinstance(pretrain_on_k_samples, int):
			raise TypeError(
				f"pretrain_on_k_samples must be int, got {type(pretrain_on_k_samples).__name__}"
			)
		if pretrain_on_k_samples < 1:
			raise ValueError(f"pretrain_on_k_samples must be >= 1, got {pretrain_on_k_samples}")
		if pretrain_on_k_samples % self.N_DIGITS != 0:
			raise ValueError(
				f"pretrain_on_k_samples must be divisible by N_DIGITS={self.N_DIGITS}, "
				f"got {pretrain_on_k_samples}."
			)
		if not isinstance(num_trial_samples, int):
			raise TypeError(
				f"num_trial_samples must be int, got {type(num_trial_samples).__name__}"
			)
		if num_trial_samples < 1:
			raise ValueError(f"num_trial_samples must be >= 1, got {num_trial_samples}")
		if num_trial_samples % self.N_DIGITS != 0:
			raise ValueError(
				f"num_trial_samples must be divisible by N_DIGITS={self.N_DIGITS}, "
				f"got {num_trial_samples}."
			)
		super().__init__(**kwargs)
		self.pretrain_on_k_samples = pretrain_on_k_samples
		self.num_trial_samples = num_trial_samples

	def pretrain_sample_count(self) -> int:
		return self.pretrain_on_k_samples

	@property
	def first_digits(self) -> tuple[int, ...]:
		return tuple(range(self.N_DIGITS))

	def experiment_id(self) -> str:
		return (
			f"cat1_sample_shuffle_finetune_K{self.pretrain_on_k_samples}_"
			f"tr{self.num_trial_samples}"
		)

	def config_fields(self) -> dict[str, Any]:
		return {
			"pretrain_on_k_samples": self.pretrain_on_k_samples,
			"num_trial_samples": self.num_trial_samples,
		}

	def _eval_loaders(self) -> dict[str, DataLoader]:
		eval_bs = 256
		all_test_idx: list[int] = []
		for d in range(self.N_DIGITS):
			all_test_idx.extend(self._test_by_digit_indices[d])
		test_perm = PermutedLabelMapDataset(
			Subset(self._test_ds, all_test_idx),
			LABEL_PERM,
		)
		return {
			"all_test_perm_supervision": DataLoader(
				test_perm, batch_size=eval_bs, shuffle=False
			),
			"all_test": make_loader(
				self._test_ds, all_test_idx, batch_size=eval_bs, shuffle=False
			),
		}

	def _build_trials(
		self, batch_size: int, seed: int, *, num_experiment_runs: int
	) -> list[list[TrialSpec]]:
		variability = self._experiment_variability.strip().lower()
		if variability != "":
			raise ValueError(
				f"{type(self).__name__} does not support experiment_variability="
				f"{self._experiment_variability!r}. Supported: ''."
			)
		per_digit = self.num_trial_samples // self.N_DIGITS
		indices_by_digit = digit_indices(self._train_ds, tuple(range(self.N_DIGITS)))
		pretrain_flat, remaining_by_digit = split_uniform_k_pretrain_remaining(
			self.pretrain_on_k_samples,
			indices_by_digit,
			self.N_DIGITS,
		)
		for d in range(self.N_DIGITS):
			rem = remaining_by_digit[d]
			if len(rem) < per_digit * num_experiment_runs:
				raise ValueError(
					f"Not enough remainder indices for digit {d} after pretrain: "
					f"need {per_digit * num_experiment_runs}, got {len(rem)}."
				)

		eval_loaders = self._eval_loaders()
		trial_pre = TrialSpec(
			name="trial_pretrain_pool_balanced_all_digits",
			train_loader=make_loader(
				self._train_ds,
				pretrain_flat,
				batch_size=batch_size,
				shuffle=True,
			),
			eval_loaders=eval_loaders,
			once_only=True,
		)

		def permuted_trial(run_idx: int) -> TrialSpec:
			flat_indices: list[int] = []
			for d in range(self.N_DIGITS):
				rem = remaining_by_digit[d]
				lo = run_idx * per_digit
				hi = (run_idx + 1) * per_digit
				flat_indices.extend(rem[lo:hi])
			post_ds = PermutedLabelMapDataset(
				Subset(self._train_ds, flat_indices),
				LABEL_PERM,
			)
			gen = torch.Generator()
			gen.manual_seed(seed + run_idx)
			loader = DataLoader(
				post_ds,
				batch_size=batch_size,
				shuffle=True,
				generator=gen,
			)
			return TrialSpec(
				name="trial_shuffle_labelperm_pool_balanced_remainder",
				train_loader=loader,
				eval_loaders=eval_loaders,
			)

		out: list[list[TrialSpec]] = []
		for run_idx in range(num_experiment_runs):
			if run_idx == 0:
				out.append([trial_pre, permuted_trial(run_idx)])
			else:
				out.append([permuted_trial(run_idx)])
		return out
