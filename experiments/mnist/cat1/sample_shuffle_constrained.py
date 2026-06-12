"""Category 1 — *individual samples*: **Shuffled (constrained digit subset)**.

Like `pretrain_then_shuffle_mislabel` with `restrain_digits`: only train images
whose *true* label lies in user-set `S`. Pretrain and post-pretrain pools are
balanced over `S` only; supervision uses a fixed cyclic relabeling on `S`
(`subset_cyclic_mislabel_map`), not the unrestricted global `LABEL_PERM` used
in the other Cat1 experiments. One shuffled trial per run (plus pretrain on run 0).
"""
from typing import Any, ClassVar

import torch
from torch.utils.data import DataLoader, Subset

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader
from experiments.mnist.pretrain_split import split_uniform_k_pretrain_remaining
from experiments.mnist.relabel_ds import PermutedLabelMapDataset
from experiments.mnist.common.label_perm import (
	parse_restrain_digits,
	subset_cyclic_mislabel_map,
)


@register_experiment
class Cat1SampleShuffleConstrained(MNISTWrapper):
	"""**Shuffled (constrained).** Pretrain on `S`, then cyclic mislabel supervision on `S`.

	`restrain_digits` defines `S` (at least two digits). `pretrain_on_k_samples`
	and `num_trial_samples` must divide `len(S)`. Each run after indexing: one trial
	of `num_trial_samples` images drawn `num_trial_samples // len(S)` per digit in
	`S` from the post-pretrain remainder, relabeled by the cyclic map on `S`, then
	batched with `shuffle=True`. Run 0 prepends one balanced pretrain trial on `S`.
	"""

	has_pretrain: ClassVar[bool] = True

	def __init__(
		self,
		*,
		pretrain_on_k_samples: int,
		num_trial_samples: int,
		restrain_digits: Any,
		**kwargs: Any,
	) -> None:
		self._restrain_digits = parse_restrain_digits(restrain_digits)
		if self._restrain_digits is None:
			raise ValueError(
				f"{type(self).__name__} requires non-empty restrain_digits in experiment_config."
			)
		self._active_digit_tuple = self._restrain_digits
		self._n_active = len(self._active_digit_tuple)
		self._trial_label_perm = subset_cyclic_mislabel_map(self._restrain_digits)

		if not isinstance(pretrain_on_k_samples, int):
			raise TypeError(
				f"pretrain_on_k_samples must be int, got {type(pretrain_on_k_samples).__name__}"
			)
		if pretrain_on_k_samples < 1:
			raise ValueError(f"pretrain_on_k_samples must be >= 1, got {pretrain_on_k_samples}")
		if pretrain_on_k_samples % self._n_active != 0:
			raise ValueError(
				f"pretrain_on_k_samples must be divisible by len(restrain_digits)={self._n_active}, "
				f"got {pretrain_on_k_samples}."
			)
		if not isinstance(num_trial_samples, int):
			raise TypeError(
				f"num_trial_samples must be int, got {type(num_trial_samples).__name__}"
			)
		if num_trial_samples < 1:
			raise ValueError(f"num_trial_samples must be >= 1, got {num_trial_samples}")
		if num_trial_samples % self._n_active != 0:
			raise ValueError(
				f"num_trial_samples must be divisible by len(restrain_digits)={self._n_active}, "
				f"got {num_trial_samples}."
			)

		super().__init__(**kwargs)
		self.pretrain_on_k_samples = pretrain_on_k_samples
		self.num_trial_samples = num_trial_samples

	def pretrain_sample_count(self) -> int:
		return self.pretrain_on_k_samples

	@property
	def first_digits(self) -> tuple[int, ...]:
		return self._active_digit_tuple

	def experiment_id(self) -> str:
		ds = "-".join(str(d) for d in self._restrain_digits)
		return (
			f"cat1_sample_shuffle_constrained_digits{ds}_"
			f"tr{self.num_trial_samples}"
		)

	def config_fields(self) -> dict[str, Any]:
		return {
			"pretrain_on_k_samples": self.pretrain_on_k_samples,
			"num_trial_samples": self.num_trial_samples,
			"restrain_digits": ",".join(str(d) for d in self._restrain_digits),
		}

	def _eval_loaders(self) -> dict[str, DataLoader]:
		eval_bs = 256
		all_test_idx: list[int] = []
		for d in self._active_digit_tuple:
			all_test_idx.extend(self._test_by_digit_indices[d])
		test_perm = PermutedLabelMapDataset(
			Subset(self._test_ds, all_test_idx),
			self._trial_label_perm,
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
		per_digit = self.num_trial_samples // self._n_active
		indices_by_digit = digit_indices(self._train_ds, self._active_digit_tuple)
		pretrain_flat, remaining_by_digit = split_uniform_k_pretrain_remaining(
			self.pretrain_on_k_samples,
			indices_by_digit,
			digits=self._active_digit_tuple,
		)
		for d in self._active_digit_tuple:
			rem = remaining_by_digit[d]
			if len(rem) < per_digit * num_experiment_runs:
				raise ValueError(
					f"Not enough remainder indices for digit {d} after pretrain: "
					f"need {per_digit * num_experiment_runs}, got {len(rem)}."
				)

		eval_loaders = self._eval_loaders()
		trial_pre = TrialSpec(
			name="trial_pretrain_pool_balanced_on_subset",
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
			for d in self._active_digit_tuple:
				rem = remaining_by_digit[d]
				lo = run_idx * per_digit
				hi = (run_idx + 1) * per_digit
				flat_indices.extend(rem[lo:hi])
			post_ds = PermutedLabelMapDataset(
				Subset(self._train_ds, flat_indices),
				self._trial_label_perm,
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
				name="trial_shuffle_cyclic_labelperm_pool_on_subset_remainder",
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
