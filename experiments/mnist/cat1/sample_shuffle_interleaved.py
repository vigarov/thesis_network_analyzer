"""Category 1 — *individual samples*: **Interleaved restrained / correct**.

Pretrain on correctly labeled `restrain_digits`, then train with a fixed
interleaved sequence: one `LABEL_PERM`-mislabeled restrained sample, then one
correctly labeled sample from a random digit outside the restrained set, cycling
through restrained digits in sorted order.
"""
from typing import Any, ClassVar

import torch
from torch.utils.data import DataLoader, Subset

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader
from experiments.mnist.pretrain_split import split_uniform_k_pretrain_remaining
from experiments.mnist.relabel_ds import ExplicitLabelIndexDataset, PermutedLabelMapDataset
from experiments.mnist.common.label_perm import LABEL_PERM, parse_restrain_digits


def _build_interleaved_entries(
	*,
	restrain_digits: tuple[int, ...],
	remaining_by_digit: dict[int, list[int]],
	outside_by_digit: dict[int, list[int]],
	num_trial_samples: int,
	run_idx: int,
	seed: int,
) -> list[tuple[int, int]]:
	"""Return ordered (train_index, supervision_label) pairs for one interleaved trial."""
	n_active = len(restrain_digits)
	n_restrained = num_trial_samples // 2
	per_digit = n_restrained // n_active
	lo = run_idx * per_digit
	hi = (run_idx + 1) * per_digit

	restrained_queues = {
		d: list(remaining_by_digit[d][lo:hi]) for d in restrain_digits
	}
	outside_digits = tuple(d for d in range(MNISTWrapper.N_DIGITS) if d not in restrain_digits)
	outside_pools = {d: list(outside_by_digit[d]) for d in outside_digits}
	gen = torch.Generator()
	gen.manual_seed(seed + run_idx)
	n_outside = len(outside_digits)

	entries: list[tuple[int, int]] = []
	rd_pos = 0
	for _ in range(n_restrained):
		d = restrain_digits[rd_pos % n_active]
		idx = restrained_queues[d].pop(0)
		entries.append((idx, LABEL_PERM[d]))
		rd_pos += 1

		choice_idx = int(torch.randint(n_outside, (1,), generator=gen).item())
		od = outside_digits[choice_idx]
		oidx = outside_pools[od].pop(0)
		entries.append((oidx, od))

	return entries


@register_experiment
class Cat1SampleShuffleInterleaved(MNISTWrapper):
	"""**Interleaved.** Pretrain on `S`, then alternate mislabeled `S` and correct outside digits.

	`restrain_digits` defines `S` (at least two digits). Each run uses one trial of
	`num_trial_samples` steps (half mislabeled restrained, half correct outside),
	presented in fixed order with `shuffle=False`. Run 0 prepends one balanced
	pretrain trial on `S` with correct labels.
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
		self._cycle_len = 2 * self._n_active

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
		if num_trial_samples % 2 != 0:
			raise ValueError(
				f"num_trial_samples must be even (half restrained, half correct outside), "
				f"got {num_trial_samples}."
			)
		if num_trial_samples % self._cycle_len != 0:
			raise ValueError(
				f"num_trial_samples must be divisible by 2 * len(restrain_digits)="
				f"{self._cycle_len}, got {num_trial_samples}."
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
		ds = "-".join(str(d) for d in self._active_digit_tuple)
		return (
			f"cat1_sample_shuffle_interleaved_digits{ds}_"
			f"tr{self.num_trial_samples}"
		)

	def config_fields(self) -> dict[str, Any]:
		return {
			"pretrain_on_k_samples": self.pretrain_on_k_samples,
			"num_trial_samples": self.num_trial_samples,
			"restrain_digits": ",".join(str(d) for d in self._active_digit_tuple),
		}

	def _eval_loaders(self) -> dict[str, DataLoader]:
		eval_bs = 256
		all_test_idx: list[int] = []
		for d in self._active_digit_tuple:
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
		per_digit = (self.num_trial_samples // 2) // self._n_active
		n_correct = self.num_trial_samples // 2
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

		outside_digits = tuple(
			d for d in range(self.N_DIGITS) if d not in self._active_digit_tuple
		)
		outside_by_digit = digit_indices(self._train_ds, outside_digits)
		need_outside = n_correct * num_experiment_runs
		for d in outside_digits:
			pool = outside_by_digit[d]
			if len(pool) < need_outside:
				raise ValueError(
					f"Not enough train indices for outside digit {d}: "
					f"need {need_outside}, got {len(pool)}."
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

		def interleaved_trial(run_idx: int) -> TrialSpec:
			entries = _build_interleaved_entries(
				restrain_digits=self._active_digit_tuple,
				remaining_by_digit=remaining_by_digit,
				outside_by_digit={
					d: outside_by_digit[d][
						run_idx * n_correct : (run_idx + 1) * n_correct
					]
					for d in outside_digits
				},
				num_trial_samples=self.num_trial_samples,
				run_idx=run_idx,
				seed=seed,
			)
			post_ds = ExplicitLabelIndexDataset(self._train_ds, entries)
			loader = DataLoader(
				post_ds,
				batch_size=batch_size,
				shuffle=False,
			)
			return TrialSpec(
				name="trial_interleaved_labelperm_restrained_correct_outside",
				train_loader=loader,
				eval_loaders=eval_loaders,
			)

		out: list[list[TrialSpec]] = []
		for run_idx in range(num_experiment_runs):
			if run_idx == 0:
				out.append([trial_pre, interleaved_trial(run_idx)])
			else:
				out.append([interleaved_trial(run_idx)])
		return out
