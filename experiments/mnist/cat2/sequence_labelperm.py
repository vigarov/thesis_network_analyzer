"""Category 2 — *digit sequences*: **Shuffled (sequential)**.

Same single-digit-per-run ordering as :class:`Cat2SequencePretrainControl`, but each
trial's targets are the global `LABEL_PERM` applied to the true digit (Category 1
style permutation, without `restrain_digits`). One trial per run after pretrain.
"""
from typing import Any, ClassVar

from torch.utils.data import DataLoader, Subset

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader
from experiments.mnist.pretrain_split import split_uniform_k_pretrain_remaining
from experiments.mnist.relabel_ds import PermutedLabelMapDataset
from experiments.mnist.common.label_perm import LABEL_PERM


@register_experiment
class Cat2SequenceLabelPerm(MNISTWrapper):
	"""**Shuffled (sequential).** Pretrain, then one digit per run with `LABEL_PERM` targets.

	Mirrors :class:`Cat2SequencePretrainControl` for *which* images are trained each run
	(balanced pretrain on all digits; then `num_trial_samples` per digit from
	remainders in digit order 0,1,…). Supervision is `LABEL_PERM[true_label]` on every
	sample (evaluation uses permuted-label loaders where applicable). No digit-subset
	constraint beyond the per-run single-digit trial construction.
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
			raise ValueError(f"pretrain_on_k_samples invalid: {pretrain_on_k_samples!r}")
		if pretrain_on_k_samples % MNISTWrapper.N_DIGITS != 0:
			raise ValueError(
				f"pretrain_on_k_samples must be divisible by 10, got {pretrain_on_k_samples}"
			)
		if not isinstance(num_trial_samples, int) or num_trial_samples < 1:
			raise ValueError(f"num_trial_samples invalid: {num_trial_samples!r}")
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
					f"{len(remaining[d])} remainder for digit {d}; need {num_trial_samples}."
				)

	def pretrain_sample_count(self) -> int:
		return self.pretrain_on_k_samples

	@property
	def first_digits(self) -> tuple[int, ...]:
		return tuple(range(self.N_DIGITS))

	def experiment_id(self) -> str:
		return (
			f"cat2_sequence_labelperm_K{self.pretrain_on_k_samples}_"
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
		if self._experiment_variability.strip() != "":
			raise ValueError(f"{type(self).__name__} does not support experiment_variability.")
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

		def digit_perm_trial(run_idx: int) -> TrialSpec:
			d = run_idx % self.N_DIGITS
			k = run_idx // self.N_DIGITS
			lo = k * self.num_trial_samples
			hi = (k + 1) * self.num_trial_samples
			rem = remaining[d]
			if len(rem) < hi:
				raise ValueError(
					f"Not enough remainder for digit {d}: need {hi}, got {len(rem)}."
				)
			slice_idx = rem[lo:hi]
			ds = PermutedLabelMapDataset(
				Subset(self._train_ds, slice_idx),
				LABEL_PERM,
			)
			loader = DataLoader(
				ds,
				batch_size=batch_size,
				shuffle=False,
			)
			return TrialSpec(
				name=f"trial_digit_{d}_labelperm_supervision_remainder",
				train_loader=loader,
				eval_loaders=eval_loaders,
			)

		out: list[list[TrialSpec]] = []
		for run_idx in range(num_experiment_runs):
			if run_idx == 0:
				out.append([trial_pre, digit_perm_trial(0)])
			else:
				out.append([digit_perm_trial(run_idx)])
		return out
