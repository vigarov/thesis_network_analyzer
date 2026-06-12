"""Experiment: Pretrain on all digits (first K total, balanced), then digitA, digitB, digitC with C relabeled as B.

Train pool: official MNIST train minus pinned eval samples (`MNISTWrapper`).

- **Pretrain** (`once_only=True`): first `pretrain_on_k_samples // 10` indices per
  digit for digits 0--9, concatenated, shuffled.
- **digitA / digitB**: first `num_trial_samples` from each digit's remainder after the pretrain prefix.
- **digitC**: same images as digitC remainder, but training targets are `digitB` (mislabeling).

Evaluation loaders use true test labels. Only the digitC *training* trial uses replaced labels.

`experiment_variability`: `""` (flat list) or `change_digits` (run 0 includes pretrain with configured
`digitA` / `digitB` / `digitC`; each later run omits pretrain and sets
`digitA <- (digitC + 2t) % 10`, `digitB <- (digitC + 2t + 1) % 10`,
`digitC <- (digitC + 2t + 2) % 10` (images of the new C, labels of the new B), for
`t = 0 .. num_experiment_runs - 2` in the follow-up runs).
"""

from typing import Any, ClassVar

from torch.utils.data import DataLoader

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader
from experiments.mnist.pretrain_split import split_uniform_k_pretrain_remaining
from experiments.mnist.relabel_ds import RelabelSubset


@register_experiment
class PretrainRelabelC(MNISTWrapper):
	"""Pretrain on all digits, then A, B, C with C trained as label B."""

	has_pretrain: ClassVar[bool] = True

	def __init__(
		self,
		digitA: int = 0,
		digitB: int = 1,
		digitC: int = 2,
		pretrain_on_k_samples: int = 5000,
		num_trial_samples: int = 100,
		**kwargs: Any,
	) -> None:
		for nm, dv in (("digitA", digitA), ("digitB", digitB), ("digitC", digitC)):
			if not 0 <= dv < MNISTWrapper.N_DIGITS:
				raise ValueError(
					f"{nm} must be in [0, {MNISTWrapper.N_DIGITS}), got {dv}"
				)
		if digitA == digitB or digitA == digitC or digitB == digitC:
			raise ValueError(
				f"digitA, digitB, digitC must be pairwise distinct; got "
				f"{digitA}, {digitB}, {digitC}"
			)
		if not isinstance(pretrain_on_k_samples, int):
			raise TypeError(
				f"pretrain_on_k_samples must be int, got {type(pretrain_on_k_samples).__name__}"
			)
		if pretrain_on_k_samples < 1:
			raise ValueError(f"pretrain_on_k_samples must be >= 1, got {pretrain_on_k_samples}")
		if pretrain_on_k_samples % MNISTWrapper.N_DIGITS != 0:
			raise ValueError(
				f"pretrain_on_k_samples must be divisible by N_DIGITS={MNISTWrapper.N_DIGITS}, "
				f"got {pretrain_on_k_samples}."
			)
		if not isinstance(num_trial_samples, int):
			raise TypeError(
				f"num_trial_samples must be int, got {type(num_trial_samples).__name__}"
			)
		if num_trial_samples < 1:
			raise ValueError(f"num_trial_samples must be >= 1, got {num_trial_samples}")
		super().__init__(digitA=digitA, digitB=digitB, **kwargs)
		self.digitC = digitC
		self.pretrain_on_k_samples = pretrain_on_k_samples
		self.num_trial_samples = num_trial_samples

	def pretrain_sample_count(self) -> int:
		return self.pretrain_on_k_samples

	@property
	def first_digits(self) -> tuple[int, ...]:
		return (self.digitA, self.digitB, self.digitC)

	def experiment_id(self) -> str:
		return (
			f"pretrain_relabelC_{self.digitA}_{self.digitB}_{self.digitC}_"
			f"K{self.pretrain_on_k_samples}_tr{self.num_trial_samples}"
		)

	def config_fields(self) -> dict[str, Any]:
		return {
			"digitA": self.digitA,
			"digitB": self.digitB,
			"digitC": self.digitC,
			"pretrain_on_k_samples": self.pretrain_on_k_samples,
			"num_trial_samples": self.num_trial_samples,
		}

	def _eval_loaders_for_digits(
		self,
		digit_a: int,
		digit_b: int,
		digit_c: int,
	) -> dict[str, DataLoader]:
		all_test_idx = sum(self._test_by_digit_indices.values(), [])
		abc_test = (
			self._test_by_digit_indices[digit_a]
			+ self._test_by_digit_indices[digit_b]
			+ self._test_by_digit_indices[digit_c]
		)
		return {
			"all_test": make_loader(
				self._test_ds, all_test_idx, batch_size=256, shuffle=False
			),
			"abc_test": make_loader(self._test_ds, abc_test, batch_size=256, shuffle=False),
			f"digitA_{digit_a}_test": make_loader(
				self._test_ds,
				self._test_by_digit_indices[digit_a],
				batch_size=256,
				shuffle=False,
			),
			f"digitB_{digit_b}_test": make_loader(
				self._test_ds,
				self._test_by_digit_indices[digit_b],
				batch_size=256,
				shuffle=False,
			),
			f"digitC_{digit_c}_test": make_loader(
				self._test_ds,
				self._test_by_digit_indices[digit_c],
				batch_size=256,
				shuffle=False,
			),
		}

	def _digit_trials(
		self,
		pretrain_indices: list[int] | None,
		remaining_by_digit: dict[int, list[int]],
		digit_a: int,
		digit_b: int,
		digit_c: int,
		batch_size: int,
	) -> tuple[TrialSpec, ...]:
		eval_loaders = self._eval_loaders_for_digits(digit_a, digit_b, digit_c)

		for d, name in (
			(digit_a, "digitA"),
			(digit_b, "digitB"),
			(digit_c, "digitC"),
		):
			rem = remaining_by_digit[d]
			if len(rem) < self.num_trial_samples:
				raise ValueError(
					f"Not enough remaining train indices for {name} (digit {d}): "
					f"need {self.num_trial_samples}, got {len(rem)}"
				)

		idx_a = remaining_by_digit[digit_a][: self.num_trial_samples]
		idx_b = remaining_by_digit[digit_b][: self.num_trial_samples]
		idx_c = remaining_by_digit[digit_c][: self.num_trial_samples]

		trial_a = TrialSpec(
			name=f"trial_digit{digit_a}",
			train_loader=make_loader(
				self._train_ds, idx_a, batch_size, shuffle=False
			),
			eval_loaders=eval_loaders,
		)
		trial_b = TrialSpec(
			name=f"trial_digit{digit_b}",
			train_loader=make_loader(
				self._train_ds, idx_b, batch_size, shuffle=False
			),
			eval_loaders=eval_loaders,
		)
		relabel_ds = RelabelSubset(self._train_ds, idx_c, digit_b)
		train_c = DataLoader(
			relabel_ds,
			batch_size=batch_size,
			shuffle=False,
		)
		trial_c = TrialSpec(
			name=f"trial_digit{digit_c}_as_label{digit_b}",
			train_loader=train_c,
			eval_loaders=eval_loaders,
		)

		if pretrain_indices is not None:
			trial_pre = TrialSpec(
				name="trial_pretrain_all_digits",
				train_loader=make_loader(
					self._train_ds,
					pretrain_indices,
					batch_size=batch_size,
					shuffle=True,
				),
				eval_loaders=eval_loaders,
				once_only=True,
			)
			return trial_pre, trial_a, trial_b, trial_c
		return trial_a, trial_b, trial_c

	def _build_trials(
		self, batch_size: int, seed: int, *, num_experiment_runs: int
	) -> list[TrialSpec] | list[list[TrialSpec]]:
		variability = self._experiment_variability.strip().lower()
		if variability not in ("", "change_digits"):
			raise ValueError(
				f"PretrainRelabelC does not support experiment_variability="
				f"{self._experiment_variability!r}. Supported: '', 'change_digits'."
			)

		indices_by_digit = digit_indices(
			self._train_ds, tuple(range(self.N_DIGITS))
		)
		pretrain_flat, remaining_by_digit = split_uniform_k_pretrain_remaining(
			self.pretrain_on_k_samples, indices_by_digit, self.N_DIGITS
		)

		s_pre, s_a, s_b, s_c = self._digit_trials(
			pretrain_flat,
			remaining_by_digit,
			self.digitA,
			self.digitB,
			self.digitC,
			batch_size,
		)
		t0_list = [s_pre, s_a, s_b, s_c]

		if variability == "":
			return t0_list

		per_run: list[list[TrialSpec]] = [t0_list]
		c0 = self.digitC
		for t in range(num_experiment_runs - 1):
			d_a = (c0 + 2 * t) % self.N_DIGITS
			d_b = (c0 + 2 * t + 1) % self.N_DIGITS
			d_c = (c0 + 2 * t + 2) % self.N_DIGITS
			trials = self._digit_trials(
				None, remaining_by_digit, d_a, d_b, d_c, batch_size
			)
			per_run.append(list(trials))
		return per_run
