"""Category 2 — *digit sequences*: **Reinforce** (three trials per run).

After balanced pretrain: (1) digit `A` correct, (2) images of `A+1` supervised as
`A`, (3) the same `A+1` images correctly labeled `A+1`. Anchor `A` steps by +2
mod 10 each run. Same sample budget story as :class:`Cat2SequenceRecover`; analogous to
the relabel/switch-digit idea in `old_relabel_experiment` but with this fixed
three-trial pattern per run.
"""
from typing import Any, ClassVar

from torch.utils.data import DataLoader

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader
from experiments.mnist.pretrain_split import split_uniform_k_pretrain_remaining
from experiments.mnist.relabel_ds import RelabelSubset


@register_experiment
class Cat2SequenceReinforce(MNISTWrapper):
	"""**Reinforce.** Pretrain then `A` correct → `A+1` as `A` → `A+1` correct.

	`digitA` fixes the run-0 anchor `A`; run `r` uses `(digitA + 2*r) % N_DIGITS`.
	Trial 2 reuses the image indices from trial 1 (digit `A+1`) with targets `A`;
	trial 3 retrains those images with true label `A+1`.
	"""

	has_pretrain: ClassVar[bool] = True

	def __init__(
		self,
		digitA: int = 0,
		*,
		pretrain_on_k_samples: int,
		num_trial_samples: int,
		**kwargs: Any,
	) -> None:
		if not 0 <= digitA < MNISTWrapper.N_DIGITS:
			raise ValueError(f"digitA must be in [0, 10), got {digitA}")
		if not isinstance(pretrain_on_k_samples, int) or pretrain_on_k_samples < 1:
			raise ValueError(f"pretrain_on_k_samples invalid: {pretrain_on_k_samples!r}")
		if pretrain_on_k_samples % MNISTWrapper.N_DIGITS != 0:
			raise ValueError(
				f"pretrain_on_k_samples must be divisible by 10, got {pretrain_on_k_samples}"
			)
		if not isinstance(num_trial_samples, int) or num_trial_samples < 1:
			raise ValueError(f"num_trial_samples invalid: {num_trial_samples!r}")
		kwargs.pop("digitB", None)
		digit_b = (digitA + 1) % MNISTWrapper.N_DIGITS
		super().__init__(digitA=digitA, digitB=digit_b, **kwargs)
		self.pretrain_on_k_samples = pretrain_on_k_samples
		self.num_trial_samples = num_trial_samples

	def pretrain_sample_count(self) -> int:
		return self.pretrain_on_k_samples

	@property
	def first_digits(self) -> tuple[int, ...]:
		return (self.digitA, (self.digitA + 1) % self.N_DIGITS)

	def experiment_id(self) -> str:
		return (
			f"cat2_sequence_reinforce_K{self.pretrain_on_k_samples}_"
			f"tr{self.num_trial_samples}_a{self.digitA}"
		)

	def config_fields(self) -> dict[str, Any]:
		return {
			"digitA": self.digitA,
			"digitB": self.digitB,
			"pretrain_on_k_samples": self.pretrain_on_k_samples,
			"num_trial_samples": self.num_trial_samples,
		}

	def _eval_loaders_for_pair(self, digit_a: int, digit_b: int) -> dict[str, DataLoader]:
		all_test_idx = sum(self._test_by_digit_indices.values(), [])
		ab_test = (
			self._test_by_digit_indices[digit_a] + self._test_by_digit_indices[digit_b]
		)
		return {
			"all_test": make_loader(
				self._test_ds, all_test_idx, batch_size=256, shuffle=False
			),
			"ab_test": make_loader(self._test_ds, ab_test, batch_size=256, shuffle=False),
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
		}

	def _triples(
		self,
		pretrain_indices: list[int] | None,
		remaining_by_digit: dict[int, list[int]],
		anchor_a: int,
		batch_size: int,
	) -> tuple[TrialSpec, ...]:
		b = (anchor_a + 1) % self.N_DIGITS
		eval_loaders = self._eval_loaders_for_pair(anchor_a, b)
		for d, nm in ((anchor_a, "a"), (b, "b")):
			rem = remaining_by_digit[d]
			if len(rem) < self.num_trial_samples:
				raise ValueError(
					f"Not enough remainder for digit {d} ({nm}): "
					f"need {self.num_trial_samples}, got {len(rem)}"
				)
		idx_a = remaining_by_digit[anchor_a][: self.num_trial_samples]
		idx_b = remaining_by_digit[b][: self.num_trial_samples]
		t_a = TrialSpec(
			name="trial_reinforce_anchor_correct",
			train_loader=make_loader(
				self._train_ds, idx_a, batch_size, shuffle=False
			),
			eval_loaders=eval_loaders,
		)
		mis_loader = DataLoader(
			RelabelSubset(self._train_ds, idx_b, anchor_a),
			batch_size=batch_size,
			shuffle=False,
		)
		t_y = TrialSpec(
			name="trial_reinforce_next_images_labeled_anchor",
			train_loader=mis_loader,
			eval_loaders=eval_loaders,
		)
		t_b = TrialSpec(
			name="trial_reinforce_next_correct",
			train_loader=make_loader(
				self._train_ds, idx_b, batch_size, shuffle=False
			),
			eval_loaders=eval_loaders,
		)
		if pretrain_indices is not None:
			t_p = TrialSpec(
				name="trial_pretrain_pool_balanced_all_digits",
				train_loader=make_loader(
					self._train_ds,
					pretrain_indices,
					batch_size=batch_size,
					shuffle=True,
				),
				eval_loaders=eval_loaders,
				once_only=True,
			)
			return t_p, t_a, t_y, t_b
		return t_a, t_y, t_b

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
		out: list[list[TrialSpec]] = []
		for r in range(num_experiment_runs):
			anchor = (self.digitA + 2 * r) % self.N_DIGITS
			if r == 0:
				out.append(list(self._triples(pretrain_flat, remaining, anchor, batch_size)))
			else:
				out.append(list(self._triples(None, remaining, anchor, batch_size)))
		return out
