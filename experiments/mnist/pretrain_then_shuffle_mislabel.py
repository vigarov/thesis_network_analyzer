"""Experiment: balanced pretrain on all digits (or a ``restrain_digits`` subset), then one trial on a permuted-label pool.

After pretrain, further ``TrialSpec`` blocks train on balanced ``num_trial_samples`` pools
drawn from the remainder (``num_trial_samples // 10`` per true digit when using all
classes, or ``num_trial_samples // |S|`` per digit when ``restrain_digits`` restricts to digit set *S*).
Supervision uses a fixed permutation of class indices: the global ``LABEL_PERM`` on
all ten digits, or a swap / cyclic shift on *S* only when ``restrain_digits`` is set.
Training steps use ``batch_size=1`` and ``shuffle=True`` on that trial's loader.

Pinned eval (activation capture) and ``all_test_perm_supervision`` use the same
permutation via :meth:`PretrainThenShuffleMislabel.eval_supervision_label`.
"""

from typing import Any, ClassVar

import torch
from torch.utils.data import DataLoader, Subset

from experiments.base import TrialSpec, register_experiment
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader
from experiments.mnist.pretrain_split import split_uniform_k_pretrain_remaining
from experiments.mnist.relabel_ds import PermutedLabelMapDataset


def parse_restrain_digits(value: Any) -> tuple[int, ...] | None:
	"""Parse optional ``restrain_digits`` from JSON/CLI into sorted unique MNIST labels.

	Accepts ``None``, ``""``, a comma-separated string (no spaces), or a sequence of ints.
	Returns ``None`` when unset/empty. Requires at least two distinct digits in ``0..9``.
	"""
	if value is None:
		return None
	if isinstance(value, str):
		s = value.strip()
		if not s:
			return None
		parts = [p.strip() for p in s.split(",") if p.strip()]
		if not parts:
			return None
		digs = [int(p) for p in parts]
	elif isinstance(value, (list, tuple)):
		if not value:
			return None
		digs = [int(x) for x in value]
	else:
		raise TypeError(
			"restrain_digits must be str, list, tuple, or None, "
			f"got {type(value).__name__}"
		)
	out = sorted(set(digs))
	for d in out:
		if d < 0 or d > 9:
			raise ValueError(f"restrain_digits must be in 0..9, got {d!r} in {out!r}")
	if len(out) < 2:
		raise ValueError(
			"restrain_digits must name at least two distinct digits "
			"(labels are shuffled only within that set)."
		)
	return tuple(out)


def subset_cyclic_mislabel_map(sorted_digits: tuple[int, ...]) -> dict[int, int]:
	"""Fixed permutation on *sorted_digits*: swap if |S|==2, else one cyclic step."""
	n = len(sorted_digits)
	if n < 2:
		raise ValueError("subset_cyclic_mislabel_map needs at least two digits")
	if n == 2:
		a, b = sorted_digits
		return {a: b, b: a}
	return {sorted_digits[i]: sorted_digits[(i + 1) % n] for i in range(n)}


LABEL_PERM = {
    0: 7,
    1: 4,
    2: 9,
    3: 1,
    4: 6,
    5: 0,
    6: 2,
    7: 5,
    8: 3,
    9: 8,
}


@register_experiment
class PretrainThenShuffleMislabel(MNISTWrapper):
	"""Pretrain on balanced K, then learn permuted labels on a balanced N-sample pool."""

	has_pretrain: ClassVar[bool] = True

	def __init__(
		self,
		digitA: int = 0,
		digitB: int = 1,
		*,
		pretrain_on_k_samples: int,
		num_trial_samples: int,
		restrain_digits: Any = None,
		**kwargs: Any,
	) -> None:
		if not isinstance(pretrain_on_k_samples, int):
			raise TypeError(
				f"pretrain_on_k_samples must be int, got {type(pretrain_on_k_samples).__name__}"
			)
		if pretrain_on_k_samples < 1:
			raise ValueError(f"pretrain_on_k_samples must be >= 1, got {pretrain_on_k_samples}")
		if not isinstance(num_trial_samples, int):
			raise TypeError(
				f"num_trial_samples must be int, got {type(num_trial_samples).__name__}"
			)
		if num_trial_samples < 1:
			raise ValueError(f"num_trial_samples must be >= 1, got {num_trial_samples}")

		self._restrain_digits = parse_restrain_digits(restrain_digits)
		self._active_digit_tuple = (
			tuple(range(self.N_DIGITS))
			if self._restrain_digits is None
			else self._restrain_digits
		)
		self._n_active = len(self._active_digit_tuple)
		self._trial_label_perm = (
			LABEL_PERM
			if self._restrain_digits is None
			else subset_cyclic_mislabel_map(self._restrain_digits)
		)
		if pretrain_on_k_samples % self._n_active != 0:
			raise ValueError(
				f"pretrain_on_k_samples must be divisible by the number of training digits "
				f"({self._n_active}), got {pretrain_on_k_samples}."
			)

		super().__init__(digitA=digitA, digitB=digitB, **kwargs)
		self.pretrain_on_k_samples = pretrain_on_k_samples
		self.num_trial_samples = num_trial_samples

	def eval_supervision_label(self, true_label: int) -> int:
		t = int(true_label)
		m = self._trial_label_perm
		if t not in m:
			return t
		return int(m[t])

	def pretrain_sample_count(self) -> int:
		return self.pretrain_on_k_samples

	@property
	def first_digits(self) -> tuple[int, ...]:
		return self._active_digit_tuple

	def experiment_id(self) -> str:
		base = (
			f"pretrain_shuffle_mislabel_K{self.pretrain_on_k_samples}_"
			f"tr{self.num_trial_samples}"
		)
		if self._restrain_digits is None:
			return base
		ds = "-".join(str(d) for d in self._restrain_digits)
		return f"{base}_digits{ds}"

	def config_fields(self) -> dict[str, Any]:
		out: dict[str, Any] = {
			"pretrain_on_k_samples": self.pretrain_on_k_samples,
			"num_trial_samples": self.num_trial_samples,
		}
		if self._restrain_digits is not None:
			out["restrain_digits"] = ",".join(str(d) for d in self._restrain_digits)
		return out

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
				test_perm,
				batch_size=eval_bs,
				shuffle=False,
			),
			"all_test": make_loader(
				self._test_ds, all_test_idx, batch_size=eval_bs, shuffle=False
			),
		}

	def _build_trials(
		self, batch_size: int, seed: int, *, num_experiment_runs: int
	) -> list[TrialSpec]:
		indices_by_digit = digit_indices(self._train_ds, self._active_digit_tuple)
		pretrain_flat, remaining_by_digit = split_uniform_k_pretrain_remaining(
			self.pretrain_on_k_samples,
			indices_by_digit,
			digits=self._active_digit_tuple,
		)

		eval_loaders = self._eval_loaders()

		trial_pre = TrialSpec(
			name="trial_pretrain_all_digits",
			train_loader=make_loader(
				self._train_ds,
				pretrain_flat,
				batch_size=batch_size,
				shuffle=True,
			),
			eval_loaders=eval_loaders,
			once_only=True,
		)
		trials = [trial_pre]
		for i in range(3):
			n_post = self.num_trial_samples
			if n_post % self._n_active != 0:
				raise ValueError(
					f"num_trial_samples must be divisible by the number of training digits "
					f"({self._n_active}); got {self.num_trial_samples}."
				)
			per_digit = n_post // self._n_active
			flat_indices: list[int] = []
			for d in self._active_digit_tuple:
				rem = remaining_by_digit[d]
				if len(rem) < per_digit:
					raise ValueError(
						f"Not enough remainder indices for digit {d} after pretrain: "
						f"need {per_digit}, got {len(rem)}."
					)
				flat_indices.extend(rem[per_digit * i:per_digit * (i + 1)])


			post_ds = PermutedLabelMapDataset(
				Subset(self._train_ds, flat_indices),
				self._trial_label_perm,
			)
			gen = torch.Generator()
			gen.manual_seed(seed)
			trial_perm = TrialSpec(
				name=f"trial_permuted_{i}",
				train_loader=DataLoader(
					post_ds,
					batch_size=1,
					shuffle=True,
					generator=gen,
				),
				eval_loaders=eval_loaders,
			)
			trials.append(trial_perm)
		return trials
