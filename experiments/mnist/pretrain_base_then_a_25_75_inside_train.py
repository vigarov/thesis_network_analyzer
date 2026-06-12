"""Experiment: Pre-train on a balanced-K subset of all digits, then one digit trial.

Uses the official MNIST train/test split:
- Training trials draw from `train=True` filtered to the relevant digits.
- Evaluation uses `train=False` filtered to the relevant digits.

Within the train set, `pretrain_on_k_samples` defines the pretrain pool: the first
`K // 10` indices per digit (in subset order) are used for pretraining; the rest of
each digit's pool forms the "remaining" subset.

Trial A (pretrain): train on all digits in the pretrain pool (shuffled).  [`once_only=True`]
Trial B: train on the first `num_trial_samples` samples of `digitA` within the
remaining subset.

Experiment runs / variability
-----------------------------
- Empty variability: flat list; pretrain runs only in run 0 via `once_only=True`.
- `change_digits`: nested list of length `num_experiment_runs`; run 0 is pretrain +
  `digitA`; later runs omit pretrain and use `(digitA + r) % 10` for run index
  `r >= 1`, each sliced to `num_trial_samples`.
"""

from typing import Any, ClassVar

from experiments.base import TrialSpec, register_experiment, NUM_POST_PRETRAIN_TRIALS
from experiments.mnist.base import MNISTWrapper, digit_indices, make_loader
from experiments.mnist.pretrain_split import split_uniform_k_pretrain_remaining


@register_experiment
class PretrainBase_thenDigitA(MNISTWrapper):
	"""Pretrain on a balanced-K subset, then a single digit trial in the remainder."""

	has_pretrain: ClassVar[bool] = True

	def __init__(
		self,
		digitA: int = 0,
		*,
		pretrain_on_k_samples: int,
		num_trial_samples: int,
		**kwargs: Any,
	):
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

		# Validate after super().__init__ so self.N_DIGITS and self._train_ds are available.
		if pretrain_on_k_samples < 1:
			raise ValueError(f"pretrain_on_k_samples must be >= 1, got {pretrain_on_k_samples}")
		if pretrain_on_k_samples % self.N_DIGITS != 0:
			raise ValueError(
				f"pretrain_on_k_samples must be divisible by N_DIGITS={self.N_DIGITS}, "
				f"got {pretrain_on_k_samples}."
			)
		self.pretrain_on_k_samples = pretrain_on_k_samples
		self.num_trial_samples = num_trial_samples

		idx_by_digit = digit_indices(self._train_ds, tuple(range(self.N_DIGITS)))
		_, remaining = split_uniform_k_pretrain_remaining(
			pretrain_on_k_samples, idx_by_digit, self.N_DIGITS
		)
		need = num_trial_samples * NUM_POST_PRETRAIN_TRIALS
		for d in range(self.N_DIGITS):
			if len(remaining[d]) < need:
				raise ValueError(
					f"pretrain_on_k_samples={pretrain_on_k_samples} leaves only "
					f"{len(remaining[d])} samples for digit {d} in the remainder; "
					f"need at least {need} (= num_trial_samples * {NUM_POST_PRETRAIN_TRIALS}) "
					f"for every digit so change_digits runs can use any anchor."
				)

	@property
	def first_digits(self) -> tuple[int, ...]:
		return (self.digitA,)

	def pretrain_sample_count(self) -> int:
		return self.pretrain_on_k_samples

	def experiment_id(self) -> str:
		return f"pretrainK{self.pretrain_on_k_samples}_tr{self.num_trial_samples}_then_{self.digitA}"

	def config_fields(self) -> dict[str, Any]:
		return {
			"digitA": self.digitA,
			"pretrain_on_k_samples": self.pretrain_on_k_samples,
			"num_trial_samples": self.num_trial_samples,
		}

	def _digit_trials(
		self,
		pretrain_indices: list[int] | None,
		remaining_indices_by_digit: dict[int, list[int]],
		digit: int,
		batch_size: int,
	) -> tuple[TrialSpec, ...]:
		eval_loaders = {
			"all_test": make_loader(
				self._test_ds,
				sum(self._test_by_digit_indices.values(), []),
				batch_size=256,
				shuffle=False,
			),
		}
		extra_eval_loaders = {
			f"digit_{digit}_test": make_loader(
				self._test_ds,
				self._test_by_digit_indices[digit],
				batch_size=256,
				shuffle=False,
			),
		}

		rem = remaining_indices_by_digit[digit]
		if len(rem) < self.num_trial_samples:
			raise ValueError(
				f"Not enough remaining train indices for digit {digit}: "
				f"need {self.num_trial_samples}, got {len(rem)}"
			)

		trial_digit = TrialSpec(
			name=f"trial_digit{digit}",
			train_loader=make_loader(
				self._train_ds,
				rem[: self.num_trial_samples],
				batch_size,
				shuffle=False,
			),
			eval_loaders=eval_loaders | extra_eval_loaders,
		)

		if pretrain_indices is not None:
			trial_pre = TrialSpec(
				name="trial_base_pretrain",
				train_loader=make_loader(
					self._train_ds, pretrain_indices, batch_size=batch_size, shuffle=True
				),
				eval_loaders=eval_loaders,
				once_only=True,
			)
			return trial_pre, trial_digit
		return (trial_digit,)

	def _build_trials(
		self, batch_size: int, seed: int, *, num_experiment_runs: int
	) -> list[TrialSpec] | list[list[TrialSpec]]:
		variability = self._experiment_variability.strip().lower()
		if variability not in ("", "change_digits"):
			raise ValueError(
				f"PretrainBase_thenDigitA does not support "
				f"experiment_variability={self._experiment_variability!r}. "
				"Supported values: '' (none), 'change_digits'."
			)

		indices_by_digit = digit_indices(self._train_ds, tuple(range(self.N_DIGITS)))
		pretrain_flat, remaining_indices_by_digit = split_uniform_k_pretrain_remaining(
			self.pretrain_on_k_samples, indices_by_digit, self.N_DIGITS
		)

		t0_pretrain, t0_digit = self._digit_trials(
			pretrain_flat,
			remaining_indices_by_digit,
			self.digitA,
			batch_size,
		)
		t0_list = [t0_pretrain, t0_digit]
		if variability == "":
			return t0_list

		per_run: list[list[TrialSpec]] = [t0_list]
		for t in range(num_experiment_runs - 1):
			d = (self.digitA + t + 1) % self.N_DIGITS
			(trial_digit,) = self._digit_trials(
				None, remaining_indices_by_digit, d, batch_size
			)
			per_run.append([trial_digit])
		return per_run
