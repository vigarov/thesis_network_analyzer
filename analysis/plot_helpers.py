"""Adds trial boundaries on plots (either for plotly plots when from_metrics) or matplotlib plots when add_trial_boundaries_mpl"""
import re
from collections.abc import Callable, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory

from experiments.mnist.common.label_perm import LABEL_PERM
from analysis.constants import _NETWORK_N_DIGITS

LIGHT_GRAY_FG_COLOR = "#78716c"

_RECOVER_REINFORCE_TRIAL_BASES: frozenset[str] = frozenset(
	{
		"trial_recover_anchor_correct",
		"trial_recover_anchor_mislabeled_as_next",
		"trial_recover_next_correct",
		"trial_reinforce_anchor_correct",
		"trial_reinforce_next_images_labeled_anchor",
		"trial_reinforce_next_correct",
	}
)
_RECOVER_REINFORCE_RUN_END_BASES: frozenset[str] = frozenset(
	{
		"trial_recover_next_correct",
		"trial_reinforce_next_correct",
	}
)


def _trial_boundaries_from_metrics(
	metrics: dict[str, np.ndarray],
	x_mode: str,
) -> tuple[list[str], list[float], bool] | None:
	"""Extract (trial_names, boundary_x_positions, use_iterations) for plotting.

	Returns None if trial boundary data is missing (backward compat with old runs).
	x_mode: 'checkpoint' or 'iteration'.
	"""
	names = metrics.get("trial_names", metrics.get("stage_names"))
	cp_idxs = metrics.get(
		"trial_end_checkpoint_idxs", metrics.get("stage_end_checkpoint_idxs")
	)
	iters = metrics.get(
		"trial_end_iterations", metrics.get("stage_end_iterations")
	)
	if names is None or cp_idxs is None:
		return None
	names = [str(n) for n in names]
	if len(names) < 1:
		return None
	use_iters = (
		x_mode == "iteration"
		and iters is not None
		and len(iters) == len(names)
	)
	boundaries: list[float] = []
	for i in range(len(names) - 1):
		if use_iters and iters is not None:
			boundaries.append(float(iters[i]) + 0.5)
		elif cp_idxs is not None:
			boundaries.append(float(cp_idxs[i]) + 0.5)
	return (names, boundaries, use_iters)


def _view_x_min_from_segment_min_idx(
	min_idx: int,
	plot_x_per_segment: Sequence[float] | np.ndarray | None,
) -> float | None:
	"""First x-coordinate of the plotted segment window (`plot_x[min_idx]`), or None."""
	if plot_x_per_segment is None or int(min_idx) <= 0:
		return None
	arr = np.asarray(plot_x_per_segment, dtype=np.float64)
	if arr.size == 0 or int(min_idx) >= arr.size:
		return None
	return float(arr[int(min_idx)])


def _view_x_max_from_segment_max_idx(
	max_idx: int | None,
	min_idx: int,
	plot_x_per_segment: Sequence[float] | np.ndarray | None,
) -> float | None:
	"""Last x-coordinate of the plotted segment window for slice `plot_x[min_idx:max_idx]` (exclusive `max_idx`)."""
	if max_idx is None or plot_x_per_segment is None:
		return None
	arr = np.asarray(plot_x_per_segment, dtype=np.float64)
	if arr.size == 0:
		return None
	mx = int(max_idx)
	mi = int(min_idx)
	if mx <= mi or mx > arr.size:
		return None
	return float(arr[mx - 1])


def _base_trial_name_for_mpl(name: str) -> str:
	"""Strip optional `run{N}_` prefix from a stored trial name."""
	s = str(name).strip()
	m = re.match(r"^run\d+_(.+)$", s)
	return m.group(1) if m else s


def _run_idx_from_trial_name(name: str) -> int | None:
	m = re.match(r"^run(\d+)_", str(name).strip())
	return int(m.group(1)) if m else None


def _recover_reinforce_short_label(
	base: str, first_digit: int, run_idx: int
) -> str | None:
	"""Digit-suffixed label for cat2 recover/reinforce trials (anchor steps +2 mod 10 per run)."""
	anchor = (first_digit + 2 * run_idx) % _NETWORK_N_DIGITS
	b = (anchor + 1) % _NETWORK_N_DIGITS
	labels: dict[str, str] = {
		"trial_recover_anchor_correct": f"{anchor}",
		"trial_recover_anchor_mislabeled_as_next": f"{anchor}as{b}",
		"trial_recover_next_correct": f"{b}",
		"trial_reinforce_anchor_correct": f"{anchor}",
		"trial_reinforce_next_images_labeled_anchor": f"{b}as{anchor}",
		"trial_reinforce_next_correct": f"{b}",
	}
	return labels.get(base)


def _abbreviate_trial_name_for_mpl(
	name: str,
	*,
	report: bool = False,
	first_digit: int | None = None,
	run_idx: int | None = None,
) -> str:
	"""Short axis label: one letter per semantic token; digits preserved (e.g. `R3D7`)."""
	s = str(name).strip()
	if not s:
		return ""

	base = _base_trial_name_for_mpl(s)
	if base in _RECOVER_REINFORCE_TRIAL_BASES:
		if first_digit is None:
			raise ValueError(
				f"first_digit is required for recover/reinforce trial {name!r}"
			)
		if run_idx is None:
			raise ValueError(
				f"run_idx is required for recover/reinforce trial {name!r}"
			)
		label = _recover_reinforce_short_label(base, first_digit, run_idx)
		if label is None:
			raise ValueError(f"unmapped recover/reinforce trial {name!r}")
		return label

	# experiments_new's long names -> compact codes for axis labels
	_trial_exact: dict[str, str] = {
		"trial_shuffle_labelperm_pool_balanced_global": "S",
		"trial_pretrain_pool_balanced_all_digits": "P",
		"trial_shuffle_labelperm_pool_balanced_remainder": "S",
		"trial_pretrain_pool_balanced_on_subset": "P",
		"trial_shuffle_cyclic_labelperm_pool_on_subset_remainder": "S",
	}
	if s in _trial_exact:
		return _trial_exact[s]

	_trial_rx: list[
		tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]
	] = [
		(
			re.compile(r"^trial_digit_(\d+)_train_correct_images$"),
			r"d\1",
		),
		(
			re.compile(r"^trial_digit_(\d+)_train_correct_images_remainder$"),
			r"d\1",
		),
		(
			re.compile(r"^trial_digit_(\d+)_labelperm_supervision_remainder$"),
			lambda m: f"d{m.group(1)}as{LABEL_PERM[int(m.group(1))]}",
		),
	]
	for rx, repl in _trial_rx:
		m = rx.match(s)
		if m is not None:
			return rx.sub(repl, s)

	m_run = re.match(r"^run(\d+)_(.+)$", s)
	if m_run:
		return f"R{m_run.group(1)}{_abbreviate_trial_name_for_mpl(m_run.group(2), report=False, first_digit=first_digit, run_idx=int(m_run.group(1)))}"

	legacy_whole: dict[str, str] = {
		"trial_pretrain_all_digits": "P",
		"trial_base_pretrain": "P",
		"trial_remaindershuffle": "V",
		"trial_remainder_shuffled": "V",
	}
	if s in legacy_whole:
		return legacy_whole[s]

	parts = [p for p in s.split("_") if p]
	out: list[str] = []
	i = 0
	while i < len(parts):
		p = parts[i]
		pl = p.lower()
		if pl in ("trial", "stage"):
			i += 1
			continue
		if pl == "pretrain":
			out.append("P")
			i += 1
			continue
		if pl == "base" and i + 1 < len(parts) and parts[i + 1].lower() == "pretrain":
			out.append("P")
			i += 2
			continue
		if pl == "all" and i + 1 < len(parts) and parts[i + 1].lower() == "digits":
			i += 2
			continue
		if pl.startswith("remainder"):
			out.append("V")
			i += 1
			continue
		if pl.startswith("permuted"):
			out.append("M")
			i += 1
			continue
		if pl == "digit" and i + 1 < len(parts) and parts[i + 1].isdigit():
			out.append("D" + parts[i + 1])
			i += 2
			continue
		if pl.startswith("digit") and len(pl) > 5 and pl[5:].isdigit():
			out.append("D" + pl[5:])
			i += 1
			continue
		if pl.startswith("label") and len(pl) > 5 and pl[5:].isdigit():
			out.append("L" + pl[5:])
			i += 1
			continue
		if pl.startswith("mislabel") or pl == "as" or pl.startswith("aslabel"):
			out.append("X")
			i += 1
			continue
		if len(p) == 1 and p.isalpha():
			out.append(p.upper())
			i += 1
			continue
		if len(p) == 2 and p[0].lower() == "d" and p[1].isdigit():
			out.append("D" + p[1])
			i += 1
			continue
		if len(p) == 2 and p[0].lower() == "l" and p[1].isdigit():
			out.append("L" + p[1])
			i += 1
			continue
		out.append(p[0].upper() if p else "")
		i += 1

	return "".join(out)


def add_trial_boundaries_mpl(
	ax: Axes,
	metrics: dict[str, np.ndarray],
	x_mode: str,
	*,
	min_idx: int = 0,
	max_idx: int | None = None,
	min_x_for_trial_labels: float | None = None,
	max_x_for_trial_labels: float | None = None,
	plot_x_per_segment: Sequence[float] | np.ndarray | None = None,
	report: bool = False,
	first_digit: int | None = None,
) -> None:
	"""Vertical dashed lines at trial/stage joins and optional trial labels.

	`first_digit` is the experiment's base `digitA` (run-0 anchor). When set,
	recover/reinforce trial labels become `A{d}`, `X{d}`, `B{d}`, `Y{d}` with
	`d` stepping as in :mod:`experiments_new.mnist.cat2.sequence_recover` /
	`sequence_reinforce` (anchor `(digitA + 2*r) % 10` per run `r`). Required
	for those trials; raises :class:`ValueError` if they appear and `first_digit` is
	`None`.

	If `plot_x_per_segment` is set (same x-axis as the plot: one value per
	checkpoint for mean curves, or one value per segment e.g. `xs[1:]` for angle
	plots), `min_idx` is the index of the first plotted point and `max_idx` is
	an exclusive end (same as Python slicing `plot_x[min_idx:max_idx]`). Boundary
	lines and trial labels outside `[plot_x[min_idx], plot_x[max_idx - 1]]` are
	skipped (half-open slice semantics).

	Otherwise every boundary in `boundary_xs` is drawn (matplotlib clips to the
	axes). Trial name annotations use `min_x_for_trial_labels` / `max_x_for_trial_labels`
	when given, else the inferred bounds from `min_idx` / `max_idx`, else
	`min_idx` as a minimum on the same scale as `mid` (legacy).
	"""
	sb = _trial_boundaries_from_metrics(metrics, x_mode)
	if sb is None:
		return
	trial_names, boundary_xs, use_iters = sb
	if not boundary_xs:
		return

	x_view_lo = _view_x_min_from_segment_min_idx(min_idx, plot_x_per_segment)
	x_view_hi = _view_x_max_from_segment_max_idx(max_idx, min_idx, plot_x_per_segment)
	for x in boundary_xs:
		if x_view_lo is not None and float(x) < x_view_lo:
			continue
		if x_view_hi is not None and float(x) > x_view_hi:
			continue
		ax.axvline(x, color="#000000", linestyle="--", linewidth=1.5, zorder=1)

	cp_idxs = metrics.get(
		"trial_end_checkpoint_idxs", metrics.get("stage_end_checkpoint_idxs")
	)
	iters = metrics.get(
		"trial_end_iterations", metrics.get("stage_end_iterations")
	)
	if cp_idxs is None:
		return

	if first_digit is not None and not (0 <= int(first_digit) < _NETWORK_N_DIGITS):
		raise ValueError(
			f"first_digit must be in [0, {_NETWORK_N_DIGITS}), got {first_digit!r}"
		)

	trans = blended_transform_factory(ax.transData, ax.transAxes)
	seq_run_idx = 0
	for i, name in enumerate(trial_names):
		if use_iters and iters is not None:
			start = 0.0 if i == 0 else float(iters[i - 1]) + 1.0
			end = float(iters[i])
		else:
			start = 0.0 if i == 0 else float(int(cp_idxs[i - 1]) + 1)
			end = float(int(cp_idxs[i]))
		mid = (start + end) / 2.0
		lab_floor = (
			min_x_for_trial_labels
			if min_x_for_trial_labels is not None
			else (x_view_lo if x_view_lo is not None else float(min_idx))
		)
		if mid < lab_floor:
			continue
		lab_ceil = (
			max_x_for_trial_labels
			if max_x_for_trial_labels is not None
			else x_view_hi
		)
		if lab_ceil is not None and mid > lab_ceil:
			continue
		s_name = str(name)
		base = _base_trial_name_for_mpl(s_name)
		rid = _run_idx_from_trial_name(s_name)
		if base in _RECOVER_REINFORCE_TRIAL_BASES:
			if first_digit is None:
				raise ValueError(
					f"first_digit is required for recover/reinforce trial {s_name!r}"
				)
			if rid is None:
				rid = seq_run_idx
		short = _abbreviate_trial_name_for_mpl(
			s_name, report=report, first_digit=first_digit, run_idx=rid
		)
		if base in _RECOVER_REINFORCE_RUN_END_BASES:
			seq_run_idx += 1
		if report:
			short = re.sub(r"^R\d+", "", short)
		ax.text(
			mid,
			1.02,
			short,
			transform=trans,
			ha="center",
			va="bottom",
			fontsize=10,
			color=LIGHT_GRAY_FG_COLOR,
			clip_on=False,
		)
