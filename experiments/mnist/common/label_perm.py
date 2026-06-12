"""Label permutation helpers (copied from legacy pretrain_then_shuffle_mislabel for a self-contained package)."""
from typing import Any


def parse_restrain_digits(value: Any) -> tuple[int, ...] | None:
	"""Parse optional ``restrain_digits`` from JSON/CLI into sorted unique MNIST labels."""
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


LABEL_PERM: dict[int, int] = {
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
