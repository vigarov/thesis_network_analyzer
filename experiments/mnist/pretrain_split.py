def split_uniform_k_pretrain_remaining(
	total_k: int,
	indices_by_digit: dict[int, list[int]],
	n_digits: int = 10,
	*,
	digits: tuple[int, ...] | None = None,
) -> tuple[list[int], dict[int, list[int]]]:
	"""Return `(pretrain_flat, remainder_by_digit)` for a balanced-K pretrain pool.

	Selects the first `total_k // n_digits` indices per digit (in the order they
	appear in `indices_by_digit`) and concatenates them into a flat pretrain list.
	The remaining per-digit indices form the trial pool.

	When *digits* is set, balances only over those class labels (length must match
	the implied divisor); *n_digits* is ignored in that case.
	"""
	if total_k < 1:
		raise ValueError(f"pretrain_on_k_samples must be >= 1, got {total_k}")
	digit_tuple = digits if digits is not None else tuple(range(n_digits))
	n = len(digit_tuple)
	if total_k % n != 0:
		raise ValueError(
			f"pretrain_on_k_samples must be divisible by n_digits={n}, got {total_k}."
		)
	k_per = total_k // n
	pretrain_parts: list[list[int]] = []
	remainder: dict[int, list[int]] = {}
	for d in digit_tuple:
		idxs = indices_by_digit[d]
		if len(idxs) < k_per:
			raise ValueError(
				f"Not enough train indices for digit {d}: need at least {k_per} for "
				f"pretrain_on_k_samples={total_k} (uniform {k_per} per digit), got {len(idxs)}"
			)
		pretrain_parts.append(idxs[:k_per])
		remainder[d] = idxs[k_per:]
	pretrain_flat: list[int] = []
	for part in pretrain_parts:
		pretrain_flat.extend(part)
	return pretrain_flat, remainder
