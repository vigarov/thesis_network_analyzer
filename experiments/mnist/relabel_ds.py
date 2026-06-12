"""Dataset wrapper: fixed subset indices with a single replacement label for every sample."""

from collections.abc import Callable

import torch
from torch.utils.data import Dataset

from experiments.mnist.base import label_to_int

class PermutedLabelMapDataset(Dataset):
	"""Wraps base_ds so each label `y` becomes `label_map[y]` (or `label_map(y)`)."""

	def __init__(
		self,
		base_ds: Dataset,
		label_map: dict[int, int] | Callable[[int], int],
	) -> None:
		self._base = base_ds
		self._label_map = label_map

	def _remap(self, y: int) -> int:
		if isinstance(self._label_map, dict):
			return int(self._label_map[y])
		return int(self._label_map(y))

	def __len__(self) -> int:
		return len(self._base)  # type: ignore[arg-type]

	def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
		x, y = self._base[i]
		yt = self._remap(label_to_int(y))
		return x, torch.tensor(yt, dtype=torch.long)


class RelabelSubset(Dataset):
	"""Subset of base_ds by indices with a fixed integer label for every sample."""

	def __init__(
		self,
		base_ds: torch.utils.data.Dataset,
		indices: list[int],
		label: int,
	) -> None:
		self._base = base_ds
		self._indices = indices
		self._label = label

	def __len__(self) -> int:
		return len(self._indices)

	def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
		x, _ = self._base[self._indices[i]]
		y = torch.tensor(self._label, dtype=torch.long)
		return x, y
