"""Base class for MNIST experiments with shared dataset handling.

Provides:
- Train-set-based normalization (mean/std computed from MNIST train split).
- Digit-index extraction utilities.
- A common ``evaluation_inputs`` implementation.

Subclasses only need to implement ``experiment_id``, ``config_fields``,
and ``build_stages``.
"""
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from experiments.base import Experiment


def digit_indices(
    dataset: datasets.MNIST | Subset,
    digits: tuple[int, ...],
) -> dict[int, list[int]]:
    """Return ``{digit: [indices]}`` for the requested *digits*."""
    per_digit: dict[int, list[int]] = {d: [] for d in digits}
    for idx in range(len(dataset)):
        label = int(dataset[idx][1])  # type: ignore[union-attr]
        if label in per_digit:
            per_digit[label].append(idx)
    return per_digit


def make_loader(
    dataset: datasets.MNIST | Subset,
    indices: list[int],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
    )


class MNISTWrapper(Experiment):
    """Shared base for MNIST digit-continual experiments.

    Handles dataset loading with normalization derived from the *train* split,
    so both train and test sets live in the same feature space.
    """

    def __init__(self, digitA: int = 1, digitB: int = 2, **kwargs):
        self.digitA = digitA
        self.digitB = digitB
        self._ensure_datasets()

    @property
    def digits(self) -> tuple[int, ...]:
        """Digits this experiment operates on (override for >2 digits)."""
        return (self.digitA, self.digitB)

    def _ensure_datasets(self):
        """Load MNIST train/test sets, normalizing both with train-set stats."""
        raw_train = datasets.MNIST(
            root="./data", train=True, download=True,
            transform=transforms.ToTensor(),
        )
        pixels = raw_train.data.float() / 255.0
        mean = pixels.mean().item()
        std = pixels.std().item()

        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ])

        full_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=tfm,
        )
        self._test_ds = datasets.MNIST(
            root="./data", train=False, download=True, transform=tfm,
        )

        # Eval inputs for activations: last 5 per digit (0-9) from train = 50 samples.
        all_digits = tuple(range(10))
        by_digit = digit_indices(full_train, all_digits)
        eval_indices_list: list[int] = []
        for d in all_digits:
            idxs = by_digit[d]
            eval_indices_list.extend(idxs[-5:])
        eval_indices_set = frozenset(eval_indices_list)
        self._eval_ds = Subset(full_train, eval_indices_list)

        # Train dataset excludes eval samples
        train_indices = [i for i in range(len(full_train)) if i not in eval_indices_set]
        self._train_ds = Subset(full_train, train_indices)

    def evaluation_inputs(
        self, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return 50 samples (5 per digit 0-9) from eval_ds for activation capture."""
        loader = DataLoader(
            self._eval_ds, batch_size=len(self._eval_ds), shuffle=False,
        )
        images, labels = next(iter(loader))
        return images.to(device), labels.to(device)
