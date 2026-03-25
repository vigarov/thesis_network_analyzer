"""Base class for MNIST experiments with shared dataset handling.

Provides:
- Train-set-based normalization (mean/std computed from MNIST train split).
- Digit-index extraction utilities.
- A common ``evaluation_inputs`` implementation.

Subclasses only need to implement ``experiment_id``, ``config_fields``,
and ``_build_stages``.
"""
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from experiments.base import Experiment


def _label_to_int(y: int | torch.Tensor) -> int:
    return int(y.item()) if isinstance(y, torch.Tensor) else int(y)


def digit_indices(
    dataset: datasets.MNIST | Subset | TensorDataset,
    digits: tuple[int, ...],
) -> dict[int, list[int]]:
    """Return ``{digit: [indices]}`` for the requested *digits*."""
    per_digit: dict[int, list[int]] = {d: [] for d in digits}
    for idx in range(len(dataset)):
        label = _label_to_int(dataset[idx][1])  # type: ignore[union-attr]
        if label in per_digit:
            per_digit[label].append(idx)
    return per_digit


def make_loader(
    dataset: datasets.MNIST | Subset | TensorDataset,
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

    #: Number of digit classes in MNIST (0--9).
    N_DIGITS: int = 10

    def __init__(self, digitA: int = 0, digitB: int = 1, **kwargs):
        super().__init__()
        self.digitA = digitA
        self.digitB = digitB
        self._ensure_datasets()
        self._test_by_digit_indices = digit_indices(self._test_ds, tuple(range(self.N_DIGITS)))
        self._eval_pinned_device: torch.device | None = None

    @property
    def first_digits(self) -> tuple[int, ...]:
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

        # Eval inputs for activations: last 5 per digit (0 .. N_DIGITS-1) from train.
        all_digits = tuple(range(self.N_DIGITS))
        train_by_digit_indices = digit_indices(full_train, all_digits)
        eval_indices_list: list[int] = []
        for d in all_digits:
            idcs = train_by_digit_indices[d]
            eval_indices_list.extend(idcs[-5:])
        eval_indices_set = frozenset(eval_indices_list)
        self._eval_ds = Subset(full_train, eval_indices_list)

        # Train dataset excludes eval samples
        train_indices = [i for i in range(len(full_train)) if i not in eval_indices_set]
        self._train_ds = Subset(full_train, train_indices)

    
    def to_device(self, device: torch.device) -> None:
        super().to_device(device) # allows the to_device() call
        if "cpu" == device.type:
            assert "cpu" in self._test_ds.data.device.type, "Test dataset must be on CPU" # type: ignore[attr-defined]
            return
        
        # In both cases, we must instantiate a temp DataLoader to actually extract the tensors (in a new dataset)
        # This is to avoid the MNISTDataset to try to instantiate a PIL image if the data is on GPU
        # (tensor.numpy() call in mnist.py, line 143)
        temp_loader = DataLoader(
            self._test_ds, batch_size=len(self._test_ds), shuffle=False,
        )
        images, labels = next(iter(temp_loader))
        self._test_ds = TensorDataset(
            images.to(device),
            labels.to(device),
        )

        temp_loader = DataLoader(
            self._eval_ds, batch_size=len(self._eval_ds), shuffle=False,
        )
        images, labels = next(iter(temp_loader))
        self._eval_ds = TensorDataset(
            images.to(device),
            labels.to(device),
        )

        self._eval_pinned_device = device


    def evaluation_inputs(
        self, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return 50 samples (5 per digit 0-9) from eval_ds for activation capture."""
        if self._eval_pinned_device is not None:
            images, labels = self._eval_ds.tensors  # type: ignore[attr-defined]
            if device != self._eval_pinned_device:
                return images.to(device), labels.to(device)
            return images, labels

        loader = DataLoader(
            self._eval_ds, batch_size=len(self._eval_ds), shuffle=False,
        )
        images, labels = next(iter(loader))
        return images.to(device), labels.to(device)
