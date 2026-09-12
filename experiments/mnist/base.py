"""Base class for image-classification experiments.

Provides:
- Dataset-agnostic loading via `experiments.dataset_registry` (MNIST or CIFAR10)
- Class-index extraction utilities.
- A common `evaluation_inputs` implementation.

Subclasses need to implement `experiment_id`, `config_fields`, and `_build_trials`. 
"""
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from experiments.base import Experiment
from experiments.dataset_registry import (
    DATA_ROOT,
    dataset_targets,
    load_train_test_datasets,
)


def label_to_int(y: int | torch.Tensor) -> int:
    return int(y.item()) if isinstance(y, torch.Tensor) else int(y)


def digit_indices(
    dataset,
    digits: tuple[int, ...],
) -> dict[int, list[int]]:
    """Return `{class_label: [indices]}` for the requested labels."""
    per_digit: dict[int, list[int]] = {d: [] for d in digits}
    labels = dataset_targets(dataset)
    for idx, label in enumerate(labels):
        if label in per_digit:
            per_digit[label].append(idx)
    return per_digit


def make_loader(
    dataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
    )


class _PinnedEvalSupervisionDataset(Dataset):
    """Pinned eval indices into full_train with supervision labels from label_from_true."""

    def __init__(
        self,
        full_train: Dataset,
        indices: list[int],
        label_from_true: Callable[[int], int],
    ) -> None:
        self._full_train = full_train
        self._indices = indices
        self._label_from_true = label_from_true

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self._indices[i]
        x, y = self._full_train[idx]
        yt = self._label_from_true(label_to_int(y))
        return x, torch.tensor(yt, dtype=torch.long)

# [TODO] : Rename to VisualExperiment
class MNISTWrapper(Experiment):
    """Shared base for experiments for both MNIST and CIFAR-10.

    Handles dataset loading with the transform/normalization procedure
    resolved from `(dataset, model, input_spec)` (see `experiments.dataset_registry`)
    Model selects normalization procedure, or dataset by default if unspecified.
    """

    # Number of classes (0-9); identical for MNIST and CIFAR-10.
    N_DIGITS: int = 10

    def __init__(
        self,
        digitA: int = 0,
        digitB: int = 1,
        *,
        dataset: str = "mnist",
        input_size: int | None = None,
        normalization: str | None = None,
        data_root: str = DATA_ROOT,
        **kwargs,
    ):
        super().__init__()
        self.digitA = digitA
        self.digitB = digitB
        self._dataset = str(dataset)
        self._input_size = input_size
        self._normalization = normalization
        self._data_root = data_root
        self._ensure_datasets()
        self._test_by_digit_indices = digit_indices(self._test_ds, tuple(range(self.N_DIGITS)))
        self._eval_pinned_device: torch.device | None = None

    def train_pool_size(self) -> int:
        return len(self._train_ds)

    @property
    def first_digits(self) -> tuple[int, ...]:
        """Digits this experiment operates on (override for >2 digits)."""
        return (self.digitA, self.digitB)

    def config_fields(self) -> dict[str, Any]:
        return {"digitA": self.digitA, "digitB": self.digitB}

    def _ensure_datasets(self):
        """Load train/test sets with the model/dataset transform/normalization."""
        full_train, self._test_ds = load_train_test_datasets(
            self._dataset,
            self._data_root,
            input_size=self._input_size,
            normalization=self._normalization,
        )

        # Eval inputs for activations: last 5 per class (0 .. N_DIGITS-1) from train.
        all_digits = tuple(range(self.N_DIGITS))
        train_by_digit_indices = digit_indices(full_train, all_digits)
        eval_indices_list: list[int] = []
        for d in all_digits:
            idcs = train_by_digit_indices[d]
            eval_indices_list.extend(idcs[-5:])
        eval_indices_set = frozenset(eval_indices_list)
        self._eval_ds = _PinnedEvalSupervisionDataset(
            full_train,
            eval_indices_list,
            lambda y_true: label_to_int(y_true), # identity by default (no relabeling)
        )

        # Train dataset excludes eval samples
        train_indices = [i for i in range(len(full_train)) if i not in eval_indices_set]
        self._train_ds = Subset(full_train, train_indices)

    
    def to_device(self, device: torch.device) -> None:
        super().to_device(device) # allows the to_device() call
        if "cpu" == device.type:
            # Only tensor-backed datasets (e.g. MNIST) expose `.data.device`;
            # CIFAR-10 stores `.data` as a NumPy array, so skip the check there.
            data = getattr(self._test_ds, "data", None)
            if isinstance(data, torch.Tensor):
                assert "cpu" in data.device.type, "Test dataset must be on CPU"
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
