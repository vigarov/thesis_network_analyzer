"""
Helper module to make dataset loding similar for both MNIST and CIFAR-10.
Handles pre-processing: 
- MNIST:
  * 28x28 + train-statistics (mean/std of train pixels) normalization
- CIFAR-10:
  * 28x28 + per-image whitening for Inception,
  * 32x32 + channel normalize for ResNet

"""
from dataclasses import dataclass

import torch
from torch.utils.data import Subset, TensorDataset
from torchvision import datasets, transforms

DATA_ROOT = "./data"

# CIFAR-10 channel statistics (see local/cifar_test.ipynb / standard values).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Both supported datasets expose 10 labels.
NUM_CLASSES = 10

# Normalization mode identifiers.
NORM_TRAIN_STATS = "train_stats"  # mean/std from train pixels (MNIST default)
NORM_PER_IMAGE_WHITENING = "per_image_whitening"  # Inception (CIFAR)
NORM_CIFAR_CHANNEL = "cifar_channel"  # ResNet / VGG (CIFAR)


class PerImageWhitening:
	"""Per-image whitening used in Zhang et al. (2017) CIFAR experiments."""

	def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
		mean = tensor.mean(dim=(1, 2), keepdim=True)
		std = tensor.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
		return (tensor - mean) / std


@dataclass(frozen=True)
class DatasetSpec:
	"""Static description of a supported dataset."""

	name: str
	torchvision_class: type
	native_size: int
	default_normalization: str


_DATASET_REGISTRY: dict[str, DatasetSpec] = {
	"mnist": DatasetSpec("mnist", datasets.MNIST, 28, NORM_TRAIN_STATS),
	"cifar10": DatasetSpec("cifar10", datasets.CIFAR10, 32, NORM_CIFAR_CHANNEL),
}


def list_datasets() -> list[str]:
	return sorted(_DATASET_REGISTRY)


def get_dataset_spec(name: str) -> DatasetSpec:
	key = str(name).strip().lower()
	if key not in _DATASET_REGISTRY:
		raise KeyError(
			f"Unknown dataset {name!r}. Available: {list_datasets()}"
		)
	return _DATASET_REGISTRY[key]


def num_classes(dataset: str) -> int:
	get_dataset_spec(dataset)  # validate name
	return NUM_CLASSES


def _resolve_size_and_norm(
	spec: DatasetSpec,
	input_size: int | None,
	normalization: str | None,
) -> tuple[int, str]:
	size = spec.native_size if input_size is None else int(input_size)
	norm = (
		spec.default_normalization
		if normalization in (None, "", "auto")
		else str(normalization)
	)
	return size, norm


def _build_transform(
	spec: DatasetSpec,
	size: int,
	norm: str,
	*,
	mean: float | None = None,
	std: float | None = None,
) -> transforms.Compose:
	steps: list = [transforms.ToTensor()]
	if size != spec.native_size:
		steps.append(transforms.CenterCrop(size))
	if norm == NORM_TRAIN_STATS:
		if mean is None or std is None:
			raise ValueError("train_stats normalization requires mean and std")
		steps.append(transforms.Normalize((mean,), (std,)))
	elif norm == NORM_PER_IMAGE_WHITENING:
		steps.append(PerImageWhitening())
	elif norm == NORM_CIFAR_CHANNEL:
		steps.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))
	else:
		raise ValueError(f"Unknown normalization mode {norm!r}")
	return transforms.Compose(steps)


def load_train_test_datasets(
	dataset: str,
	data_root: str = DATA_ROOT,
	*,
	input_size: int | None = None,
	normalization: str | None = None,
):
	"""Return `(full_train, test)` torchvision datasets with the resolved transform.

	`input_size` / `normalization` are derived from the model 
	(see `models.AnalyzableModel.input_spec`)
	"""
	spec = get_dataset_spec(dataset)
	size, norm = _resolve_size_and_norm(spec, input_size, normalization)

	mean = std = None
	if norm == NORM_TRAIN_STATS:
		raw_train = spec.torchvision_class(
			root=data_root, train=True, download=True,
			transform=transforms.ToTensor(),
		)
		pixels = raw_train.data.float() / 255.0
		mean = pixels.mean().item()
		std = pixels.std().item()

	tfm = _build_transform(spec, size, norm, mean=mean, std=std)
	full_train = spec.torchvision_class(
		root=data_root, train=True, download=True, transform=tfm,
	)
	test_ds = spec.torchvision_class(
		root=data_root, train=False, download=True, transform=tfm,
	)
	return full_train, test_ds


def dataset_targets(dataset) -> list[int]:
	"""
	Avoids applying the (potentially expensive) image transform just to read
	labels. Falls back to per-item access for datasets without `targets`.
	"""
	if isinstance(dataset, Subset):
		base = dataset_targets(dataset.dataset)
		return [base[i] for i in dataset.indices]
	if isinstance(dataset, TensorDataset):
		return [int(t) for t in dataset.tensors[1]]
	targets = getattr(dataset, "targets", None)
	if targets is not None:
		if isinstance(targets, torch.Tensor):
			return [int(t) for t in targets.tolist()]
		return [int(t) for t in targets]
	out: list[int] = []
	for i in range(len(dataset)):
		y = dataset[i][1]
		out.append(int(y.item()) if isinstance(y, torch.Tensor) else int(y))
	return out
