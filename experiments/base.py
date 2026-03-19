"""Base class and registry for experiments."""
import abc
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader


@dataclass
class StageSpec:
    """Describes one training stage."""

    name: str
    train_loader: DataLoader
    eval_loaders: dict[str, DataLoader] = field(default_factory=dict)
    is_turning_point: bool = False


class Experiment(abc.ABC):
    """Abstract base class that every experiment must implement."""

    @abc.abstractmethod
    def experiment_id(self) -> str:
        """Unique identifier used in the results directory path."""

    @abc.abstractmethod
    def config_fields(self) -> dict[str, Any]:
        """Return the experiment-specific config fields that are
        optimizer-independent and must be frozen across reruns."""

    @abc.abstractmethod
    def build_stages(self, batch_size: int, seed: int) -> list[StageSpec]:
        """Construct dataloaders for each training stage.

        Returns an ordered list of StageSpec objects. The training loop
        iterates through them sequentially.
        """

    @abc.abstractmethod
    def evaluation_inputs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a representative (inputs, labels) batch used for
        activation capture at checkpoints. The experiment controls which
        samples are used so this can vary beyond two-digit scenarios."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_EXPERIMENT_REGISTRY: dict[str, type[Experiment]] = {}


def register_experiment(cls: type[Experiment]) -> type[Experiment]:
    key = cls.__name__
    _EXPERIMENT_REGISTRY[key] = cls
    return cls


def get_experiment(name: str, **kwargs: Any) -> Experiment:
    if name not in _EXPERIMENT_REGISTRY:
        raise KeyError(
            f"Unknown experiment '{name}'. "
            f"Available: {sorted(_EXPERIMENT_REGISTRY)}"
        )
    return _EXPERIMENT_REGISTRY[name](**kwargs)


def list_experiments() -> list[str]:
    return sorted(_EXPERIMENT_REGISTRY)
