"""Base class and registry for experiments."""
import abc
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader


@dataclass
class StageSpec:
    """Describes one training stage."""

    name: str
    train_loader: DataLoader
    eval_loaders: dict[str, DataLoader] = field(default_factory=dict)
    post_stage_callback: Callable[..., None] | None = field(default=None, repr=False)
    once_only: bool = False


class Experiment(abc.ABC):
    """Abstract base class that every experiment must implement."""

    def __init__(self) -> None:
        self._stages_built: bool = False
        self._trial_variability: str = ""

    def set_trial_var(self, trial_variability: str) -> None:
        """Set the trial variability string before calling build_stages.
            Must be called before build_stages(). 
        """
        if self._stages_built:
            raise RuntimeError(
                "set_trial_var() must be called before build_stages(); "
                "it is invalid after stages have been built."
            )
        self._trial_variability = trial_variability

    def to_device(self, device: torch.device) -> None:
        """Pinning data to device to be done before building stages.
        """
        if self._stages_built:
            raise RuntimeError(
                "to_device() must be called before build_stages(); "
                "it is invalid after stages have been built."
            )

    @abc.abstractmethod
    def experiment_id(self) -> str:
        """Unique identifier used in the results directory path."""

    @abc.abstractmethod
    def config_fields(self) -> dict[str, Any]:
        """Return the experiment-specific config fields that are
        optimizer-independent and must be frozen across reruns."""

    def build_stages(
        self, batch_size: int, seed: int, *, num_trials: int
    ) -> "list[StageSpec] | list[list[StageSpec]]":
        """Construct dataloaders for each training stage.

        Returns either a flat list[StageSpec] (same stages reused each trial)
        or a list[list[StageSpec]] (one list per trial; len must equal num_trials).
        """
        stages = self._build_stages(batch_size, seed, num_trials=num_trials)
        self._stages_built = True
        return stages

    @abc.abstractmethod
    def _build_stages(
        self, batch_size: int, seed: int, *, num_trials: int
    ) -> "list[StageSpec] | list[list[StageSpec]]":
        """Return an ordered list of StageSpec (flat) or one list per trial (nested)."""

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
