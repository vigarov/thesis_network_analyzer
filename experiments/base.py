"""Base class and registry for experiments."""
import abc
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

import torch
from torch.utils.data import DataLoader


@dataclass
class TrialSpec:
    """Describes one training trial (one sequenced block in the training schedule)."""

    name: str
    train_loader: DataLoader
    eval_loaders: dict[str, DataLoader] = field(default_factory=dict)
    post_trial_callback: Callable[..., None] | None = field(default=None, repr=False)
    once_only: bool = False


#: Number of post-pretrain training trials; each uses `num_trial_samples` (experiment-specific).
NUM_POST_PRETRAIN_TRIALS: int = 3


class Experiment(abc.ABC):
    """Abstract base class that every experiment must implement."""

    #: When True, this experiment includes a dedicated pretrain trial; subclasses must
    #: override `pretrain_sample_count`. `train_pool_size` is implemented on
    #: bases that define a finite training pool (e.g. `MNISTWrapper`).
    has_pretrain: ClassVar[bool] = False

    def __init__(self) -> None:
        self._trials_built: bool = False
        self._experiment_variability: str = ""

    def train_pool_size(self) -> int:
        """Size of the experiment training pool (samples available for training trials)."""
        raise NotImplementedError(
            f"{type(self).__name__}.train_pool_size() is not implemented."
        )

    def pretrain_sample_count(self) -> int:
        """Number of training samples used in the pretrain trial (meaningful only if `has_pretrain`)."""
        raise NotImplementedError(
            f"{type(self).__name__}.pretrain_sample_count() is not implemented."
        )

    def set_experiment_variability(self, experiment_variability: str) -> None:
        """Set the experiment variability string before calling build_trials.
            Must be called before build_trials().
        """
        if self._trials_built:
            raise RuntimeError(
                "set_experiment_variability() must be called before build_trials(); "
                "it is invalid after trials have been built."
            )
        self._experiment_variability = experiment_variability

    def to_device(self, device: torch.device) -> None:
        """Pinning data to device to be done before building trials.
        """
        if self._trials_built:
            raise RuntimeError(
                "to_device() must be called before build_trials(); "
                "it is invalid after trials have been built."
            )

    @abc.abstractmethod
    def experiment_id(self) -> str:
        """Unique identifier used in the results directory path."""

    @abc.abstractmethod
    def config_fields(self) -> dict[str, Any]:
        """Return the experiment-specific config fields that are
        optimizer-independent and must be frozen across reruns."""

    def build_trials(
        self, batch_size: int, seed: int, *, num_experiment_runs: int
    ) -> "list[TrialSpec] | list[list[TrialSpec]]":
        """Construct dataloaders for each training trial.

        Returns either a flat list[TrialSpec] (same trials reused each experiment run)
        or a list[list[TrialSpec]] (one list per run; len must equal num_experiment_runs).
        """
        trials = self._build_trials(batch_size, seed, num_experiment_runs=num_experiment_runs)
        self._trials_built = True
        return trials

    @abc.abstractmethod
    def _build_trials(
        self, batch_size: int, seed: int, *, num_experiment_runs: int
    ) -> "list[TrialSpec] | list[list[TrialSpec]]":
        """Return an ordered list of TrialSpec (flat) or one list per experiment run (nested)."""

    @abc.abstractmethod
    def evaluation_inputs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a representative (inputs, labels) batch used for
        activation capture at checkpoints. The experiment controls which
        samples are used so this can vary beyond two-digit scenarios."""


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


def get_registered_experiment_class(name: str) -> type[Experiment]:
    """Return the registered experiment class for name without instantiating it."""
    if name not in _EXPERIMENT_REGISTRY:
        raise KeyError(
            f"Unknown experiment '{name}'. "
            f"Available: {sorted(_EXPERIMENT_REGISTRY)}"
        )
    return _EXPERIMENT_REGISTRY[name]


def list_experiments() -> list[str]:
    return sorted(_EXPERIMENT_REGISTRY)
