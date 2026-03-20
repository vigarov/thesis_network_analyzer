"""Base utilities and registry for models."""

import abc
from typing import Any
import torch.nn as nn


class AnalyzableModel(nn.Module, abc.ABC):
    """Base class for models used in the analysis framework.

    Subclasses declare which layers contain the "clickable units" that appear
    in the visualization, and provide metadata about each unit.
    """

    @abc.abstractmethod
    def model_id(self) -> str:
        """Unique identifier used in the results directory path."""

    @abc.abstractmethod
    def config_fields(self) -> dict[str, Any]:
        """Model-specific config fields frozen across reruns."""

    @abc.abstractmethod
    def clickable_units(self) -> list[dict[str, Any]]:
        """Return a list of unit descriptors for the visualization.

        Each descriptor is a dict with at least:
            - node_id: str  (canonical: ``scope|layer_name|unit_type|unit_index`` via
              :func:`models.unit_node_id.format_unit_node_id`; must parse with
              :func:`models.unit_node_id.parse_unit_node_id`)
            - layer_name: str
            - unit_index: int
            - unit_type: "neuron" | "channel"
        """

    @abc.abstractmethod
    def hookable_layers(self) -> dict[str, nn.Module]:
        """Return {layer_name: module} for layers where forward hooks
        should capture activations."""


def apply_xavier_init(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


def get_activation(name: str) -> nn.Module:
    name_lower = name.lower()
    if name_lower not in ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {sorted(ACTIVATION_MAP)}"
        )
    return ACTIVATION_MAP[name_lower]()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, type[AnalyzableModel]] = {}


def register_model(cls: type[AnalyzableModel]) -> type[AnalyzableModel]:
    key = cls.__name__
    _MODEL_REGISTRY[key] = cls
    return cls


def get_model(name: str, **kwargs: Any) -> AnalyzableModel:
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)
