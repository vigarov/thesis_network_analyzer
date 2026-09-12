"""Base utilities and registry for models."""

import abc
import math
from typing import Any, cast

import torch.nn as nn


def parse_he_init(value: Any) -> tuple[bool, int]:
    """Parse `he_init` into (all_layers, k).

    If `all_layers` is True, every Linear/Conv2d gets He (Kaiming) init.
    Otherwise the first `k` such modules get He init; the rest use Gaussian
    noise with variance `init_epsilon` (handled by the caller).

    `"all"` or any negative integer means all layers. Non-negative `k` means
    the first `k` weight layers only.
    """
    if isinstance(value, bool):
        raise ValueError("he_init must not be a boolean")
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "all":
            return (True, 0)
        try:
            iv = int(s, 10)
        except ValueError as e:
            raise ValueError(f"Invalid he_init string: {value!r}") from e
        if iv < 0:
            return (True, 0)
        return (False, iv)
    if isinstance(value, float):
        if value < 0:
            return (True, 0)
        if not value.is_integer():
            raise ValueError(f"he_init must be integral, got {value!r}")
        return (False, int(value))
    if isinstance(value, int):
        if value < 0:
            return (True, 0)
        return (False, value)
    raise TypeError(f"he_init must be int, str, or float, got {type(value).__name__}")


_KAIMING_FOR_ACTIVATION: dict[str, tuple[str, float]] = {
    "relu": ("relu", 0.0),
    "leaky_relu": ("leaky_relu", 0.01),
    "tanh": ("tanh", 0.0),
    "sigmoid": ("sigmoid", 0.0),
    "elu": ("relu", 0.0),
    "gelu": ("relu", 0.0),
}


def _kaiming_nonlinearity(activation: str) -> tuple[str, float]:
    return _KAIMING_FOR_ACTIVATION.get(activation.lower(), ("relu", 0.0))


def apply_model_weight_init(
    model: nn.Module,
    *,
    he_init: Any,
    init_epsilon: float,
    activation: str = "relu",
) -> None:
    """Initialize Linear/Conv2d weights: first K modules Kaiming (He), rest N(0, init_epsilon).

    Traversal order matches `model.modules()` (depth-first). Biases are zero for He-init layers and
    N(0, init_epsilon) for the random-init layers.
    """
    if init_epsilon < 0:
        raise ValueError("init_epsilon must be non-negative")
    weight_modules = [
        m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))
    ]
    n = len(weight_modules)
    all_he, k_req = parse_he_init(he_init)
    if all_he:
        k_eff = n
    elif k_req > n:
        print(
            f"WARNING: he_init={k_req} exceeds number of weight layers ({n}); "
            "He-initializing all layers."
        )
        k_eff = n
    else:
        k_eff = k_req

    nonlin, neg_slope = _kaiming_nonlinearity(activation)
    std_rand = math.sqrt(init_epsilon)

    scale_getter = getattr(model, "he_init_module_scale", None)
    for i, m in enumerate(weight_modules):
        if i < k_eff:
            scale = (
                float(scale_getter(m))  # type: ignore[arg-type]
                if callable(scale_getter)
                else 1.0
            )
            if isinstance(m, nn.Conv2d) and scale != 1.0:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=scale * math.sqrt(2.0 / n))
            else:
                nn.init.kaiming_normal_(
                    m.weight,
                    a=neg_slope,
                    mode="fan_in",
                    nonlinearity=cast(Any, nonlin),
                )
                if scale != 1.0:
                    m.weight.mul_(scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            nn.init.normal_(m.weight, mean=0.0, std=std_rand)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=std_rand)


class AnalyzableModel(nn.Module, abc.ABC):
    """Base class for models used in the analysis framework.

    Subclasses declare which layers contain the "tracked units" whose
    activations, weights, and optimizer signals are captured during training,
    and provide metadata about each unit.
    """

    @abc.abstractmethod
    def model_id(self) -> str:
        """Unique identifier used in the results directory path."""

    @abc.abstractmethod
    def config_fields(self) -> dict[str, Any]:
        """Model-specific config fields frozen across reruns."""

    @abc.abstractmethod
    def tracked_units(self) -> list[dict[str, Any]]:
        """Return a list of unit (neurons) to be captured/tracked.

        Each descriptor is a dict with at least:
            - node_id: str  (canonical: `scope|layer_name|unit_type|unit_index` via
              `models.unit_node_id.format_unit_node_id`; must parse with
              `models.unit_node_id.parse_unit_node_id`)
            - layer_name: str
            - unit_index: int
            - unit_type: "neuron" | "channel"
        """

    @abc.abstractmethod
    def hookable_layers(self) -> dict[str, nn.Module]:
        """Return {layer_name: module} for layers where forward hooks
        should capture activations."""

    @property
    @abc.abstractmethod
    def input_spec(self) -> tuple[int | None, str | None]:
        """Returns model-specific `(input_size, normalization)` for the dataset normalization.

        `None` for either field means the dataset default applies (e.g. MNIST
        train-statistics normalization without resizing).
        """

    def freeze_backbone(self) -> None:
        """For more complex models (e.g.: Inception/ResNet), freeze the convolutional extractors.
        noop otherwise
        """
        return None

    def he_init_module_scale(self, module: nn.Module) -> float:
        """Per-module scale applied during He init (1.0 = standard Kaiming)."""
        return 1.0


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
