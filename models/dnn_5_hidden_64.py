"""DNN with 5 hidden layers of 64 units each, for MNIST classification."""
from typing import Any

import torch.nn as nn

from models.base import AnalyzableModel, get_activation, register_model

HIDDEN_SIZE = 64
N_HIDDEN = 5

@register_model
class DNN5Hidden64(AnalyzableModel):
    """Fully-connected network: 784 (28x28 flattened) -> 5 (layers) x64 -> 10 (classes)."""

    def __init__(self, activation: str = "relu", num_classes: int = 10):
        super().__init__()
        self._activation_name = activation
        self._num_classes = num_classes

        layers: list[nn.Module] = []
        in_features = 784
        for i in range(N_HIDDEN):
            layers.append(nn.Linear(in_features, HIDDEN_SIZE))
            layers.append(get_activation(activation))
            in_features = HIDDEN_SIZE
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(HIDDEN_SIZE, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        return self.head(x)

    def model_id(self) -> str:
        return f"dnn_{N_HIDDEN}x{HIDDEN_SIZE}"

    def config_fields(self) -> dict[str, Any]:
        return {
            "architecture": "dnn",
            "hidden_layers": N_HIDDEN,
            "hidden_size": HIDDEN_SIZE,
            "activation": self._activation_name,
            "num_classes": self._num_classes,
        }

    def clickable_units(self) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        for layer_idx in range(N_HIDDEN):
            layer_name = f"hidden.{layer_idx * 2}"
            for neuron_idx in range(HIDDEN_SIZE):
                units.append({
                    "node_id": f"dnn:{layer_name}:neuron_{neuron_idx}",
                    "layer_name": layer_name,
                    "unit_index": neuron_idx,
                    "unit_type": "neuron",
                })
        return units

    def hookable_layers(self) -> dict[str, nn.Module]:
        layers: dict[str, nn.Module] = {}
        for layer_idx in range(N_HIDDEN):
            name = f"hidden.{layer_idx * 2}"
            layers[name] = self.hidden[layer_idx * 2]
        return layers
