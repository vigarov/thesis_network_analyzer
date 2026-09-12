"""Shared 5x64 DNN used in both the standalon DNN (for MNIST) 
and the DNN classification head (for Inception and ResNet on CIFAR10)
"""
from typing import Any

import torch.nn as nn

from models.base import get_activation
from models.unit_node_id import format_unit_node_id

HIDDEN_SIZE = 64
N_HIDDEN = 5

# Scope used in unit node-ids for the DNN classification head (kept as "dnn"
# so head units are represented identically regardless of the backbone in front).
HEAD_SCOPE = "dnn"


def build_dnn_head(
    in_features: int,
    *,
    hidden_size: int = HIDDEN_SIZE,
    n_hidden: int = N_HIDDEN,
    num_classes: int = 10,
    activation: str = "relu",
) -> tuple[nn.Sequential, nn.Linear]:
    """Create the common "5 hidden layers and out classification head" """
    layers: list[nn.Module] = []
    f = in_features
    for _ in range(n_hidden):
        layers.append(nn.Linear(f, hidden_size))
        layers.append(get_activation(activation))
        f = hidden_size
    hidden = nn.Sequential(*layers)
    head = nn.Linear(hidden_size, num_classes)
    return hidden, head


def dnn_head_tracked_units(
    scope: str = HEAD_SCOPE,
    *,
    n_hidden: int = N_HIDDEN,
    hidden_size: int = HIDDEN_SIZE,
    num_classes: int = 10,
) -> list[dict[str, Any]]:
    """Tracked units for the DNN (hidden neurons + output neurons)."""
    units: list[dict[str, Any]] = []
    for layer_idx in range(n_hidden):
        layer_name = f"hidden.{layer_idx * 2}"
        for neuron_idx in range(hidden_size):
            units.append({
                "node_id": format_unit_node_id(scope, layer_name, "neuron", neuron_idx),
                "layer_name": layer_name,
                "unit_index": neuron_idx,
                "unit_type": "neuron",
            })
    for k in range(num_classes):
        units.append({
            "node_id": format_unit_node_id(scope, "head", "neuron", k),
            "layer_name": "head",
            "unit_index": k,
            "unit_type": "neuron",
        })
    return units


def dnn_head_hookable_layers(
    hidden: nn.Sequential,
    head: nn.Linear,
    *,
    n_hidden: int = N_HIDDEN,
) -> dict[str, nn.Module]:
    """Forward-hook targets for the DNN (hiiden layers + output)."""
    layers: dict[str, nn.Module] = {}
    for layer_idx in range(n_hidden):
        name = f"hidden.{layer_idx * 2}"
        layers[name] = hidden[layer_idx * 2]
    layers["head"] = head
    return layers
