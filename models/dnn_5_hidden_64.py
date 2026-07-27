"""DNN with 5 hidden layers of 64 units each, for MNIST classification."""
from typing import Any

import torch.nn as nn

from models.base import AnalyzableModel, register_model
from models.dnn_head import (
	HEAD_SCOPE,
	build_dnn_head,
	dnn_head_tracked_units,
	dnn_head_hookable_layers,
)

HIDDEN_SIZE = 64
N_HIDDEN = 5
FLATTEN_IMAGE_SIZE = 28 * 28

@register_model
class DNN5Hidden64(AnalyzableModel):
	"""Fully-connected network: 784 (28x28 flattened) -> 5 (layers) x64 -> 10 (classes)."""

	def __init__(self, activation: str = "relu", num_classes: int = 10):
		super().__init__()
		self._activation_name = activation
		self._num_classes = num_classes

		self.hidden, self.head = build_dnn_head(
			FLATTEN_IMAGE_SIZE,
			hidden_size=HIDDEN_SIZE,
			n_hidden=N_HIDDEN,
			num_classes=num_classes,
			activation=activation,
		)

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

	def tracked_units(self) -> list[dict[str, Any]]:
		return dnn_head_tracked_units(
			HEAD_SCOPE,
			n_hidden=N_HIDDEN,
			hidden_size=HIDDEN_SIZE,
			num_classes=self._num_classes,
		)

	def hookable_layers(self) -> dict[str, nn.Module]:
		return dnn_head_hookable_layers(self.hidden, self.head, n_hidden=N_HIDDEN)

	@property
	def input_spec(self) -> tuple[int | None, str | None]:
		return (None, None)
