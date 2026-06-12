"""CNN with 3 conv layers (kernel 5, 32 out-channels) + FC 32 + classification head.

UNUSED FOR NOW [TODO: remove]

"""
from typing import Any
import torch.nn as nn

from models.base import AnalyzableModel, get_activation, register_model
from models.unit_node_id import format_unit_node_id

CONV_HIDDEN = 32
FC_HIDDEN = 32
K_SIZE = 5
PADDING = 2

@register_model
class CNN3ConvK5Out32FC32(AnalyzableModel):
	"""3-conv + FC-32 network for MNIST classification.

	Architecture:
		Conv2d(1, 32, 5) -> act -> Conv2d(32, 32, 5) -> act ->
		Conv2d(32, 32, 5) -> act -> AdaptiveAvgPool -> FC(32, 32) -> act -> FC(32, 10)
	"""

	def __init__(self, activation: str = "relu", num_classes: int = 10):
		super().__init__()
		self._activation_name = activation
		self._num_classes = num_classes

		# 3 Sequentials instead of only one to select more easily clickable/hookable units
		self.conv1 = nn.Sequential(nn.Conv2d(1, CONV_HIDDEN, kernel_size=K_SIZE, padding=2), get_activation(activation))
		self.conv2 = nn.Sequential(nn.Conv2d(CONV_HIDDEN, CONV_HIDDEN, kernel_size=K_SIZE, padding=2), get_activation(activation))
		self.conv3 = nn.Sequential(nn.Conv2d(CONV_HIDDEN, CONV_HIDDEN, kernel_size=K_SIZE, padding=2), get_activation(activation))
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(nn.Linear(CONV_HIDDEN, FC_HIDDEN), get_activation(activation))
		self.head = nn.Linear(FC_HIDDEN, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.pool(x)
		x = x.view(x.size(0), -1) # flatten
		x = self.fc(x)
		return self.head(x)

	def model_id(self) -> str:
		return f"cnn_3conv_k{K_SIZE}_{CONV_HIDDEN}_fc{FC_HIDDEN}"

	def config_fields(self) -> dict[str, Any]:
		return {
			"architecture": "cnn",
			"conv_layers": 3,
			"kernel_size": K_SIZE,
			"out_channels": CONV_HIDDEN,
			"fc_hidden": FC_HIDDEN,
			"activation": self._activation_name,
			"num_classes": self._num_classes,
		}

	def clickable_units(self) -> list[dict[str, Any]]:
		units: list[dict[str, Any]] = []
		for conv_name in ("conv1", "conv2", "conv3"):
			for ch in range(CONV_HIDDEN):
				units.append({
					"node_id": format_unit_node_id(
						"cnn", conv_name, "channel", ch
					),
					"layer_name": conv_name,
					"unit_index": ch,
					"unit_type": "channel",
				})
		for neuron_idx in range(FC_HIDDEN):
			units.append({
				"node_id": format_unit_node_id("cnn", "fc", "neuron", neuron_idx),
				"layer_name": "fc",
				"unit_index": neuron_idx,
				"unit_type": "neuron",
			})
		for k in range(self._num_classes):
			units.append({
				"node_id": format_unit_node_id("cnn", "head", "neuron", k),
				"layer_name": "head",
				"unit_index": k,
				"unit_type": "neuron",
			})
		return units

	def hookable_layers(self) -> dict[str, nn.Module]:
		return {
			"conv1": self.conv1[0],
			"conv2": self.conv2[0],
			"conv3": self.conv3[0],
			"fc": self.fc[0],
			"head": self.head,
		}
