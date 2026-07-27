"""32-layer CIFAR ResNet backbone, but 
with the 5-layer DNN instead of a single-layer fully-connected classification layer at the end.

Input is 32x32x3 with channel-wise CIFAR normalization ; conv. backbone output is of dimension I = 148.
"""
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import AnalyzableModel, register_model
from models.dnn_head import (
	HIDDEN_SIZE,
	N_HIDDEN,
	HEAD_SCOPE,
	build_dnn_head,
	dnn_head_tracked_units,
	dnn_head_hookable_layers,
)
from experiments.dataset_registry import NORM_CIFAR_CHANNEL

DEFAULT_BASE_PLANES = 37
BACKBONE_INIT_SCALE = 0.5


class _BasicBlock(nn.Module):
	def __init__(self, in_planes: int, planes: int, stride: int = 1):
		super().__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=True)
		self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=True)
		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
			)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = F.relu(self.conv1(x))
		out = self.conv2(out)
		out += self.shortcut(x)
		return F.relu(out)


class _ResNetBackbone(nn.Module):
	"""32-layer CIFAR ResNet trunk ([5,5,5] blocks) producing a feature vector."""

	def __init__(self, base_planes: int = DEFAULT_BASE_PLANES):
		super().__init__()
		self.in_planes = base_planes
		self.conv1 = nn.Conv2d(3, base_planes, 3, stride=1, padding=1, bias=True)
		self.layer1 = self._make_layer(base_planes, 5, stride=1)
		self.layer2 = self._make_layer(base_planes * 2, 5, stride=2)
		self.layer3 = self._make_layer(base_planes * 4, 5, stride=2)
		self.out_features = base_planes * 4
		self._init_weights()

	def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
		strides = [stride] + [1] * (num_blocks - 1)
		layers: list[nn.Module] = []
		for s in strides:
			layers.append(_BasicBlock(self.in_planes, planes, s))
			self.in_planes = planes
		return nn.Sequential(*layers)

	def _init_weights(self, init_scale: float = BACKBONE_INIT_SCALE) -> None:
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				# Without scaling, the activations, and hence gradients, are very large. 
				# This is problematic for optimizers like SGD which don't scale gradients using other signals.
				nn.init.normal_(m.weight, 0.0, init_scale * math.sqrt(2.0 / n))
				if m.bias is not None:
					nn.init.zeros_(m.bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = F.relu(self.conv1(x))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		return F.avg_pool2d(out, out.size(3))


@register_model
class ResNetCIFAR(AnalyzableModel):
	"""32-layer CIFAR ResNet backbone (frozen during simulation) + shared DNN head."""

	def __init__(
		self,
		activation: str = "relu",
		num_classes: int = 10,
		base_planes: int = DEFAULT_BASE_PLANES,
	):
		super().__init__()
		self._activation_name = activation
		self._num_classes = num_classes
		self._base_planes = base_planes
		self._backbone_frozen = False

		self.backbone = _ResNetBackbone(base_planes=base_planes)
		self.hidden, self.head = build_dnn_head(
			self.backbone.out_features,
			hidden_size=HIDDEN_SIZE,
			n_hidden=N_HIDDEN,
			num_classes=num_classes,
			activation=activation,
		)

	def forward(self, x):
		x = self.backbone(x)
		x = x.view(x.size(0), -1)
		x = self.hidden(x)
		return self.head(x)

	def he_init_module_scale(self, module: nn.Module) -> float:
		if isinstance(module, nn.Conv2d):
			for backbone_module in self.backbone.modules():
				if module is backbone_module:
					return BACKBONE_INIT_SCALE
		return 1.0

	def freeze_backbone(self) -> None:
		self._backbone_frozen = True
		for p in self.backbone.parameters():
			p.requires_grad_(False)
		self.backbone.eval()

	def train(self, mode: bool = True):
		super().train(mode)
		if self._backbone_frozen:
			self.backbone.eval()
		return self

	def model_id(self) -> str:
		return f"resnet32_dnn_{N_HIDDEN}x{HIDDEN_SIZE}"

	def config_fields(self) -> dict[str, Any]:
		return {
			"architecture": "resnet32",
			"backbone": "resnet32_cifar",
			"backbone_features": self.backbone.out_features,
			"base_planes": self._base_planes,
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
		return dnn_head_hookable_layers(
			self.hidden, self.head, n_hidden=N_HIDDEN
		)

	@property
	def input_spec(self) -> tuple[int | None, str | None]:
		return (32, NORM_CIFAR_CHANNEL)
