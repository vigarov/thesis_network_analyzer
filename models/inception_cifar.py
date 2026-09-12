"""Small Inception backbone (Zhang et al. 2017, Fig.3), but 
with the 5-layer DNN instead of a single-layer fully-connected classification layer at the end.

Input is 28x28x3 (cropped-image at the center for CIFAR10) with per-image whitening ; conv. backbone output is of dimension I = 336.
"""
import math
from typing import Any

import torch
import torch.nn as nn

from models.base import AnalyzableModel, register_model
from models.dnn_head import (
    HIDDEN_SIZE,
    N_HIDDEN,
    HEAD_SCOPE,
    build_dnn_head,
    dnn_head_tracked_units,
    dnn_head_hookable_layers,
)
from experiments.dataset_registry import NORM_PER_IMAGE_WHITENING

BACKBONE_FEATURES = 336


class _ConvModule(nn.Module):
    def __init__(self, in_channel, C=96, K=3, S=1, padding:str|int="same"):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, C, (K, K), (S, S), padding=padding, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class _InceptionModule(nn.Module):
    def __init__(self, in_channel, Ch1, Ch2):
        super().__init__()
        self.conv_module_1 = _ConvModule(in_channel, Ch1, 1, 1)
        self.conv_module_2 = _ConvModule(in_channel, Ch2, 3, 1)

    def forward(self, x):
        return torch.cat([self.conv_module_1(x), self.conv_module_2(x)], 1)


class _DownsampleModule(nn.Module):
    def __init__(self, in_channel, Ch3):
        super().__init__()
        self.conv_module = _ConvModule(in_channel, Ch3, 3, 2, padding=1)
        self.max_pool = nn.MaxPool2d((3, 3), stride=2, ceil_mode=True)

    def forward(self, x):
        return torch.cat([self.conv_module(x), self.max_pool(x)], 1)


class _InceptionBackbone(nn.Module):
    """20-layer small Inception trunk producing a 336-dim feature vector."""

    def __init__(self, init_scale: float = 0.5):
        super().__init__()
        self.conv_module = _ConvModule(3, 96, 3, 1, padding=1)
        self.Inception_module_1 = _InceptionModule(96, 32, 32)
        self.Inception_module_2 = _InceptionModule(64, 32, 48)
        self.downsample_module_1 = _DownsampleModule(80, 80)
        self.Inception_module_3 = _InceptionModule(160, 112, 48)
        self.Inception_module_4 = _InceptionModule(160, 96, 64)
        self.Inception_module_5 = _InceptionModule(160, 80, 80)
        self.Inception_module_6 = _InceptionModule(160, 48, 96)
        self.downsample_module_2 = _DownsampleModule(144, 96)
        self.Inception_module_7 = _InceptionModule(240, 176, 160)
        self.Inception_module_8 = _InceptionModule(336, 176, 160)
        self.mean_pooling = nn.AvgPool2d((7, 7))
        self._init_weights(init_scale)

    def _init_weights(self, init_scale: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, init_scale * math.sqrt(2.0 / n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_module(x)
        x = self.Inception_module_1(x)
        x = self.Inception_module_2(x)
        x = self.downsample_module_1(x)
        x = self.Inception_module_3(x)
        x = self.Inception_module_4(x)
        x = self.Inception_module_5(x)
        x = self.Inception_module_6(x)
        x = self.downsample_module_2(x)
        x = self.Inception_module_7(x)
        x = self.Inception_module_8(x)
        return self.mean_pooling(x)


@register_model
class InceptionCIFAR(AnalyzableModel):
    """Small-Inception backbone (frozen during simulation) + shared 5x64 DNN head."""

    def __init__(self, activation: str = "relu", num_classes: int = 10):
        super().__init__()
        self._activation_name = activation
        self._num_classes = num_classes
        self._backbone_frozen = False

        self.backbone = _InceptionBackbone()
        self.hidden, self.head = build_dnn_head(
            BACKBONE_FEATURES,
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
        return f"inception_small_dnn_{N_HIDDEN}x{HIDDEN_SIZE}"

    def config_fields(self) -> dict[str, Any]:
        return {
            "architecture": "inception_small",
            "backbone": "inception_small_cifar",
            "backbone_features": BACKBONE_FEATURES,
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
        return (28, NORM_PER_IMAGE_WHITENING)
