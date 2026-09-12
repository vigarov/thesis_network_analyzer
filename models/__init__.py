"""Model registry - import submodules to trigger registration."""

from models.base import (
    AnalyzableModel,
    apply_model_weight_init,
    get_activation,
    get_model,
    list_models,
    register_model,
    parse_he_init,
)

import models.dnn_5_hidden_64
import models.inception_cifar
import models.resnet_cifar
