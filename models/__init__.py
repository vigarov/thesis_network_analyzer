"""Model registry - import submodules to trigger registration."""

from models.base import (
    AnalyzableModel,
    apply_model_weight_init,
    get_activation,
    get_model,
    list_models,
    register_model,
    validate_he_init,
)

import models.dnn_5_hidden_64  # noqa: F401
import models.cnn_3conv_k5_out32_then_fc32  # noqa: F401
