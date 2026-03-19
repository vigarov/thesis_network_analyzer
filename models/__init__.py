"""Model registry - import submodules to trigger registration."""

from models.base import (
    AnalyzableModel,
    apply_xavier_init,
    get_activation,
    get_model,
    list_models,
    register_model,
)

import models.dnn_5_hidden_64  # noqa: F401
import models.cnn_3conv_k5_out64_then_fc64  # noqa: F401
