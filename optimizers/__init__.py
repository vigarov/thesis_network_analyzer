"""Optimizer signal extractor registry - import submodules to register."""

from optimizers.base import (
    OptimizerSignalExtractor,
    get_extractor,
    list_extractors,
    register_extractor,
)

import optimizers.sgd_extractor  # noqa: F401
import optimizers.adam_extractor  # noqa: F401
import optimizers.adagrad_extractor  # noqa: F401
