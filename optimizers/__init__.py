"""Optimizer signal extractor registry - import submodules to register."""

from optimizers.base import (
    OptimizerSignalExtractor,
    get_extractor,
    list_extractors,
    register_extractor,
)

import optimizers.sgd_extractor 
import optimizers.adam_extractor 
import optimizers.adagrad_extractor 
import optimizers.pure_shampoo_extractor 
import optimizers.grafted_shampoo_extractor 
