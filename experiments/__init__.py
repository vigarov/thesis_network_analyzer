"""Experiment registry - import submodules to trigger registration."""

from experiments.base import (
    Experiment,
    StageSpec,
    get_experiment,
    list_experiments,
    register_experiment,
)

import experiments.mnist  # noqa: F401  - triggers registration
