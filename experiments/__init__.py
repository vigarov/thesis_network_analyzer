"""Experiment registry - import submodules to trigger registration."""

from experiments.base import (
	Experiment,
	TrialSpec,
	get_experiment,
	list_experiments,
	register_experiment,
)

import experiments.mnist  # noqa: F401  - triggers registration
import experiments_new.mnist  # noqa: F401
