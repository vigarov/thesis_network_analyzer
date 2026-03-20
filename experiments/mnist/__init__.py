"""MNIST continual-digit experiments - import submodules to register."""

from experiments.mnist.base import MNISTWrapper  # noqa: F401

import experiments.mnist.digitA_then_digitB_75_25  # noqa: F401
import experiments.mnist.pretrain25_then_digitA_then_digitB_25_75_inside_train  # noqa: F401
import experiments.mnist.pretrain_excl_then_digitA_perturb_then_digitB  # noqa: F401
