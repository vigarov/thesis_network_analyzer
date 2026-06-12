"""MNIST continual-digit experiments - import submodules to register."""

from experiments.mnist.base import MNISTWrapper  # noqa: F401

import experiments.mnist.digitA_then_digitB_75_25  # noqa: F401
import experiments.mnist.pretrain_base_then_a_25_75_inside_train  # noqa: F401
import experiments.mnist.pretrain_a_mislabel_b_then_b  # noqa: F401
import experiments.mnist.pretrain_control_base  # noqa: F401
import experiments.mnist.control_base  # noqa: F401
import experiments.mnist.pretrain_then_shuffle_mislabel  # noqa: F401
import experiments.mnist.old_relabel_experiment  # noqa: F401
