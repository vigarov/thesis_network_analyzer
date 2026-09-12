# Keys omitted from the training fingerprint and from training-config equality checks.
# `framework_version` is ignored if present in an old config.json on disk.
FINGERPRINT_EXCLUDE_KEYS = frozenset(
    {
        "internal_keep_tensors",
        "device",
        "save_model_cp",
        "save_model",
        "force",
        "training_config_fingerprint",
        "framework_version",
    }
)

# placeholder expanded to all available optimizers/experiments (depending on context)
ALL_AVAILABLE_SELECTION = "all"

# Default dataset when a config/CLI does not specify 
DEFAULT_DATASET = "mnist"

# Human-friendly optimizer CLI names -> registry extractor class name.
# Kept in sync with `OPTIMIZER_SHORTHAND` in `run_simulation` (derived from this dict).
INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS: dict[str, str] = {
    "sgd": "SGDExtractor",
    "adam": "AdamExtractor",
    "adagrad": "AdaGradExtractor",
    "pure_shampoo": "PureShampooExtractor",
    "grafted_shampoo": "GraftedShampooExtractor",
}

# Checkpoint filename inside each `use_initial_model` bundle subfolder (try first match).
INITIAL_MODEL_CHECKPOINT_FILENAMES: tuple[str, ...] = ("model.pt", "model.pth")

# Required optimizer state filename inside each `use_initial_model` bundle subfolder.
INITIAL_MODEL_OPTIMIZER_STATE_FILENAME = "optimizer.pt"

# Default model weight-init seeds when `--multi_seeds` is passed without a value.
DEFAULT_MULTI_SEEDS: list[int] = [6, 7, 14, 30, 31, 35, 51, 68, 90, 96]

# Default base learning rates when `--multi_lr` is passed without a value
# DEFAULT_LR * [{0.5, 1} * {10^(-1) , 10)}]
DEFAULT_MULTI_LRS: list[float] = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

# Keys omitted from pretrain config fingerprint and equality checks.
PRETRAIN_FINGERPRINT_EXCLUDE_KEYS = frozenset(
    {
        "device",
        "save",
        "force",
        "out_dir",
        "pretrain_config_fingerprint",
    }
)

# Pretrain fields that match to skip re-training
PRETRAIN_REUSE_KEYS: tuple[str, ...] = (
    "base_lr",
    "train_k_samples",
    "model_class",
    "dataset",
)

# Sentinel for argparse: bare `--multi_seeds` (no CSV value).
MULTI_SEEDS_USE_DEFAULT = "<DEFAULT_MULTI_SEEDS>"

# Sentinel for argparse: bare `--multi_lr` (no CSV value).
MULTI_LRS_USE_DEFAULT = "<DEFAULT_MULTI_LRS>"

# Threshold-based pretraining defaults (see compute_results/pretrain.py).
BS_LINEAR_BRACKET_THRESHOLD = 80
DEFAULT_TRAIN_K_SAMPLES = 1000
DEFAULT_THRESHOLD_ACC = 0.8
EXPERT_TRAIN_K_SAMPLES = 20000
EXPERT_TRAIN_K_SAMPLES_CIFAR10 = 40000
EXPERT_THRESHOLD_ACC = 0.95
EXPERT_THRESHOLD_ACC_CIFAR10 = 0.7
