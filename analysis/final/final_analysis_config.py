"""Shared configuration for FINAL compare / additional plots and analyse_sim_results."""
from pathlib import Path

from analysis.dataset_filter import INCEPTION_MODEL_ID, MNIST_MODEL_ID, RESNET_MODEL_ID

DATASET_MODEL_IDS: dict[str, frozenset[str]] = {
    "mnist": frozenset({MNIST_MODEL_ID}),
    "cifar10": frozenset({INCEPTION_MODEL_ID, RESNET_MODEL_ID}),
}

RESULTS_ANALYSIS_SUBDIR: dict[str, str] = {
    "mnist": "mnist",
    "cifar10": "cifar",
}

EXPERT_MODEL_DIR_BY_DATASET: dict[str, str] = {
    "mnist": "pretrained_models/mnist/expert/!OPT/",
    "cifar10": "pretrained_models/cifar10_!ARCH/expert/!OPT/!SD/",
}

DEFAULT_SCORE_FACTORS: dict[str, float] = {
    "angle_score": 1.0,
    "length_score": 1.0,
    "n_partial_decay_score": 0.0,
    "n_active_decay_score": 0.0,
    "robustness_score": 1.0,
}

_DEFAULT_FIND_PEAKS_KWARGS: dict[str, float | int] = {
    "prominence": 0.5,
    "distance": 12,
    "height": 1.5,
}

_DEFAULT_FIND_PEAKS_KWARGS_CONTROLGLOBAL_SHUFFLE_ALL: dict[str, float | int] = {
    "prominence": 3.0,
    "distance": 10,
    "height": 1.5,
}

FIND_PEAKS_KWARGS_BY_DATASET: dict[str, dict[str, float | int]] = {
    "mnist": dict(_DEFAULT_FIND_PEAKS_KWARGS),
    "cifar10": dict(_DEFAULT_FIND_PEAKS_KWARGS),
}

FIND_PEAKS_KWARGS_CONTROLGLOBAL_SHUFFLE_ALL_BY_DATASET: dict[str, dict[str, float | int]] = {
    "mnist": dict(_DEFAULT_FIND_PEAKS_KWARGS_CONTROLGLOBAL_SHUFFLE_ALL),
    "cifar10": dict(_DEFAULT_FIND_PEAKS_KWARGS_CONTROLGLOBAL_SHUFFLE_ALL),
}

TRAIN_LOSS_LINE_ALPHA = 0.2
PEAK_MARKER_SIZE = 22.0
PEAK_MARKER_LINEWIDTH = 0.4


def dataset_key(dataset: str) -> str:
    """Normalize notebook/CLI dataset name to loader key (`mnist` or `cifar`)."""
    return dataset[:5]


def require_dataset(dataset: str) -> None:
    if dataset not in DATASET_MODEL_IDS:
        raise ValueError(
            f"DATASET must be one of {sorted(DATASET_MODEL_IDS)}, got {dataset!r}"
        )


def model_ids_for_dataset(dataset: str) -> frozenset[str]:
    require_dataset(dataset)
    return DATASET_MODEL_IDS[dataset]


def save_dir_for_dataset(project_root: Path | str, dataset: str) -> Path:
    require_dataset(dataset)
    return Path(project_root) / "results_analysis" / RESULTS_ANALYSIS_SUBDIR[dataset]


def plot_dir_for_dataset(dataset: str, *, additional: bool = False) -> str:
    require_dataset(dataset)
    key = RESULTS_ANALYSIS_SUBDIR[dataset]
    sub = "additional" if additional else ""
    if sub:
        return f"local/save/plots/FINAL/{key}/{sub}/"
    return f"local/save/plots/FINAL/{key}/"
