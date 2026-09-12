"""Per-architecture plot iteration helpers for CIFAR FINAL notebooks."""
from collections.abc import Iterator
from typing import Any

from analysis.dataset_filter import (
    INCEPTION_MODEL_ID,
    RESNET_MODEL_ID,
    mid_to_arch_slug,
)
from analysis.final.final_analysis_config import dataset_key, model_ids_for_dataset

CIFAR_MODELID_ORDER = (INCEPTION_MODEL_ID, RESNET_MODEL_ID)

ResultsTree = dict[str, dict[str, dict[str, list[str]]]]
AllResults = dict[tuple[str, str, str, str], dict[str, Any]]

# (model_id, arch_slug, filename_suffix, title_suffix)
PlotContext = tuple[str, str | None, str, str]


def _arch_title_label(arch_slug: str | None) -> str:
    if arch_slug is None:
        return ""
    return arch_slug.replace("_", " ").title()


def _context_for_mid(model_id: str) -> PlotContext:
    if model_id.startswith("inception") or model_id.startswith("resnet"):
        slug = mid_to_arch_slug(model_id)
        label = _arch_title_label(slug)
        return (model_id, slug, f"_{slug}", f" — {label}")
    return (model_id, None, "", "")


def model_ids_in_results(all_results: AllResults) -> list[str]:
    return sorted({model_id for (_, model_id, _, _) in all_results})


def iter_model_plot_contexts(dataset: str, all_results: AllResults) -> Iterator[PlotContext]:
    """MNIST: one context. CIFAR: inception then resnet (only if present in results).
    We chose to make this an iterator so that we can keep the original `for` loop syntax in the various analysis notebooks.
    Yields `(model_id, arch_slug, filename_suffix, title_suffix)`.
    """
    key = dataset_key(dataset)
    if key == "cifar":
        present = set(model_ids_in_results(all_results))
        for model_id in CIFAR_MODELID_ORDER:
            if model_id in present:
                yield _context_for_mid(model_id)
        return

    expected = model_ids_for_dataset(dataset)
    present = [model_id for model_id in model_ids_in_results(all_results) if model_id in expected]
    if not present:
        for model_id in expected:
            yield _context_for_mid(model_id)
        return
    for model_id in present:
        yield _context_for_mid(model_id)


def filter_all_results(all_results: AllResults, model_id: str) -> AllResults:
    return {key: payload for key, payload in all_results.items() if key[1] == model_id}


def filter_tree(tree: ResultsTree, model_id: str) -> ResultsTree:
    out: ResultsTree = {}
    for eid, models in tree.items():
        if model_id not in models:
            continue
        out[eid] = {model_id: models[model_id]}
    return out


def with_arch_filename(filename: str, filename_suffix: str) -> str:
    return insert_filename_suffix(filename, filename_suffix)


def with_arch_title(title: str, title_suffix: str) -> str:
    if not title_suffix:
        return title
    return f"{title}{title_suffix}"


def insert_filename_suffix(filename: str, suffix: str) -> str:
    """Insert `suffix` before the file extension (e.g. `_inception`)."""
    if not suffix:
        return filename
    if "." in filename:
        stem, ext = filename.rsplit(".", 1)
        return f"{stem}{suffix}.{ext}"
    return f"{filename}{suffix}"
