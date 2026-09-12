"""Experiment id → display label mapping and sort order (no label string constants — pass `ExperimentDisplayLabels`)."""
from typing import Iterable, NamedTuple

from analysis.scoring_helpers import is_pretrain_shuffle_mislabel_experiment


class ExperimentDisplayLabels(NamedTuple):
    shuffle_all: str
    shuffle_nopre: str
    seq_nopre: str
    seq_pre: str
    shuffle_seq: str
    no_pretrain: str


_LEGACY_DISPLAY_SHUFFLE_SEQ = "Shuffle (seq)"


def normalize_experiment_display_label(
    lab: str,
    labels: ExperimentDisplayLabels,
) -> str:
    """Map legacy plain-text labels to current display strings (e.g. after LaTeX renames)."""
    s = str(lab)
    if s == _LEGACY_DISPLAY_SHUFFLE_SEQ:
        return labels.shuffle_seq
    return s


def experiment_display_sort_rank(lab: str, labels: ExperimentDisplayLabels) -> int:
    """Sort key: Cat1 (control → subset shuffle → shuffle all), then Cat2 sequence family, then legacy."""
    lab = normalize_experiment_display_label(lab, labels)
    if lab in (labels.shuffle_nopre, "Shuffle (no pretrain)", "Control Shuffle (✗PT)"):
        return 100
    # Digit-subset shuffles only ($\text{Shuffle}_{\{...\}}$), not Shuffle_all / Shuffle_seq
    if lab.startswith(r"$\text{Shuffle}_{\{"):
        return 101
    if lab == labels.shuffle_all:
        return 102
    if lab in (labels.seq_nopre, "Control (✗PT)", "Control (no pretrain)"):
        return 200
    if lab in (labels.seq_pre, "Control (✔PT)", "Control (pretrain seq)"):
        return 201
    if lab == labels.shuffle_seq:
        return 202
    if lab == "Recover":
        return 203
    if lab == "Reinforce":
        return 204
    if lab == "Control":
        return 205
    if lab == labels.no_pretrain:
        return 300
    return 999


def sort_experiment_display_names(
    labels_iter: Iterable[str],
    labels: ExperimentDisplayLabels,
) -> list[str]:
    uniq = sorted(
        {normalize_experiment_display_label(str(lab), labels) for lab in labels_iter}
    )

    def key(lab: str) -> tuple[int, str]:
        return (experiment_display_sort_rank(lab, labels), lab)

    return sorted(uniq, key=key)


def rename_experiments_for_labels(exp_name: str, labels: ExperimentDisplayLabels) -> str:
    s = str(exp_name)
    if s.startswith("digit_"):
        return labels.no_pretrain

    _c1con = "cat1_sample_shuffle_constrained_digits"
    _c1int = "cat1_sample_shuffle_interleaved_digits"
    if s.startswith(_c1con) and "_tr" in s:
        mid = s[len(_c1con) :].split("_tr", 1)[0]
        if mid:
            digits = tuple(int(p) for p in mid.split("-") if p != "")
            inner = ",".join(str(d) for d in digits)
            return r"$\text{Shuffle}_{\{" + inner + r"\}}$"
    if s.startswith(_c1int) and "_tr" in s:
        mid = s[len(_c1int) :].split("_tr", 1)[0]
        if mid:
            digits = tuple(int(p) for p in mid.split("-") if p != "")
            inner = ",".join(str(d) for d in digits)
            return r"$\text{Shuffle}_{\{" + inner + r"\}}$"
    if s.startswith("cat1_sample_shuffle_control_tr"):
        return labels.shuffle_nopre
    if s.startswith("cat1_sample_shuffle_finetune_"):
        return labels.shuffle_all

    if "pretrain_shuffle_mislabel" in s:
        if "_digits" in s:
            suffix = s.rsplit("_digits", 1)[1]
            digits = tuple(int(part) for part in suffix.split("-") if part != "")
            inner = ",".join(str(d) for d in digits)
            return r"$\text{Shuffle}_{\{" + inner + r"\}}$"
        return labels.shuffle_all

    if is_pretrain_shuffle_mislabel_experiment(s):
        return labels.shuffle_all

    if "cat2_sequence_recover" in s:
        return "Recover"
    if "cat2_sequence_reinforce" in s:
        return "Reinforce"
    if "cat2_sequence_labelperm" in s:
        return labels.shuffle_seq
    if s.startswith("cat2_sequence_pretrain_control_"):
        return labels.seq_pre
    if s.startswith("cat2_sequence_control_tr"):
        return labels.seq_nopre

    if "mislabel" in s and "shuffle_mislabel" not in s:
        return "Recover"
    if "relabel" in s:
        return "Reinforce"
    return "Control"


def two_row_cat12_from_sorted(
    exps: list[str],
    labels: ExperimentDisplayLabels,
) -> tuple[list[str], list[str], list[str]]:
    """Split sorted display labels: Cat1 (sort ranks 100-102), Cat2 sequence (200-205); remainder is `other`."""
    row1 = [e for e in exps if 100 <= experiment_display_sort_rank(e, labels) <= 102]
    row2 = [e for e in exps if 200 <= experiment_display_sort_rank(e, labels) <= 205]
    in_rows = set(row1) | set(row2)
    other = [e for e in exps if e not in in_rows]
    return row1, row2, other
