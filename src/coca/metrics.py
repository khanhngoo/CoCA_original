"""Evaluation metrics for confidence-first outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    accuracy: float
    parse_success_rate: float
    ece: float
    brier: float
    auroc: float | None
    count: int


def brier_score(confidences: Sequence[float], labels: Sequence[bool]) -> float:
    if len(confidences) != len(labels):
        raise ValueError("confidences and labels must have the same length")
    if not confidences:
        return 0.0
    return sum((conf - float(label)) ** 2 for conf, label in zip(confidences, labels)) / len(
        confidences
    )


def expected_calibration_error(
    confidences: Sequence[float],
    labels: Sequence[bool],
    bins: int = 10,
) -> float:
    if len(confidences) != len(labels):
        raise ValueError("confidences and labels must have the same length")
    if not confidences:
        return 0.0

    total = len(confidences)
    ece = 0.0
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        if idx == bins - 1:
            members = [
                (conf, label)
                for conf, label in zip(confidences, labels)
                if low <= conf <= high
            ]
        else:
            members = [
                (conf, label)
                for conf, label in zip(confidences, labels)
                if low <= conf < high
            ]
        if not members:
            continue
        avg_conf = sum(conf for conf, _ in members) / len(members)
        avg_acc = sum(float(label) for _, label in members) / len(members)
        ece += (len(members) / total) * abs(avg_conf - avg_acc)
    return ece


def auroc_score(confidences: Sequence[float], labels: Sequence[bool]) -> float | None:
    if len(confidences) != len(labels):
        raise ValueError("confidences and labels must have the same length")
    positives = sum(bool(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(confidences, labels), key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(ranked):
        jdx = idx + 1
        while jdx < len(ranked) and ranked[jdx][0] == ranked[idx][0]:
            jdx += 1
        avg_rank = (idx + 1 + jdx) / 2.0
        rank_sum += avg_rank * sum(bool(label) for _, label in ranked[idx:jdx])
        idx = jdx

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def compute_eval_metrics(
    confidences: Sequence[float | None],
    labels: Sequence[bool],
    bins: int = 10,
) -> EvalMetrics:
    if len(confidences) != len(labels):
        raise ValueError("confidences and labels must have the same length")
    valid_pairs = [(conf, label) for conf, label in zip(confidences, labels) if conf is not None]
    valid_confidences = [conf for conf, _ in valid_pairs]
    valid_labels = [label for _, label in valid_pairs]

    accuracy = sum(float(label) for label in labels) / len(labels) if labels else 0.0
    parse_success_rate = len(valid_pairs) / len(labels) if labels else 0.0
    return EvalMetrics(
        accuracy=accuracy,
        parse_success_rate=parse_success_rate,
        ece=expected_calibration_error(valid_confidences, valid_labels, bins=bins),
        brier=brier_score(valid_confidences, valid_labels),
        auroc=auroc_score(valid_confidences, valid_labels),
        count=len(labels),
    )

