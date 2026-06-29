"""CoCA reward computation for confidence-first math rollouts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Sequence

from coca.prompts import parse_confidence, split_confidence_answer


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
FINAL_RE = re.compile(
    r"(?:final answer|answer is|therefore|thus)\s*:?\s*([^\n]+)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


@dataclass(frozen=True, slots=True)
class AnswerScore:
    correct: bool
    predicted: str
    gold: str
    method: str


@dataclass(frozen=True, slots=True)
class GroupRewards:
    confidences: list[float | None]
    answer_rewards: list[float]
    confidence_rewards: list[float]
    answer_advantages: list[float]
    confidence_advantages: list[float]
    gesr: float


def extract_final_answer(text: str) -> str:
    """Extract a likely final answer from a generated answer segment."""

    stripped = text.strip()
    if not stripped:
        return ""

    boxed = BOXED_RE.findall(stripped)
    if boxed:
        return boxed[-1].strip()

    final_matches = FINAL_RE.findall(stripped)
    if final_matches:
        return final_matches[-1].strip().rstrip(".")

    numbers = NUMBER_RE.findall(stripped)
    if numbers:
        return numbers[-1].replace(",", "")

    return stripped.splitlines()[-1].strip().rstrip(".")


def normalize_answer(text: str) -> str:
    text = extract_final_answer(text)
    text = text.strip().strip("$").strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    while text.endswith((".", ";", ",")):
        text = text[:-1]
    return text.lower()


def score_math_answer(generated_answer: str, gold_answer: str) -> AnswerScore:
    """Score a generated answer against gold with math-verify, then exact fallback."""

    predicted = extract_final_answer(generated_answer)
    try:
        from math_verify import parse, verify

        parsed_gold = parse(gold_answer)
        parsed_predicted = parse(predicted)
        if parsed_gold and parsed_predicted and verify(parsed_gold, parsed_predicted):
            return AnswerScore(True, predicted, gold_answer, "math_verify")
    except Exception:
        pass

    correct = normalize_answer(predicted) == normalize_answer(gold_answer)
    return AnswerScore(correct, predicted, gold_answer, "normalized_exact")


def score_completion(completion: str, gold_answer: str) -> tuple[float | None, AnswerScore]:
    confidence_text, answer_text = split_confidence_answer(completion)
    confidence = parse_confidence(confidence_text or "")
    return confidence, score_math_answer(answer_text, gold_answer)


def normalize_advantages(values: Sequence[float], eps: float = 1e-8) -> list[float]:
    if not values:
        return []
    mean = fmean(values)
    variance = fmean((value - mean) ** 2 for value in values)
    std = math.sqrt(variance)
    if std <= eps:
        return [0.0 for _ in values]
    return [(value - mean) / (std + eps) for value in values]


def compute_group_rewards(
    confidences: Sequence[float | None],
    correctness: Sequence[bool],
    invalid_confidence_reward: float = -1.0,
    eps: float = 1e-8,
) -> GroupRewards:
    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must have the same length")
    if not confidences:
        raise ValueError("at least one rollout is required")

    answer_rewards = [1.0 if value else 0.0 for value in correctness]
    gesr = fmean(answer_rewards)
    confidence_rewards = [
        -((confidence - gesr) ** 2) if confidence is not None else invalid_confidence_reward
        for confidence in confidences
    ]
    return GroupRewards(
        confidences=list(confidences),
        answer_rewards=answer_rewards,
        confidence_rewards=confidence_rewards,
        answer_advantages=normalize_advantages(answer_rewards, eps=eps),
        confidence_advantages=normalize_advantages(confidence_rewards, eps=eps),
        gesr=gesr,
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return fmean(values) if values else 0.0

