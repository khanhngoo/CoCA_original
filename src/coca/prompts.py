"""Prompt formatting and confidence segment parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


CONF_OPEN = "<confidence>"
CONF_CLOSE = "</confidence>"

# Paper App. B.2: fixed confidence-first system prompt.
DEFAULT_SYSTEM_PROMPT = (
    "You need to provide the answer as well as its confidence level to follow-up "
    "questions. The confidence level is a number between 0 and 1 (inclusive) enclosed "
    "within <confidence> </confidence> tags. The final format that must be followed is: "
    "<confidence> confidence level here </confidence> answer here"
)

# Paper App. B.2: task-specific user-prompt suffix that elicits a \boxed{} final answer.
USER_PROMPT_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."

CONFIDENCE_RE = re.compile(
    r"<confidence>\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*%?)\s*</confidence>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class SegmentMasks:
    confidence: list[bool]
    answer: list[bool]
    text: str
    confidence_span: tuple[int, int] | None
    answer_span: tuple[int, int]


def build_messages(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": problem.strip() + USER_PROMPT_SUFFIX},
    ]


def build_coca_prompt(problem: str) -> str:
    return (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"Problem:\n{problem.strip()}{USER_PROMPT_SUFFIX}\n\n"
        f"Response format:\n{CONF_OPEN}0.00{CONF_CLOSE}\n"
    )


def format_coca_prompt(tokenizer: Any, problem: str) -> str:
    """Use a model chat template when present, otherwise use a plain prompt."""

    messages = build_messages(problem)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return build_coca_prompt(problem)
    return build_coca_prompt(problem)


def split_confidence_answer(text: str) -> tuple[str | None, str]:
    match = CONFIDENCE_RE.search(text)
    if match is None:
        return None, text.strip()
    return match.group(1).strip(), text[match.end() :].strip()


def parse_confidence(text: str) -> float | None:
    """Parse a confidence score from a completion or raw confidence string."""

    raw, _ = split_confidence_answer(text)
    candidate = raw if raw is not None else text.strip()
    if not candidate:
        return None

    percent = candidate.endswith("%")
    candidate = candidate[:-1].strip() if percent else candidate
    try:
        value = float(candidate)
    except ValueError:
        return None

    if percent:
        value /= 100.0
    elif 1.0 < value <= 100.0:
        value /= 100.0

    if 0.0 <= value <= 1.0:
        return value
    return None


def confidence_char_span(text: str) -> tuple[int, int] | None:
    """Return the character span covering the full confidence segment."""

    match = CONFIDENCE_RE.search(text)
    if match is None:
        return None
    return match.span()


def tokens_to_confidence_close(text: str, tokenizer: Any) -> int | None:
    close_idx = text.lower().find(CONF_CLOSE)
    if close_idx < 0:
        return None
    prefix = text[: close_idx + len(CONF_CLOSE)]
    return len(tokenizer(prefix, add_special_tokens=False)["input_ids"])


def _offset_masks(
    offsets: list[tuple[int, int]],
    confidence_span_value: tuple[int, int] | None,
    answer_span: tuple[int, int],
) -> tuple[list[bool], list[bool]]:
    confidence_mask: list[bool] = []
    answer_mask: list[bool] = []
    for start, end in offsets:
        if end <= start:
            confidence_mask.append(False)
            answer_mask.append(False)
            continue
        confidence_mask.append(
            confidence_span_value is not None
            and start < confidence_span_value[1]
            and end > confidence_span_value[0]
        )
        answer_mask.append(start < answer_span[1] and end > answer_span[0])
    return confidence_mask, answer_mask


def segment_completion_token_masks(tokenizer: Any, completion_ids: list[int]) -> SegmentMasks:
    """Find confidence and answer token masks for generated completion tokens."""

    text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    conf_span = confidence_char_span(text)
    answer_start = conf_span[1] if conf_span is not None else 0
    answer_span = (answer_start, len(text))

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = list(encoded["offset_mapping"])
        if len(encoded["input_ids"]) == len(completion_ids):
            confidence_mask, answer_mask = _offset_masks(offsets, conf_span, answer_span)
            return _ensure_nonempty_masks(
                SegmentMasks(confidence_mask, answer_mask, text, conf_span, answer_span)
            )
    except Exception:
        pass

    offsets = []
    previous = ""
    for idx in range(len(completion_ids)):
        current = tokenizer.decode(completion_ids[: idx + 1], skip_special_tokens=False)
        offsets.append((len(previous), len(current)))
        previous = current
    confidence_mask, answer_mask = _offset_masks(offsets, conf_span, answer_span)
    return _ensure_nonempty_masks(
        SegmentMasks(confidence_mask, answer_mask, text, conf_span, answer_span)
    )


def _ensure_nonempty_masks(masks: SegmentMasks) -> SegmentMasks:
    if any(masks.confidence) or not masks.confidence:
        return masks

    fallback_len = min(8, len(masks.confidence))
    confidence = [idx < fallback_len for idx in range(len(masks.confidence))]
    answer = [idx >= fallback_len for idx in range(len(masks.answer))]
    return SegmentMasks(confidence, answer, masks.text, masks.confidence_span, masks.answer_span)

