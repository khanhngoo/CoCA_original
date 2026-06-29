"""Dataset loading and normalization for Big-Math-RL-Verified."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


PROBLEM_KEYS: Sequence[str] = ("problem", "question", "prompt", "input")
ANSWER_KEYS: Sequence[str] = ("answer", "final_answer", "target", "solution")


@dataclass(frozen=True, slots=True)
class MathExample:
    problem: str
    answer: str
    source: str | None = None
    domain: str | None = None
    metadata: dict[str, Any] | None = None


class NormalizedMathDataset:
    """Thin adapter that normalizes rows lazily for torch DataLoader use."""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> MathExample:
        return normalize_example(self.dataset[index])


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def normalize_example(row: Mapping[str, Any]) -> MathExample:
    """Normalize a dataset row into the fields CoCA needs."""

    problem = _first_present(row, PROBLEM_KEYS)
    answer = _first_present(row, ANSWER_KEYS)
    if problem is None:
        raise KeyError(f"Could not find a problem field in columns: {sorted(row.keys())}")
    if answer is None:
        raise KeyError(f"Could not find an answer field in columns: {sorted(row.keys())}")

    metadata = {
        key: value
        for key, value in row.items()
        if key not in set(PROBLEM_KEYS) | set(ANSWER_KEYS)
    }
    return MathExample(
        problem=str(problem).strip(),
        answer=str(answer).strip(),
        source=str(row["source"]).strip() if row.get("source") is not None else None,
        domain=str(row["domain"]).strip() if row.get("domain") is not None else None,
        metadata=metadata,
    )


def _load_split(
    args: list[str],
    kwargs: dict[str, Any],
    limit: int | None,
    token: str | bool | None,
) -> Any:
    """Call datasets.load_dataset with a modern `token` kwarg + legacy fallback, then limit."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install with `pip install -e .`."
        ) from exc

    if token is not None:
        kwargs["token"] = token
    try:
        dataset = load_dataset(*args, **kwargs)
    except TypeError:
        if "token" in kwargs:
            kwargs["use_auth_token"] = kwargs.pop("token")
        dataset = load_dataset(*args, **kwargs)

    if limit is not None and hasattr(dataset, "select"):
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def load_big_math_dataset(
    dataset_name: str = "SynthLabsAI/Big-Math-RL-Verified",
    split: str = "train",
    config_name: str | None = None,
    limit: int | None = None,
    token: str | bool | None = None,
) -> Any:
    """Load Big-Math with a modern `token` kwarg and a legacy fallback."""

    dataset_path = str(dataset_name)
    if dataset_path.endswith((".jsonl", ".json")):
        args: list[str] = ["json"]
        kwargs: dict[str, Any] = {"split": split, "data_files": dataset_path}
    else:
        args = [dataset_name]
        if config_name:
            args.append(config_name)
        kwargs = {"split": split}
    return _load_split(args, kwargs, limit, token)


def iter_normalized(dataset: Iterable[Mapping[str, Any]]) -> Iterable[MathExample]:
    for row in dataset:
        yield normalize_example(row)


# Math evaluation benchmarks (App. B.3). Each entry maps a short name to the HF dataset id,
# config, split, and the row keys for the problem and gold answer.
EVAL_BENCHMARKS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "dataset_id": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "problem_key": "question",
        "answer_key": "answer",
        "answer_kind": "gsm8k",  # gold answer lives after the final "#### " marker
    },
    "math500": {
        "dataset_id": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
        "answer_kind": "plain",
    },
}


def extract_gold_answer(raw: str, answer_kind: str) -> str:
    """Pull the final gold answer string out of a benchmark row."""

    text = str(raw).strip()
    if answer_kind == "gsm8k" and "####" in text:
        text = text.split("####")[-1].strip()
    return text.replace(",", "")


def normalize_eval_row(row: Mapping[str, Any], spec: Mapping[str, Any]) -> MathExample:
    problem = row.get(spec["problem_key"])
    answer = row.get(spec["answer_key"])
    if problem is None:
        raise KeyError(f"Missing problem key {spec['problem_key']!r} in columns {sorted(row.keys())}")
    if answer is None:
        raise KeyError(f"Missing answer key {spec['answer_key']!r} in columns {sorted(row.keys())}")
    return MathExample(
        problem=str(problem).strip(),
        answer=extract_gold_answer(answer, spec["answer_kind"]),
        source=spec["dataset_id"],
    )


def load_eval_dataset(
    name: str,
    limit: int | None = None,
    token: str | bool | None = None,
) -> list[MathExample]:
    """Load and normalize a math eval benchmark into MathExample rows."""

    if name not in EVAL_BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {sorted(EVAL_BENCHMARKS)}")
    spec = EVAL_BENCHMARKS[name]

    args: list[str] = [spec["dataset_id"]]
    if spec["config"]:
        args.append(spec["config"])
    kwargs: dict[str, Any] = {"split": spec["split"]}
    dataset = _load_split(args, kwargs, limit, token)
    return [normalize_eval_row(row, spec) for row in dataset]
