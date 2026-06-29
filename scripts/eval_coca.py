"""Evaluate a CoCA checkpoint on Big-Math style examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coca.data import iter_normalized, load_big_math_dataset, load_eval_dataset
from coca.metrics import compute_eval_metrics
from coca.model_utils import load_inference_model
from coca.prompts import format_coca_prompt, parse_confidence, tokens_to_confidence_close
from coca.rewards import score_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--benchmark",
        choices=("bigmath", "gsm8k", "math500"),
        default="bigmath",
        help="Eval set. 'bigmath' uses --dataset/--split; others load fixed math benchmarks.",
    )
    parser.add_argument("--dataset", default="SynthLabsAI/Big-Math-RL-Verified")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from tqdm.auto import tqdm

    if args.benchmark == "bigmath":
        raw = load_big_math_dataset(
            dataset_name=args.dataset,
            split=args.split,
            config_name=args.dataset_config,
            limit=args.limit,
            token=args.hf_token,
        )
        examples = list(iter_normalized(raw))
    else:
        examples = load_eval_dataset(args.benchmark, limit=args.limit, token=args.hf_token)

    model, tokenizer = load_inference_model(args.base_model, args.checkpoint)
    device = next(model.parameters()).device

    confidences: list[float | None] = []
    labels: list[bool] = []
    ttc_values: list[int] = []
    generations: list[dict[str, object]] = []

    for example in tqdm(examples, desc="evaluating"):
        prompt = format_coca_prompt(tokenizer, example.problem)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        prompt_len = int(encoded["attention_mask"].sum().item())
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[0, prompt_len:].tolist()
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
        confidence, score = score_completion(completion, example.answer)
        confidences.append(confidence)
        labels.append(score.correct)
        close_tokens = tokens_to_confidence_close(completion, tokenizer)
        if close_tokens is not None:
            ttc_values.append(close_tokens)
        generations.append(
            {
                "problem": example.problem,
                "gold": example.answer,
                "completion": completion,
                "confidence": confidence,
                "correct": score.correct,
                "predicted": score.predicted,
            }
        )

    metrics = compute_eval_metrics(confidences, labels)
    payload = {
        "accuracy": metrics.accuracy,
        "parse_success_rate": metrics.parse_success_rate,
        "ece": metrics.ece,
        "brier": metrics.brier,
        "auroc": metrics.auroc,
        "count": metrics.count,
        "avg_tokens_to_confidence": sum(ttc_values) / len(ttc_values) if ttc_values else None,
    }
    print(json.dumps(payload, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps({"metrics": payload, "generations": generations}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
