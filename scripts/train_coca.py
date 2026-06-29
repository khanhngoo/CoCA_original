"""Train CoCA segmented GRPO on Big-Math-RL-Verified."""

from __future__ import annotations

import argparse
from pathlib import Path

from coca.config import DatasetConfig, GenerationConfig, ModelConfig, TrainingConfig
from coca.data import load_big_math_dataset
from coca.model_utils import load_policy_model, load_tokenizer
from coca.trainer import CoCATrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", default="SynthLabsAI/Big-Math-RL-Verified")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="outputs/coca-qwen2.5-1.5b-lora")
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--wandb-project", default=None, help="wandb project name; omit to disable logging")
    parser.add_argument("--wandb-run-name", default=None, help="wandb run name (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mixed_precision = None if args.mixed_precision in (None, "no") else args.mixed_precision
    dataset_config = DatasetConfig(
        name=args.dataset,
        split=args.split,
        config_name=args.dataset_config,
        limit=args.limit_train_samples,
        token=args.hf_token,
    )
    model_config = ModelConfig(
        name=args.model,
        load_in_4bit=args.load_in_4bit,
        use_lora=not args.no_lora,
    )
    training_config = TrainingConfig(
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
        group_size=args.group_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        clip_epsilon=args.clip_epsilon,
        kl_beta=args.kl_beta,
        seed=args.seed,
        mixed_precision=mixed_precision,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )

    dataset = load_big_math_dataset(
        dataset_name=dataset_config.name,
        split=dataset_config.split,
        config_name=dataset_config.config_name,
        limit=dataset_config.limit,
        token=dataset_config.token,
    )
    tokenizer = load_tokenizer(model_config.name, trust_remote_code=model_config.trust_remote_code)
    model = load_policy_model(model_config)
    trainer = CoCATrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        train_config=training_config,
        generation_config=generation_config,
    )
    trainer.train()


if __name__ == "__main__":
    main()

