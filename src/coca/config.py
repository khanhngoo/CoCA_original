"""Configuration objects for CoCA training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetConfig:
    name: str = "SynthLabsAI/Big-Math-RL-Verified"
    split: str = "train"
    config_name: str | None = None
    limit: int | None = None
    token: str | bool | None = None


@dataclass(slots=True)
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    trust_remote_code: bool = True
    load_in_4bit: bool = False
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 1.0
    # Paper (App. B.1) uses no nucleus/top-k sampling during training; 1.0 disables truncation.
    top_p: float = 1.0
    do_sample: bool = True


@dataclass(slots=True)
class TrainingConfig:
    output_dir: Path = Path("outputs/coca-qwen2.5-1.5b-lora")
    max_steps: int = 1000
    per_device_batch_size: int = 1
    group_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-6  # paper App. B.1
    weight_decay: float = 0.0
    warmup_steps: int = 25
    max_grad_norm: float = 1.0
    clip_epsilon: float = 0.2
    advantage_epsilon: float = 1e-8
    kl_beta: float = 0.0  # paper CoCA objective (Eq. 10-11) omits KL; set >0 to add GRPO KL (Eq. 3)
    mixed_precision: str | None = None
    logging_steps: int = 1
    save_steps: int = 100
    seed: int = 42
    invalid_confidence_reward: float = -1.0
    wandb_project: str | None = None
    wandb_run_name: str | None = None


@dataclass(slots=True)
class EvalConfig:
    checkpoint: Path | None = None
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    split: str = "train"
    limit: int = 200
    max_new_tokens: int = 512
    temperature: float = 0.0


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert nested dataclasses into JSON-friendly dictionaries."""

    data = asdict(obj)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data

