"""Model and tokenizer loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from coca.config import ModelConfig


QWEN_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = True) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers before loading models.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _torch_dtype() -> Any:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _flash_attention_available() -> bool:
    """True only when CUDA + a half-precision dtype + the flash-attn package are all present."""

    import importlib.util

    import torch

    if not torch.cuda.is_available():
        return False
    if _torch_dtype() not in (torch.bfloat16, torch.float16):
        return False
    return importlib.util.find_spec("flash_attn") is not None


def load_policy_model(config: ModelConfig) -> Any:
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Install transformers, peft, and bitsandbytes before loading the policy model."
        ) from exc

    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_torch_dtype(),
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs: dict[str, Any] = dict(
        trust_remote_code=config.trust_remote_code,
        torch_dtype=_torch_dtype(),
        quantization_config=quantization_config,
        device_map="auto" if config.load_in_4bit else None,
    )
    # FlashAttention-2 speeds rollout generation (memory-bandwidth bound). Only request it on
    # CUDA with a bf16/fp16 dtype, and fall back to the default ("eager"/SDPA) attention if the
    # flash-attn package is unavailable so the run never hard-fails on attention backend.
    if _flash_attention_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(config.name, **model_kwargs)
    except (ImportError, ValueError) as exc:
        if model_kwargs.pop("attn_implementation", None) is None:
            raise
        print(f"flash_attention_2 unavailable ({exc}); falling back to default attention.")
        model = AutoModelForCausalLM.from_pretrained(config.name, **model_kwargs)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    if config.use_lora:
        if config.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=QWEN_LORA_TARGETS,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model


def load_inference_model(
    base_model: str,
    checkpoint: str | Path | None = None,
    trust_remote_code: bool = True,
) -> tuple[Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Install transformers and peft before evaluation.") from exc

    tokenizer_source = str(checkpoint) if checkpoint and Path(checkpoint).exists() else base_model
    tokenizer = load_tokenizer(tokenizer_source, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        torch_dtype=_torch_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if checkpoint and Path(checkpoint).exists() and (Path(checkpoint) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(checkpoint))
    elif checkpoint:
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            trust_remote_code=trust_remote_code,
            torch_dtype=_torch_dtype(),
            device_map="auto" if torch.cuda.is_available() else None,
        )
    model.eval()
    return model, tokenizer

