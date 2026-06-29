"""Segmented GRPO trainer for confidence-first CoCA rollouts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from coca.config import GenerationConfig, TrainingConfig, dataclass_to_dict
from coca.data import MathExample, NormalizedMathDataset
from coca.prompts import format_coca_prompt, segment_completion_token_masks
from coca.rewards import compute_group_rewards, mean, score_completion


@dataclass(slots=True)
class RolloutSample:
    full_ids: list[int]
    completion_ids: list[int]
    confidence_mask: list[bool]
    answer_mask: list[bool]
    confidence: float | None
    correct: bool
    answer_advantage: float
    confidence_advantage: float
    answer_reward: float
    confidence_reward: float
    gesr: float
    text: str
    # Per-token logprobs over `full_ids[1:]` captured under the rollout policy (theta_old)
    # and the reference policy (LoRA adapters disabled). These freeze the sampling-time
    # distribution so the PPO ratio and KL term are well-defined at update time.
    old_logprobs: list[float]
    ref_logprobs: list[float]


class CoCATrainer:
    """A compact trainer implementing the paper's segmented GRPO objective."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        train_config: TrainingConfig,
        generation_config: GenerationConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = NormalizedMathDataset(dataset)
        self.train_config = train_config
        self.generation_config = generation_config
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def train(self) -> None:
        import torch
        from accelerate import Accelerator
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from tqdm.auto import trange
        from transformers import get_scheduler, set_seed

        set_seed(self.train_config.seed)
        random.seed(self.train_config.seed)
        output_dir = Path(self.train_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_configs(output_dir)

        use_wandb = self.train_config.wandb_project is not None
        if use_wandb and accelerator.is_local_main_process:
            try:
                import wandb
                wandb.init(
                    project=self.train_config.wandb_project,
                    name=self.train_config.wandb_run_name,
                    config={
                        **dataclass_to_dict(self.train_config),
                        **dataclass_to_dict(self.generation_config),
                    },
                )
            except ImportError:
                use_wandb = False
                print("wandb not installed — run `pip install wandb` to enable logging.")

        accelerator = Accelerator(
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            mixed_precision=self.train_config.mixed_precision,
        )
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.train_config.per_device_batch_size,
            shuffle=True,
            collate_fn=lambda batch: batch,
        )
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.train_config.warmup_steps,
            num_training_steps=self.train_config.max_steps,
        )
        self.model, optimizer, dataloader, scheduler = accelerator.prepare(
            self.model,
            optimizer,
            dataloader,
            scheduler,
        )

        data_iter = _cycle(dataloader)
        progress = trange(self.train_config.max_steps, disable=not accelerator.is_local_main_process)
        for step in progress:
            batch: list[MathExample] = next(data_iter)
            with accelerator.accumulate(self.model):
                samples = self._rollout_batch(batch, accelerator.device)
                loss, stats = self._loss(samples, accelerator.device)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_config.max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process and step % self.train_config.logging_steps == 0:
                stats["loss"] = float(loss.detach().cpu())
                progress.set_postfix({key: round(value, 4) for key, value in stats.items()})
                if use_wandb:
                    import wandb
                    wandb.log(stats, step=step)

            if (
                accelerator.is_local_main_process
                and self.train_config.save_steps > 0
                and (step + 1) % self.train_config.save_steps == 0
            ):
                self._save_checkpoint(accelerator, output_dir / f"step-{step + 1}")

        if accelerator.is_local_main_process:
            self._save_checkpoint(accelerator, output_dir)
            if use_wandb:
                import wandb
                wandb.finish()

    def _rollout_batch(self, batch: list[MathExample], device: Any) -> list[RolloutSample]:
        import torch

        self.model.eval()
        all_samples: list[RolloutSample] = []
        for example in batch:
            prompt = format_coca_prompt(self.tokenizer, example.problem)
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_len = int(encoded["attention_mask"].sum().item())
            gen_kwargs: dict[str, Any] = dict(
                max_new_tokens=self.generation_config.max_new_tokens,
                do_sample=self.generation_config.do_sample,
                temperature=self.generation_config.temperature,
                num_return_sequences=self.train_config.group_size,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            # Paper trains without nucleus/top-k sampling to preserve the model's intrinsic
            # output distribution; only pass top_p when it actually truncates (< 1.0).
            if self.generation_config.top_p < 1.0:
                gen_kwargs["top_p"] = self.generation_config.top_p
            with torch.no_grad():
                # Re-enable KV cache for generation (gradient_checkpointing disables it globally).
                _set_use_cache(self.model, True)
                generated = self.model.generate(**encoded, **gen_kwargs)
                _set_use_cache(self.model, False)

            group_partials = []
            for sequence in generated:
                ids = sequence.tolist()
                completion_ids = ids[prompt_len:]
                masks = segment_completion_token_masks(self.tokenizer, completion_ids)
                confidence, answer_score = score_completion(masks.text, example.answer)
                group_partials.append((ids, completion_ids, masks, confidence, answer_score))

            rewards = compute_group_rewards(
                [item[3] for item in group_partials],
                [item[4].correct for item in group_partials],
                invalid_confidence_reward=self.train_config.invalid_confidence_reward,
                eps=self.train_config.advantage_epsilon,
            )

            # Freeze the sampling-time (theta_old) and reference logprobs for every rollout in
            # the group. Both are computed with grad disabled; the reference uses the base model
            # by disabling the LoRA adapters (no second model loaded).
            group_ids = [item[0] for item in group_partials]
            old_group = self._frozen_logprobs(group_ids, device, use_reference=False)
            ref_group = self._frozen_logprobs(group_ids, device, use_reference=True)

            for idx, (ids, completion_ids, masks, confidence, answer_score) in enumerate(
                group_partials
            ):
                all_samples.append(
                    RolloutSample(
                        full_ids=ids,
                        completion_ids=completion_ids,
                        confidence_mask=masks.confidence,
                        answer_mask=masks.answer,
                        confidence=confidence,
                        correct=answer_score.correct,
                        answer_advantage=rewards.answer_advantages[idx],
                        confidence_advantage=rewards.confidence_advantages[idx],
                        answer_reward=rewards.answer_rewards[idx],
                        confidence_reward=rewards.confidence_rewards[idx],
                        gesr=rewards.gesr,
                        text=masks.text,
                        old_logprobs=old_group[idx],
                        ref_logprobs=ref_group[idx],
                    )
                )
        self.model.train()
        return all_samples

    def _frozen_logprobs(
        self, group_ids: list[list[int]], device: Any, use_reference: bool
    ) -> list[list[float]]:
        """Per-token logprobs over each sequence's [1:] positions, under no_grad.

        When `use_reference` is True the LoRA adapters are disabled so the base model acts as
        the frozen reference policy. If the model has no adapters (full fine-tune), the
        reference falls back to the live weights and the KL term degenerates to ~0.
        """

        import torch
        from torch.nn.utils.rnn import pad_sequence

        ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in group_ids]
        lengths = [len(ids) for ids in group_ids]
        input_ids = pad_sequence(
            ids_tensors, batch_first=True, padding_value=self.pad_token_id
        ).to(device)
        positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        attention_mask = (
            positions < torch.tensor(lengths, dtype=torch.long).unsqueeze(1)
        ).long().to(device)

        disable_adapter = getattr(self.model, "disable_adapter", None)
        with torch.no_grad():
            if use_reference and callable(disable_adapter):
                with disable_adapter():
                    logprobs = _token_logprobs(self.model, input_ids, attention_mask)
            else:
                logprobs = _token_logprobs(self.model, input_ids, attention_mask)

        # Slice each row back to its own completion length (positions are over [1:]).
        result: list[list[float]] = []
        for idx, length in enumerate(lengths):
            result.append(logprobs[idx, : length - 1].detach().cpu().tolist())
        return result

    def _loss(self, samples: list[RolloutSample], device: Any) -> tuple[Any, dict[str, float]]:
        import torch

        if not samples:
            raise ValueError("Cannot compute loss for an empty rollout batch")

        (
            input_ids,
            attention_mask,
            confidence_mask,
            answer_mask,
            old_logprobs,
            ref_logprobs,
        ) = self._collate(samples, device)

        # Only the live policy participates in autograd; theta_old and the reference policy were
        # frozen at rollout time (see _frozen_logprobs).
        new_logprobs = _token_logprobs(self.model, input_ids, attention_mask)
        old_logprobs = old_logprobs.to(new_logprobs.dtype)
        ref_logprobs = ref_logprobs.to(new_logprobs.dtype)

        ratio = torch.exp(new_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.train_config.clip_epsilon,
            1.0 + self.train_config.clip_epsilon,
        )

        confidence_adv = torch.tensor(
            [sample.confidence_advantage for sample in samples],
            dtype=new_logprobs.dtype,
            device=device,
        ).unsqueeze(1)
        answer_adv = torch.tensor(
            [sample.answer_advantage for sample in samples],
            dtype=new_logprobs.dtype,
            device=device,
        ).unsqueeze(1)

        shifted_conf_mask = confidence_mask[:, 1:]
        shifted_answer_mask = answer_mask[:, 1:]
        conf_objective = torch.minimum(
            ratio * confidence_adv,
            clipped_ratio * confidence_adv,
        ) * shifted_conf_mask
        answer_objective = torch.minimum(
            ratio * answer_adv,
            clipped_ratio * answer_adv,
        ) * shifted_answer_mask

        segment_mask = shifted_conf_mask + shifted_answer_mask
        denom = segment_mask.sum().clamp_min(1.0)
        loss = -(conf_objective.sum() + answer_objective.sum()) / denom

        kl_value = 0.0
        if self.train_config.kl_beta > 0.0:
            # GRPO k3 KL estimator against the frozen reference policy, masked to the
            # confidence/answer segments: kl ~= exp(ref - new) - (ref - new) - 1 >= 0.
            log_diff = ref_logprobs - new_logprobs
            kl_per_token = (torch.exp(log_diff) - log_diff - 1.0) * segment_mask
            kl = kl_per_token.sum() / denom
            loss = loss + self.train_config.kl_beta * kl
            kl_value = float(kl.detach().cpu())

        valid_conf = [sample.confidence for sample in samples if sample.confidence is not None]
        stats = {
            "accuracy": mean(sample.answer_reward for sample in samples),
            "gesr": mean(sample.gesr for sample in samples),
            "confidence_reward": mean(sample.confidence_reward for sample in samples),
            "answer_reward": mean(sample.answer_reward for sample in samples),
            "parse_success": len(valid_conf) / len(samples),
            "kl": kl_value,
        }
        return loss, stats

    def _collate(
        self, samples: list[RolloutSample], device: Any
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        ids_tensors = [torch.tensor(sample.full_ids, dtype=torch.long) for sample in samples]
        lengths = torch.tensor([len(sample.full_ids) for sample in samples], dtype=torch.long)
        input_ids = pad_sequence(ids_tensors, batch_first=True, padding_value=self.pad_token_id).to(
            device
        )
        positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        attention_mask = (positions < lengths.unsqueeze(1)).long().to(device)

        # Logprob grids are over the shifted (label) positions, i.e. width seq_len - 1.
        logprob_width = input_ids.shape[1] - 1
        confidence_masks = []
        answer_masks = []
        old_logprob_rows = []
        ref_logprob_rows = []
        for sample in samples:
            prompt_len = len(sample.full_ids) - len(sample.completion_ids)
            full_conf = [False] * prompt_len + sample.confidence_mask
            full_answer = [False] * prompt_len + sample.answer_mask
            pad_len = input_ids.shape[1] - len(full_conf)
            confidence_masks.append(torch.tensor(full_conf + [False] * pad_len, dtype=torch.float32))
            answer_masks.append(torch.tensor(full_answer + [False] * pad_len, dtype=torch.float32))

            old_pad = logprob_width - len(sample.old_logprobs)
            ref_pad = logprob_width - len(sample.ref_logprobs)
            old_logprob_rows.append(
                torch.tensor(sample.old_logprobs + [0.0] * old_pad, dtype=torch.float32)
            )
            ref_logprob_rows.append(
                torch.tensor(sample.ref_logprobs + [0.0] * ref_pad, dtype=torch.float32)
            )
        return (
            input_ids,
            attention_mask,
            torch.stack(confidence_masks).to(device),
            torch.stack(answer_masks).to(device),
            torch.stack(old_logprob_rows).to(device),
            torch.stack(ref_logprob_rows).to(device),
        )

    def _save_checkpoint(self, accelerator: Any, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(path, save_function=accelerator.save)
        self.tokenizer.save_pretrained(path)

    def _save_configs(self, output_dir: Path) -> None:
        payload = {
            "training": dataclass_to_dict(self.train_config),
            "generation": dataclass_to_dict(self.generation_config),
        }
        (output_dir / "coca_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _token_logprobs(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    import torch
    import torch.nn.functional as F

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    shifted_attention = attention_mask[:, 1:].to(token_logprobs.dtype)
    return token_logprobs * shifted_attention


def _set_use_cache(model: Any, value: bool) -> None:
    base = getattr(model, "base_model", model)
    if hasattr(base, "config"):
        base.config.use_cache = value


def _cycle(iterable: Iterable[Any]) -> Iterable[Any]:
    while True:
        for item in iterable:
            yield item
