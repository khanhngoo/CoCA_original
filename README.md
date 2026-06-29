# CoCA Confidence-First RL MVP

This repo recreates the core architecture from **Confidence Before Answering: A Paradigm Shift for Efficient LLM Uncertainty Estimation** as a runnable MVP.

It trains a Qwen instruction model to emit:

```text
<confidence>0.73</confidence>
...answer...
```

before answering, then applies segmented GRPO-style credit assignment:

- confidence tokens receive a calibration reward against group empirical success rate.
- answer tokens receive a math correctness reward.

## Environment

The system `python3` in this workspace is Python 3.13, while the ML stack is expected to work best on Python 3.10 or 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

If the dataset is gated for your account:

```bash
huggingface-cli login
```

## Dataset

The default loader uses:

```python
from datasets import load_dataset

ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified")
```

Expected columns are `problem`, `answer`, `source`, `domain`, and `llama8b_solve_rate`.

## Train

Small LoRA smoke run:

```bash
python -m scripts.train_coca \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset SynthLabsAI/Big-Math-RL-Verified \
  --output-dir outputs/coca-qwen2.5-1.5b-lora \
  --max-steps 10 \
  --limit-train-samples 64 \
  --group-size 4 \
  --per-device-batch-size 1
```

Longer run:

```bash
python -m scripts.train_coca \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset SynthLabsAI/Big-Math-RL-Verified \
  --output-dir outputs/coca-qwen2.5-1.5b-lora \
  --max-steps 1000 \
  --group-size 4
```

Use `--load-in-4bit` when training on a constrained GPU with bitsandbytes available.

## Evaluate

On a held-out math benchmark (GSM8K or MATH-500):

```bash
python -m scripts.eval_coca \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint outputs/coca-qwen2.5-1.5b-lora \
  --benchmark gsm8k \
  --limit 200
```

`--benchmark` accepts `gsm8k`, `math500`, or `bigmath` (the default, which uses
`--dataset`/`--split`). The evaluator reports accuracy, confidence parse success, ECE, Brier
score, AUROC when defined, and average tokens consumed before the confidence close tag.

## Deviations from the paper

This MVP targets single-GPU LoRA training, so a few settings differ from the paper
(Appendix B). All are CLI-overridable:

- **LoRA, not full fine-tune.** The reference policy for the KL term (Eq. 3) is obtained by
  disabling the LoRA adapters (`model.disable_adapter()`) rather than loading a second frozen
  model. With `--no-lora` there is no separate reference and the KL term degenerates to ~0.
- **`max_new_tokens` defaults to 512** (paper uses 4096). Raise it via `--max-new-tokens` if
  memory allows.
- **Eval is math-only** (GSM8K, MATH-500) via lightweight loaders, not the full OpenCompass
  8-benchmark suite (no code / factual-QA tasks).

Paper-matched defaults already applied: learning rate `1e-6`, temperature `1.0`, no nucleus
sampling (`top_p=1.0`), KL coefficient `0.04`, and the exact confidence-first system prompt
plus the `\boxed{}` user-prompt suffix.

## Tests

Pure parser/reward/metric tests do not require PyTorch or Hugging Face packages:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

