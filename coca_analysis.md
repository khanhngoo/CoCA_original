# CoCA Codebase vs Paper Analysis

## What's Correctly Implemented

| Paper | Code |
|---|---|
| Confidence-first format `<confidence>s</confidence> y^a` | `prompts.py`: `CONF_OPEN/CLOSE`, `format_coca_prompt()` |
| GESR = mean of group answer rewards (Eq. 7) | `rewards.py`: `gesr = fmean(answer_rewards)` |
| Confidence reward = `-(s - GESR)²` (Eq. 8) | `rewards.py`: `-(confidence - gesr) ** 2` |
| Answer reward = `𝟙(AnsCorrect)` (Eq. 6) | `rewards.py`: `1.0 if correct else 0.0` |
| Separate advantages for confidence/answer (Eq. 9) | `rewards.py`: `normalize_advantages()` called separately |
| Segmented PPO-clip loss (Eq. 10-11) | `trainer.py`: `conf_objective` + `answer_objective` with separate masks |
| KL penalty term `-β𝔼[KL(π\|\|π_ref)]` (Eq. 3) | `trainer.py`: placeholder, adds `torch.zeros` — **not implemented** |
| Token-level segment masks | `prompts.py`: `segment_completion_token_masks()` with offset mapping |
| math_verify + normalized exact fallback | `rewards.py`: `score_math_answer()` |
| ECE, Brier, AUROC metrics | `metrics.py`: all present |
| LoRA on Qwen2.5-1.5B/3B/7B | `config.py`, `train_coca.py`: CLI args |

---

## What's Missing / Wrong

### 1. KL divergence — stub only
Paper (Eq. 3): `L_GRPO = clipped_obj - β·KL(π_new || π_ref)`. Code adds `torch.zeros`. No frozen reference model loaded anywhere.

### 2. `π_old` is wrong — same model used for old/new logprobs
Paper: `ρ_{i,t}(θ) = π_θ(y_t|x,y_{<t}) / π_{θ_old}(y_t|x,y_{<t})`. Old policy should be a snapshot from **before** the update.

Code (`trainer.py:_loss()`):
```python
self.model.eval()
with torch.no_grad():
    old_logprobs = _token_logprobs(self.model, ...)  # same model!
new_logprobs = _token_logprobs(self.model, ...)
```
Both calls use same `self.model` weights at same step → ratio always ~1 → clipping never activates. Paper requires `θ_old` snapshot (set `θ_old ← θ` after each update, Algorithm 1 line 15).

### 3. No top-k/nucleus sampling during rollout
Paper (Appendix B.1): "No additional sampling strategies (e.g., top-k, nucleus sampling) are used during training." Code uses `top_p=0.95` by default — contradicts paper.

### 4. Wrong hyperparameters

| Param | Paper | Code default |
|---|---|---|
| Learning rate | `1e-6` | `5e-6` |
| Max generation length | 4096 tokens | 512 tokens |
| Global batch size | 128×16 | 1 (no multi-GPU setup) |
| top_p | none (greedy/temp-only) | 0.95 |
| Training framework | MindSpeed-RL on Ascend 910 | plain Accelerate |

### 5. Prompt format differs
Paper (Appendix B.2):
```
System: "You need to provide the answer as well as its confidence level...
<confidence> confidence level here </confidence> answer here"

User: "{question} Please reason step by step, and put your final answer within \boxed{}."
```

Code (`prompts.py`):
```python
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math assistant. Before solving, output your calibrated "
    "probability of answering correctly as a decimal from 0 to 1 inside "
    "<confidence>...</confidence>. Then solve the problem..."
)
# No \boxed{} instruction in user prompt
```
Missing `\boxed{}` instruction → `extract_final_answer()` may find fewer boxed answers.

### 6. Evaluation uses OpenCompass — not implemented
Paper evals on: AIME2024/2025, MATH-500, GSM8K, HumanEval, MBPP(s), SimpleQA, TriviaQA. Code `eval_coca.py` only evaluates on the **training dataset** (Big-Math-RL-Verified). No held-out benchmark eval, no OpenCompass integration.

### 7. `avg_tokens_to_confidence` metric — partially broken
Paper reports TTC (token consumption to confidence). Code has `tokens_to_confidence_close()` in `prompts.py` but `eval_coca.py` reports `avg_tokens_to_confidence: null` in smoke run — never wired up properly.

---

## Assumptions Made by Codebase (not in paper)

1. **LoRA only** — paper trains full model (MindSpeed-RL), code defaults `use_lora=True`
2. **`invalid_confidence_reward = -1.0`** when parse fails — paper doesn't specify this
3. **`_ensure_nonempty_masks()`** fallback assigns first 8 tokens as "confidence" when parse fails — paper assumes model always emits the tag
4. **No reference policy** — `kl_beta=0.0` default sidesteps the unimplemented KL entirely

---

## Summary

Core algorithm (GESR, Brier reward, segmented advantages, PPO-clip masks) is **correctly implemented**. Critical bug: **`π_old = π_new`** — importance sampling ratio is trivially 1 everywhere, making PPO-clip equivalent to vanilla policy gradient. Everything else is either missing (KL, eval benchmarks) or misconfigured (LR, max_tokens, prompt format, top_p).
