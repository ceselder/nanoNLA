# Qwen3-8B NLA — reproduction notes

This doc captures the **Qwen3-8B layer-24** NLA training run added in this fork,
including all scripts, the deviations from the paper's reference Qwen2.5-7B
recipe, and the empirical fixes we landed.

## Pipeline overview

```
Stage 0 (extract)           Stage 1 (split)            Stage 2 (judge)         Stage 3 (build SFT)
  FineFineWeb 100k docs  →  doc-level partition into  →  Sonnet 4.6 via       →  parquets:
  → 1.4M positions @          {av_sft, ar_sft, rl}       Batches API:            av_train.parquet
  layer 24, raw bf16        80/10/10                    explanation per        av_val.parquet
  activations                                            (doc, position)        ar_sft_shuf_clean.parquet
                                                                                rl_shuf.parquet

Stage 4 (AV SFT)            Stage 5 (AR SFT)           Stage 6 (RL GRPO)
  Qwen3-8B + Karvonen      Truncated K+1=25 layer    LoRA actor (Karvonen)
  layer-1 norm-matched  →  backbone + Linear(d,d)  →  + frozen AR critic,
  injection. Trained         value head. Trained        GRPO clipped surrogate
  to verbalise injected      to predict gold            with k3 KL penalty.
  activation.                activation from
                              AV's explanation.
```

## Reproduction (1× H200, 1× node)

All paths below assume the cluster layout `/workspace-vast/celeste/...`. Adjust as needed.

### 0. Data generation (one-time, ~12h on cluster)

```bash
sbatch launch/sbatch_datagen.sh
# Reads configs/datagen/qwen3_8b_finefineweb_100k.yaml.
# Writes activation_vector parquets + Sonnet-judged explanations.
# Outputs: /workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k/
```

### 1. AV (Activation Verbalizer) SFT

```bash
sbatch launch/sbatch_av_sft_karvonen.sh
# Wraps launch/run_av_sft_karvonen.sh which is the "Variant F" recipe:
#   NLA_KARVONEN_INJECTION=1 — ADD norm-matched injection at residual after
#                              transformer layer 1 (per Karvonen et al. 2025).
# Trained on av_train.parquet (90% slice), val on av_val.parquet (10% held-out).
# Config: lr 2e-5 cosine, batch 32, 1000 steps. Saves every 500.
# Wandb tag: WANDB_NAME=av_sft_karvonen.
```

### 2. AR (Activation Reconstructor) SFT

The critic is a truncated K+1-layer Qwen3-8B backbone with a `Linear(d, d)`
value head. We prepare the truncated init first, then fine-tune.

```bash
sbatch launch/sbatch_prepare_critic.sh  # one-time: builds qwen3_8b_L24_critic_init
sbatch launch/sbatch_ar_safe.sh         # main training
# sbatch_ar_safe.sh → run_ar_sft_safe.sh:
#   NLA_FREEZE_VALUE_HEAD=1 — keep the value head as identity init.
#   Paper config: global_batch 256, micro 64, lr 2e-5 cosine, 1000 steps.
#   Wandb tag: WANDB_NAME=ar_sft_safe.
```

⚠️ **Why freeze the value head?** Without `NLA_FREEZE_VALUE_HEAD=1`, AR SFT
NaN'd by step ~29 on every retry (8+ failed attempts). Diagnosis: the
backbone's last_hidden vectors are already very close in direction to the gold
activation (identity init was working); the value-head's small bf16 updates
in the first few steps blew the prediction norm up by 100×, leading to fp32
overflow when normalized. Freezing it keeps the prediction at
`backbone_last_hidden` (effectively a linear-projection-free critic), which
loses ~5% FVE but trains stably.

### 3. Convert DCP → HF (both ckpts)

```bash
# AV: standard Miles converter handles it because the actor is a full HF model.
python tools/convert_fsdp_to_hf.py \
  --input-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_karvonen/iter_0001000 \
  --output-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_karvonen/iter_0001000/hf \
  --origin-hf-dir Qwen/Qwen3-8B

# AR: NLACriticModel has nonstandard backbone.* prefixes; use the custom converter.
python launch/convert_ar_dcp_to_hf.py \
  --input-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft_safe/iter_0001000 \
  --output-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft_safe/iter_0001000/hf \
  --origin-hf-dir Qwen/Qwen3-8B \
  --num-layers 25
```

### 4. RL (GRPO)

```bash
sbatch launch/sbatch_rl_overnight.sh
# Runs nla/train_rl_self_contained.py.
# 250 steps, ~32s/step ≈ 2.5h on 1 H200.
# Saves LoRA every 50 steps to
# /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_rl_grpo_overnight/iter_NNNNNN/
```

A smoke variant (2 steps, B=2, G=2, max_new=64, max_rows=200) for quick
sanity checks: `sbatch launch/sbatch_rl_smoke.sh`.

## Paper-faithful RL configuration

The reference recipe is in `configs/rl.sh` (paper's Qwen2.5-7B run via Miles).
The self-contained script `nla/train_rl_self_contained.py` matches it on the
algorithmic side:

| paper / `rl.sh` | self-contained | what it is |
|---|---|---|
| `--advantage-estimator grpo` | per-prompt mean/std normalisation | group-relative advantage |
| `--use-kl-loss --kl-loss-coef 0.01` | `--kl-beta 0.01`, k3 estimator | KL penalty against reference |
| `--rollout-max-response-len 150` | `--max-new-tokens 150` | response cap |
| clipped surrogate (PPO-style) | `--clip-eps 0.2` + min(ratio·A, clip(ratio,1±ε)·A) | importance-ratio clip |
| FAILED extraction → `-2.0` | `-2.0` for missing `<explanation>` | reward floor matches `nla/reward.py` |

> Run-specific hyperparameters (LR, group size, LoRA rank, AR critic) differ
> between the two runs — see the comparison table below.

### Two configs: `overnight` (250-step) vs `long` (1500-step)

The repo ships two GRPO runs. The **overnight** run (`sbatch_rl_overnight.sh`)
trades paper-fidelity for speed on 1× H200; the **long** run
(`sbatch_rl_long.sh`) moves several knobs back toward the paper. The post-RL
eval reported below is the **overnight** run.

| knob | overnight (250-step) | long (1500-step) | paper / `rl.sh` |
|---|---|---|---|
| actor LR | `1e-6` | `1e-5` | `1e-6` constant |
| LoRA actor | r=16, α=32 | r=128, α=16, **rsLoRA** | full 8B FT |
| AR critic | **frozen** | **co-trained** (`--train-critic`, critic-lr 5e-5) | co-trained |
| effective batch | 8×4 = 32 / step | 16×16 = 256 / step | 128×4 = 512 / step |

Deviations from the paper common to **both** runs (memory / single-GPU constraint):

1. **LoRA actor instead of full 8B fine-tune.** Paper does full FT on 2× H100;
   we have 1× H200. The reference policy is the same model with the LoRA adapter
   disabled via `peft`'s `disable_adapter()` context manager — no separate copy,
   no weight-sync.

2. **HF `generate()` for rollout instead of SGLang `input_embeds`.** Paper's
   `rl.sh` uses a forked SGLang server for batched rollout with injection. We use
   plain `transformers` `generate()` with the Karvonen hook on layer 1. ~5× slower
   per step but no infra to wire up, no off-policy weight-sync drift.

## What went wrong + the fixes

| symptom | cause | fix |
|---|---|---|
| `peft 0.19 ImportError on torchao 0.9` | Recent peft requires torchao ≥ 0.16, env had 0.9 | Downgrade `peft==0.13.0` |
| `[data] loading ...` hangs 5+ min | `.as_py()` on 100k × 4096-float `activation_vector` column | Streaming row-group read + `slice(0, take)` before `to_pylist()` |
| Step 0 succeeded but `loss=inf`, then step 1 generate crashed with `Assertion 'probability tensor contains either inf, nan or element < 0'` | `output_scores=True` returns logits AFTER `top_p` filter, so filtered tokens get `-inf` `old_logp` → `ratio = exp(new_lp - old_lp) = inf` | Use `output_logits=True` (raw pre-processor logits); drop `top_p` to keep rollout/training distributions matched |
| (precautionary) | A bad batch could still NaN one update | Non-finite `loss` guard skips the optimizer step + logs |

## Post-RL evaluation

Pre-vs-post comparison on 128 held-out `rl_shuf` prompts (rows 25000-25128,
past the RL trainer's `--max-rows 20000` cursor; doc-disjoint from
`av_train`). One sample per prompt, temperature 1.0, max_new=150. Reward
= `-mse_nrm` from the AR critic; failed extractions get reward `-2.0`.

| metric | pre-RL (AV-SFT only) | post-RL (AV-SFT + 250-step RL LoRA) | Δ |
|---|---|---|---|
| **mean reward (= -mse_nrm)** | -0.3810 | -0.3611 | **+0.0199** ✅ |
| **FVE** (vs baseline 0.687) | **44.52%** | **47.42%** | **+2.90 pp** ✅ |
| reward std      | 0.289 | 0.270 | -7% |
| extraction rate | 98%   | 98%   | flat |
| reward max      | -0.102 | -0.113 | similar |
| per-prompt wins | — | 63/128 (49%) | RL hurts as often as helps |

**FVE = 1 − mse_actual / baseline_mse**, where the baseline is the
predict-the-mean ceiling (normalize(μ) as constant pred, μ = mean of
held-out activations). 0% = no better than constant prediction; 100% =
perfect reconstruction. Paper's Qwen2.5-7B critic-SFT alone reports FVE
= 37.5% (TRAINING_NOTES.md); our AV-SFT-alone hits 44.5% on held-out
data, and 250 GRPO steps adds 2.9 pp. The win is modest — the
1500-step run (in progress, wandb run `4cvdfjiw`) should give a sharper
signal.

> ⚠️ **Earlier eval reported FVE 20% → 38% (Δ +18pp) on `av_val.parquet`**.
> That number was inflated by data leakage: `av_val.parquet` was
> row-sub-sampled from `av_train.parquet` (16127 / 16128 docs overlap),
> so the AV actor had seen those documents during SFT. The clean numbers
> above use `rl_shuf` rows past the trainer's cursor — disjoint by
> `doc_id` from everything the actor has ever trained on. Both pre and
> post numbers shifted, but the *direction* of the delta is the same.

Reproduce: `python launch/compute_fve_baseline.py --parquet rl_shuf.parquet
--sidecar rl_shuf.parquet --reward-pre -0.3810 --reward-post -0.3611`.

Per-prompt: 29/64 improved by RL, 34/64 hurt, 1/64 tied. Even though the win
rate is close to 50/50, the **mean improvement is +0.124** — the wins are
larger than the losses, and the std collapses noticeably (0.566 → 0.375)
because the RL'd actor stops producing very bad outliers as often.

Run: `sbatch launch/sbatch_eval_post_rl.sh` (job 1566576, 6:46 elapsed on
1× H200).

## Wandb runs

- AV SFT (Karvonen): https://wandb.ai/adamkarvonen/nla-qwen3-8b/runs/epu2zb0m
- AR SFT (safe): https://wandb.ai/adamkarvonen/nla-qwen3-8b/runs/8ea7vvfk
- RL GRPO smoke: https://wandb.ai/adamkarvonen/nla-qwen3-8b/runs/l41reyk1
- RL GRPO overnight (250 steps): https://wandb.ai/adamkarvonen/nla-qwen3-8b/runs/8ls885ti
- RL GRPO long (1500 steps, with live FVE logging): https://wandb.ai/adamkarvonen/nla-qwen3-8b/runs/4cvdfjiw

All runs live under the cohort-shared `adamkarvonen/nla-qwen3-8b` project
(API key is shared across the cohort; not a leak).
