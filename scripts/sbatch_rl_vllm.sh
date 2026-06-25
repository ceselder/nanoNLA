#!/bin/bash
#SBATCH --job-name=qwen3_rl_grpo_vllm
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

# vLLM-rollout GRPO with TRL-style colocate weight sync.
#
# Same algorithmic config as the v4 HF-generate run (B=16 G=16 paper-faithful,
# co-trained AR, rsLoRA r=128, LRs 1e-5 / 5e-5), but rollouts go through vLLM
# with vllm-lens SteeringVector for Karvonen injection. Expected ~3-5×
# speedup vs HF-generate (rollouts dominate step time).
#
# Weight sync every 20 steps: merge LoRA into base, push state_dict into
# vLLM via collective_rpc("load_weights"), unmerge LoRA. TIS-clip (cap=2.0)
# in the GRPO loss handles residual vLLM/HF engine mismatch.

set -euo pipefail
# Use the existing vllm-lens venv (has torch 2.9.1+cu128 matching cluster's
# CUDA 12.8 driver). The auto-bootstrapped nla_vllm venv pulled torch
# 2.11.0+cu130 which failed at runtime; pinning torch + vllm + peft together
# in the existing venv is the working combo. Build the venv with
# launch/install_vllm_lens.sh — current validated combo (see that script):
#   vllm==0.19.0 (cu128) + vllm_lens==1.1.0 + transformers==4.57.1
#   + peft + bitsandbytes + wandb.
# (An earlier combo used vllm 0.16.0; 0.19.0 is what install_vllm_lens.sh
# pins now — the hook breaks on vLLM >= 0.22.)
VENV=/workspace-vast/celeste/envs/vllm-lens
source "$VENV/bin/activate"
# Allow pickle-based serialisation so we can pass a lambda to apply_model
# (needed for weight-broadcast: `llm.apply_model(lambda m: m.load_weights(items))`).
# msgpack can't serialise functions; pickle can. Single-tenant cluster — safe.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
: "${WANDB_API_KEY:?set WANDB_API_KEY in your shell}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace-vast/celeste/nla-experiments:${PYTHONPATH:-}

DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k

cd /workspace-vast/celeste/nla-experiments

# EXPERIMENTAL vLLM-rollout path (unverified — see nla/train_rl_vllm.py docstring).
# The verified recipe is scripts/sbatch_rl_fixed.sh (HF-generate, single GPU 4-bit).
python -m nla.train_rl_vllm \
  --i-understand-experimental \
  --av-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_karvonen/iter_0001000/hf \
  --ar-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft_safe/iter_0001000/hf \
  --rl-parquet $DATA/rl_shuf.parquet \
  --sidecar $DATA/rl_shuf.parquet \
  --save-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_rl_grpo_vllm \
  --num-steps 1000 \
  --batch-prompts 16 \
  --group-size 16 \
  --max-new-tokens 150 \
  --temperature 1.0 \
  --lr 1e-5 \
  --kl-beta 0.01 \
  --clip-eps 0.2 \
  --lora-r 128 \
  --lora-alpha 16 \
  --use-rslora \
  --train-critic \
  --critic-lr 5e-5 \
  --logp-micro-batch 2 \
  --critic-micro-batch 4 \
  --max-rows 30000 \
  --save-every 100 \
  --eval-every 10 \
  --eval-n-prompts 20 \
  --eval-skip-rows 35000 \
  --max-grad-norm 1.0 \
  --vllm-gpu-mem 0.35 \
  --vllm-max-len 1024 \
  --vllm-tp 4 \
  --vllm-sync-every 20 \
  --tis-cap 2.0 \
  --wandb-project nla-qwen3-8b \
  --wandb-name qwen3_8b_L24_rl_grpo_vllm_v1 \
  --seed 0
