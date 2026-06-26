#!/bin/bash
# Gemma-4-26B-A4B NLA warmstart on a 2×B200 box (RunPod), reusing the slim
# Qwen release's (text, summary) pairs and re-extracting Gemma activations.
#
# Pipeline: regen AV activations → regen AR activations → AV-SFT → AR-SFT.
# AR-SFT logs held-out FVE (doc-disjoint av_sft pairs) every --heldout-every.
#
# Run from the repo root inside the gemma venv:
#   bash scripts/run_gemma4_warmstart.sh [smoke]
# "smoke" caps rows for a cheap end-to-end validation.
set -euo pipefail
source /workspace/.env   # HF_TOKEN, WANDB_API_KEY, HF_HOME
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/gemma-nla/nanoNLA
PY=/usr/local/bin/python   # system python has the working matched torch/tv + extras
cd /workspace/gemma-nla/nanoNLA

MODEL=google/gemma-4-26B-A4B
LAYER=20
CHAR="㊗"
SLIM=/workspace/gemma-nla/qwen_slim
OUT=/workspace/gemma-nla/data
CKPT=/workspace/gemma-nla/ckpts
mkdir -p "$OUT" "$CKPT"

SMOKE="${1:-}"
if [ "$SMOKE" = "smoke" ]; then
  MAXROWS_REGEN="--max-rows 4000"; STEP_ARG="--num-steps 60"; WARMUP=5; SAVE=60; HELDOUT_ROWS=400; TAG=smoke
else
  # ⚠️ DON'T FORGET THIS — TRAIN ON THE FULL THING: --epochs 1 = exactly one
  # full pass over ALL ~247k SFT rows (train_sft computes steps from the real
  # row count). The old --num-steps 1000 was ~6.5% of one epoch and floored
  # the cosine LR early → undersold FVE.
  MAXROWS_REGEN=""; STEP_ARG="--epochs 1"; WARMUP=50; SAVE=250; HELDOUT_ROWS=1000; TAG=v2_fullep
fi
# Effective batch via grad-accum (memory-neutral; micro-batch stays 16 for the
# eager-MoE memory). grad_accum 4 → eff batch 64, which smooths the loss vs
# batch 16. (Paper used 256; we keep 64 by choice.) With --epochs 1 the step
# count auto-scales (~3.9k for 247k at eff 64).
GRAD_ACCUM=4
[ "$SMOKE" = "smoke" ] && GRAD_ACCUM=1

echo "===== [1/4] regen AV activations (Gemma layer $LAYER) ====="
# Sequential, device_map=auto (model split across the visible GPUs). Reliable
# (smoke-proven); the eager Gemma-4 MoE is memory-heavy so batch stays at 16.
if [ -f "$OUT/av_sft_gemma.parquet" ]; then echo "  exists — skipping"; else
$PY -m scripts.gemma_warmstart_from_slim \
  --slim "$SLIM/av_sft_shuf.parquet" --qwen-sidecar "$SLIM/av_sft_shuf.parquet.nla_meta.yaml" \
  --out "$OUT/av_sft_gemma.parquet" --mode av \
  --base-model "$MODEL" --layer "$LAYER" --injection-char "$CHAR" \
  --experts-implementation eager --batch-size 16 $MAXROWS_REGEN
fi

echo "===== [2/4] regen AR activations ====="
if [ -f "$OUT/ar_sft_gemma.parquet" ]; then echo "  exists — skipping"; else
$PY -m scripts.gemma_warmstart_from_slim \
  --slim "$SLIM/ar_sft_shuf.parquet" --qwen-sidecar "$SLIM/ar_sft_shuf.parquet.nla_meta.yaml" \
  --out "$OUT/ar_sft_gemma.parquet" --mode ar \
  --base-model "$MODEL" --layer "$LAYER" --injection-char "$CHAR" \
  --experts-implementation eager --batch-size 16 $MAXROWS_REGEN
fi

echo "===== [3/4] AV-SFT (verbalizer) ====="
# --no-gradient-checkpointing: the eager Gemma-4 MoE has data-dependent tensor
# counts that break non-reentrant checkpoint's recompute check; 26B fits a
# 183GB B200 without GC anyway.
$PY -m nla.train_sft --mode av --base-ckpt "$MODEL" --experts-implementation eager \
  --no-gradient-checkpointing \
  --parquet "$OUT/av_sft_gemma.parquet" --sidecar "$OUT/av_sft_gemma.parquet" \
  --save-dir "$CKPT/gemma4_av_sft_$TAG" \
  $STEP_ARG --batch-size 16 --gradient-accumulation-steps $GRAD_ACCUM --use-lora --lora-r 128 --lora-alpha 16 \
  --lr 3e-5 --min-lr 3e-6 --save-every $SAVE \
  --wandb-project nla-gemma4-26b --wandb-name "gemma4_av_sft_$TAG"

echo "===== [4/4] AR-SFT (reconstructor, K+1=$((LAYER+1)) layers) + held-out FVE ====="
$PY -m nla.train_sft --mode ar --base-ckpt "$MODEL" --experts-implementation eager \
  --parquet "$OUT/ar_sft_gemma.parquet" --sidecar "$OUT/ar_sft_gemma.parquet" \
  --save-dir "$CKPT/gemma4_ar_sft_$TAG" \
  $STEP_ARG --batch-size 16 --gradient-accumulation-steps $GRAD_ACCUM --use-lora --lora-r 128 --lora-alpha 16 \
  --ar-num-layers $((LAYER+1)) --lr 3e-5 --min-lr 3e-6 \
  --save-every $SAVE \
  --heldout-parquet "$OUT/av_sft_gemma.parquet" --heldout-rows $HELDOUT_ROWS --heldout-every 50 \
  --wandb-project nla-gemma4-26b --wandb-name "gemma4_ar_sft_$TAG"

echo "===== DONE — AV + AR warmstart complete. wandb project: nla-gemma4-26b ====="
