#!/bin/bash
# Parallel Gemma-4 warmstart: AV-SFT on GPU0, AR-SFT on GPU1 (independent jobs,
# one per B200) → ~half the wall-clock of the sequential run_gemma4_warmstart.sh.
# Both --epochs 1 (full pass over all rows) at eff batch 64 (micro 16 × ga 4).
# Activations must already be extracted (av_sft_gemma.parquet / ar_sft_gemma.parquet).
set -euo pipefail
source /workspace/.env
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/gemma-nla/nanoNLA
PY=/usr/local/bin/python
cd /workspace/gemma-nla/nanoNLA
MODEL=google/gemma-4-26B-A4B
OUT=/workspace/gemma-nla/data
CKPT=/workspace/gemma-nla/ckpts
TAG=v2_fullep

echo "===== AV-SFT on GPU0 (background) ====="
CUDA_VISIBLE_DEVICES=0 $PY -m nla.train_sft --mode av --base-ckpt "$MODEL" \
  --experts-implementation eager --no-gradient-checkpointing \
  --parquet "$OUT/av_sft_gemma.parquet" --sidecar "$OUT/av_sft_gemma.parquet" \
  --save-dir "$CKPT/gemma4_av_sft_$TAG" \
  --epochs 1 --batch-size 16 --gradient-accumulation-steps 4 \
  --use-lora --lora-r 128 --lora-alpha 16 \
  --lr 3e-5 --min-lr 3e-6 --save-every 250 \
  --wandb-project nla-gemma4-26b --wandb-name "gemma4_av_sft_$TAG" \
  > /workspace/logs/av_v2.log 2>&1 &
AVPID=$!
echo "  AV pid $AVPID (GPU0)"

sleep 120   # stagger so the two 26B loads don't race on the MFS snapshot

echo "===== AR-SFT on GPU1 (background) ====="
CUDA_VISIBLE_DEVICES=1 $PY -m nla.train_sft --mode ar --base-ckpt "$MODEL" \
  --experts-implementation eager \
  --parquet "$OUT/ar_sft_gemma.parquet" --sidecar "$OUT/ar_sft_gemma.parquet" \
  --save-dir "$CKPT/gemma4_ar_sft_$TAG" \
  --epochs 1 --batch-size 16 --gradient-accumulation-steps 4 --ar-num-layers 21 \
  --use-lora --lora-r 128 --lora-alpha 16 \
  --lr 3e-5 --min-lr 3e-6 --save-every 250 \
  --heldout-parquet "$OUT/av_sft_gemma.parquet" --heldout-rows 1000 --heldout-every 50 \
  --wandb-project nla-gemma4-26b --wandb-name "gemma4_ar_sft_$TAG" \
  > /workspace/logs/ar_v2.log 2>&1 &
ARPID=$!
echo "  AR pid $ARPID (GPU1)"

wait $AVPID; wait $ARPID
echo "===== DONE — AV + AR warmstart complete (parallel). wandb: nla-gemma4-26b ====="
