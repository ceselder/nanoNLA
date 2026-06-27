#!/bin/bash
# Gemma-4-26B-A4B NLA RL (GRPO) on a B200 box, from the AV+AR warmstarts.
# Extracts the RL-split activations (doc-disjoint from av/ar), then co-trains
# the actor (Karvonen layer-1 inject) against the AR critic. Built-in held-out
# FVE eval every --eval-every; no external (judge) evals.
#
#   bash scripts/run_gemma4_rl.sh
set -euo pipefail
source /workspace/.env
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/gemma-nla/nanoNLA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/local/bin/python
cd /workspace/gemma-nla/nanoNLA

MODEL=google/gemma-4-26B-A4B
LAYER=20
CHAR="㊗"
SLIM=/workspace/gemma-nla/qwen_slim
OUT=/workspace/gemma-nla/data
CKPT=/workspace/gemma-nla/ckpts
# Use the v2 full-epoch warmstarts (latest iter_* in each dir). AV-SFT v2 loss
# ~1.5; AR-SFT v2 held-out FVE 58.1% (vs v1's 48.4%).
AV=$(ls -d $CKPT/gemma4_av_sft_v2_fullep/iter_* 2>/dev/null | sort | tail -1)
AR=$(ls -d $CKPT/gemma4_ar_sft_v2_fullep/iter_* 2>/dev/null | sort | tail -1)
[ -z "$AV" ] || [ -z "$AR" ] && { echo "missing v2 AV/AR checkpoint (AV=$AV AR=$AR)"; exit 1; }
echo "[rl] AV=$AV"; echo "[rl] AR=$AR"

echo "===== [1/2] regen RL-split activations (Gemma layer $LAYER, ~40k slice) ====="
# mode=ar: no 'response' column needed (rl_shuf has none) and the sidecar gets
# the critic suffix the reward path uses. The reused 'prompt' column is the
# actor prompt (with <INJECT>), which RL needs.
if [ -f "$OUT/rl_gemma.parquet" ]; then echo "  exists — skipping"; else
$PY -m scripts.gemma_warmstart_from_slim \
  --slim "$SLIM/rl_shuf.parquet" --qwen-sidecar "$SLIM/rl_shuf.parquet.nla_meta.yaml" \
  --out "$OUT/rl_gemma.parquet" --mode ar \
  --base-model "$MODEL" --layer "$LAYER" --injection-char "$CHAR" \
  --experts-implementation eager --batch-size 16 --max-rows 40000
fi

echo "===== [2/2] GRPO RL (co-trained critic, held-out FVE eval) ====="
$PY -m nla.train_rl_self_contained \
  --av-ckpt "$AV" --ar-ckpt "$AR" --base-ckpt "$MODEL" \
  --quant none --experts-implementation eager \
  --rl-parquet "$OUT/rl_gemma.parquet" --sidecar "$OUT/rl_gemma.parquet" \
  --save-dir "$CKPT/gemma4_rl_v1" \
  --num-steps 500 --batch-prompts 8 --group-size 8 \
  --max-new-tokens 150 --temperature 1.0 \
  --lr 1e-5 --kl-beta 0.01 --clip-eps 0.2 \
  --train-critic --critic-lr 5e-5 \
  --logp-micro-batch 2 --max-rows 30000 \
  --save-every 50 --eval-every 10 --eval-n-prompts 20 --eval-skip-rows 35000 \
  --max-grad-norm 1.0 \
  --wandb-project nla-gemma4-26b --wandb-name gemma4_rl_v1 --seed 0

echo "===== DONE — Gemma-4 RL complete. wandb project: nla-gemma4-26b ====="
