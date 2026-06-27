#!/bin/bash
# Gemma-4-26B-A4B NLA RL (GRPO) — DATA-PARALLEL across both B200s via torchrun.
# Each rank holds a full model replica on its own GPU and processes a disjoint
# stride-shard of the prompts; LoRA + critic grads are gloo-all-reduced (mean)
# before each optim.step, so 2 ranks == one 2×-bigger batch (numerically). See
# the DDP block + _allreduce_grads_ in nla/train_rl_self_contained.py.
#
#   DERISK (2 steps, confirm FVE ~17-25% + both GPUs + no deadlock):
#     STEPS=2 bash scripts/run_gemma4_rl_ddp.sh
#   FULL RUN:
#     bash scripts/run_gemma4_rl_ddp.sh
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
OUT=/workspace/gemma-nla/data
CKPT=/workspace/gemma-nla/ckpts
AV=$(ls -d $CKPT/gemma4_av_sft_v2_fullep/iter_* 2>/dev/null | sort | tail -1)
AR=$(ls -d $CKPT/gemma4_ar_sft_v2_fullep/iter_* 2>/dev/null | sort | tail -1)
[ -z "$AV" ] || [ -z "$AR" ] && { echo "missing v2 AV/AR ckpt (AV=$AV AR=$AR)"; exit 1; }
echo "[rl-ddp] AV=$AV"; echo "[rl-ddp] AR=$AR"

NPROC="${NPROC:-6}"          # data-parallel ranks (1 per GPU) — 6×B200
STEPS="${STEPS:-500}"        # set STEPS=2 for the derisk
BATCH="${BATCH:-96}"         # GLOBAL prompts/step (must be divisible by NPROC); 96 -> 16/rank
GROUP="${GROUP:-8}"          # paper group size (paper: 128×8; we run 96×8)
SAVEDIR="${SAVEDIR:-$CKPT/gemma4_rl_ddp}"
[ "$STEPS" = "2" ] && SAVEDIR="$CKPT/gemma4_rl_ddp_derisk"

[ -f "$OUT/rl_gemma.parquet" ] || { echo "rl_gemma.parquet missing — run the single-GPU script once to regen it"; exit 1; }

echo "===== GRPO RL (DDP, ${NPROC} ranks, global batch ${BATCH}x${GROUP}, ${STEPS} steps) ====="
# torchrun --standalone: single node, auto MASTER_ADDR/PORT. Each rank uses
# cuda:LOCAL_RANK (both GPUs visible — this is the RunPod box, NOT the SLURM
# cluster, so setting per-rank devices via LOCAL_RANK is correct here).
torchrun --standalone --nproc_per_node="$NPROC" -m nla.train_rl_self_contained \
  --av-ckpt "$AV" --ar-ckpt "$AR" --base-ckpt "$MODEL" \
  --quant none --experts-implementation eager \
  --rl-parquet "$OUT/rl_gemma.parquet" --sidecar "$OUT/rl_gemma.parquet" \
  --save-dir "$SAVEDIR" \
  --num-steps "$STEPS" --batch-prompts "$BATCH" --group-size "$GROUP" \
  --max-new-tokens 150 --temperature 1.0 \
  --lr 1e-5 --kl-beta 0.01 --clip-eps 0.2 \
  --train-critic --critic-lr 5e-5 \
  --logp-micro-batch 4 --critic-micro-batch 16 \
  --rollout-chunk 160 --score-chunk 64 \
  --max-rows 30000 \
  --save-every 50 --eval-every 10 --eval-n-prompts 20 --eval-skip-rows 35000 \
  --max-grad-norm 1.0 \
  --wandb-project nla-gemma4-26b --wandb-name "gemma4_rl_ddp_${BATCH}x${GROUP}" --seed 0

echo "===== DONE — DDP RL complete. wandb project: nla-gemma4-26b ====="
