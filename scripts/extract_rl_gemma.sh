#!/bin/bash
# Extract the Gemma-4 RL-split activations (~40k doc-disjoint slice) on GPU1,
# while AV-SFT finishes on GPU0. Produces rl_gemma.parquet (+ sidecar) so the
# RL run can skip extraction and start immediately.
set -euo pipefail
source /workspace/.env
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/gemma-nla/nanoNLA
cd /workspace/gemma-nla/nanoNLA
CUDA_VISIBLE_DEVICES=1 /usr/local/bin/python -m scripts.gemma_warmstart_from_slim \
  --slim /workspace/gemma-nla/qwen_slim/rl_shuf.parquet \
  --qwen-sidecar /workspace/gemma-nla/qwen_slim/rl_shuf.parquet.nla_meta.yaml \
  --out /workspace/gemma-nla/data/rl_gemma.parquet --mode ar \
  --base-model google/gemma-4-26B-A4B --layer 20 --injection-char "㊗" \
  --experts-implementation eager --batch-size 16 --max-rows 40000
echo "RL_EXTRACT_DONE"
