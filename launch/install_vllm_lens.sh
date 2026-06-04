#!/bin/bash
# Install vllm-lens — the vLLM activation-injection plugin, used as the FAST RL
# rollout backend: inject the AV's activation at the marker via
# SteeringVector(norm_match=True) (== the Karvonen formula) during vLLM generation.
# ~5x faster than the HF generate() rollout path.
#
# TWO HARD-WON VERSION PINS — do not bump blindly:
#
#   vllm==0.19.0
#     vllm-lens 1.1.0 (the latest, released 2026-04-14) was built against vLLM
#     0.19.0. vLLM 0.22+ (2026-05-29) refactored GPUModelRunner, after which the
#     injection hook crashes with
#         AttributeError: 'GPUModelRunner' object has no attribute 'input_batch'
#     and then SILENTLY SKIPS injection — generations look fine but NO vector is
#     ever injected. 0.19.0 is the matched version where the hook fires.
#
#   --torch-backend=cu128
#     vLLM 0.22's default wheel is cu130 (needs NVIDIA driver >= 580). The cluster
#     driver is 570 (CUDA 12.8) -> cu130 fails at import with `libcudart.so.13:
#     cannot open shared object file`. 0.19.0's default wheel is cu128, which runs
#     on driver 570 directly.
#
# VERIFIED: greedy injection on Qwen3-0.6B — scale-0 == baseline (control), scale-8
# dramatically diverges (random vector steers the model into degenerate output).
# See smoke_vllm_lens.py.
#
# Kept in its own venv (not the nla/sglang env) since it pins a specific vLLM.
set -euo pipefail
VENV=${1:-/workspace-vast/celeste/envs/vllm-lens}

uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" "vllm==0.19.0" "vllm-lens==1.1.0" --torch-backend=cu128

echo "=== verify (imports vllm._C -> exercises libcudart) ==="
"$VENV/bin/python" - <<'PY'
import torch, vllm, vllm_lens
from vllm import LLM, SamplingParams
from vllm_lens import SteeringVector
print(f"OK — torch {torch.__version__} (cuda {torch.version.cuda}) | "
      f"vllm {vllm.__version__} | vllm_lens {vllm_lens.__version__}")
PY
