"""Decisive 'is the AR loaded correctly from the start' test.

Loads the critic EXACTLY as train_rl_self_contained does (init_critic_from_base
+ inject LoRA via lora_attention_targets + load ar_lora_value_head.safetensors),
on cuda:1 (idle GPU — does NOT touch the RL job on cuda:0), then scores the GOLD
explanations from the AV-split parquet and reports FVE. If this reproduces the
AR-SFT held-out number (~58%), the AR is provably loaded correctly and the low
on-policy RL FVE is purely the actor-quality gap.
"""
import sys, glob, json, torch
import numpy as np
sys.path.insert(0, "/workspace/gemma-nla/nanoNLA")
from transformers import AutoTokenizer
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file
from nla.train_sft import init_critic_from_base, load_heldout_explanation_pairs, heldout_fve_mse
from nla.config import load_nla_config
from nla.schema import compute_predict_mean_baselines, resolve_target_scale
from nla.arch_adapters import lora_attention_targets

BASE = "google/gemma-4-26B-A4B"
AR = sorted(glob.glob("/workspace/gemma-nla/ckpts/gemma4_ar_sft_v2_fullep/iter_*"))[-1]
GOLD = "/workspace/gemma-nla/data/av_sft_gemma.parquet"  # has `response` = gold summaries + activations
DEV = "cuda:1"
N = 1000

print(f"[ar] {AR}", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
cfg = load_nla_config(GOLD, tok)
mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
ar_meta = json.load(open(AR + "/ar_meta.json"))
print(f"[cfg] mse_scale_f={mse_scale_f:.3f} ar_meta={ar_meta}", flush=True)

# --- build critic EXACTLY as train_rl does ---
critic = init_critic_from_base(
    BASE, ar_meta["ar_num_layers"], torch.bfloat16, None,
    device_map=None, strip_final_norm=ar_meta.get("final_norm_stripped", False),
    experts_implementation="eager",
).to(DEV)
inject_adapter_in_model(LoraConfig(
    r=ar_meta["lora_r"], lora_alpha=ar_meta["lora_alpha"], lora_dropout=0.0,
    bias="none", task_type="CAUSAL_LM", use_rslora=True,
    target_modules=lora_attention_targets(critic.backbone.config),
), critic.backbone)
sd = load_file(AR + "/ar_lora_value_head.safetensors")
miss, unexp = critic.load_state_dict(sd, strict=False)
n_lora = sum(1 for k in sd if "lora_" in k)
print(f"[critic] loaded {len(sd)} AR tensors ({n_lora} lora), unexpected={len(unexp)}", flush=True)
assert n_lora > 0 and not unexp, f"AR load mismatch! unexpected={unexp[:3]}"
critic.eval()

# --- gold pairs + paper raw-var baseline ---
pairs = load_heldout_explanation_pairs(GOLD, N)
acts = torch.tensor(np.stack([a for _, a in pairs]), dtype=torch.float32)
_meannorm, baseline = compute_predict_mean_baselines(acts, mse_scale_f)
print(f"[fve] {len(pairs)} gold pairs | baseline(rawvar)={baseline:.4f}", flush=True)

h_mse, n_scored = heldout_fve_mse(critic, tok, pairs, cfg.critic_prompt_template, mse_scale_f, DEV)
fve = (1.0 - h_mse / baseline) * 100.0
print(f"\n[RESULT] GOLD-explanation FVE via the RL-load-path critic: {fve:.1f}%  "
      f"(mse={h_mse:.4f}, n={n_scored})", flush=True)
print(f"[verdict] AR-SFT held-out was ~58%. If this is ~55-60% -> AR loads correctly; "
      f"low on-policy RL FVE is the actor gap. If this is ~20% -> AR load is broken.", flush=True)
