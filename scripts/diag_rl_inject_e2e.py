"""End-to-end injection test in the ACTUAL generate() path.

Loads the RL actor exactly as train_rl does (base + AV-SFT LoRA), registers the
same Karvonen hook, and generates greedily for the same prompt under three
conditions:
  (A) real activation injected
  (B) a DIFFERENT row's activation injected
  (C) no injection (vectors_ref = None)

If injection actually fires: (A) and (B) describe DIFFERENT concepts, and both
differ from (C). If the hook silently no-ops (e.g. embed-hook never captured
input_ids for Gemma-4), all three are identical -> injection is broken.
"""
import sys, glob, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, "/workspace/gemma-nla/nanoNLA")
from nla.train_rl_self_contained import _register_karvonen_hook, build_prompt_text, cjk_fraction
from nla.config import load_nla_config

BASE = "google/gemma-4-26B-A4B"
SIDE = "/workspace/gemma-nla/data/rl_gemma.parquet"
AV = sorted(glob.glob("/workspace/gemma-nla/ckpts/gemma4_av_sft_v2_fullep/iter_*"))[-1]
print(f"[av] {AV}", flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
cfg = load_nla_config(SIDE, tok)
print(f"[cfg] inj_id={cfg.injection_token_id} left={cfg.injection_left_neighbor_id} right={cfg.injection_right_neighbor_id}", flush=True)

base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    device_map="cuda:0", experts_implementation="eager",
)
actor = PeftModel.from_pretrained(base, AV, adapter_name="default", is_trainable=False)
actor.set_adapter("default")
actor.eval()

vectors_ref = [None]
_register_karvonen_hook(actor, vectors_ref, cfg.injection_token_id,
                        cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id, layer_idx=1)

pf = pq.ParquetFile(SIDE)
rg = pf.read_row_group(0, columns=["prompt", "activation_vector"])
prompts = rg.column("prompt").to_pylist()
acts = rg.column("activation_vector").to_pylist()


def gen(ids, v):
    vectors_ref[0] = v
    try:
        with torch.no_grad():
            out = actor.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=100, do_sample=False, top_p=1.0, top_k=0,
                repetition_penalty=1.0, pad_token_id=tok.eos_token_id,
            )
    finally:
        vectors_ref[0] = None
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


for i in [0, 1, 2]:
    ptext = build_prompt_text(prompts[i], cfg.injection_char, tok)
    ids = torch.tensor([tok.encode(ptext, add_special_tokens=False)], device="cuda:0")
    act_i = torch.tensor(acts[i], dtype=torch.float32, device="cuda:0").unsqueeze(0)
    j = (i + 5) % len(acts)
    act_j = torch.tensor(acts[j], dtype=torch.float32, device="cuda:0").unsqueeze(0)

    t_real = gen(ids, act_i)
    t_other = gen(ids, act_j)
    t_none = gen(ids, None)

    print(f"\n================= row {i} =================", flush=True)
    print(f"[A real act {i}] cjk={cjk_fraction(t_real):.2f} :: {t_real[:300]!r}")
    print(f"[B other act {j}] cjk={cjk_fraction(t_other):.2f} :: {t_other[:300]!r}")
    print(f"[C no-inject ] cjk={cjk_fraction(t_none):.2f} :: {t_none[:300]!r}")
    print(f"  A==B? {t_real.strip()==t_other.strip()}   A==C? {t_real.strip()==t_none.strip()}")

print("\n[VERDICT] If A==B==C -> injection NOT firing. If all differ & A is concept-specific -> injection works.", flush=True)
