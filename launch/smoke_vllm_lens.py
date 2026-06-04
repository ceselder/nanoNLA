"""Definitive greedy test: does vllm-lens injection actually change the output?

Greedy (temperature=0) → deterministic. Same prompt, inject at the marker with
scale 0 (no-op control), 1 (real), 8 (strong). If scale>0 diverges from baseline
and scale=0 matches it, injection is genuinely applied (not sampling noise).
"""
import argparse
import os

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm_lens import SteeringVector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--layer", type=int, default=1)
    args = p.parse_args()
    os.environ.setdefault("HF_HOME", "/workspace-vast/pretrained_ckpts")

    d_model = AutoConfig.from_pretrained(args.model).hidden_size
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.5,
              max_model_len=512, enforce_eager=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    inj = tok.encode("㈎", add_special_tokens=False)
    assert len(inj) == 1
    user = "You are a meticulous AI researcher describing an activation vector.\n\n<concept>㈎</concept>\n\nPlease provide an explanation."
    prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    marker_pos = [i for i, t in enumerate(ids) if t == inj[0]][0]

    torch.manual_seed(0)
    act = torch.randn(1, d_model, dtype=torch.float32)

    def gen(scale):
        if scale is None:
            sp = SamplingParams(temperature=0.0, max_tokens=50)
        else:
            sv = SteeringVector(activations=act, layer_indices=[args.layer],
                                scale=float(scale), norm_match=True,
                                position_indices=[marker_pos])
            sp = SamplingParams(temperature=0.0, max_tokens=50,
                                extra_args={"apply_steering_vectors": [sv]})
        return llm.generate([prompt], sp)[0].outputs[0].text

    base = gen(None)
    print(f"\nBASELINE (no inject): {base[:130]!r}", flush=True)
    results = {}
    for s in [0.0, 1.0, 8.0]:
        out = gen(s)
        results[s] = out
        print(f"scale={s:>4}  differs_from_baseline={out != base}  : {out[:130]!r}", flush=True)

    control_ok = results[0.0] == base          # scale 0 must equal baseline
    injects = results[8.0] != base             # strong scale must diverge
    print(f"\nCONTROL(scale0==baseline)={control_ok}  STRONG_INJECT_DIVERGES={injects}", flush=True)
    print("VERDICT: INJECTION_WORKS" if (control_ok and injects) else "VERDICT: INJECTION_NO_EFFECT", flush=True)


if __name__ == "__main__":
    main()
