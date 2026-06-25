"""Live architecture probe for the Gemma-4 NLA port — run before training.

Loads the model on CPU/meta where possible and verifies every arch-dependent
path the NLA pipeline relies on resolves correctly for Gemma-4's multimodal
MoE wrapper (which nests the text decoder under .model.language_model and
carries a vision tower we must not touch):

  1. resolve_text_config → hidden_size
  2. resolve_decoder_layers → the text decoder stack (count, not vision)
  3. get_input_embeddings works
  4. the karvonen-hook layer walk finds .layers
  5. LoRA target_modules (q/k/v/o_proj) exist in the TEXT decoder, and whether
     the vision tower ALSO matches them (→ would need scoping)
  6. tokenizer.apply_chat_template works (transformers v5 API)
  7. injection marker ㊗ is single-token; compute_canonical_neighbors works

Run: python -m scripts.gemma4_arch_probe --model google/gemma-4-26B-A4B
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config, resolve_embed_scale
from nla.schema import compute_canonical_neighbors

ACTOR_T = ("You are a meticulous AI researcher ... <concept>{injection_char}</concept>\n\n"
           "Please provide an explanation.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-26B-A4B")
    p.add_argument("--char", default="㊗")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[tok] {tok.__class__.__name__}")
    ids = tok.encode(args.char, add_special_tokens=False)
    print(f"[6] marker {args.char!r} -> {ids} ({'OK single-token' if len(ids)==1 else 'BAD multi-token'})")
    try:
        msg = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                      tokenize=False, add_generation_prompt=True)
        print(f"[6] apply_chat_template OK ({len(msg)} chars)")
    except Exception as e:
        print(f"[6] apply_chat_template FAILED: {type(e).__name__}: {e}")
    try:
        l, r = compute_canonical_neighbors(tok, ACTOR_T, args.char, ids[0])
        print(f"[7] neighbors L={l} R={r}")
    except Exception as e:
        print(f"[7] neighbors FAILED: {type(e).__name__}: {e}")

    print(f"[load] {args.model} (bf16, device_map=auto)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    print(f"[model] top type: {type(model).__name__}")
    tc = resolve_text_config(model.config)
    print(f"[1] resolve_text_config.hidden_size = {getattr(tc,'hidden_size',None)} "
          f"(expect 2816); model_type={getattr(tc,'model_type',None)}")
    print(f"[1] resolve_embed_scale = {resolve_embed_scale(model.config):.3f} (expect sqrt(2816)=53.07)")
    try:
        layers = resolve_decoder_layers(model)
        print(f"[2] resolve_decoder_layers → {len(layers)} layers (expect 30); "
              f"layer0 type {type(layers[0]).__name__}")
    except Exception as e:
        print(f"[2] resolve_decoder_layers FAILED: {type(e).__name__}: {e}")
    try:
        emb = model.get_input_embeddings()
        print(f"[3] get_input_embeddings: {type(emb).__name__} weight {tuple(emb.weight.shape)}")
    except Exception as e:
        print(f"[3] get_input_embeddings FAILED: {type(e).__name__}: {e}")

    # [4] karvonen walk (mirror the trainer logic)
    target = model
    path = []
    while not hasattr(target, "layers"):
        if hasattr(target, "model"): target = target.model; path.append("model")
        elif hasattr(target, "language_model"): target = target.language_model; path.append("language_model")
        elif hasattr(target, "transformer"): target = target.transformer; path.append("transformer")
        else: break
    print(f"[4] karvonen walk: {'.'.join(path)} → "
          f"{'.layers OK ('+str(len(target.layers))+')' if hasattr(target,'layers') else 'FAILED'}")

    # [5] LoRA target presence in text decoder vs vision tower
    tgt = {"q_proj", "k_proj", "v_proj", "o_proj"}
    text_hits, vis_hits = set(), set()
    for n, _ in model.named_modules():
        leaf = n.split(".")[-1]
        if leaf in tgt:
            if "vision" in n or "vision_tower" in n: vis_hits.add(leaf)
            else: text_hits.add(leaf)
    print(f"[5] LoRA targets in text decoder: {sorted(text_hits)}")
    print(f"[5] LoRA targets ALSO in vision tower: {sorted(vis_hits)} "
          f"{'← MUST scope target_modules to language_model!' if vis_hits else '(none — safe)'}")
    print("PROBE DONE")


if __name__ == "__main__":
    main()
