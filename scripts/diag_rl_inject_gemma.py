"""Decisive check: does the RL rollout encode path actually place the marker
token (== sidecar inj_id, with correct neighbors) for Gemma-4?

Runs the EXACT rollout path: build_prompt_text(row["prompt"]) ->
encode(add_special_tokens=False), then counts (ids == inj_id) and inspects the
neighbors. If matches==0 the injection hook silently no-ops -> actor never sees
the activation -> FVE craters with no crash.
"""
import sys, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/gemma-nla/nanoNLA")
from nla.train_rl_self_contained import build_prompt_text
from nla.config import load_nla_config

BASE = "google/gemma-4-26B-A4B"
SIDE = "/workspace/gemma-nla/data/rl_gemma.parquet"

tok = AutoTokenizer.from_pretrained(BASE)
cfg = load_nla_config(SIDE, tok)
print(f"[sidecar] inj_id={cfg.injection_token_id} left={cfg.injection_left_neighbor_id} "
      f"right={cfg.injection_right_neighbor_id} char={cfg.injection_char!r}")
print(f"[encode(char) bare] {tok.encode(cfg.injection_char, add_special_tokens=False)}")

pf = pq.ParquetFile(SIDE)
rg = pf.read_row_group(0, columns=["prompt"])
prompts = rg.column("prompt").to_pylist()

bad = 0
for i in range(min(3, len(prompts))):
    row = prompts[i]
    ptext = build_prompt_text(row, cfg.injection_char, tok)
    ids = tok.encode(ptext, add_special_tokens=False)
    t = torch.tensor(ids)
    matches = (t == cfg.injection_token_id).nonzero().flatten().tolist()
    print(f"\n=== row {i}: {len(matches)} marker match(es) in rollout-encoded prompt (len {len(ids)}) ===")
    if not matches:
        bad += 1
        # Where did the char go? find the char in the string, show local encode.
        j = ptext.find(cfg.injection_char)
        print(f"  !!! NO MARKER TOKEN. char at str-pos {j}. context: {ptext[max(0,j-30):j+30]!r}")
        # encode just the local window to see what id the char became in-context
        if j >= 0:
            win = ptext[max(0, j-10):j+10]
            print(f"  local-window encode: {tok.encode(win, add_special_tokens=False)}")
        continue
    for p in matches:
        L = ids[p-1] if p > 0 else None
        R = ids[p+1] if p+1 < len(ids) else None
        okL = "OK" if L == cfg.injection_left_neighbor_id else "MISMATCH"
        okR = "OK" if R == cfg.injection_right_neighbor_id else "MISMATCH"
        print(f"  pos {p}: left={L} [{okL}]  right={R} [{okR}]")
        print(f"  window decode: {tok.decode(ids[max(0,p-6):p+7])!r}")

print(f"\n[VERDICT] {'BROKEN — markerless rollout prompts -> silent no-inject' if bad else 'marker present in rollout path'}")
