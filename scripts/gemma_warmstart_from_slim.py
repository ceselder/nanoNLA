"""Build a Gemma warmstart split from a slim (text+summary) NLA dataset.

The published warmstart dataset (e.g. ceselder/qwen3-8b-nla-L24-finefineweb-100k)
is slim — it carries the reusable, model-AGNOSTIC parts (the prompt with the
`<INJECT>` placeholder, the Claude summary in `response`, the source
`detokenized_text_truncated`, `doc_id`) but NOT activations. Activations are
model-SPECIFIC, so to warmstart an NLA for a DIFFERENT target model we:

  1. reuse the (text, summary) pairs verbatim — no new API calls, no cost;
  2. re-extract activations from the NEW model at a chosen layer, at the last
     token of `detokenized_text_truncated` (the truncation point the summary
     describes); and
  3. write a NEW sidecar for the new model's tokenizer (injection token,
     neighbors, critic suffix all change with the tokenizer).

Rows whose text exceeds the new model's `--max-length` are DROPPED (their last
token would be a truncation artifact, not the text end the summary describes).

Output parquet matches the stage-3 schema the trainers expect (av_sft: prompt,
response, activation_vector, doc_id, ...; ar_sft: prompt, activation_vector,
doc_id, ...), plus a `{out}.nla_meta.yaml` sidecar.

Usage:
  python -m scripts.gemma_warmstart_from_slim \
    --slim qwen_slim/av_sft_shuf.parquet --qwen-sidecar qwen_slim/av_sft_shuf.parquet.nla_meta.yaml \
    --out gemma/av_sft_shuf.parquet --mode av \
    --base-model google/gemma-4-26B-A4B --layer 20 --injection-char ㊗ --max-length 4096
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from nla.datagen.extractors import HFExtractor
from nla.datagen.injection_tokens import compute_critic_suffix_ids
from nla.schema import SCALE_SQRT_D, compute_canonical_neighbors


def build_gemma_sidecar(tokenizer, qwen_sidecar_path, *, mode, base_model, layer,
                        d_model, injection_char, n_rows):
    """Construct the dataset sidecar for the new model's tokenizer.

    Templates are tokenizer-agnostic text → reused from the source sidecar.
    Token IDs / neighbors / critic suffix are recomputed for THIS tokenizer.
    """
    src = yaml.safe_load(Path(qwen_sidecar_path).read_text())
    templates = src["prompt_templates"]
    actor_t = templates.get("av") or templates["actor"]
    critic_t = templates.get("ar") or templates.get("critic")

    inj_ids = tokenizer.encode(injection_char, add_special_tokens=False)
    assert len(inj_ids) == 1, (
        f"injection char {injection_char!r} → {inj_ids} ({len(inj_ids)} tokens) in "
        f"{base_model}; must be exactly ONE token. Pick a different marker."
    )
    inj_id = inj_ids[0]
    left, right = compute_canonical_neighbors(tokenizer, actor_t, injection_char, inj_id)
    suffix_ids = compute_critic_suffix_ids(tokenizer, critic_t) if mode == "ar" else None

    meta = {
        "kind": "nla_dataset",
        "schema_version": 1,
        "stage": f"{mode}_sft",
        "row_count": n_rows,
        "extraction": {
            "base_model": base_model,
            "d_model": d_model,
            "layer_index": layer,
            "norm": "none",
            "injection_scale": None,        # raw inject (Karvonen norm-matches)
            "mse_scale": SCALE_SQRT_D,       # direction-only critic MSE
            "regenerated_from": str(qwen_sidecar_path),
        },
        "tokens": {
            "injection_char": injection_char,
            "injection_token_id": inj_id,
            "injection_left_neighbor_id": left,
            "injection_right_neighbor_id": right,
            "critic_suffix_ids": suffix_ids,
        },
        "prompt_templates": {"actor": actor_t, "critic": critic_t},
        "created_by": "scripts.gemma_warmstart_from_slim",
    }
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slim", required=True, help="slim source parquet (text + summary)")
    p.add_argument("--qwen-sidecar", required=True, help="source sidecar (for templates)")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", required=True, choices=["av", "ar"])
    p.add_argument("--base-model", required=True)
    p.add_argument("--layer", type=int, required=True,
                   help="extraction layer K — captures the OUTPUT of block K "
                        "(= hidden_states[K+1]); AR is then truncated to K+1 blocks")
    p.add_argument("--injection-char", required=True,
                   help="single-token-in-this-tokenizer marker (e.g. ㊗ for Gemma-4)")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=2048, help="rows per parquet write")
    p.add_argument("--max-rows", type=int, default=None, help="cap (smoke runs)")
    p.add_argument("--experts-implementation", default=None,
                   help="MoE experts kernel: 'eager' for Gemma-4 MoE on Blackwell "
                        "(B200/sm_100), whose fused torch._grouped_mm is sm_90-only.")
    args = p.parse_args()

    pf = pq.ParquetFile(args.slim)
    cols = pf.schema_arrow.names
    assert "detokenized_text_truncated" in cols, f"slim parquet lacks source text: {cols}"
    assert "activation_vector" not in cols, "input already has activations"
    need = ["prompt", "doc_id", "detokenized_text_truncated"] + (["response"] if args.mode == "av" else [])
    for c in need:
        assert c in cols, f"slim parquet missing column {c!r} (have {cols})"

    print(f"[gemma-warmstart] {args.base_model} layer={args.layer} char={args.injection_char!r} "
          f"mode={args.mode} max_len={args.max_length}", flush=True)
    _mk = {"experts_implementation": args.experts_implementation} if args.experts_implementation else None
    ext = HFExtractor(model_name=args.base_model, max_length=args.max_length,
                      batch_size=args.batch_size, model_kwargs=_mk)
    d_model = ext.d_model
    print(f"[gemma-warmstart] d_model={d_model}, building sidecar...", flush=True)
    meta = build_gemma_sidecar(
        ext.tokenizer, args.qwen_sidecar, mode=args.mode, base_model=args.base_model,
        layer=args.layer, d_model=d_model, injection_char=args.injection_char, n_rows=0,
    )
    print(f"[gemma-warmstart] inj_id={meta['tokens']['injection_token_id']} "
          f"neighbors={meta['tokens']['injection_left_neighbor_id']}/"
          f"{meta['tokens']['injection_right_neighbor_id']} "
          f"suffix={meta['tokens']['critic_suffix_ids']}", flush=True)

    vec_type = pa.list_(pa.float32(), d_model)
    writer = None
    out_cols = need
    done = 0
    kept = 0
    dropped = 0
    for batch in pf.iter_batches(batch_size=args.chunk_size, columns=need):
        if args.max_rows is not None and kept >= args.max_rows:
            break
        tbl = pa.Table.from_batches([batch])
        texts = tbl.column("detokenized_text_truncated").to_pylist()
        results = ext.extract(texts, args.layer)
        # Drop rows the new tokenizer truncated (last token = artifact, not text end).
        keep_mask = [len(r.token_ids) < args.max_length for r in results]
        vecs, rows_keep = [], []
        for i, (r, keep) in enumerate(zip(results, keep_mask)):
            if keep:
                vecs.append(r.hidden_states[-1].tolist())
                rows_keep.append(i)
        dropped += len(results) - len(rows_keep)
        if not rows_keep:
            done += tbl.num_rows
            continue
        sub = tbl.take(pa.array(rows_keep))
        sub = sub.append_column("activation_vector", pa.array(vecs, type=vec_type))
        sub = sub.append_column("activation_layer", pa.array([args.layer] * len(rows_keep), pa.int32()))
        if writer is None:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(args.out, sub.schema)
        writer.write_table(sub)
        done += tbl.num_rows
        kept += len(rows_keep)
        if done % (args.chunk_size * 5) < args.chunk_size:
            print(f"  {done} processed | {kept} kept | {dropped} dropped(>maxlen)", flush=True)
    if writer is not None:
        writer.close()

    meta["row_count"] = kept
    Path(f"{args.out}.nla_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    print(f"[gemma-warmstart] DONE → {args.out} ({kept} rows, {dropped} dropped) "
          f"+ sidecar", flush=True)


if __name__ == "__main__":
    main()
