"""Regenerate the activation_vector column for a slim NLA dataset parquet.

The published NLA datasets (e.g. ceselder/qwen3-8b-nla-L24-finefineweb-100k)
ship WITHOUT the activation_vector column to stay small — the raw vectors are
deterministic given the input text + layer, so they are regenerated here rather
than stored (~7.6 GB of float arrays -> ~0.8 GB of text/provenance).

For each row we re-run `detokenized_text_truncated` through the base model and
take the layer-K hidden state at the FINAL token — which is the original
extraction position by construction, since stage 0 truncated the text to end
exactly at it. Output matches the original stage-0 schema (activation_vector
re-added, raw / unnormalized — norm="none").

Usage:
    python tools/regenerate_activations.py \\
        --in av_sft_shuf.parquet --out av_sft_shuf.full.parquet \\
        --base-model Qwen/Qwen3-8B
"""

import argparse

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen.extractors import HFExtractor


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="inp", required=True, help="slim parquet (no activation_vector)")
    p.add_argument("--out", required=True, help="output parquet (activation_vector re-added)")
    p.add_argument("--base-model", required=True, help="HF base model, e.g. Qwen/Qwen3-8B")
    p.add_argument("--chunk-size", type=int, default=512, help="rows per write (bounds memory)")
    p.add_argument("--batch-size", type=int, default=16, help="model forward batch size")
    args = p.parse_args()

    pf = pq.ParquetFile(args.inp)
    assert "activation_vector" not in pf.schema_arrow.names, (
        "input already has an activation_vector column — nothing to regenerate"
    )
    assert "detokenized_text_truncated" in pf.schema_arrow.names, (
        "input lacks detokenized_text_truncated — cannot regenerate without the source text"
    )

    ext = HFExtractor(model_name=args.base_model, batch_size=args.batch_size)
    vec_type = pa.list_(pa.float32(), ext.d_model)

    writer = None
    done = 0
    for batch in pf.iter_batches(batch_size=args.chunk_size):
        tbl = pa.Table.from_batches([batch])
        texts = tbl.column("detokenized_text_truncated").to_pylist()
        layers = set(tbl.column("activation_layer").to_pylist())
        assert len(layers) == 1, f"chunk mixes activation_layer values: {layers}"
        layer = layers.pop()

        results = ext.extract(texts, layer)
        # the prefix ends exactly at the extraction token -> it is the last token
        vecs = [r.hidden_states[-1].tolist() for r in results]
        tbl = tbl.append_column("activation_vector", pa.array(vecs, type=vec_type))

        if writer is None:
            writer = pq.ParquetWriter(args.out, tbl.schema)
        writer.write_table(tbl)
        done += tbl.num_rows
        print(f"  {done} rows", flush=True)

    if writer is not None:
        writer.close()
    print(f"wrote {done} rows with activation_vector -> {args.out}")


if __name__ == "__main__":
    main()
