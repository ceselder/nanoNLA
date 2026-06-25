# Training an NLA on a new model

End-to-end recipe for training a Natural Language Autoencoder (AV verbalizer +
AR reconstructor) on **any** decoder LM:
**activation extraction → AV/AR warm-start SFT → GRPO RL**, all logged to Weights
& Biases. Everything runs from this repo's self-contained trainers
(`nla/train_sft.py`, `nla/train_rl_self_contained.py`) — no external RL
framework needed. The exact verified invocations live in
`scripts/sbatch_{av,ar}_sft_lora_fixed.sh` and `scripts/sbatch_rl_fixed.sh`;
`scripts/smoke_fixed_pipeline.sh` exercises the whole chain in minutes.

Prereqs on the box:

```bash
export HF_TOKEN=...           # model + corpus download
export WANDB_API_KEY=...      # logging
export ANTHROPIC_API_KEY=...  # stage-2 explanations only
```

## 0 · Choose the layer

NLAs read a residual-stream layer about **two-thirds deep** (semantic features
have formed; the unembedding hasn't taken over). **No code change is needed for a
new model** — `run_pipeline` just reads `base_model` + `layer_index` from the
datagen YAML (`resolve()` passes an unknown model straight through):

```
layer_index = (2 * num_layers) // 3      # Qwen3-0.6B 28→18 · Qwen3-8B 36→24 · Llama-3.3-70B 80→53
```

(If you'll reuse it, optionally add a `ModelPreset` to
`nla/datagen/model_presets.py` and reference it with `model: <key>`.)

## 1 · Generate data

Copy a `configs/datagen/*.yaml`, edit the head, run the orchestrator:

```yaml
base_model: <your/model>
layer_index: <2/3 depth>
output_dir: /data/nla/<run>
corpus:
  name: <hf-dataset | local.parquet>    # any text corpus; a .parquet path uses Dataset.from_parquet
  split: train
  text_column: text
  start: 0
  length: 100000                        # docs × positions_per_doc = #activations
stage0: {positions_per_doc: 10, chunk_size: 256, seed: 42,
         extractor_kwargs: {batch_size: 12, max_length: 4096}}
stage1: {av_sft_frac: 0.25, ar_sft_frac: 0.25, rl_frac: 0.50, seed: 42}
stage2:
  provider_cls: nla.datagen.providers.BatchAnthropicProvider
  provider_kwargs: {model: claude-sonnet-4-6, max_tokens: 300, max_batch_size: 10000}
  chunk_size: 50000
stage3: {keep_debug_metadata: true}
shuffle: {enabled: true, seed: 42}
storage_cls: nla.datagen.storage.LocalStorage
```

```bash
python -m nla.datagen.run_pipeline --config configs/datagen/<your>.yaml
#  → av_sft_shuf.parquet · ar_sft_shuf.parquet · rl_shuf.parquet  (+ .nla_meta.yaml sidecars)
```

Stage 0 (extraction) is the **only model-specific step** — it forward-hooks
`layer_index` and writes raw activations. Stages 1–3 are model-agnostic.
Smoke-test extraction cheaply with `--stages 0,1` (skips the paid API
explanations); a few dozen docs is enough to confirm the dims are right.

> **Using the published dataset?** The HF release
> ([ceselder/qwen3-8b-nla-L24-finefineweb-100k](https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k))
> ships **slim** — text + provenance, **no `activation_vector` column** (the
> vectors are deterministic given text+layer, so they're regenerated rather than
> stored). The trainers below require that column, so first re-add it:
> ```bash
> python tools/regenerate_activations.py \
>   --in av_sft_shuf.parquet --out av_sft_shuf.full.parquet \
>   --base-model Qwen/Qwen3-8B --max-length 4096
> ```
> (repeat for ar_sft / rl) and point `--parquet` at the `.full.parquet`. Only
> needed for the published slim release — your own `run_pipeline` output already
> has the column.

## 2 · AV SFT — verbalizer (activation → text)

`nla/train_sft.py --mode av` loads the base model (4-bit + LoRA by default for
single-GPU budgets), hooks the Karvonen norm-matched injection at the layer-1
output, and trains cross-entropy on response tokens only:

```bash
python -m nla.train_sft --mode av --base-ckpt <your/model> \
  --parquet /data/nla/<run>/av_sft_shuf.parquet \
  --sidecar /data/nla/<run>/av_sft_shuf.parquet \
  --save-dir /ckpts/nla/<run>_av \
  --num-steps 1000 --batch-size 64 --gradient-accumulation-steps 1 \
  --use-lora --quant 4bit --lora-r 128 --lora-alpha 16 \
  --lr 3e-5 --min-lr 3e-6 --lr-warmup-steps 50 --max-grad-norm 1.0 \
  --save-every 500 --wandb-project nla-<run> --wandb-name av_sft --seed 0
```

(Verified invocation: `scripts/sbatch_av_sft_lora_fixed.sh`.)

## 3 · AR SFT — reconstructor (text → activation)

Same entry point with `--mode ar`. The trainer truncates the base model
in-process to `--ar-num-layers` blocks + a `Linear(d, d)` value head — **set
`--ar-num-layers` to `layer_index + 1`** (the critic needs the *output of*
block K, so block K must exist). The final RMSNorm is stripped by default
(`--strip-final-norm`, recorded in `ar_meta.json` inside the checkpoint):

```bash
python -m nla.train_sft --mode ar --base-ckpt <your/model> \
  --parquet /data/nla/<run>/ar_sft_shuf.parquet \
  --sidecar /data/nla/<run>/ar_sft_shuf.parquet \
  --save-dir /ckpts/nla/<run>_ar \
  --num-steps 1000 --batch-size 64 --gradient-accumulation-steps 1 \
  --use-lora --quant 4bit --lora-r 128 --lora-alpha 16 \
  --ar-num-layers <layer_index + 1> \
  --lr 3e-5 --min-lr 3e-6 --lr-warmup-steps 50 --max-grad-norm 1.0 \
  --save-every 500 --wandb-project nla-<run> --wandb-name ar_sft --seed 0
```

(Verified invocation: `scripts/sbatch_ar_sft_lora_fixed.sh`.)

For an honest held-out FVE during training, pass `--heldout-parquet` pointing
at a doc-disjoint parquet (e.g. the AV split — disjoint from AR data by
stage-1 construction). Post-hoc, `scripts/eval_ar_heldout.py` runs the same
doc-disjoint eval (plus a shuffled control) against a saved checkpoint.
Checkpoints save in HF format directly — no conversion step.

## 4 · RL — GRPO (reward = −reconstruction MSE)

The self-contained trainer rolls out the AV with HF `generate()` under the
Karvonen hook, scores with the AR critic, and does GRPO. **`--train-critic` is
required in practice**: with a frozen critic the reward is static, advantages
collapse to ≈0, and RL does nothing (FVE stays dead-flat).

```bash
python -m nla.train_rl_self_contained \
  --av-ckpt /ckpts/nla/<run>_av/iter_0001000 \
  --ar-ckpt /ckpts/nla/<run>_ar/iter_0001000 \
  --base-ckpt <your/model> --quant 4bit \
  --rl-parquet /data/nla/<run>/rl_shuf.parquet \
  --sidecar    /data/nla/<run>/rl_shuf.parquet \
  --save-dir   /ckpts/nla/<run>_rl \
  --num-steps 500 --batch-prompts 16 --group-size 16 \
  --max-new-tokens 150 --temperature 1.0 \
  --lr 1e-5 --kl-beta 0.01 --clip-eps 0.2 \
  --train-critic --critic-lr 5e-5 \
  --logp-micro-batch 2 --max-rows 30000 \
  --save-every 50 --eval-every 10 --eval-n-prompts 20 --eval-skip-rows 35000 \
  --max-grad-norm 1.0 \
  --wandb-project nla-<run> --wandb-name rl_grpo --seed 0
```

`--av-ckpt` is the AV LoRA dir; `--ar-ckpt` is the AR LoRA dir (must contain
`ar_meta.json`); both apply onto `--base-ckpt`. Resume with
`--resume-from-lora <iter_dir> --start-step <N>` (see
`scripts/sbatch_rl_fixed_resume.sh` for a self-chaining SLURM variant).

(Verified invocation: `scripts/sbatch_rl_fixed.sh`. A faster multi-GPU
rollout path exists in `nla/train_rl_vllm.py`.)

Every stage streams to wandb; checkpoints land in each `--save-dir`. That's the
whole loop — extraction → warm-start → RL.

---

### Validated

The datagen/extraction path here is **smoke-tested on Qwen3-0.6B** (layer 18,
no preset, no code change): stage 0 produced correct 1024-dim activations and
stage 1 split them cleanly. The SFT/RL commands above are the exact ones from
the verified Qwen3-8B fixed-pipeline run
(`scripts/sbatch_{av,ar}_sft_lora_fixed.sh`, `scripts/sbatch_rl_fixed.sh`)
with the model/paths parameterized.
