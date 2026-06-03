# Training an NLA on a new model

End-to-end recipe for training a Natural Language Autoencoder (AV verbalizer +
AR reconstructor) on **any** decoder LM:
**activation extraction → AV/AR warm-start SFT → GRPO RL**, all logged to Weights
& Biases. Generalizes the worked [Qwen3-8B run](qwen3_8b_run.md); install
Miles + SGLang first per [setup.md](setup.md).

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

## 2 · AV SFT — verbalizer (activation → text)

```bash
export NLA_KARVONEN_INJECTION=1                  # norm-matched injection (the key knob)
export INSTRUCT_MODEL=<your/model>
export AV_SFT_PARQUET=/data/nla/<run>/av_sft_shuf.parquet
export INJ_SCALE=raw
export SAVE_DIR=/ckpts/nla/<run>_av
export WANDB_PROJECT=nla-<run>  WANDB_NAME=av_sft
cd "$(python -c 'import miles,os;print(os.path.dirname(miles.__file__))')/.."
bash configs/actor_sft.sh \
  --actor-num-gpus-per-node 2 --attn-implementation sdpa \
  --rollout-batch-size 32 --global-batch-size 32 --micro-batch-size 4 \
  --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
  --num-rollout 1000 --save-interval 500 --use-wandb --wandb-project "$WANDB_PROJECT"
```

## 3 · AR SFT — reconstructor (text → activation)

Build the truncated critic init (a copy of the model truncated to your layer +
a `Linear(d, d)` head), then SFT it:

```bash
python -m nla.scripts.prepare_critic_checkpoint \
  --base-model <your/model> --num-layers <layer_index> \
  --dataset-sidecar /data/nla/<run>/ar_sft_shuf.parquet \
  --output /ckpts/nla/<run>_critic_init

export AR_SFT_PARQUET=/data/nla/<run>/ar_sft_shuf.parquet
export NLA_FREEZE_VALUE_HEAD=1                    # stability (see qwen3_8b_run.md)
export CRITIC_INIT_CKPT=/ckpts/nla/<run>_critic_init
export SAVE_DIR=/ckpts/nla/<run>_ar
export WANDB_PROJECT=nla-<run>  WANDB_NAME=ar_sft
cd "$(python -c 'import miles,os;print(os.path.dirname(miles.__file__))')/.."
bash configs/critic_sft.sh \
  --actor-num-gpus-per-node 2 --attn-implementation sdpa \
  --rollout-batch-size 256 --global-batch-size 256 --micro-batch-size 64 \
  --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
  --num-rollout 1000 --save-interval 200 --use-wandb --wandb-project "$WANDB_PROJECT"
```

Convert both checkpoints to HF (`tools/convert_fsdp_to_hf.py` for the AV;
`launch/convert_ar_dcp_to_hf.py --num-layers <layer_index+1>` for the AR — see
[qwen3_8b_run.md §3](qwen3_8b_run.md)).

## 4 · RL — GRPO (reward = −reconstruction MSE)

The self-contained trainer rolls out the AV with HF `generate()` under the
Karvonen hook, scores with the frozen AR, and does GRPO:

```bash
python -m nla.train_rl_self_contained \
  --av-ckpt /ckpts/nla/<run>_av/iter_0001000/hf \
  --ar-ckpt /ckpts/nla/<run>_ar/iter_0001000/hf \
  --rl-parquet /data/nla/<run>/rl_shuf.parquet \
  --sidecar    /data/nla/<run>/rl_shuf.parquet \
  --save-dir   /ckpts/nla/<run>_rl \
  --num-steps 250 --batch-prompts 8 --group-size 4 \
  --lr 1e-6 --kl-beta 0.01 --clip-eps 0.2 --lora-r 16 --lora-alpha 32 \
  --wandb-project nla-<run> --wandb-name rl_grpo
```

Every stage streams to wandb; checkpoints land in each `SAVE_DIR`. That's the
whole loop — extraction → warm-start → RL.

---

### Validated

The datagen/extraction path here is **smoke-tested on Qwen3-0.6B** (layer 18,
no preset, no code change): stage 0 produced correct 1024-dim activations and
stage 1 split them cleanly. The SFT/RL commands above are the exact ones from the
[Qwen3-8B production run](qwen3_8b_run.md) with the model/paths parameterized.
