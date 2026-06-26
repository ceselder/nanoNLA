"""NLA actor + critic inference — single-file, local (no SGLang, no nla deps).

An NLA (Natural Language Autoencoder) pair is two fine-tuned LMs that together
map activation vectors to natural language and back:

  ACTOR  (activation verbalizer)  : hidden-state vector  →  text
                                    [inject the vector via the train-time
                                     KARVONEN layer-1 ADD hook, then autoregress]

  CRITIC (activation reconstructor): text  →  hidden-state vector
                                    [truncated K+1-layer LM + Linear(d,d)
                                     head, extract at final token]

The round-trip — extract → ACTOR verbalizes → CRITIC reconstructs → MSE against
original — measures how well the verbalization captured the vector's content.

Injection is the SAME mechanism the model was trained with: a forward hook on
the OUTPUT of layer 1 that norm-matched-ADDs the activation at the marker token
(`karvonen_inject_in_residual`), NOT embedding replacement. So the client holds
the full model and generates locally — it does not use an SGLang server.

This file contains both halves:
  NLAClient  — actor inference (loads the model, registers the Karvonen hook)
  NLACritic  — load critic + reconstruct + score (optional, pure torch)

Ship alongside HF-format NLA actor + critic checkpoint dirs (each with
config.json, safetensors, tokenizer files, nla_meta.yaml). A LoRA-adapter actor
is fine too — pass base_ckpt.

Dependencies:
    uv pip install torch transformers safetensors pyyaml numpy peft
    # Optional (for --parquet CLI): uv pip install pyarrow

Multi-GPU: device_map="auto" (default) shards the model across all visible GPUs
(naive pipeline MP); the Karvonen hook fires on layer-1's output wherever that
shard lives, so it is device-agnostic. Use device_map="cuda:0" to pin one GPU.
For Gemma-4 MoE on Blackwell (B200), pass
model_kwargs={"experts_implementation": "eager"}.

Usage:
    client = NLAClient("./actor_hf")                  # full model
    client = NLAClient("./av_lora", base_ckpt="google/gemma-4-26B-A4B")  # LoRA
    text = client.generate(activation_vector)         # activation: [d_model]
    texts = client.generate_batch(vectors, temperature=0.7)

    # custom prompt (must contain <INJECT> where the vector goes):
    text = client.generate(v, prompt="What is: <concept><INJECT></concept>?")
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ─── Constants ──────────────────────────────────────────────────────────────

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
INJECT_PLACEHOLDER = "<INJECT>"
# Embedding weight key suffixes across HF architectures (Llama/Qwen/Mistral/
# Gemma use embed_tokens; GPT-2 uses wte; Falcon uses word_embeddings).
_EMBED_KEY_SUFFIXES = ("embed_tokens.weight", "wte.weight", "word_embeddings.weight")


# ─── Sidecar config ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NLAConfig:
    d_model: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    actor_prompt_template: str
    # L2-norm the vector gets rescaled to before injection. MANDATORY — the
    # model learned with this exact scale; raw-magnitude vectors are OOD.
    # Qwen7B: 150. Gemma-3-12B: 80000 (√d embed scaling inflates residual norms).
    injection_scale: float


def load_nla_config(
    checkpoint_dir: str | Path,
    tokenizer: Any,
    injection_scale_override: float | None = None,
) -> NLAConfig:
    """Parse {checkpoint_dir}/nla_meta.yaml and assert against live tokenizer.

    Catches the two most common silent-failure modes BEFORE the first request:
      - tokenizer version drift → injection char tokenizes differently
      - prompt template drift → neighbors no longer match
    Both produce CJK-flavoured output if not caught (the marker char's own
    embedding gets verbalized as the activation).
    """
    meta_path = Path(checkpoint_dir) / "nla_meta.yaml"
    assert meta_path.exists(), (
        f"no nla_meta.yaml at {checkpoint_dir!r}. Not an NLA checkpoint — "
        f"the sidecar ships alongside config.json/safetensors. If you "
        f"received a checkpoint without it, ask the provider for the sidecar."
    )
    meta = yaml.safe_load(meta_path.read_text())

    kind = meta["kind"]
    assert kind in ("nla_model", "nla_dataset"), f"unknown sidecar kind: {kind!r}"
    d_model = meta["d_model"] if kind == "nla_model" else meta["extraction"]["d_model"]

    # injection_scale: legacy/embedding-replace knob. Karvonen injection uses
    # RAW vectors (norm-matched at layer 1), so generation never reads it —
    # it's kept on NLAConfig only for informational parity. Absent → 0.0.
    inj_scale = meta.get("extraction", {}).get("injection_scale")
    if inj_scale is None:
        inj_scale = injection_scale_override if injection_scale_override is not None else 0.0

    t = meta["tokens"]
    cfg = NLAConfig(
        d_model=d_model,
        injection_char=t["injection_char"],
        injection_token_id=t["injection_token_id"],
        injection_left_neighbor_id=t["injection_left_neighbor_id"],
        injection_right_neighbor_id=t["injection_right_neighbor_id"],
        actor_prompt_template=meta["prompt_templates"].get("av")
                              or meta["prompt_templates"]["actor"],
        injection_scale=float(inj_scale),
    )

    # encode(), NOT convert_tokens_to_ids(): byte-level BPE tokenizers (Qwen,
    # GPT-2) key on the byte-string representation, not the unicode char.
    # convert_tokens_to_ids('㈎') → None for Qwen; encode('㈎') → [149705].
    live_inj = tokenizer.encode(cfg.injection_char, add_special_tokens=False)
    assert live_inj == [cfg.injection_token_id], (
        f"tokenizer drift: {cfg.injection_char!r} → {live_inj}, sidecar says "
        f"[{cfg.injection_token_id}]. Multi-token = char split = wrong "
        f"tokenizer or vocab changed."
    )
    assert live_inj[0] != tokenizer.unk_token_id, (
        f"{cfg.injection_char!r} maps to UNK"
    )

    # Verify neighbors by tokenizing the canonical prompt. One-step
    # apply_chat_template(tokenize=True) handles BOS correctly for all
    # architectures (Gemma template includes <bos>; Qwen has none).
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )
    # transformers v5 returns a BatchEncoding (dict-like); v4 returned list[int].
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    matches = [i for i, tok in enumerate(ids) if tok == cfg.injection_token_id]
    assert len(matches) == 1, (
        f"injection token appears {len(matches)}× in canonical prompt "
        f"(expected 1). Template: {content!r}"
    )
    p = matches[0]
    assert 0 < p < len(ids) - 1
    assert ids[p - 1] == cfg.injection_left_neighbor_id, (
        f"left neighbor drift: {ids[p-1]} vs sidecar "
        f"{cfg.injection_left_neighbor_id}"
    )
    assert ids[p + 1] == cfg.injection_right_neighbor_id, (
        f"right neighbor drift: {ids[p+1]} vs sidecar "
        f"{cfg.injection_right_neighbor_id}"
    )

    return cfg


# ─── Embedding table (load without materializing full model) ────────────────

def load_embedding_only(
    checkpoint_dir: str | Path,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Embedding:
    """Load ONLY the input embedding weight tensor from safetensors.

    safe_open reads the single key lazily (~2s for a 12B model vs ~30s for
    the full model). Returns a plain nn.Embedding — if the model's embedding
    class does extra work in forward (Gemma-3: ×√d), apply that scale
    separately after lookup (see resolve_embed_scale).
    """
    root = Path(checkpoint_dir)

    def _find_key(keys: list[str], where: str) -> str:
        m = [k for k in keys if k.endswith(_EMBED_KEY_SUFFIXES)]
        assert len(m) == 1, (
            f"expected exactly one input-embedding key in {where} "
            f"(suffixes {_EMBED_KEY_SUFFIXES!r}), got {m!r}"
        )
        return m[0]

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        key = _find_key(list(weight_map), str(index_path))
        shard = root / weight_map[key]
    else:
        shard = root / "model.safetensors"
        assert shard.exists(), f"no model.safetensors or .index.json at {root!r}"
        with safe_open(str(shard), framework="pt") as f:
            key = _find_key(list(f.keys()), str(shard))

    with safe_open(str(shard), framework="pt") as f:
        weight = f.get_tensor(key).to(dtype)

    vocab, d = weight.shape
    embed = torch.nn.Embedding(vocab, d, _weight=weight)
    embed.requires_grad_(False)
    embed.eval()
    return embed


# Explicit registry of model_types whose embedding forward() multiplies by √d.
# Mirrors nla/arch_adapters.py::_SCALED_EMBED_MODEL_TYPES — keep in sync.
# This file is OSS-standalone so cannot import arch_adapters; the registry is
# small and the drift hazard of a prefix-match (a hypothetical "phi-gemma-moe"
# would spuriously match .startswith("gemma")) is worse than a duplicated set.
_SCALED_EMBED_MODEL_TYPES = frozenset({
    "gemma", "gemma2", "gemma3", "gemma3_text", "t5",
})


def resolve_embed_scale(checkpoint_dir: str | Path) -> float:
    """1.0 for Qwen/Llama/Mistral; √hidden_size for Gemma/T5.

    Gemma3TextScaledWordEmbedding.forward() multiplies by √d (≈62 for
    d=3840). load_embedding_only returns a plain nn.Embedding, so that
    multiply never happens — all token embeddings are 62× too small.
    The injection vector (from residual-stream extraction) IS at full
    scale, so it dominates everything else → garbage.

    If your arch also scales embeddings in forward(), add its model_type
    to _SCALED_EMBED_MODEL_TYPES.
    """
    config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)
    model_type = getattr(text_cfg, "model_type", "") or ""
    if model_type in _SCALED_EMBED_MODEL_TYPES:
        return math.sqrt(text_cfg.hidden_size)
    return 1.0


# ─── Pure injection math ────────────────────────────────────────────────────

def normalize_activation(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    """Rescale to target_scale L2-norm. Zeros stay zero. Norm in fp32."""
    norm_fp32 = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm_fp32 / target_scale).to(v.dtype)


def karvonen_inject_in_residual(
    input_ids: torch.Tensor,      # [B, T]
    resid: torch.Tensor,          # [B, T, d] — output of the inject layer
    vectors: torch.Tensor,        # [N, d]  — RAW activation(s), N marker sites
    inj_id: int, left_id: int, right_id: int,
) -> torch.Tensor:
    """Karvonen norm-matched ADD injection: h'_p = h_p + ||h_p|| * v/||v||.

    THIS is the injection the models are trained with (norm-matched ADD on the
    residual stream entering layer 2, i.e. the OUTPUT of layer index 1) — NOT
    embedding replacement. Vectors are RAW (unnormalized); the norm-match
    against the live residual does the scaling. Valid marker = inj_id with
    sidecar neighbors at p±1; found count must equal vectors.shape[0] or CRASH.
    """
    seq_len = input_ids.shape[-1]
    assert input_ids.shape == resid.shape[:-1]
    assert vectors.ndim == 2 and vectors.shape[1] == resid.shape[-1]
    out = resid.clone()
    vectors = vectors.to(out.device, out.dtype)
    vec_idx = 0
    for b, p in (input_ids == inj_id).nonzero().tolist():
        if p == 0 or p == seq_len - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        h_p = out[b, p].clone()
        v_unit = vectors[vec_idx] / (vectors[vec_idx].norm() + 1e-9)
        out[b, p] = h_p + h_p.norm() * v_unit
        vec_idx += 1
    assert vec_idx == vectors.shape[0], (
        f"found {vec_idx} injection sites with correct neighbors, expected "
        f"{vectors.shape[0]}. Template drift, tokenizer mismatch, or prompt "
        f"missing the injection marker."
    )
    return out


def _walk_to_decoder(module):
    """Descend to the module holding `.layers` — follows .model / .language_model
    (Gemma-3/4 multimodal) / .transformer. Mirrors nla.arch_adapters."""
    target = module
    for _ in range(8):
        if hasattr(target, "layers"):
            return target
        if hasattr(target, "model"):
            target = target.model
        elif hasattr(target, "language_model"):
            target = target.language_model
        elif hasattr(target, "transformer"):
            target = target.transformer
        else:
            break
    raise AssertionError(f"could not find .layers from {type(module).__name__}")


def register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1):
    """Register the train-time Karvonen injection hook on `model`.

    An embedding hook stashes input_ids; the layer-`layer_idx` hook ADDs the
    norm-matched activation (from vectors_ref[0], a [N,d] tensor set per
    forward) at the marker positions. No-op on cache steps (seq_len < 2) and
    when no vector / no marker is present.
    """
    state = {"input_ids": None}

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["input_ids"] = ids
        return output

    def layer_hook(module, args, output):
        resid, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        ids = state["input_ids"]
        v = vectors_ref[0]
        if ids is None or resid.shape[1] < 2 or v is None or v.shape[0] == 0:
            return output
        if (ids == inj_id).sum().item() == 0:
            return output
        injected = karvonen_inject_in_residual(
            ids.to(resid.device), resid, v.to(resid.device), inj_id, left_id, right_id,
        )
        return injected if rest is None else (injected, *rest)

    model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    _walk_to_decoder(model).layers[layer_idx].register_forward_hook(layer_hook)


# ─── Client ─────────────────────────────────────────────────────────────────

class NLAClient:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        base_ckpt: str | Path | None = None,
        device_map: str = "auto",
        injection_scale_override: float | None = None,
        dtype: torch.dtype = torch.bfloat16,
        model_kwargs: dict | None = None,
    ):
        """
        checkpoint_dir: HF-format actor dir (full model OR a LoRA adapter dir)
            with nla_meta.yaml. If it's a LoRA adapter, set base_ckpt.
        base_ckpt: base model for a LoRA actor (None if checkpoint_dir is full).
        device_map: passed to from_pretrained. "auto" shards across all visible
            GPUs (naive MP) — the Karvonen layer-1 hook is device-agnostic, so
            this is naturally multi-GPU; use "cuda:0" to pin to one GPU.
        injection_scale_override: legacy sidecar knob; Karvonen injection uses
            RAW vectors (norm-matched at the layer), so this is unused for
            generation and kept only so old sidecars load.
        model_kwargs: extra from_pretrained kwargs, e.g.
            {"experts_implementation": "eager"} for Gemma-4 MoE on Blackwell.

        NOTE: this injects via the train-time Karvonen layer-1 ADD (a forward
        hook on the live model), NOT embedding replacement — so it must hold
        the full model, not just the embedding, and does not use SGLang.
        """
        checkpoint_dir = Path(checkpoint_dir)
        tok_src = str(base_ckpt) if base_ckpt is not None else str(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        self.cfg = load_nla_config(
            checkpoint_dir, self.tokenizer,
            injection_scale_override=injection_scale_override,
        )

        load_src = str(base_ckpt) if base_ckpt is not None else str(checkpoint_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            load_src, torch_dtype=dtype, device_map=device_map,
            trust_remote_code=True, **(model_kwargs or {}),
        )
        if base_ckpt is not None:
            # LoRA actor: attach the trained adapter onto the base.
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(checkpoint_dir))
        self.model.eval()

        # Karvonen layer-1 ADD hook — the injection the model was trained with.
        self._vectors_ref = [None]
        register_karvonen_hook(
            self.model, self._vectors_ref,
            self.cfg.injection_token_id,
            self.cfg.injection_left_neighbor_id,
            self.cfg.injection_right_neighbor_id,
        )
        self._embed_device = self.model.get_input_embeddings().weight.device

        print(
            f"[NLAClient] {checkpoint_dir.name}: d_model={self.cfg.d_model} "
            f"injection=Karvonen-ADD@layer1 device_map={device_map} "
            f"inj_char={self.cfg.injection_char!r}(id={self.cfg.injection_token_id})"
        )

    # ─── Core inference step ──────────────────────────────────────────────

    def _build_input_ids(self, prompt_content: str | None) -> torch.Tensor:
        """Tokenize the actor prompt (with the injection marker). Returns [1, T].

        prompt_content: user message WITH <INJECT> placeholder. None uses the
        sidecar's canonical actor template (recommended — train distribution).
        """
        if prompt_content is None:
            content = self.cfg.actor_prompt_template.format(
                injection_char=self.cfg.injection_char
            )
        else:
            assert INJECT_PLACEHOLDER in prompt_content, (
                f"custom prompt must contain {INJECT_PLACEHOLDER!r}"
            )
            content = prompt_content.replace(INJECT_PLACEHOLDER, self.cfg.injection_char)
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True,
        )
        # transformers v5 returns a BatchEncoding (dict-like); v4 a list[int].
        input_ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    @torch.no_grad()
    def _local_generate(self, v_raw: torch.Tensor, prompt_content: str | None,
                        **sampling: object) -> str:
        """Inject the RAW activation via the Karvonen layer-1 hook and decode."""
        assert torch.isfinite(v_raw).all(), "activation has NaN/Inf"
        ids_t = self._build_input_ids(prompt_content).to(self._embed_device)
        # vectors_ref carries the raw [1, d] vector; the hook norm-matches it
        # onto the layer-1 residual at the marker. No injection_scale rescale.
        self._vectors_ref[0] = v_raw.float().view(1, -1).to(self._embed_device)
        sp = {"do_sample": True, "temperature": 1.0, "max_new_tokens": 200,
              "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
              "pad_token_id": self.tokenizer.eos_token_id}
        sp.update(sampling)
        try:
            out = self.model.generate(
                input_ids=ids_t, attention_mask=torch.ones_like(ids_t), **sp,
            )
        finally:
            self._vectors_ref[0] = None
        return self.tokenizer.decode(out[0, ids_t.shape[1]:], skip_special_tokens=True)

    # ─── Public API ───────────────────────────────────────────────────────

    def generate(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        **sampling: object,
    ) -> str:
        """Decode one activation vector.

        activation:  [d_model] raw vector — injected via Karvonen norm-match (raw, unscaled).
        prompt:      user-message content with <INJECT> marker. Default (None)
                     uses the sidecar's actor template — RECOMMENDED.
        extract_explanation:  strip <explanation> tags. False returns raw gen
                     (useful for debugging — if ALL outputs are CJK, or
                     describe a CJK char in English, injection likely failed).
        sampling:    HF generate() kwargs (temperature, max_new_tokens, top_p, ...).

        Known-noisy inputs (don't over-interpret poor decodes from these):
        - Early-sequence positions (first ~10 tokens): layer-K has seen few
          tokens, residual stream hasn't accumulated signal. Decodes trend
          toward training prior.
        - Occasional high-norm activations (some models produce rare spikes,
          e.g. Qwen layer-20 early newlines at ~14k vs typical ~100-170).
          Seen during training but rare — unsurprising if decode is poor.
        """
        v = torch.as_tensor(np.asarray(activation, dtype=np.float32))
        assert v.numel() == self.cfg.d_model, (
            f"activation length {v.numel()} != d_model {self.cfg.d_model}"
        )

        text = self._local_generate(v, prompt, **sampling)

        if not extract_explanation:
            return text
        m = EXPLANATION_RE.search(text)
        if m is None:
            # Truncated gen (bump max_new_tokens) or model drift. Return
            # partial; log loudly so caller notices.
            print(f"[NLAClient] WARNING: no <explanation> tags. "
                  f"Raw[:200]={text[:200]!r}")
            return text
        return m.group(1).strip()

    def generate_batch(
        self,
        activations: Iterable[Iterable[float] | np.ndarray | torch.Tensor],
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        **sampling: object,
    ) -> list[str]:
        """Sequential local generations (one model.generate per vector)."""
        return [self.generate(v, prompt=prompt,
                              extract_explanation=extract_explanation,
                              **sampling)
                for v in activations]


# ─── CRITIC (activation reconstructor) ───────────────────────────────────────
#
# Optional — the actor is usable standalone. The critic closes the autoencoder
# loop: explanation text → predicted activation vector. MSE against the original
# gives a fidelity score (the RL training reward). Useful if you want to
# rank/filter actor decodes by how reliably they tracked the input.
#
# Architecture: first K+1 layers of the base model (K = extraction layer, e.g.
# K=20 for Qwen → 21 layers kept), final LayerNorm replaced with Identity,
# lm_head stripped, Linear(d,d) value_head bolted on. Extract at tokens[-1]
# (the prompt ends with a fixed suffix like '</text> <summary>').
#
# Checkpoint layout:
#   critic_hf/config.json           — num_hidden_layers = K+1 (pre-truncated)
#   critic_hf/model-*.safetensors   — truncated backbone
#   critic_hf/value_head.safetensors — the Linear head, loaded separately
#   critic_hf/nla_meta.yaml         — mse_scale + critic prompt template

# Final LayerNorm attribute name varies by arch. Qwen2/Llama/Mistral: "norm".
# GPT-2-style: "ln_f". Some: "final_layernorm". Extend if a new arch fails the
# constructor's assert with a clear message.
_FINAL_LN_ATTRS = ("norm", "final_layernorm", "ln_f")


class NLACritic:
    """Load an NLA critic and compute reconstruction MSE.

    Usage:
        critic = NLACritic("./critic_hf", device="cuda:0")
        mse, cos = critic.score(actor_output_text, original_activation)

    Both are returned because they carry identical information — MSE = 2(1−cos)
    under the L2-norm-to-√d normalization — but cos is usually the more
    intuitive thing to report externally. People know what cos=0.9 means;
    MSE=0.2 needs a lookup table. Pick one and be consistent.

        cos=1.0  → MSE=0.0   perfect
        cos=0.9  → MSE=0.2   good decode (typical for clean positions)
        cos=0.5  → MSE=1.0   mediocre
        cos=0.0  → MSE=2.0   orthogonal
        cos=−1.0 → MSE=4.0   antipodal (never seen in practice)

    On mse_scale vs injection_scale — different things, don't confuse them:

      injection_scale (e.g. 150 for Qwen) is the L2 norm the ACTOR expects
      vectors at — it matches the training-data distribution of activation
      norms. Get it wrong → the vector is OOD → injection fails → CJK output.

      mse_scale (√d_model ≈ 59.87 for Qwen) makes `.mean()` produce the
      d-agnostic `2(1-cos)` value. With both vectors at L2=s, per-element
      MSE is `2s²(1-cos)/d`; choosing s=√d makes s²/d=1. So the multiply
      IS load-bearing — without it you'd get `2(1-cos)/d ≈ 0.0005`. The √d
      choice also kept training-time gradient magnitudes reasonable. The
      returned MSE is already the final answer; don't rescale.
    """

    def __init__(self, checkpoint_dir: str | Path, *,
                 device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
        checkpoint_dir = Path(checkpoint_dir)
        meta = yaml.safe_load((checkpoint_dir / "nla_meta.yaml").read_text())
        assert meta["role"] in ("critic", "ar"), (
            f"sidecar role={meta['role']!r}, expected 'critic' or 'ar'. "
            f"Point NLACritic at the AR (reconstructor) checkpoint, not the AV."
        )
        ms = meta["extraction"]["mse_scale"]
        assert ms is not None, (
            f"sidecar mse_scale is None (raw-MSE mode). NLACritic.score() is "
            f"direction-only (2(1-cos)) and requires a numeric mse_scale; this "
            f"checkpoint was trained without normalization and is not supported here."
        )
        self.mse_scale: float = float(ms)
        self.template: str = (meta["prompt_templates"].get("ar")
                              or meta["prompt_templates"]["critic"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), trust_remote_code=True
        )
        # BOS invariant: training tokenized critic prompts with
        # add_special_tokens=True (reward.py, nla_generate.py). For Gemma/Llama
        # this prepends BOS; for Qwen (bos_token=None) it's a no-op. Dropping
        # BOS shifts position-0 meaning → degraded reconstruction everywhere
        # (observed: Gemma fve_nrm 0.31 vs 0.77). reconstruct() below uses
        # add_special_tokens=True — this assert catches if that ever flips.
        probe = self.tokenizer("x", add_special_tokens=True)["input_ids"]
        bos = self.tokenizer.bos_token_id
        assert bos is None or probe[0] == bos, (
            f"tokenizer has bos_token_id={bos} but add_special_tokens=True "
            f"produced first token {probe[0]}. Critic was trained with BOS "
            f"prefix — reconstruct() must match."
        )

        # config.json already has the truncated num_hidden_layers (K+1) — the
        # checkpoint was produced by training, not on-the-fly truncation here.
        backbone = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir), torch_dtype=dtype, trust_remote_code=True,
        )
        # Strip lm_head (critic never emits logits) and final LN (value head
        # sees raw residual-stream output of block K, not the normed version).
        backbone.lm_head = torch.nn.Identity()
        inner = backbone.model  # Qwen2ForCausalLM.model → Qwen2Model
        for attr in _FINAL_LN_ATTRS:
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break
        else:
            raise AssertionError(
                f"no final-LN attribute on {type(inner).__name__} — tried "
                f"{_FINAL_LN_ATTRS!r}. Add the arch's attr name to that list."
            )

        d = backbone.config.hidden_size
        self.value_head = torch.nn.Linear(d, d, bias=False, dtype=dtype)
        head_path = checkpoint_dir / "value_head.safetensors"
        assert head_path.exists(), (
            f"no value_head.safetensors at {checkpoint_dir!r}. NLA critic "
            f"checkpoints ship this alongside config.json — it's the trained "
            f"reconstruction head, not derivable from the backbone."
        )
        self.value_head.load_state_dict(load_file(str(head_path)))

        self.backbone = backbone.to(device).eval()
        self.value_head = self.value_head.to(device).eval()
        self.device = device
        print(f"[NLACritic] {backbone.config.num_hidden_layers} layers  "
              f"d_model={d}  mse_scale={self.mse_scale:.2f}")

    @torch.inference_mode()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        """Explanation text → predicted activation vector (raw, unnormalized)."""
        prompt = self.template.format(explanation=explanation)
        # add_special_tokens=True: Gemma critic was trained with BOS prefix
        # (critic_prompt_template is a raw string, not chat-template-processed).
        # Qwen has bos_token=None so this is a no-op there. Omitting BOS for
        # Gemma shifts position-0 meaning → degraded reconstruction everywhere.
        ids = self.tokenizer(prompt, return_tensors="pt",
                             add_special_tokens=True)["input_ids"].to(self.device)
        h = self.backbone.model(ids, use_cache=False).last_hidden_state[0, -1]  # last token
        return self.value_head(h).float().cpu()

    def score(self, explanation: str,
              original: np.ndarray | torch.Tensor) -> tuple[float, float]:
        """(direction-MSE, cos-sim). Both pred+gold L2-normalized to mse_scale
        before MSE → MSE = 2(1-cos), range [0, 4]. Orthogonal = 2."""
        pred = self.reconstruct(explanation)
        gold = torch.as_tensor(np.asarray(original, dtype=np.float32))
        pred_n = pred / pred.norm().clamp_min(1e-12) * self.mse_scale
        gold_n = gold / gold.norm().clamp_min(1e-12) * self.mse_scale
        mse = ((pred_n - gold_n) ** 2).mean().item()
        cos = (pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item()
        return mse, cos


# ─── CLI ────────────────────────────────────────────────────────────────────

def _main() -> None:
    """Feed vectors from a parquet's activation_vector column, or smoke-test
    with one random vector. ALL outputs in CJK (or English describing a CJK
    char)? Injection likely failed — see README §Debugging."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="HF-format NLA actor dir (with nla_meta.yaml); "
                    "a full model, or a LoRA adapter dir (then pass --base-ckpt)")
    ap.add_argument("--base-ckpt", default=None,
                    help="base model if `checkpoint` is a LoRA adapter dir")
    ap.add_argument("--device-map", default="auto",
                    help="'auto' (multi-GPU naive MP) or e.g. 'cuda:0'")
    ap.add_argument("--experts-implementation", default=None,
                    help="'eager' for Gemma-4 MoE on Blackwell/B200")
    ap.add_argument("--parquet", default=None,
                    help="Parquet with activation_vector column. Default: "
                         "smoke-test with one random vector.")
    ap.add_argument("--n", type=int, default=3, help="rows to sample from parquet")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--injection-scale", type=float, default=None,
                    help="Override sidecar value (OOD — only if sidecar is "
                         "wrong/missing)")
    ap.add_argument("--prompt", default=None,
                    help="Custom user content with <INJECT> marker. Default: "
                         "sidecar's actor template (recommended).")
    ap.add_argument("--raw", action="store_true",
                    help="Print raw output (no tag extraction)")
    args = ap.parse_args()

    client = NLAClient(
        args.checkpoint,
        base_ckpt=args.base_ckpt,
        device_map=args.device_map,
        injection_scale_override=args.injection_scale,
        model_kwargs=({"experts_implementation": args.experts_implementation}
                      if args.experts_implementation else None),
    )

    if args.parquet is None:
        print("[smoke] No parquet — generating for one random unit vector.")
        v = np.random.randn(client.cfg.d_model).astype(np.float32)
        out = client.generate(
            v, prompt=args.prompt,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            extract_explanation=not args.raw,
        )
        print(f"\n{out}\n")
        return

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.parquet)
    batch = next(pf.iter_batches(batch_size=args.n, columns=["activation_vector"]))
    # flatten→reshape avoids to_pylist()'s O(n×d) Python-float creation
    flat = batch.column("activation_vector").flatten().to_numpy(
        zero_copy_only=False).astype(np.float32)
    vecs = flat.reshape(len(batch), -1)

    for i, v in enumerate(vecs):
        out = client.generate(
            v, prompt=args.prompt,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            extract_explanation=not args.raw,
        )
        print(f"─── [{i}]  ||v||={np.linalg.norm(v):.1f} ─────────────────────")
        print(out)
        print()


if __name__ == "__main__":
    _main()
