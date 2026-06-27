"""Self-contained NLA GRPO training: Karvonen injection, LoRA actor.

Architecture:
  - Actor: Qwen3-8B + Karvonen layer-1 ADD injection (from AV-SFT ckpt).
           Wrapped with LoRA so backbone stays frozen (memory: 16GB base + ~100MB
           LoRA + small Adam states + activations all fit on one H200).
  - Reference policy: same model with LoRA adapter disabled (PEFT context manager).
           No second model copy.
  - Critic: NLACriticModel (truncated K+1-layer backbone + value head). Frozen,
            bf16, eval-only — produces predicted activation given the actor's
            explanation. Reward = -MSE(pred, gold).
  - Rollout: HF model.generate() with the same Karvonen hook used in training.
             Slower than vLLM but no weight-sync complexity. On-policy.

GRPO objective (DeepSeekMath / DeepSeek-R1):
  L = -E[min(r * A, clip(r, 1-eps, 1+eps) * A)] + beta * KL(pi || pi_ref)
  where r = exp(log_p_new - log_p_old), token-level
        A = group-relative reward, per-prompt baseline
        KL ≈ exp(log_p_ref - log_p_new) - (log_p_ref - log_p_new) - 1 (k3 estimator)

Per step:
  1. Sample B prompts from rl_shuf.parquet (each carries a gold activation v).
  2. Generate G samples per prompt with sampling temperature.
     Collect old log_probs from generate's output_scores.
  3. Extract <explanation>; failed extractions get reward = -2.0 (paper default,
     equals MSE on fully-orthogonal unit vectors — i.e. maximally bad).
  4. Score with critic → r_ij = -mse_nrm.
  5. Group-relative advantage: A_ij = (r_ij - mean_j) / std_j (per prompt group).
  6. Training-mode forward of the actor: compute new log_probs (LoRA active).
  7. Reference forward (same batch, LoRA disabled): compute ref log_probs.
  8. GRPO loss, backward + Adam.
"""

import argparse
import math
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import (LoraConfig, PeftModel, get_peft_model,
                  inject_adapter_in_model, prepare_model_for_kbit_training)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import wandb

from nla.config import load_nla_config
from nla.injection import karvonen_inject_in_residual
from nla.models import NLACriticModel
from nla.schema import (
    compute_predict_mean_baselines,
    extract_explanation,
    normalize_activation,
    resolve_target_scale,
)
from nla.train_sft import _resolve_device_map, init_critic_from_base


def cjk_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if "CJK" in unicodedata.name(c, "")) / len(text)


def _register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1):
    """Attach a forward hook on layer `layer_idx` of the HF actor.

    The hook reads `vectors_ref[0]` (a [N, d] tensor set by the caller before
    each forward). N must equal the number of marker positions in the current
    input_ids. Hook is a no-op when seq_len < 2 (autoregressive cache-step
    forwards pass a single new token; no marker present).
    """
    # input_ids isn't passed to layer hooks — capture it via an embedding hook
    # that stashes a thread-local ref.
    state = {"input_ids": None}

    def embed_hook(module, args, kwargs, output):
        # args[0] is input_ids in HF embeddings; sometimes passed as kwarg.
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["input_ids"] = ids
        return output

    def layer_hook(module, args, output):
        if isinstance(output, tuple):
            resid, *rest = output
        else:
            resid, rest = output, None
        input_ids = state["input_ids"]
        if input_ids is None or resid.shape[1] < 2:
            return output
        # vectors_ref is a list with one tensor; updated by caller pre-forward.
        v = vectors_ref[0]
        if v is None or v.shape[0] == 0:
            return output
        # Match the marker count to vectors expected.
        matches_count = (input_ids == inj_id).sum().item()
        if matches_count == 0:
            return output
        # Only inject when marker count matches available vectors — otherwise
        # we'd assert. (Should always match in this flow.)
        injected = karvonen_inject_in_residual(
            input_ids, resid, v, inj_id, left_id, right_id,
        )
        if rest is None:
            return injected
        return (injected, *rest)

    emb_handle = model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    base = model.base_model if hasattr(model, "base_model") else model
    # Walk to the decoder stack. Follows .model (Qwen/Llama) AND .language_model
    # (Gemma-3/4 multimodal wrappers nest the text decoder under
    # .model.language_model). Stops at the module that actually holds .layers.
    target = base
    while not hasattr(target, "layers"):
        if hasattr(target, "model"):
            target = target.model
        elif hasattr(target, "language_model"):
            target = target.language_model
        elif hasattr(target, "transformer"):
            target = target.transformer
        else:
            raise AssertionError(
                f"could not find .layers walking from {type(target).__name__}; "
                f"extend the karvonen-hook layer walk for this architecture"
            )
    layer_handle = target.layers[layer_idx].register_forward_hook(layer_hook)
    return emb_handle, layer_handle


def load_rl_dataset(parquet_path, n_max=None):
    """Streaming load — reads only the columns we need, only the rows we need.

    Full-table read of a 3.7GB parquet with .as_py() on every row's 4096-float
    activation_vector takes 5+ minutes; rowgroup-by-rowgroup streaming with
    early stop is sub-second for n_max=200 and ~30s for n_max=10000.
    """
    import pyarrow.parquet as pq_inner
    pf = pq_inner.ParquetFile(parquet_path)
    rows = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=["prompt", "activation_vector"])
        n_in_rg = rg.num_rows
        # Slice first — to_pylist() on a 5000-row column with 4096-float
        # activations is the bottleneck (~30s); take only what we need.
        take = n_in_rg if n_max is None else min(n_max - len(rows), n_in_rg)
        rg = rg.slice(0, take)
        prompts = rg.column("prompt").to_pylist()
        acts = rg.column("activation_vector").to_pylist()
        for p, a in zip(prompts, acts):
            rows.append({"prompt": p, "activation": a})
    return rows


def build_prompt_text(prompt_msgs, inject_char, tokenizer):
    """Apply chat template; substitute <INJECT> placeholder."""
    msgs = [
        {**m, "content": m["content"].replace("<INJECT>", inject_char)}
        if isinstance(m.get("content"), str)
        else m
        for m in prompt_msgs
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def rollout_one_prompt(
    actor, tokenizer, prompt_text, activation, vectors_ref,
    inj_id, group_size, max_new_tokens, temperature, device,
    eos_ids=None,
):
    """Generate `group_size` samples for one prompt; capture old log-probs per response token."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    batched = prompt_t.expand(group_size, -1).contiguous()
    v_batch = activation.unsqueeze(0).expand(group_size, -1).contiguous().to(device).float()
    vectors_ref[0] = v_batch
    try:
        gen_out = actor.generate(
            input_ids=batched,
            attention_mask=torch.ones_like(batched),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            # Explicitly override sampler — Qwen3's generation_config.json may
            # set top_p<1.0 / top_k>0 / repetition_penalty by default. Any of
            # these introduce -inf in the sampler logits, breaking the
            # importance ratio (exp(new_lp - old_lp) → inf).
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_logits=True,  # RAW pre-processor logits — scores are post-top_p
                                 # and assign -inf to filtered tokens, which would
                                 # make exp(new_lp - old_lp) = inf in GRPO loss.
        )
    finally:
        vectors_ref[0] = None
    full_ids = gen_out.sequences  # [G, prompt_len + new_len]
    # gen_out.logits: tuple of [G, V], RAW pre-softmax model logits at each step.
    # Captured under the SAME hook-applied forward used for training, so old_lp
    # and new_lp come from the same model snapshot at step 0 (then drift only
    # as LoRA weights update).
    scores = gen_out.logits  # tuple of [G, V]
    prompt_len = prompt_t.shape[1]
    # Trim at ANY stop id, not just tokenizer.eos_token_id: Qwen3's
    # generation_config lists two ([151645 <|im_end|>, 151643 <|endoftext|>]);
    # a sample terminating via the second would otherwise keep one forced-pad
    # token in the loss.
    if eos_ids is None:
        eos_ids = {tokenizer.eos_token_id}
    responses = []
    for g in range(group_size):
        resp_ids = full_ids[g, prompt_len:].tolist()
        # Trim at the first stop token (inclusive). Batched generate() pads
        # samples that finish early to the group max with pad(=eos) tokens;
        # those positions were never sampled from the policy. Without
        # trimming, old_logp/new_logp cover the pads too and the GRPO
        # surrogate + KL get gradient on garbage positions.
        n_real = next(
            (i + 1 for i, t in enumerate(resp_ids) if t in eos_ids),
            len(resp_ids),
        )
        resp_ids = resp_ids[:n_real]
        text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        # Collect old log_p for each REAL generated token.
        old_logp = []
        for t, step_logits in enumerate(scores):
            if t >= n_real:
                break
            lp = F.log_softmax(step_logits[g].float(), dim=-1)
            old_logp.append(lp[resp_ids[t]].item())
        responses.append({
            "text": text,
            "full_ids": full_ids[g, : prompt_len + n_real],
            "prompt_len": prompt_len,
            "old_logp": torch.tensor(old_logp, dtype=torch.float32),
            "n_resp": len(old_logp),
        })
    return responses


def critic_predict(critic, input_ids, attention_mask, mse_scale_f):
    """Forward the critic and produce a per-sample prediction vector.

    Architecture tweak (vs upstream NLACriticModel.forward):
      pred = value_head(normalize(backbone_last_hidden, mse_scale))

    The upstream forward does value_head(backbone_last_hidden) directly,
    which leaves value_head's input norm unbounded. With bf16+Adam on a
    near-identity value_head, that's exactly the path that NaN'd AR SFT
    8+ times. Normalising the backbone-last-hidden BEFORE the value_head
    bounds the input to a fixed norm (mse_scale), so a tiny weight update
    can't blow up the output norm by 100×. At identity init the two
    formulas agree (after the loss's final normalize), so swapping is
    backward-compatible with AR-SFT checkpoints.

    Returns: [B, d_model] fp32 pred tensor. Caller is responsible for
    grad / no_grad context; this function does not toggle.
    """
    cout = critic(input_ids=input_ids, attention_mask=attention_mask)
    backbone_last = cout.backbone_last_hidden  # [B, T, D] (bf16)
    if attention_mask is not None:
        last_idx = attention_mask.sum(dim=1) - 1
    else:
        last_idx = torch.full(
            (input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device,
        )
    bs = input_ids.shape[0]
    last_h = backbone_last[
        torch.arange(bs, device=input_ids.device), last_idx
    ].float()  # [B, D]
    last_h_norm = normalize_activation(last_h, mse_scale_f)
    pred = critic.value_head(
        last_h_norm.to(critic.value_head.weight.dtype)
    ).float()
    return pred


def score_with_critic(
    critic, tokenizer, explanations, activations, template, mse_scale_f, device,
):
    """Returns list of rewards (None for failed extractions)."""
    rewards = []
    for expl, act in zip(explanations, activations):
        if expl is None:
            rewards.append(None)
            continue
        text = template.format(explanation=expl)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 1024:
            rewards.append(None)
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = critic_predict(critic, x, None, mse_scale_f)[0]  # [d]
        gold = act.to(device).float()
        pred_n = normalize_activation(pred.unsqueeze(0), mse_scale_f)[0]
        gold_n = normalize_activation(gold.unsqueeze(0), mse_scale_f)[0]
        mse = F.mse_loss(pred_n, gold_n).item()
        if not math.isfinite(mse):
            rewards.append(None)
            continue
        rewards.append(-mse)
    return rewards


def grpo_update_microbatched(
    actor, optim, tokenizer, full_ids_list, prompt_lens, activations,
    old_logps_list, advantages, vectors_ref, device,
    micro_batch=2, clip_eps=0.2, kl_beta=0.04, max_grad_norm=1.0,
):
    """Fused micro-batched forward+loss+backward for GRPO.

    Each micro-batch: forward (LoRA on, grad) → ref forward (LoRA off, no grad)
    → per-chunk GRPO loss → backward → release graph → next chunk. Single
    optim.step() at the end. Peak memory = one micro-batch graph instead of
    N retained graphs (which is what OOMs at B*G=256).

    Returns (mean_loss, grad_norm, metrics_dict).
    """
    optim.zero_grad()
    n = len(full_ids_list)
    sample_losses_log = []
    sample_kls_log = []
    sample_clipfrac_log = []
    advantages = advantages.detach()  # no grad through advantage
    for cs in range(0, n, micro_batch):
        idxs = list(range(cs, min(cs + micro_batch, n)))
        bs = len(idxs)
        max_len = max(full_ids_list[i].numel() for i in idxs)
        pad_id = tokenizer.eos_token_id
        batch_ids = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for row, i in enumerate(idxs):
            L = full_ids_list[i].numel()
            batch_ids[row, :L] = full_ids_list[i].to(device)
            attn[row, :L] = 1
        v_batch = torch.stack(
            [activations[i].to(device).float() for i in idxs], dim=0,
        )
        # --- new_logp (with grad) ---
        vectors_ref[0] = v_batch
        try:
            new_logits = actor(input_ids=batch_ids, attention_mask=attn).logits
        finally:
            vectors_ref[0] = None
        new_logp = F.log_softmax(new_logits.float(), dim=-1)
        # --- ref_logp: switch to the frozen "reference" adapter (= AV-SFT init).
        #     Not disable_adapter() — the policy is a LoRA, so that would anchor
        #     KL to the bare base instead of the SFT init. ---
        vectors_ref[0] = v_batch
        try:
            with torch.no_grad():
                actor.set_adapter("reference")
                ref_logits = actor(input_ids=batch_ids, attention_mask=attn).logits
        finally:
            actor.set_adapter("default")
            vectors_ref[0] = None
        ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
        del ref_logits
        # --- per-sample GRPO loss for this chunk ---
        chunk_losses = []
        for row, i in enumerate(idxs):
            L = full_ids_list[i].numel()
            p_len = prompt_lens[i]
            if L <= p_len:
                continue
            target_ids = batch_ids[row, p_len:L]
            pred_idx = torch.arange(p_len - 1, L - 1, device=device)
            new_lp = (
                new_logp[row].index_select(0, pred_idx)
                .gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            )
            ref_lp = (
                ref_logp[row].index_select(0, pred_idx)
                .gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                .detach()
            )
            old_lp = old_logps_list[i].to(device).detach()
            if new_lp.numel() == 0 or old_lp.numel() != new_lp.numel():
                continue
            ratio = torch.exp(new_lp - old_lp)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            A = advantages[i]
            surrogate = torch.minimum(ratio * A, clipped * A)
            delta = ref_lp - new_lp
            kl = torch.exp(delta) - delta - 1.0
            per_tok = -(surrogate - kl_beta * kl)
            sample_loss = per_tok.mean()
            chunk_losses.append(sample_loss)
            sample_kls_log.append(kl.detach().mean().item())
            sample_clipfrac_log.append(
                ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean().item()
            )
        # Free retained logp / logits before backward to bound peak.
        del new_logits, ref_logp
        if not chunk_losses:
            del new_logp
            continue
        # Scale so summed chunk losses give batch-mean.
        chunk_loss = torch.stack(chunk_losses).sum() / n
        chunk_loss.backward()
        sample_losses_log.append(chunk_loss.item() * n / len(chunk_losses))
        del new_logp
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in actor.parameters() if p.requires_grad], max_grad_norm,
    )
    gn = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
    # Guard BEFORE stepping: clip_grad_norm_ does not sanitize nan/inf (it
    # scales by max_norm/total_norm, and nan propagates). Stepping Adam on
    # non-finite grads corrupts the moment estimates AND the weights.
    if math.isfinite(gn):
        optim.step()
    else:
        optim.zero_grad(set_to_none=True)
        print(f"[grpo] non-finite grad norm ({gn}) — skipping optimizer step",
              flush=True)
    metrics = {
        "kl_mean": float(np.mean(sample_kls_log)) if sample_kls_log else 0.0,
        "clip_frac": float(np.mean(sample_clipfrac_log)) if sample_clipfrac_log else 0.0,
    }
    mean_loss = float(np.mean(sample_losses_log)) if sample_losses_log else 0.0
    return mean_loss, gn, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", required=True,
                   help="AV-SFT LoRA adapter dir (sits on --base-ckpt).")
    p.add_argument("--ar-ckpt", required=True,
                   help="AR-LoRA dir (ar_lora_value_head.safetensors + ar_meta.json).")
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B",
                   help="Base the AV/AR LoRA adapters sit on (4-bit if --quant 4bit).")
    p.add_argument("--quant", choices=["none", "4bit"], default="4bit")
    p.add_argument("--experts-implementation", default=None,
                   help="MoE experts kernel (Gemma-4 etc.): 'eager' to avoid the "
                        "fused torch._grouped_mm (sm_90-only) on Blackwell/B200.")
    p.add_argument("--device-map", choices=["single", "auto"], default="single")
    p.add_argument("--max-gpu-mem", type=int, default=0)
    p.add_argument("--rl-parquet", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--batch-prompts", type=int, default=8,
                   help="prompts per step")
    p.add_argument("--group-size", type=int, default=4,
                   help="samples per prompt (for group baseline)")
    p.add_argument("--max-new-tokens", type=int, default=150)  # paper's rollout cap
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True,
                   help="Use rsLoRA scaling (alpha/sqrt(r) instead of alpha/r). "
                        "Default ON because we use r=128 where vanilla LoRA's "
                        "alpha/r=0.125 collapses the effective learning rate.")
    p.add_argument("--train-critic", action="store_true", default=False,
                   help="Co-train the AR critic (paper-faithful). Adds a "
                        "separate optimizer for the critic and supervised MSE "
                        "loss on (explanation, gold_activation) pairs each step.")
    p.add_argument("--critic-lr", type=float, default=1e-5)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False,
                   help="Recompute activations during backward (saves ~50% "
                        "activation memory at ~30%% compute cost). Off by "
                        "default — 8-bit Adam on critic gives bigger savings.")
    p.add_argument("--critic-micro-batch", type=int, default=4,
                   help="Micro-batch size for the critic's training-time forward. "
                        "Single full-batch forward OOMs at B*G=256.")
    p.add_argument("--logp-micro-batch", type=int, default=2)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume-from-lora", type=str, default=None,
                   help="Directory containing a saved LoRA adapter (iter_NNNNNN); "
                        "loaded as the policy adapter so training continues from "
                        "those weights (KL reference stays the AV-SFT init). If "
                        "the dir has a critic/ subdir, the co-trained critic is "
                        "resumed too.")
    p.add_argument("--start-step", type=int, default=0,
                   help="Initial step counter — useful when resuming so wandb "
                        "x-axis lines up with the previous run.")
    p.add_argument("--eval-every", type=int, default=10,
                   help="Run a held-out qualitative eval every N steps. "
                        "Logs explanation texts to wandb Table; 0 disables.")
    p.add_argument("--eval-n-prompts", type=int, default=20,
                   help="Number of fixed held-out prompts for per-step eval.")
    p.add_argument("--eval-skip-rows", type=int, default=30000,
                   help="Take eval prompts from rl_shuf rows starting here "
                        "(past --max-rows training cursor).")
    p.add_argument("--max-rows", type=int, default=None,
                   help="cap rows from rl parquet (avoids 3.7GB full-load for smoke runs)")
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-beta", type=float, default=0.04)
    p.add_argument("--wandb-project", default="nla-qwen3-8b")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--external-evals", default="",
        help="Comma-sep list of evals/ IDs to run every --eval-every step "
             "(e.g. 'hallucination,karvonen_confusion'). Empty = none.",
    )
    p.add_argument("--eval-n-hallucination", type=int, default=40,
                   help="N held-out prompts for hallucination eval.")
    p.add_argument("--eval-n-karvonen", type=int, default=97,
                   help="N records (out of 97 filtered) for karvonen_confusion eval.")
    p.add_argument("--judge-key-env", default="ANTHROPIC_API_KEY_FALLBACK",
                   help="Env var holding the judge API key (default: high-prio).")
    p.add_argument("--judge-concurrency", type=int, default=32,
                   help="Parallel judge calls (Anthropic sync, NOT batch API).")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda"
    os.environ.setdefault("HF_HOME", "/workspace-vast/pretrained_ckpts")

    # old_logp is computed from generate's RAW logits (output_logits). At
    # temperature 1.0 that IS the sampling distribution; at any other
    # temperature the importance ratio no longer corrects for the actual
    # behavior policy and the surrogate is silently biased.
    assert args.temperature == 1.0, (
        f"--temperature {args.temperature} != 1.0: old_logp comes from raw "
        f"logits, so the GRPO importance ratio is only valid at T=1. "
        f"Remove this assert only if you also temperature-scale old/new logp."
    )
    # Eval rows are taken past --eval-skip-rows; training samples rows[:max_rows].
    # Without this cap the training pool contains the literal eval rows.
    if args.eval_every > 0 and (args.eval_n_prompts > 0 or args.external_evals.strip()):
        assert args.max_rows is not None and args.max_rows <= args.eval_skip_rows, (
            f"evals enabled but --max-rows ({args.max_rows}) is unset or exceeds "
            f"--eval-skip-rows ({args.eval_skip_rows}) — training would include "
            f"the eval rows themselves."
        )

    # ---- tokenizer + nla config ----
    # From --base-ckpt, NOT hardcoded — the sidecar asserts below catch a
    # wrong-family tokenizer, but only if we load the one the run targets.
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.sidecar, tokenizer)
    inj_id = cfg.injection_token_id
    left_id = cfg.injection_left_neighbor_id
    right_id = cfg.injection_right_neighbor_id
    inject_char = cfg.injection_char
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    template = cfg.critic_prompt_template
    assert template is not None, "critic_prompt_template missing"
    print(f"[cfg] inj_id={inj_id} mse_scale_f={mse_scale_f} d_model={cfg.d_model}")

    # ---- actor: (4-bit) base + AV-SFT LoRA ("default", trainable) + frozen
    #      "reference" adapter (= AV-SFT init) for the KL anchor.
    #      The policy is now a LoRA *on* the frozen base, so disable_adapter()
    #      would anchor KL to the bare base, not the SFT init — hence a second
    #      frozen adapter. (See feedback_rl_ref_policy.)
    quant_config = None
    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    dmap, max_mem = _resolve_device_map(args.device_map, args.max_gpu_mem, quant_config)
    print(f"[actor] base={args.base_ckpt} + AV-LoRA={args.av_ckpt} "
          f"(quant={args.quant}, device_map={args.device_map})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=quant_config, device_map=dmap, max_memory=max_mem,
        **({"experts_implementation": args.experts_implementation}
           if args.experts_implementation else {}),
    )
    if dmap is None:
        base = base.to(device)
    if quant_config is not None:
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=args.gradient_checkpointing,
        )
    # Policy adapter: AV-SFT init, or a saved RL LoRA when resuming. The KL
    # "reference" adapter ALWAYS comes from the AV-SFT ckpt — resuming must
    # not move the KL anchor.
    policy_ckpt = args.resume_from_lora or args.av_ckpt
    if args.resume_from_lora:
        print(f"[actor] RESUMING policy LoRA from {args.resume_from_lora} "
              f"(KL reference stays {args.av_ckpt})")
    actor = PeftModel.from_pretrained(
        base, policy_ckpt, adapter_name="default", is_trainable=True,
    )
    actor.load_adapter(args.av_ckpt, adapter_name="reference")  # frozen KL ref
    actor.set_adapter("default")
    if args.resume_from_lora:
        # Sanity check: a resumed adapter should differ from the reference.
        # diff == 0 is either the historical resume-ignored bug OR a genuinely
        # zero-update leg (e.g. all extractions failed → adv ≡ 0); a hard
        # assert here would brick-loop the self-chaining sbatch on the latter,
        # so warn loudly instead. NOTE: this does NOT catch a partial adapter
        # load (PEFT load_adapter never raises on missing keys) — fresh-init
        # tensors also differ from the reference.
        _diff = 0.0
        _sd = actor.state_dict()
        for n in list(_sd):
            if ".default." in n and "lora_" in n:
                _ref = _sd.get(n.replace(".default.", ".reference."))
                if _ref is not None:
                    _diff += (_sd[n].float() - _ref.float()).pow(2).sum().item()
        print(f"[actor] resumed; sum((lora_default - lora_reference)²) = {_diff:.3e}")
        if _diff == 0.0:
            print(f"[actor] WARNING: resumed adapter is IDENTICAL to the AV-SFT "
                  f"reference — either {args.resume_from_lora} is untrained or "
                  f"the resume load silently failed.", flush=True)
    # The reference adapter is the frozen KL anchor — pin requires_grad False
    # explicitly. (PEFT's set_adapter toggles trainability when switching
    # adapters during the update loop; the optimizer snapshot taken below
    # protects against drift today, but be explicit so a future optimizer
    # rebuild can't silently start training the anchor.)
    for _n, _p in actor.named_parameters():
        if ".reference." in _n:
            _p.requires_grad_(False)
    actor.print_trainable_parameters()
    actor.train()
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable()
        actor.enable_input_require_grads()
        print(f"[actor] gradient_checkpointing ENABLED")

    # ---- critic: AR-LoRA (4-bit base + injected LoRA + value_head). Rebuild
    #      the exact structure train_sft saved, then load the adapter + head. ----
    import json as _json
    from safetensors.torch import load_file as _load_file
    # When resuming and the resume dir has a saved co-trained critic, load
    # that instead of the AR-SFT init — otherwise the reward model snaps back
    # to its SFT state and the reward scale is discontinuous across the resume.
    ar_src = Path(args.ar_ckpt)
    if args.resume_from_lora is not None:
        _resumed_critic = Path(args.resume_from_lora) / "critic"
        if (_resumed_critic / "ar_lora_value_head.safetensors").exists():
            ar_src = _resumed_critic
            print(f"[critic] RESUMING co-trained critic from {ar_src}")
    ar_meta = _json.loads((ar_src / "ar_meta.json").read_text())
    print(f"[critic] AR-LoRA from {ar_src}: {ar_meta}")
    assert ar_meta.get("quant") != "4bit" or quant_config is not None, (
        f"AR-LoRA was trained on a 4-bit backbone (ar_meta quant=4bit) but this "
        f"run uses --quant {args.quant}: the LoRA's baked-in quantization-error "
        f"compensation would silently mismatch a bf16 backbone."
    )
    ar_quant = quant_config if ar_meta.get("quant") == "4bit" else None
    ar_dmap, ar_maxmem = _resolve_device_map(args.device_map, args.max_gpu_mem, ar_quant)
    critic = init_critic_from_base(
        args.base_ckpt, ar_meta["ar_num_layers"], torch.bfloat16,
        ar_quant, device_map=ar_dmap, max_memory=ar_maxmem,
        # Checkpoints record whether their backbone ran with the final RMSNorm
        # stripped (design §4) or kept (pre-2026-06 ckpts). Must match training.
        strip_final_norm=ar_meta.get("final_norm_stripped", False),
        experts_implementation=args.experts_implementation,
    )
    if ar_dmap is None:
        critic = critic.to(device)
    # Target the SAME modules AR-SFT did (lora_attention_targets), NOT the flat
    # ar_meta list: on multimodal models (Gemma) the flat ["q_proj",...] suffix-
    # matches a rogue Gemma4ClippableLinear *q_proj outside language_model that
    # PEFT can't wrap, AND would mismatch the saved adapter keys. The scoped
    # regex restricts to language_model attention — matching AR-SFT exactly.
    from nla.arch_adapters import lora_attention_targets
    inject_adapter_in_model(LoraConfig(
        r=ar_meta["lora_r"], lora_alpha=ar_meta["lora_alpha"], lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
        target_modules=lora_attention_targets(critic.backbone.config),
    ), critic.backbone)
    _ar_sd = _load_file(str(ar_src / "ar_lora_value_head.safetensors"))
    _miss, _unexp = critic.load_state_dict(_ar_sd, strict=False)
    # `missing` is always the whole frozen backbone (uninformative);
    # `unexpected` non-empty means a key-schema drift left the reward model
    # at init — fail loudly, not via a print nobody reads.
    _n_lora = sum(1 for k in _ar_sd if "lora_" in k)
    assert _n_lora > 0 and not _unexp, (
        f"AR weights load mismatch: {_n_lora} lora tensors in file, "
        f"unexpected={_unexp[:3]} — PEFT key-schema drift?"
    )
    print(f"[critic] loaded {len(_ar_sd)} AR tensors "
          f"(missing={len(_miss)} backbone keys, unexpected=0)")
    # Freeze everything; conditionally unfreeze LoRA + value_head for co-training.
    for p_ in critic.parameters():
        p_.requires_grad_(False)
    critic_optim = None
    if args.train_critic:
        # Per paper §RL training: AR is co-trained simultaneously with AV on
        # the SAME explanations the actor produces this step. Loss = MSE against
        # the gold activation, normalised. AR's gradient does NOT flow back into
        # the actor (the explanation tokens are discrete — gradient stops there
        # automatically). Both backbone AND value_head train; the bf16+Adam
        # blow-up that NaN'd AR SFT is now neutralised by critic_predict's
        # normalize-before-value_head trick (bounds value_head input norm).
        # Co-train only the AR LoRA adapters + value_head (4-bit base frozen).
        for n_, p_ in critic.named_parameters():
            if ("lora_" in n_) or n_.startswith("value_head"):
                p_.requires_grad_(True)
        critic_trainable = [p for p in critic.parameters() if p.requires_grad]
        try:
            import bitsandbytes as _bnb
            critic_optim = _bnb.optim.AdamW8bit(
                critic_trainable, lr=args.critic_lr, betas=(0.9, 0.95),
                weight_decay=0.0,
            )
        except ImportError:
            critic_optim = torch.optim.AdamW(
                critic_trainable, lr=args.critic_lr, betas=(0.9, 0.95),
                weight_decay=0.0,
            )
        n_trainable = sum(p.numel() for p in critic_trainable)
        print(f"[critic] CO-TRAINED, lr={args.critic_lr}, "
              f"trainable={n_trainable/1e9:.2f}B (backbone + value_head)")
    else:
        print(f"[critic] FROZEN (eval-only scorer)")
    critic.eval()  # Qwen3 has no dropout — eval mode is fine for both grad/no-grad
    print(f"[critic] value_head shape={tuple(critic.value_head.weight.shape)}")

    # ---- karvonen hook on actor ----
    vectors_ref = [None]
    _register_karvonen_hook(actor, vectors_ref, inj_id, left_id, right_id, layer_idx=1)

    # All stop ids for rollout EOS-trimming (Qwen3 lists two in its
    # generation_config; trimming on tokenizer.eos_token_id alone would leave
    # one forced-pad token in the loss for sequences stopping on the other).
    eos_ids = {tokenizer.eos_token_id}
    _gc_eos = getattr(getattr(actor, "generation_config", None), "eos_token_id", None)
    if _gc_eos is not None:
        eos_ids.update(_gc_eos if isinstance(_gc_eos, (list, tuple)) else [_gc_eos])
    eos_ids.discard(None)
    print(f"[rollout] EOS-trim ids: {sorted(eos_ids)}")

    # ---- dataset ----
    print(f"[data] loading {args.rl_parquet} (max_rows={args.max_rows})", flush=True)
    rows = load_rl_dataset(args.rl_parquet, n_max=args.max_rows)
    print(f"[data] {len(rows)} rows", flush=True)

    # ---- FVE baseline: predict-the-mean MSE on this dataset ----
    # FVE = 1 - mse_actual / baseline_mse, with the PAPER's baseline:
    # E[||v_norm - μ||²], the raw variance of the normalized distribution
    # (≈0.72 on Qwen 7B-class). NOTE: runs before 2026-06-09 used the looser
    # "meannorm" baseline MSE(v_norm, normalize(μ)) (≈0.94), which inflates
    # FVE vs the paper's definition — old wandb curves are not comparable.
    # Both are logged; `fve` uses the paper definition.
    _act_stack = torch.tensor(
        [r["activation"] for r in rows[: min(len(rows), 4000)]],
        dtype=torch.float32,
    )
    fve_baseline_meannorm, fve_baseline = compute_predict_mean_baselines(
        _act_stack, mse_scale_f,
    )
    del _act_stack
    print(f"[fve] predict-the-mean baseline mse_nrm = {fve_baseline:.4f} "
          f"(paper def; meannorm baseline = {fve_baseline_meannorm:.4f})",
          flush=True)

    # ---- optimizer ----
    # 8-bit Adam (bitsandbytes) for both actor LoRA and critic — block-wise
    # int8 quantization of (m, v) state cuts optimizer memory ~4×. "Paged"
    # variant CPU-offloads pages under memory pressure. Standard choice for
    # memory-constrained LLM fine-tuning; numerically equivalent to fp32 Adam
    # within bf16 noise for our use case.
    try:
        import bitsandbytes as bnb
        _adam_cls = bnb.optim.AdamW8bit
        print(f"[optim] using bitsandbytes AdamW8bit (bnb {bnb.__version__})")
    except ImportError:
        _adam_cls = torch.optim.AdamW
        print(f"[optim] bitsandbytes unavailable, falling back to torch AdamW (fp32 m,v)")
    trainable = [p for p in actor.parameters() if p.requires_grad]
    optim = _adam_cls(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # ---- wandb ----
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pending_idxs = list(range(len(rows)))
    rng.shuffle(pending_idxs)
    cursor = 0

    # ---- Fixed held-out eval prompts, DOC-DISJOINT from training rows.
    # Stage-1 only guarantees disjointness BETWEEN av_sft/ar_sft/rl FILES;
    # within rl_shuf.parquet, rows past --eval-skip-rows can share doc_id
    # with rows before it (the file is row-shuffled, not doc-partitioned
    # internally). Without explicit filtering we measured ~50% doc-overlap.
    # Fix: scan training-window (rows 0..eval_skip_rows) to collect doc_ids,
    # then take eval rows past the cursor whose doc_id is NOT in that set.
    eval_rows = []
    if args.eval_every > 0 and args.eval_n_prompts > 0:
        import pyarrow.parquet as _pq
        _pf = _pq.ParquetFile(args.rl_parquet)
        # Pass 1: training-window doc_ids
        _train_doc_ids: set = set()
        _seen = 0
        for _rg_idx in range(_pf.num_row_groups):
            if _seen >= args.eval_skip_rows:
                break
            _rg = _pf.read_row_group(_rg_idx, columns=["doc_id"])
            _ids = _rg.column("doc_id").to_pylist()
            _nrg = len(_ids)
            _take = min(_nrg, args.eval_skip_rows - _seen)
            _train_doc_ids.update(_ids[:_take])
            _seen += _nrg
        # Pass 2: doc-disjoint rows past the cursor
        _seen = 0
        for _rg_idx in range(_pf.num_row_groups):
            if len(eval_rows) >= args.eval_n_prompts:
                break
            _rg = _pf.read_row_group(
                _rg_idx, columns=["prompt", "activation_vector", "doc_id"],
            )
            _n = _rg.num_rows
            if _seen + _n <= args.eval_skip_rows:
                _seen += _n
                continue
            _start = max(0, args.eval_skip_rows - _seen)
            _prompts = _rg.column("prompt").to_pylist()
            _acts = _rg.column("activation_vector").to_pylist()
            _dids = _rg.column("doc_id").to_pylist()
            for _i in range(_start, _n):
                if _dids[_i] in _train_doc_ids:
                    continue
                eval_rows.append({"prompt": _prompts[_i], "activation": _acts[_i]})
                if len(eval_rows) >= args.eval_n_prompts:
                    break
            _seen += _n
        print(f"[eval] {len(eval_rows)} doc-disjoint prompts loaded "
              f"(rows past {args.eval_skip_rows}, excluding "
              f"{len(_train_doc_ids)} training doc_ids)", flush=True)
    eval_table_data = []  # accumulates [step, idx, reward, fve, extracted, explanation]

    # ---- External evals (evals/ pluggable, run every --eval-every step) ----
    # Reuses the trainer's `vectors_ref` so each eval shares the injection hook
    # already attached to the actor (no stacked hooks, no extra Qwen3-8B load).
    # Judge calls go through ANTHROPIC_API_KEY_FALLBACK (high-prio) in a
    # ThreadPoolExecutor — see evals/base.py.
    external_evals = []
    if args.external_evals.strip():
        if not os.environ.get(args.judge_key_env):
            raise RuntimeError(
                f"--external-evals requires ${args.judge_key_env} to be set "
                f"(judge API key for Sonnet 4.6). See CLAUDE.md."
            )
        # Side-effect imports register the eval classes via @register decorator.
        import evals.hallucination  # noqa: F401
        import evals.karvonen_confusion  # noqa: F401
        from evals.base import EvalConfig
        from evals.registry import get_eval

        n_samples_for = {
            "hallucination": args.eval_n_hallucination,
            "karvonen_confusion": args.eval_n_karvonen,
        }
        for eid in [s.strip() for s in args.external_evals.split(",") if s.strip()]:
            ev_cfg = EvalConfig(
                output_dir=save_dir / "eval_runs",
                n_samples=n_samples_for.get(eid, 40),
                seed=args.seed,
                eval_skip_rows=args.eval_skip_rows,
                parquet_path=args.rl_parquet,
                judge_model="claude-sonnet-4-6",
                judge_temperature=0.0,
                judge_max_concurrency=args.judge_concurrency,
                anthropic_api_key_env=args.judge_key_env,
            )
            ev_cls = get_eval(eid)
            ev = ev_cls(ev_cfg)
            ev.setup(actor, critic, tokenizer, cfg, device,
                     shared_vectors_ref=vectors_ref)
            external_evals.append(ev)
            print(f"[external eval] {eid} ready (n={ev_cfg.n_samples}, "
                  f"judge={ev_cfg.judge_model}, key=${args.judge_key_env})",
                  flush=True)

    # History for the multi-line Pareto chart.
    pareto_history: list[dict] = []

    for step in range(args.start_step, args.num_steps):
        t0 = time.time()
        # ---- batch select ----
        if cursor + args.batch_prompts > len(pending_idxs):
            rng.shuffle(pending_idxs)
            cursor = 0
        batch_idxs = pending_idxs[cursor : cursor + args.batch_prompts]
        cursor += args.batch_prompts

        # ---- rollouts ----
        actor.eval()
        all_full_ids = []
        all_prompt_lens = []
        all_activations = []
        all_explanations = []
        all_response_text = []
        all_prompt_group = []
        all_old_logps = []  # 1-D tensor per sample
        for gi, row_idx in enumerate(batch_idxs):
            row = rows[row_idx]
            prompt_text = build_prompt_text(row["prompt"], inject_char, tokenizer)
            activation = torch.tensor(row["activation"], dtype=torch.float32)
            responses = rollout_one_prompt(
                actor, tokenizer, prompt_text, activation, vectors_ref,
                inj_id, args.group_size, args.max_new_tokens, args.temperature, device,
                eos_ids=eos_ids,
            )
            for r in responses:
                expl = extract_explanation(r["text"])
                all_full_ids.append(r["full_ids"])
                all_prompt_lens.append(r["prompt_len"])
                all_activations.append(activation)
                all_explanations.append(expl)
                all_response_text.append(r["text"])
                all_prompt_group.append(gi)
                all_old_logps.append(r["old_logp"].to(device))

        # ---- scoring ----
        rewards = score_with_critic(
            critic, tokenizer, all_explanations, all_activations,
            template, mse_scale_f, device,
        )
        # Paper's FAILED_EXTRACTION_REWARD = -2.0 (nla/reward.py): MSE on
        # fully-orthogonal unit vectors is 2.0, so this is the "worst possible"
        # critic outcome. Same penalty as a fully wrong direction.
        rewards_filled = [-2.0 if r is None else r for r in rewards]
        rewards_t = torch.tensor(rewards_filled, dtype=torch.float32, device=device)

        # ---- GRPO group-relative advantage (per-prompt mean & std) ----
        group_t = torch.tensor(all_prompt_group, dtype=torch.long, device=device)
        adv = torch.zeros_like(rewards_t)
        for gi in range(args.batch_prompts):
            mask = group_t == gi
            if mask.sum() == 0:
                continue
            group_r = rewards_t[mask]
            mu = group_r.mean()
            sd = group_r.std() if group_r.numel() > 1 else torch.tensor(1.0, device=device)
            adv[mask] = (group_r - mu) / (sd + 1e-6)

        # ---- GRPO update: fused forward+loss+backward per micro-batch ----
        # Previous code did all forwards then all backwards, which retained
        # every micro-batch's compute graph and OOM'd at B*G=256. The fused
        # version releases each chunk's graph before starting the next.
        actor.train()
        mean_loss_val, grad_norm_val, grpo_metrics = grpo_update_microbatched(
            actor, optim, tokenizer,
            all_full_ids, all_prompt_lens, all_activations,
            all_old_logps, adv, vectors_ref, device,
            micro_batch=args.logp_micro_batch,
            clip_eps=args.clip_eps, kl_beta=args.kl_beta,
            max_grad_norm=args.max_grad_norm,
        )
        # Build a scalar-tensor stand-in for the existing logging path that
        # expects a `loss` tensor with .item().
        loss = torch.tensor(mean_loss_val, device=device)
        grad_norm = torch.tensor(grad_norm_val, device=device)
        if not math.isfinite(mean_loss_val):
            print(
                f"step {step}: loss={mean_loss_val} non-finite "
                f"(kl={grpo_metrics.get('kl_mean')}, "
                f"clip_frac={grpo_metrics.get('clip_frac')}). Skipping critic update.",
                flush=True,
            )
            # The helper already refused to optim.step() on a non-finite grad
            # norm, so weights are intact; skip the critic update + logging.
            continue

        # ---- AR critic co-training (paper-faithful, optional) ----
        # Per paper §RL: "Update the AR by one step of gradient descent on the
        # regression loss ||h_l − AR_θ(z)||²_2". Inputs z = the explanations the
        # actor just produced this step; targets h_l = the gold activations.
        # Gradient from this update does NOT flow into the actor (z is discrete).
        critic_loss_val = float("nan")
        critic_grad_norm_val = float("nan")
        if args.train_critic and critic_optim is not None:
            crit_inputs = []
            crit_golds = []
            for expl, act in zip(all_explanations, all_activations):
                if expl is None:
                    continue
                text = template.format(explanation=expl)
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > 1024 or len(ids) == 0:
                    continue
                crit_inputs.append(torch.tensor(ids, dtype=torch.long))
                crit_golds.append(act)
            if crit_inputs:
                # Micro-batch the critic update — single forward on 256 sequences
                # × 200 tokens × 5.5B-param critic with grad blows past 130GB.
                # Accumulate gradient across micro-batches, single step at the
                # end (loss is divided by total bs so it averages correctly).
                bs_total = len(crit_inputs)
                pad_id = tokenizer.eos_token_id
                critic_optim.zero_grad()
                accumulated = 0.0
                finite = True
                cmb = max(1, args.critic_micro_batch)
                for cs in range(0, bs_total, cmb):
                    chunk = list(range(cs, min(cs + cmb, bs_total)))
                    max_len = max(crit_inputs[i].numel() for i in chunk)
                    bs = len(chunk)
                    batch_ids = torch.full(
                        (bs, max_len), pad_id, dtype=torch.long, device=device,
                    )
                    attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
                    for row, i in enumerate(chunk):
                        L = crit_inputs[i].numel()
                        batch_ids[row, :L] = crit_inputs[i].to(device)
                        attn[row, :L] = 1
                    pred = critic_predict(critic, batch_ids, attn, mse_scale_f)
                    gold = torch.stack([crit_golds[i] for i in chunk]).to(device).float()
                    pred_n = normalize_activation(pred, mse_scale_f)
                    gold_n = normalize_activation(gold, mse_scale_f)
                    # Scale so the sum across micro-batches = MSE over full batch.
                    chunk_loss = F.mse_loss(pred_n, gold_n) * (bs / bs_total)
                    if not torch.isfinite(chunk_loss):
                        print(f"step {step}: critic loss non-finite (chunk {cs}), skipping", flush=True)
                        finite = False
                        break
                    chunk_loss.backward()
                    accumulated += chunk_loss.item()
                if finite:
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        critic_trainable, args.max_grad_norm,
                    )
                    critic_optim.step()
                    critic_loss_val = accumulated  # already the full-batch mean
                    critic_grad_norm_val = (
                        critic_grad_norm.item()
                        if hasattr(critic_grad_norm, "item")
                        else float(critic_grad_norm)
                    )

        # ---- logging ----
        valid_rewards = [r for r in rewards if r is not None]
        n_valid = len(valid_rewards)
        n_total = len(rewards)
        extraction_rate = n_valid / n_total if n_total else 0
        mean_cjk = (
            sum(cjk_fraction(t) for t in all_response_text) / max(len(all_response_text), 1)
        )
        # Response lengths come from the rollout's old_logps (one entry per sample).
        n_resps_t = torch.tensor(
            [lp.numel() for lp in all_old_logps], dtype=torch.float32, device=device,
        )
        # FVE on valid (non-extraction-failed) samples — gives an
        # interpretable curve in wandb that maps to paper's reported numbers.
        # Use valid rewards only so extraction failures don't bias FVE down.
        fve = (
            1.0 - (-float(np.mean(valid_rewards))) / fve_baseline
            if valid_rewards else float("nan")
        )
        log = {
            "step": step,
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "reward_mean": float(np.mean(valid_rewards)) if valid_rewards else float("nan"),
            "reward_std": float(np.std(valid_rewards)) if valid_rewards else float("nan"),
            "reward_min": float(np.min(valid_rewards)) if valid_rewards else float("nan"),
            "reward_max": float(np.max(valid_rewards)) if valid_rewards else float("nan"),
            "fve": fve,
            "fve_pct": fve * 100.0,
            "fve_baseline": fve_baseline,
            "fve_baseline_meannorm": fve_baseline_meannorm,
            # FVE under the old (pre-2026-06-09) meannorm baseline, for
            # comparing against historical wandb curves only.
            "fve_pct_meannorm": (
                (1.0 - (-float(np.mean(valid_rewards))) / fve_baseline_meannorm) * 100.0
                if valid_rewards else float("nan")
            ),
            "advantage_mean": adv.mean().item(),
            "advantage_std": adv.std().item(),
            "extraction_rate": extraction_rate,
            "mean_cjk": mean_cjk,
            "mean_resp_len": n_resps_t.mean().item(),
            "kl_mean": grpo_metrics.get("kl_mean", 0.0),
            "clip_frac": grpo_metrics.get("clip_frac", 0.0),
            "critic_loss": critic_loss_val,
            "critic_grad_norm": critic_grad_norm_val,
            "wall_s": time.time() - t0,
        }
        crit_str = (
            f"| crit {critic_loss_val:.4f} " if args.train_critic else ""
        )
        print(
            f"step {step:04d} | loss {log['loss']:.4f} | r {log['reward_mean']:.3f} "
            f"| FVE {log['fve_pct']:.1f}% {crit_str}| kl {log['kl_mean']:.4f} | "
            f"clip {log['clip_frac']:.2%} | ext {extraction_rate:.0%} | "
            f"t {log['wall_s']:.0f}s",
            flush=True,
        )

        # ---- per-step eval: every N steps, run actor (current weights) on a
        # FIXED set of held-out prompts and log explanations as a wandb Table.
        # Lets you scrub through the run and watch explanations evolve.
        if args.eval_every > 0 and step % args.eval_every == 0:
            actor.eval()
            eval_rewards_s = []
            eval_records = []
            for ei, row in enumerate(eval_rows):
                prompt_text = build_prompt_text(row["prompt"], inject_char, tokenizer)
                activation = torch.tensor(row["activation"], dtype=torch.float32)
                ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                pt = torch.tensor([ids], dtype=torch.long, device=device)
                vectors_ref[0] = activation.unsqueeze(0).to(device).float()
                try:
                    with torch.no_grad():
                        gen = actor.generate(
                            input_ids=pt, attention_mask=torch.ones_like(pt),
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True, temperature=1.0,
                            top_p=1.0, top_k=0, repetition_penalty=1.0,
                            pad_token_id=tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                        )
                finally:
                    vectors_ref[0] = None
                resp = tokenizer.decode(
                    gen.sequences[0, pt.shape[1]:], skip_special_tokens=True,
                )
                expl = extract_explanation(resp)
                e_reward = -2.0
                if expl is not None:
                    ctext = template.format(explanation=expl)
                    cids = tokenizer.encode(ctext, add_special_tokens=False)
                    if 0 < len(cids) <= 1024:
                        x = torch.tensor([cids], dtype=torch.long, device=device)
                        with torch.no_grad():
                            pred = critic_predict(critic, x, None, mse_scale_f)[0]
                        gold = activation.to(device).float()
                        pn = normalize_activation(pred.unsqueeze(0), mse_scale_f)[0]
                        gn = normalize_activation(gold.unsqueeze(0), mse_scale_f)[0]
                        mse = F.mse_loss(pn, gn).item()
                        if math.isfinite(mse):
                            e_reward = -mse
                eval_rewards_s.append(e_reward)
                eval_records.append({
                    "step": step, "idx": ei, "reward": e_reward,
                    "fve": (1.0 - (-e_reward) / fve_baseline) if e_reward > -2.0 else float("nan"),
                    "extracted": expl is not None,
                    "explanation": expl if expl is not None else "<extraction failed>",
                })
            # Aggregate eval scalars
            valid_e = [r for r in eval_rewards_s if r > -2.0]
            log["eval/reward_mean"] = (
                float(np.mean(eval_rewards_s)) if eval_rewards_s else float("nan")
            )
            log["eval/reward_mean_valid"] = (
                float(np.mean(valid_e)) if valid_e else float("nan")
            )
            log["eval/fve_pct"] = (
                (1.0 - (-float(np.mean(valid_e))) / fve_baseline) * 100.0
                if valid_e else float("nan")
            )
            log["eval/extraction_rate"] = (
                sum(1 for r in eval_records if r["extracted"]) / len(eval_records)
                if eval_records else 0.0
            )
            # Persistent table — accumulates across the whole run.
            for r in eval_records:
                eval_table_data.append([
                    r["step"], r["idx"], r["reward"], r["fve"],
                    r["extracted"], r["explanation"][:500],
                ])
            if not args.no_wandb:
                log["eval/samples"] = wandb.Table(
                    columns=["step", "idx", "reward", "fve", "extracted", "explanation"],
                    data=list(eval_table_data),
                )
            print(
                f"  [eval@{step}] reward {log['eval/reward_mean']:.3f} "
                f"| FVE {log['eval/fve_pct']:.1f}% "
                f"| ext {log['eval/extraction_rate']:.0%}",
                flush=True,
            )
            # Print 3 sample explanations so the log itself shows how outputs
            # evolve. Pick indices 0, 7, 14 — spread across the eval set.
            for _ei in (0, 7, 14):
                if _ei < len(eval_records):
                    _r = eval_records[_ei]
                    _expl = _r["explanation"][:200].replace("\n", " ")
                    print(
                        f"    [eval@{step} idx={_ei} r={_r['reward']:.3f}] {_expl}",
                        flush=True,
                    )

            # ---- External evals (hallucination, karvonen_confusion, …) ----
            # Each eval reuses `actor` + `vectors_ref`, so no extra GPU load.
            # Judge calls are parallelised inside each eval's evaluate().
            for ev in external_evals:
                try:
                    result = ev.evaluate(step)
                    for k, v in result.metrics.items():
                        log[f"eval/{ev.id}/{k}"] = v
                    if not args.no_wandb and result.table_rows:
                        cols = list(result.table_rows[0].keys())
                        log[f"eval/{ev.id}/rollouts"] = wandb.Table(
                            columns=cols,
                            data=[[r[c] for c in cols] for r in result.table_rows],
                        )
                except Exception as _ev_err:
                    print(f"[external eval {ev.id}@{step}] FAILED: "
                          f"{type(_ev_err).__name__}: {_ev_err}", flush=True)

            # ---- Pareto chart: FVE (capability) vs hallucinations (faithful-
            # ness) vs Karvonen captures-quirk (depth) on a shared X-axis.
            # Tells you at a glance whether RL is buying capability at the
            # cost of faithfulness or whether it's Pareto-improving.
            pareto_history.append({
                "step": step,
                "fve_pct": float(log.get("eval/fve_pct", float("nan"))),
                "halluc_x10": float(log.get(
                    "eval/hallucination/hallucinations_mean", float("nan"))) * 10.0,
                "captures_quirk_pct": float(log.get(
                    "eval/karvonen_confusion/captures_quirk_rate", float("nan"))) * 100.0,
                "clean_rate_pct": float(log.get(
                    "eval/hallucination/clean_rate", float("nan"))) * 100.0,
            })
            if not args.no_wandb and len(pareto_history) >= 2:
                _xs = [h["step"] for h in pareto_history]
                log["pareto/capability_vs_faithfulness"] = wandb.plot.line_series(
                    xs=_xs,
                    ys=[
                        [h["fve_pct"] for h in pareto_history],
                        [h["halluc_x10"] for h in pareto_history],
                        [h["captures_quirk_pct"] for h in pareto_history],
                        [h["clean_rate_pct"] for h in pareto_history],
                    ],
                    keys=["FVE %", "hallucinations × 10",
                          "captures_quirk %", "clean_rate %"],
                    title="Capability vs faithfulness (shared X = step)",
                    xname="step",
                )

        if not args.no_wandb:
            wandb.log(log, step=step)

        # ---- save LoRA periodically ----
        if (step + 1) % args.save_every == 0:
            out_dir = save_dir / f"iter_{step + 1:06d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Critic FIRST: resume keys off the actor's adapter_config.json,
            # so a crash between the two saves must leave critic-without-actor
            # (resume fails loudly) rather than actor-without-critic (resume
            # silently falls back to the SFT critic → reward discontinuity).
            if args.train_critic:
                from safetensors.torch import save_file as _save_file
                crit_dir = out_dir / "critic"
                crit_dir.mkdir(exist_ok=True)
                _crit_sd = {n: p_.detach().cpu().contiguous()
                            for n, p_ in critic.named_parameters()
                            if ("lora_" in n) or n.startswith("value_head")}
                _save_file(_crit_sd, str(crit_dir / "ar_lora_value_head.safetensors"))
                (crit_dir / "ar_meta.json").write_text(_json.dumps(ar_meta, indent=2))
            actor.save_pretrained(str(out_dir))
            print(f"[save] LoRA → {out_dir}"
                  + (" (+ co-trained critic)" if args.train_critic else ""))

    print("done.")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
