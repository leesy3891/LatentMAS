"""
analyze_latent_entropy.py  (v4 — stability metrics refactor)
=============================================================
Measures per-step **normalized entropy**, **JS divergence**, **cosine
similarity**, **angular distance**, and **perplexity** of hidden states
across all methods, strictly reflecting the multi-agent execution pipeline
defined in ``run.py`` and ``methods/latent_mas.py``.

Changes from v3
----------------
1. **Metric taxonomy refactored**:
   - Normalized entropy in [0, 1] replaces raw entropy as primary confidence metric.
   - Jensen-Shannon divergence in [0, 1] replaces raw KL as primary drift metric.
   - Angular distance in [0, 1] added alongside cosine similarity in [-1, 1].
   - Raw KL divergence removed from tracked metrics.

2. **Decision-time vs post-update split** (decode agents):
   - Stage A (decision_time): metrics computed from the hidden state *before*
     token sampling (normalized entropy, JS divergence vs previous decision).
   - Stage B (post_update): metrics computed *after* the sampled token is fed
     back (JS divergence, cosine similarity, angular distance vs previous
     post-update state).

3. **Latent-step category** (latent agents):
   - Latent recurrence steps are NOT decoded-token steps and are recorded
     under an explicit ``latent_step`` category with confidence and hidden
     drift sub-metrics.

4. **Boundary metrics fixed**:
   - Now uses true current-agent start state (step-0 after prompt prefill) vs
     previous-agent end state, instead of end-to-end comparison.
   - Boundary rows include JS divergence, cosine similarity, angular distance,
     source/target roles, and boundary type.

5. **Perplexity-based error propagation analysis**:
   - Perplexity = exp(entropy) computed at every analysis step.
   - LatentMAS: per latent step; TextMAS: up to 80 sampled steps per agent.
   - Aggregated plots with interpretation rule.

6. **Top-token analysis**: Fixed top-5 with Jaccard overlap (replaces Gaussian
   cumulative-mass cutoff).

7. **File naming**: ``_stability_results.json`` replaces ``_entropy_results.json``.

Usage (unchanged CLI; analysis adapts internally):
  # latent_mas
  CUDA_VISIBLE_DEVICES=3 python analyze_latent_entropy.py \\
      --method latent_mas --model_name Qwen/Qwen3-4B --task aime2024 \\
      --latent_steps 80 --prompt sequential --max_samples -1

  # text_mas
  CUDA_VISIBLE_DEVICES=3 python analyze_latent_entropy.py \\
      --method text_mas --model_name Qwen/Qwen3-4B --task aime2024 \\
      --prompt sequential --max_samples -1

  # baseline
  CUDA_VISIBLE_DEVICES=3 python analyze_latent_entropy.py \\
      --method baseline --model_name Qwen/Qwen3-4B --task aime2024 \\
      --prompt sequential --max_samples -1
"""

import argparse
import json
import math
import os
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import ModelWrapper, _past_length
from data import (
    load_aime2024, load_aime2025, load_gsm8k,
    load_gpqa_diamond, load_arc_easy, load_arc_challenge,
    load_mbppplus, load_humanevalplus, load_medqa,
)
from utils import (
    set_seed, auto_device,
    extract_gsm8k_answer, normalize_answer,
    extract_markdown_python_block, run_with_timeout,
)

# ── Extracted visualization, logging, and analysis utilities ──────────
from scripts import (
    # Constants
    EXEC_IDX_COLORS, METRIC_KEYS,
    # Color / label helpers
    get_exec_color, _build_exec_label, make_prefix,
    # Plotting
    plot_per_agent_metric, plot_concatenated_overview,
    plot_boundary_metrics, plot_perplexity, run_all_plots,
    # TXT logging
    write_txt_case, write_txt_summary,
    # JSON logging
    build_step_semantics, save_json_results,
    # Top-token probability analysis (deferred batch, fixed top-5)
    batch_project_hidden_states,
    TOP_K,
)

# ── prompt builders (latent_mas & text_mas use different ones) ──────────
try:
    from prompts import (
        build_agent_message_sequential_latent_mas,
        build_agent_message_hierarchical_latent_mas,
        build_agent_messages_sequential_text_mas,
        build_agent_messages_hierarchical_text_mas,
    )
    _HAS_PROMPTS = True
except ImportError:
    _HAS_PROMPTS = False

# ── agent definitions ──────────────────────────────────────────────────
try:
    from methods import default_agents
except ImportError:
    Agent = namedtuple("Agent", ["name", "role"])
    def default_agents():
        return [
            Agent("Planner", "planner"),
            Agent("Critic",  "critic"),
            Agent("Refiner", "refiner"),
            Agent("Judger",  "judger"),
        ]

# ═══════════════════════════════════════════════════════════════════════
# Dataset loaders
# ═══════════════════════════════════════════════════════════════════════

TASK_LOADERS = {
    "gsm8k":         lambda: load_gsm8k(split="test"),
    "aime2024":      lambda: load_aime2024(split="train"),
    "aime2025":      lambda: load_aime2025(split="train"),
    "gpqa":          lambda: load_gpqa_diamond(split="test"),
    "arc_easy":      lambda: load_arc_easy(split="test"),
    "arc_challenge": lambda: load_arc_challenge(split="test"),
    "mbppplus":      lambda: load_mbppplus(split="test"),
    "humanevalplus": lambda: load_humanevalplus(split="test"),
    "medqa":         lambda: load_medqa(split="test"),
}


# ═══════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_lm_head(model):
    """Locate the language model head (output embeddings)."""
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Cannot locate lm_head / output embeddings.")
    return lm_head


@torch.no_grad()
def compute_distribution_metrics(
    logits: torch.Tensor,
    prev_probs: Optional[torch.Tensor] = None,
    vocab_size: Optional[int] = None,
) -> Dict:
    """Compute distribution-space metrics from logits.

    Args:
        logits:     [1, V] or [V] raw logits from lm_head.
        prev_probs: [1, V] or [V] probability distribution from the previous
                    step (used for JS divergence).  None if no previous step.
        vocab_size: vocabulary size for normalized entropy.  If None, inferred
                    from logits.shape[-1].

    Returns:
        Dict with keys:
            logits              — [1, V] tensor (detached, float32)
            log_probs           — [1, V] tensor
            probs               — [1, V] tensor
            normalized_entropy  — float in [0, 1]
            raw_entropy         — float (nats, used for perplexity)
            js_divergence       — float in [0, 1] or None
            perplexity          — float = exp(raw_entropy)
    """
    logits_f = logits.float().view(1, -1)                        # [1, V]
    V = vocab_size or logits_f.shape[-1]

    log_probs = torch.log_softmax(logits_f, dim=-1)              # [1, V]
    probs = log_probs.exp()                                      # [1, V]

    # Raw entropy:  H = -sum p * log(p)
    raw_entropy = -(probs * log_probs).sum(dim=-1).item()

    # Normalized entropy:  H / log(V)  in [0, 1]
    log_V = math.log(V) if V > 1 else 1.0
    norm_entropy = raw_entropy / log_V

    # Perplexity: exp(H)
    perplexity = math.exp(raw_entropy)

    # Jensen-Shannon divergence vs previous distribution
    js_div = None
    if prev_probs is not None:
        prev_p = prev_probs.float().view(1, -1)
        curr_p = probs
        m = 0.5 * (curr_p + prev_p)
        # Clamp to avoid log(0)
        m_clamped = m.clamp_min(1e-12)
        curr_clamped = curr_p.clamp_min(1e-12)
        prev_clamped = prev_p.clamp_min(1e-12)
        kl_curr_m = (curr_p * (curr_clamped.log() - m_clamped.log())).sum(dim=-1)
        kl_prev_m = (prev_p * (prev_clamped.log() - m_clamped.log())).sum(dim=-1)
        # JS in [0, ln(2)] with base-e; normalize by ln(2) to get [0, 1]
        js_raw = 0.5 * kl_curr_m + 0.5 * kl_prev_m
        js_div = (js_raw / math.log(2)).clamp(0.0, 1.0).item()

    return {
        "logits":             logits_f.detach(),
        "log_probs":          log_probs.detach(),
        "probs":              probs.detach(),
        "normalized_entropy": norm_entropy,
        "raw_entropy":        raw_entropy,
        "js_divergence":      js_div,
        "perplexity":         perplexity,
    }


@torch.no_grad()
def compute_hidden_drift_metrics(
    hidden: torch.Tensor,
    prev_hidden: Optional[torch.Tensor] = None,
) -> Dict:
    """Compute hidden-space drift metrics between adjacent hidden states.

    Args:
        hidden:      [1, D] or [D] current hidden state.
        prev_hidden: [1, D] or [D] previous hidden state, or None at step-0.

    Returns:
        Dict with:
            cosine_similarity  — float in [-1, 1] or None
            angular_distance   — float in [0, 1]  or None
    """
    if prev_hidden is None:
        return {"cosine_similarity": None, "angular_distance": None}

    cos = F.cosine_similarity(
        hidden.float().view(1, -1),
        prev_hidden.float().view(1, -1),
        dim=-1,
    ).item()

    # Angular distance = arccos(clamp(cos)) / pi  in [0, 1]
    cos_clamped = max(-1.0, min(1.0, cos))
    ang = math.acos(cos_clamped) / math.pi

    return {"cosine_similarity": cos, "angular_distance": ang}


# ═══════════════════════════════════════════════════════════════════════
# Core: latent recurrence for one non-judger agent  (LatentMAS)
# ═══════════════════════════════════════════════════════════════════════
#
# Ground truth (models.py -> generate_latent_batch):
#   1. Run full forward pass on the agent's prompt tokens (with KV cache).
#   2. Take last_hidden = hidden_states[-1][:, -1, :].
#   3. For each latent step:
#        a. latent_vec = _apply_latent_realignment(last_hidden, model)
#        b. Feed latent_vec as inputs_embeds ([1,1,D]) with KV cache.
#        c. Update last_hidden from the new output.
#   4. Return accumulated KV cache.
#
# The analysis mirrors this exactly, adding per-step metric recording.
#
# Latent-step metrics are recorded under the ``latent_step`` category:
#   - confidence: normalized_entropy, js_divergence (vs previous latent step)
#   - hidden drift: cosine_similarity, angular_distance (vs previous hidden)
# These are NOT decode-agent decision_time/post_update metrics.

@torch.no_grad()
def run_latent_agent_with_metrics(
    model_wrapper: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_steps: int,
    past_kv: Optional[Tuple] = None,
) -> Tuple:
    """Execute one non-judger agent via latent recurrence, recording
    per-step metrics under the ``latent_step`` category.

    Returns:
        (metrics_dict, updated_past_kv,
         start_hidden, start_log_probs, start_probs,
         end_hidden, end_log_probs,
         all_hiddens, perplexity_values)

    start_hidden / start_log_probs / start_probs:
        The step-0 state immediately after prompt prefill.  Used for
        true boundary comparison with the previous agent's end state.

    end_hidden / end_log_probs:
        Final state after all latent recurrence steps.

    all_hiddens: list of [1, D] tensors on CPU, length = latent_steps + 1
        (step-0 after prompt prefill, then one per latent recurrence step).

    perplexity_values: list of floats, one per step (including step-0).
    """
    model = model_wrapper.model
    device = model_wrapper.device
    lm_head = _get_lm_head(model)
    vocab_size = lm_head.weight.shape[0]

    # ── Extend attention mask for existing KV cache ──
    attn = attention_mask.to(device)
    if past_kv is not None:
        plen = _past_length(past_kv)
        if plen > 0:
            past_mask = torch.ones(
                (attn.shape[0], plen), dtype=attn.dtype, device=device,
            )
            attn = torch.cat([past_mask, attn], dim=-1)

    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attn,
        past_key_values=past_kv,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, D]

    # ── Metric accumulators ──
    norm_entropies: List[Optional[float]] = []
    js_divs: List[Optional[float]] = []
    cosines: List[Optional[float]] = []
    angular_dists: List[Optional[float]] = []
    perplexities: List[float] = []
    all_hiddens: List[torch.Tensor] = []

    # ── Step-0: after processing the agent's prompt tokens ──
    logits_0 = lm_head(last_hidden.to(lm_head.weight.dtype))
    dist_0 = compute_distribution_metrics(logits_0, prev_probs=None,
                                          vocab_size=vocab_size)

    norm_entropies.append(dist_0["normalized_entropy"])
    js_divs.append(None)           # no previous step
    cosines.append(None)           # no previous hidden
    angular_dists.append(None)     # no previous hidden
    perplexities.append(dist_0["perplexity"])
    all_hiddens.append(last_hidden.detach().cpu())

    # Save step-0 state for boundary comparison
    start_hidden = last_hidden.detach().clone()
    start_log_probs = dist_0["log_probs"].detach().clone()
    start_probs = dist_0["probs"].detach().clone()

    prev_probs = dist_0["probs"]
    prev_hidden = last_hidden.clone()

    # Track the most recent dist for end-state (handles latent_steps == 0)
    latest_dist = dist_0

    # ── Latent recurrence (matches generate_latent_batch exactly) ──
    for _ in range(latent_steps):
        latent_vec = model_wrapper._apply_latent_realignment(last_hidden, model)
        latent_embed = latent_vec.unsqueeze(1)  # [1, 1, D]

        plen = _past_length(past)
        latent_mask = torch.ones((1, plen + 1), dtype=torch.long, device=device)

        outputs = model(
            inputs_embeds=latent_embed,
            attention_mask=latent_mask,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        logits_s = lm_head(last_hidden.to(lm_head.weight.dtype))
        dist_s = compute_distribution_metrics(logits_s, prev_probs=prev_probs,
                                              vocab_size=vocab_size)
        drift_s = compute_hidden_drift_metrics(last_hidden, prev_hidden)

        norm_entropies.append(dist_s["normalized_entropy"])
        js_divs.append(dist_s["js_divergence"])
        cosines.append(drift_s["cosine_similarity"])
        angular_dists.append(drift_s["angular_distance"])
        perplexities.append(dist_s["perplexity"])
        all_hiddens.append(last_hidden.detach().cpu())

        prev_probs = dist_s["probs"]
        prev_hidden = last_hidden.clone()
        latest_dist = dist_s

    # Build the metrics dict using the latent_step schema
    metrics = {
        "normalized_entropy":  norm_entropies,
        "js_divergence":       js_divs,
        "cosine_similarity":   cosines,
        "angular_distance":    angular_dists,
        "n_steps":             latent_steps,
        "metric_category":     "latent_step",
    }

    end_hidden = last_hidden.detach().clone()
    end_log_probs = latest_dist["log_probs"].detach().clone()

    return (metrics, past,
            start_hidden, start_log_probs, start_probs,
            end_hidden, end_log_probs,
            all_hiddens, perplexities)


# ═══════════════════════════════════════════════════════════════════════
# Core: autoregressive decode for one agent  (TextMAS / judger / baseline)
# ═══════════════════════════════════════════════════════════════════════
#
# Ground truth:
#   - TextMAS (text_mas.py): Each agent gets a fresh forward pass (no KV
#     cache sharing).  Context is accumulated text.
#   - LatentMAS judger: Gets the accumulated KV cache from all prior
#     latent agents, then decodes autoregressively.
#   - Baseline: Single forward pass + decode, no multi-agent pipeline.
#
# Metric phases per decode step:
#   Stage A -- decision_time (before sampling):
#     - normalized_entropy
#     - js_divergence vs previous decision-time distribution
#   Stage B -- post_update (after feeding sampled token back):
#     - js_divergence vs previous post-update distribution
#     - cosine_similarity vs previous post-update hidden
#     - angular_distance vs previous post-update hidden

@torch.no_grad()
def run_decode_agent_with_metrics(
    model_wrapper: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 2048,
    n_metric_steps: int = 80,
    past_kv: Optional[Tuple] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> Tuple:
    """Decode tokens for one agent, recording per-step metrics with explicit
    decision-time / post-update separation for the first *n_metric_steps*
    decoded tokens.

    Returns:
        (metrics_dict, decoded_text, total_decoded_tokens,
         start_hidden, start_log_probs, start_probs,
         end_hidden, end_log_probs,
         all_decision_perplexities)

    start_hidden / start_log_probs / start_probs:
        Step-0 state after prompt prefill, for boundary comparison.

    end_hidden / end_log_probs:
        Final hidden state and log-probs after all decoding.

    all_decision_perplexities:
        Perplexity at every decision-time step (including step-0 prefill).
    """
    model = model_wrapper.model
    device = model_wrapper.device
    lm_head = _get_lm_head(model)
    vocab_size = lm_head.weight.shape[0]

    # ── Extend attention mask for existing KV cache ──
    attn = attention_mask.to(device)
    if past_kv is not None:
        plen = _past_length(past_kv)
        if plen > 0:
            past_mask = torch.ones(
                (attn.shape[0], plen), dtype=attn.dtype, device=device,
            )
            attn = torch.cat([past_mask, attn], dim=-1)

    # Prefill
    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attn,
        past_key_values=past_kv,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1, :]

    # ── Metric accumulators ──
    # Decision-time metrics (Stage A)
    dt_norm_entropies: List[Optional[float]] = []
    dt_js_divs: List[Optional[float]] = []
    # Post-update metrics (Stage B)
    pu_js_divs: List[Optional[float]] = []
    pu_cosines: List[Optional[float]] = []
    pu_angular_dists: List[Optional[float]] = []
    # Perplexity (from decision-time)
    all_perplexities: List[float] = []

    # ── Step-0 (prefill): decision-time snapshot ──
    logits_0 = lm_head(last_hidden.to(lm_head.weight.dtype))
    dist_0 = compute_distribution_metrics(logits_0, prev_probs=None,
                                          vocab_size=vocab_size)

    dt_norm_entropies.append(dist_0["normalized_entropy"])
    dt_js_divs.append(None)        # no previous decision
    pu_js_divs.append(None)        # no post-update at step-0
    pu_cosines.append(None)
    pu_angular_dists.append(None)
    all_perplexities.append(dist_0["perplexity"])

    # Save step-0 state for boundary comparison
    start_hidden = last_hidden.detach().clone()
    start_log_probs = dist_0["log_probs"].detach().clone()
    start_probs = dist_0["probs"].detach().clone()

    # Tracking state for decision-time JS
    prev_decision_probs = dist_0["probs"]
    # Tracking state for post-update drift
    prev_post_update_probs: Optional[torch.Tensor] = None
    prev_post_update_hidden: Optional[torch.Tensor] = None

    # ── Autoregressive decoding ──
    generated_ids: List[int] = []
    eos_id = model_wrapper.tokenizer.eos_token_id

    for step in range(max_new_tokens):
        # ── Stage A: decision-time metrics (current hidden, before sampling) ──
        logits_dt = lm_head(last_hidden.to(lm_head.weight.dtype))
        logits_f = logits_dt.float()

        # Sampling
        if temperature > 0:
            probs_sample = F.softmax(logits_f / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs_sample, descending=True, dim=-1)
            cumsum = sorted_probs.cumsum(dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
        else:
            next_token = logits_f.argmax(dim=-1, keepdim=True)

        next_token = next_token.squeeze(-1)
        tok_id = next_token[0].item()
        generated_ids.append(tok_id)

        if tok_id == eos_id:
            break

        # Record decision-time metrics for first n_metric_steps
        if step < n_metric_steps:
            dist_dt = compute_distribution_metrics(
                logits_dt, prev_probs=prev_decision_probs,
                vocab_size=vocab_size,
            )
            dt_norm_entropies.append(dist_dt["normalized_entropy"])
            dt_js_divs.append(dist_dt["js_divergence"])
            all_perplexities.append(dist_dt["perplexity"])
            prev_decision_probs = dist_dt["probs"]

        # ── Forward the sampled token ──
        next_input = next_token.unsqueeze(-1)
        plen = _past_length(past)
        new_mask = torch.ones((1, plen + 1), dtype=torch.long, device=device)

        outputs = model(
            input_ids=next_input,
            attention_mask=new_mask,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        # ── Stage B: post-update metrics (after token feedback) ──
        if step < n_metric_steps:
            logits_pu = lm_head(last_hidden.to(lm_head.weight.dtype))
            dist_pu = compute_distribution_metrics(
                logits_pu, prev_probs=prev_post_update_probs,
                vocab_size=vocab_size,
            )
            drift_pu = compute_hidden_drift_metrics(
                last_hidden, prev_post_update_hidden,
            )
            pu_js_divs.append(dist_pu["js_divergence"])
            pu_cosines.append(drift_pu["cosine_similarity"])
            pu_angular_dists.append(drift_pu["angular_distance"])

            prev_post_update_probs = dist_pu["probs"]
            prev_post_update_hidden = last_hidden.clone()

    decoded_text = model_wrapper.tokenizer.decode(
        generated_ids, skip_special_tokens=True,
    ).strip()

    # The final end state for boundary comparison
    logits_end = lm_head(last_hidden.to(lm_head.weight.dtype))
    dist_end = compute_distribution_metrics(logits_end, vocab_size=vocab_size)
    end_hidden = last_hidden.detach().clone()
    end_log_probs = dist_end["log_probs"].detach().clone()

    # ── Build unified metrics dict ──
    # For plotting compatibility, the primary METRIC_KEYS arrays use:
    #   normalized_entropy -> decision-time values
    #   js_divergence      -> post-update values
    #   cosine_similarity  -> post-update values
    #   angular_distance   -> post-update values
    # The full detail is available in the JSON under decision_time / post_update.
    metrics = {
        # Flat arrays for plotting (unified view)
        "normalized_entropy":  dt_norm_entropies,
        "js_divergence":       pu_js_divs,
        "cosine_similarity":   pu_cosines,
        "angular_distance":    pu_angular_dists,
        "n_steps":             len(dt_norm_entropies) - 1,  # excluding step-0
        "metric_category":     "decode",
        # Detailed phase-separated metrics for JSON
        "decision_time": {
            "normalized_entropy": dt_norm_entropies,
            "js_divergence":      dt_js_divs,
        },
        "post_update": {
            "js_divergence":    pu_js_divs,
            "cosine_similarity": pu_cosines,
            "angular_distance":  pu_angular_dists,
        },
    }
    return (metrics, decoded_text, len(generated_ids),
            start_hidden, start_log_probs, start_probs,
            end_hidden, end_log_probs,
            all_perplexities)


# ═══════════════════════════════════════════════════════════════════════
# Prompt builders  (dispatch to the correct one per method x topology)
# ═══════════════════════════════════════════════════════════════════════

def build_prompt_latent(role: str, question: str, args) -> List[Dict]:
    """Build prompt for LatentMAS agents.

    Ground truth note: In latent_mas.py, context is always "" for all agents.
    Communication between agents happens exclusively through the KV cache,
    not through text context.
    """
    if not _HAS_PROMPTS:
        return [
            {"role": "system",
             "content": "You are a helpful assistant. Think step by step."},
            {"role": "user", "content": question},
        ]
    if args.prompt == "hierarchical":
        return build_agent_message_hierarchical_latent_mas(
            role=role, question=question, context="",
            method="latent_mas", args=args,
        )
    return build_agent_message_sequential_latent_mas(
        role=role, question=question, context="",
        method="latent_mas", args=args,
    )


def build_prompt_text(role: str, question: str, context: str, args) -> List[Dict]:
    """Build prompt for TextMAS agents.

    Ground truth note: In text_mas.py, context accumulates the text output
    of all prior agents.  Each agent receives a fresh prompt with full
    context -- no KV cache is shared.
    """
    if not _HAS_PROMPTS:
        content = f"{question}\n\n{context}" if context else question
        return [
            {"role": "system",
             "content": "You are a helpful assistant. Think step by step."},
            {"role": "user", "content": content},
        ]
    if args.prompt == "hierarchical":
        return build_agent_messages_hierarchical_text_mas(
            role=role, question=question, context=context,
            method="text_mas", args=args,
        )
    return build_agent_messages_sequential_text_mas(
        role=role, question=question, context=context,
        method="text_mas", args=args,
    )


# ═══════════════════════════════════════════════════════════════════════
# Answer evaluation  (mirroring run.py logic exactly)
# ═══════════════════════════════════════════════════════════════════════

def evaluate_answer(final_text: str, item: Dict, task: str) -> Tuple[str, str, bool]:
    if task in ["mbppplus", "humanevalplus"]:
        pred = extract_markdown_python_block(final_text)
        gold = item.get("gold", "")
        if pred is None:
            return "", gold, False
        code = pred + "\n" + gold
        ok, _ = run_with_timeout(code, timeout=10)
        return pred or "", gold, ok

    elif task in ["aime2024", "aime2025"]:
        pred = normalize_answer(extract_gsm8k_answer(final_text))
        gold = str(item.get("gold", "")).strip()
        try:
            ok = int(pred) == int(gold)
        except (ValueError, TypeError):
            ok = False
        return pred, gold, ok

    else:
        pred = normalize_answer(extract_gsm8k_answer(final_text))
        gold = str(item.get("gold", "")).strip()
        ok = (pred == gold) if (pred and gold) else False
        return pred, gold, ok


# ═══════════════════════════════════════════════════════════════════════
# Tokenise helper  (single-item, returns [1, L])
# ═══════════════════════════════════════════════════════════════════════

def _tokenize_prompt(model_wrapper: ModelWrapper, prompt_text: str, device):
    enc = model_wrapper.tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ═══════════════════════════════════════════════════════════════════════
# Inter-agent boundary metric computation  (FIXED)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_boundary_metrics(
    current_start_hidden: torch.Tensor,
    current_start_probs: torch.Tensor,
    previous_end_hidden: Optional[torch.Tensor],
    previous_end_probs: Optional[torch.Tensor],
) -> Dict:
    """Compute JS divergence, cosine similarity, and angular distance at
    an agent boundary.

    Uses the TRUE start state of the current agent (step-0 after prompt
    prefill) vs the TRUE end state of the previous agent.

    Args:
        current_start_hidden:  [1, D] hidden at current agent step-0.
        current_start_probs:   [1, V] probs at current agent step-0.
        previous_end_hidden:   [1, D] hidden at end of previous agent (or None).
        previous_end_probs:    [1, V] probs at end of previous agent (or None).

    Returns:
        Dict with boundary_js_divergence, boundary_cosine_similarity,
        boundary_angular_distance (all None if no previous agent).
    """
    if previous_end_hidden is None or previous_end_probs is None:
        return {
            "boundary_js_divergence":     None,
            "boundary_cosine_similarity": None,
            "boundary_angular_distance":  None,
        }

    # JS divergence between current start and previous end
    curr_p = current_start_probs.float().view(1, -1)
    prev_p = previous_end_probs.float().view(1, -1)
    m = 0.5 * (curr_p + prev_p)
    m_clamped = m.clamp_min(1e-12)
    curr_clamped = curr_p.clamp_min(1e-12)
    prev_clamped = prev_p.clamp_min(1e-12)
    kl_curr_m = (curr_p * (curr_clamped.log() - m_clamped.log())).sum(dim=-1)
    kl_prev_m = (prev_p * (prev_clamped.log() - m_clamped.log())).sum(dim=-1)
    js_raw = 0.5 * kl_curr_m + 0.5 * kl_prev_m
    js_div = (js_raw / math.log(2)).clamp(0.0, 1.0).item()

    # Hidden-space drift
    drift = compute_hidden_drift_metrics(current_start_hidden, previous_end_hidden)

    return {
        "boundary_js_divergence":     js_div,
        "boundary_cosine_similarity": drift["cosine_similarity"],
        "boundary_angular_distance":  drift["angular_distance"],
    }


# ═══════════════════════════════════════════════════════════════════════
# Perplexity sampling for TextMAS
# ═══════════════════════════════════════════════════════════════════════

def _sample_perplexity_steps(
    perplexities: List[float],
    max_points: int = 80,
) -> List[float]:
    """Sample up to max_points evenly-spaced perplexity values from a list.

    If the list has fewer than max_points elements, return all of them.
    """
    n = len(perplexities)
    if n <= max_points:
        return list(perplexities)
    indices = np.linspace(0, n - 1, max_points, dtype=int)
    return [perplexities[i] for i in indices]


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent hidden-state stability analysis: "
                    "normalized entropy, JS divergence, cosine similarity, "
                    "angular distance, perplexity."
    )
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "text_mas", "latent_mas"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--task", type=str, default="aime2024",
                        choices=list(TASK_LOADERS.keys()))
    parser.add_argument("--latent_steps", type=int, default=80)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="sequential",
                        choices=["sequential", "hierarchical"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--bin_size", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="example_logs")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--n_metric_steps", type=int, default=80,
                        help="How many decoded-token steps to record metrics for "
                             "(per decode agent)")
    parser.add_argument("--max_txt_cases", type=int, default=30,
                        help="Max cases written to the TXT case-study file")
    # Flags consumed by ModelWrapper / prompt builders
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--text_mas_context_length", type=int, default=-1)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    prefix = make_prefix(args.task, args.model_name, args.method)
    os.makedirs(args.out_dir, exist_ok=True)

    agents = default_agents()
    agent_roles = [a.role for a in agents]
    agent_names = [a.name for a in agents]

    print(f"[Config] method={args.method}, model={args.model_name}, "
          f"task={args.task}, device={device}")
    print(f"         agents={agent_names}")
    if args.method == "latent_mas":
        print(f"         latent_steps={args.latent_steps}, "
              f"realign={args.latent_space_realign}")
    print(f"         n_metric_steps (decode)={args.n_metric_steps}")

    # ── Step semantics note ──
    print()
    print("=" * 60)
    print("STEP SEMANTICS NOTE")
    print("=" * 60)
    if args.method == "latent_mas":
        print(f"  LatentMAS non-judger agents: each 'step' = 1 latent recurrence")
        print(f"    (hidden -> realign -> forward with KV cache).  Total = {args.latent_steps} per agent.")
        print(f"    Metrics category: latent_step (confidence + hidden drift).")
        print(f"  LatentMAS judger: each 'step' = 1 decoded token.")
        print(f"    Total = up to {args.max_new_tokens} tokens, metrics for first {args.n_metric_steps}.")
        print(f"    Metrics split: decision_time (confidence) + post_update (drift).")
        print(f"  These step types are NOT directly comparable across agents.")
    elif args.method == "text_mas":
        print(f"  TextMAS all agents: each 'step' = 1 decoded token.")
        print(f"    Metrics recorded for first {args.n_metric_steps} tokens per agent.")
        print(f"    Metrics split: decision_time (confidence) + post_update (drift).")
        print(f"  No KV cache sharing -- each agent gets a fresh forward pass.")
    else:
        print(f"  Baseline: each 'step' = 1 decoded token.")
        print(f"    Metrics recorded for first {args.n_metric_steps} tokens.")
        print(f"    Metrics split: decision_time (confidence) + post_update (drift).")
    print("=" * 60)
    print()

    # ── Load model (always HF, no vLLM for analysis) ──
    model_wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)

    # ── Load dataset ──
    dataset = list(TASK_LOADERS[args.task]())
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
    print(f"[Data] {args.task}: {len(dataset)} cases\n")

    # ── Prepare TXT file ──
    txt_path = os.path.join(args.out_dir, f"{prefix}_case_study.txt")
    txt_file = open(txt_path, "w", encoding="utf-8")

    # ── Agent name mapping for TextMAS hierarchical context formatting ──
    _HIER_NAME_MAP = {
        "Planner": "Math Agent",  "Critic": "Science Agent",
        "Refiner": "Code Agent",  "Judger": "Task Summrizer",
    }

    # ═════════════════════════════════════════════════════════════════
    # Main loop
    # ═════════════════════════════════════════════════════════════════

    data_rows: List[Dict] = []          # flat intra-agent metric rows
    boundary_rows: List[Dict] = []      # inter-agent boundary metrics
    perplexity_rows: List[Dict] = []    # per-agent perplexity data
    cases_meta: List[Dict] = []         # per-case result metadata
    n_correct = 0
    # Deferred hidden-state storage for latent_mas top-token analysis
    all_cases_hidden_records: List[Dict] = []

    for case_idx, item in enumerate(tqdm(dataset, desc="Analyzing")):
        question = item["question"]
        agent_records: List[Dict] = []   # temporary per-case, used for TXT
        final_text = ""
        exec_idx = 0   # global execution counter within this case

        # Track end-of-agent state for boundary metrics
        prev_agent_end_hidden: Optional[torch.Tensor] = None
        prev_agent_end_probs: Optional[torch.Tensor] = None
        prev_agent_role: Optional[str] = None
        prev_agent_type: Optional[str] = None

        # ─────────────────────────────────────────────────
        #  LatentMAS
        # ─────────────────────────────────────────────────
        if args.method == "latent_mas":
            past_kv: Optional[Tuple] = None
            case_hidden_records: List[Dict] = []

            for agent in agents:
                messages = build_prompt_latent(agent.role, question, args)
                prompt_text = model_wrapper.render_chat(
                    messages, add_generation_prompt=True,
                )
                if args.think:
                    prompt_text = f"{prompt_text}<think>"

                input_ids, attn_mask = _tokenize_prompt(
                    model_wrapper, prompt_text, device,
                )

                if agent.role != "judger":
                    # ── Latent recurrence agent ──
                    (metrics, past_kv,
                     start_hidden, start_log_probs, start_probs,
                     end_hidden, end_log_probs,
                     all_hiddens, ppl_values) = \
                        run_latent_agent_with_metrics(
                            model_wrapper, input_ids, attn_mask,
                            latent_steps=args.latent_steps,
                            past_kv=past_kv,
                        )
                    ag_type = "latent"
                    ag_info = {
                        "name": agent.name, "role": agent.role,
                        "type": ag_type, "exec_idx": exec_idx, **metrics,
                    }

                    # Store hidden states for deferred top-token analysis
                    case_hidden_records.append({
                        "agent_name": agent.name,
                        "agent_role": agent.role,
                        "exec_idx":   exec_idx,
                        "hiddens":    all_hiddens,
                    })

                    # ── Boundary: true start vs previous end ──
                    boundary = compute_boundary_metrics(
                        start_hidden, start_probs,
                        prev_agent_end_hidden, prev_agent_end_probs,
                    )
                    boundary["transition"] = (
                        f"exec{exec_idx-1}->exec{exec_idx}" if exec_idx > 0
                        else None
                    )
                    boundary["case_idx"] = case_idx
                    boundary["source_exec_idx"] = exec_idx - 1 if exec_idx > 0 else None
                    boundary["target_exec_idx"] = exec_idx
                    boundary["source_role"] = prev_agent_role
                    boundary["target_role"] = agent.role
                    boundary["boundary_type"] = (
                        f"{prev_agent_type}->latent" if prev_agent_type else None
                    )
                    if boundary["transition"] is not None:
                        boundary_rows.append(boundary)

                    # Update previous-agent tracking with END state
                    prev_agent_end_hidden = end_hidden.clone()
                    prev_agent_end_probs = end_log_probs.exp().detach().clone()
                    prev_agent_role = agent.role
                    prev_agent_type = "latent"

                    # Perplexity row (all latent steps)
                    perplexity_rows.append({
                        "case_idx":          case_idx,
                        "exec_idx":          exec_idx,
                        "agent_role":        agent.role,
                        "agent_type":        "latent",
                        "perplexity_values": ppl_values,
                    })

                else:
                    # ── Judger: decode with accumulated KV cache ──
                    past_for_dec = past_kv if args.latent_steps > 0 else None
                    (metrics, decoded_text, n_decoded,
                     start_hidden, start_log_probs, start_probs,
                     end_hidden, end_log_probs,
                     ppl_values) = \
                        run_decode_agent_with_metrics(
                            model_wrapper, input_ids, attn_mask,
                            max_new_tokens=args.max_new_tokens,
                            n_metric_steps=args.n_metric_steps,
                            past_kv=past_for_dec,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                    final_text = decoded_text
                    ag_type = "decode"
                    ag_info = {
                        "name": agent.name, "role": agent.role,
                        "type": ag_type, "exec_idx": exec_idx,
                        "decoded_tokens": n_decoded, **metrics,
                    }

                    # ── Boundary: true start vs previous end ──
                    boundary = compute_boundary_metrics(
                        start_hidden, start_probs,
                        prev_agent_end_hidden, prev_agent_end_probs,
                    )
                    boundary["transition"] = (
                        f"exec{exec_idx-1}->exec{exec_idx}" if exec_idx > 0
                        else None
                    )
                    boundary["case_idx"] = case_idx
                    boundary["source_exec_idx"] = exec_idx - 1 if exec_idx > 0 else None
                    boundary["target_exec_idx"] = exec_idx
                    boundary["source_role"] = prev_agent_role
                    boundary["target_role"] = agent.role
                    boundary["boundary_type"] = (
                        f"{prev_agent_type}->decode" if prev_agent_type else None
                    )
                    if boundary["transition"] is not None:
                        boundary_rows.append(boundary)

                    # Perplexity: sample 80 points from decode agent
                    sampled_ppl = _sample_perplexity_steps(ppl_values, max_points=80)
                    perplexity_rows.append({
                        "case_idx":          case_idx,
                        "exec_idx":          exec_idx,
                        "agent_role":        agent.role,
                        "agent_type":        "decode",
                        "perplexity_values": sampled_ppl,
                    })

                agent_records.append(ag_info)

                # ── Emit flat rows for plotting ──
                for mk in METRIC_KEYS:
                    data_rows.append({
                        "case_idx":   case_idx,
                        "metric":     mk,
                        "agent_role": agent.role,
                        "exec_idx":   exec_idx,
                        "agent_type": ag_type,
                        "values":     metrics[mk],
                    })
                exec_idx += 1

            # Store this case's hidden records for deferred top-token analysis
            if case_hidden_records:
                all_cases_hidden_records.append({
                    "case_idx": case_idx,
                    "agents": case_hidden_records,
                })

        # ─────────────────────────────────────────────────
        #  TextMAS
        # ─────────────────────────────────────────────────
        elif args.method == "text_mas":
            context = ""

            for agent in agents:
                messages = build_prompt_text(
                    agent.role, question, context, args,
                )
                prompt_text = model_wrapper.render_chat(
                    messages, add_generation_prompt=True,
                )
                input_ids, attn_mask = _tokenize_prompt(
                    model_wrapper, prompt_text, device,
                )

                # TextMAS: ALL agents decode (no KV cache from prior agents)
                (metrics, decoded_text, n_decoded,
                 start_hidden, start_log_probs, start_probs,
                 end_hidden, end_log_probs,
                 ppl_values) = \
                    run_decode_agent_with_metrics(
                        model_wrapper, input_ids, attn_mask,
                        max_new_tokens=args.max_new_tokens,
                        n_metric_steps=args.n_metric_steps,
                        past_kv=None,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )

                ag_type = "decode"
                ag_info = {
                    "name": agent.name, "role": agent.role,
                    "type": ag_type, "exec_idx": exec_idx,
                    "decoded_tokens": n_decoded, **metrics,
                }
                agent_records.append(ag_info)

                # ── Boundary: true start vs previous end ──
                boundary = compute_boundary_metrics(
                    start_hidden, start_probs,
                    prev_agent_end_hidden, prev_agent_end_probs,
                )
                boundary["transition"] = (
                    f"exec{exec_idx-1}->exec{exec_idx}" if exec_idx > 0
                    else None
                )
                boundary["case_idx"] = case_idx
                boundary["source_exec_idx"] = exec_idx - 1 if exec_idx > 0 else None
                boundary["target_exec_idx"] = exec_idx
                boundary["source_role"] = prev_agent_role
                boundary["target_role"] = agent.role
                boundary["boundary_type"] = (
                    f"{prev_agent_type}->decode" if prev_agent_type else None
                )
                if boundary["transition"] is not None:
                    boundary_rows.append(boundary)

                # Update tracking with END state
                prev_agent_end_hidden = end_hidden.clone()
                prev_agent_end_probs = end_log_probs.exp().detach().clone()
                prev_agent_role = agent.role
                prev_agent_type = "decode"

                # Perplexity: sample 80 points from each agent's decode
                sampled_ppl = _sample_perplexity_steps(ppl_values, max_points=80)
                perplexity_rows.append({
                    "case_idx":          case_idx,
                    "exec_idx":          exec_idx,
                    "agent_role":        agent.role,
                    "agent_type":        "decode",
                    "perplexity_values": sampled_ppl,
                })

                for mk in METRIC_KEYS:
                    data_rows.append({
                        "case_idx":   case_idx,
                        "metric":     mk,
                        "agent_role": agent.role,
                        "exec_idx":   exec_idx,
                        "agent_type": ag_type,
                        "values":     metrics[mk],
                    })
                exec_idx += 1

                if agent.role != "judger":
                    label = (_HIER_NAME_MAP.get(agent.name, agent.name)
                             if args.prompt == "hierarchical" else agent.name)
                    context += f"[{label}]:\n{decoded_text}\n\n"
                else:
                    final_text = decoded_text

        # ─────────────────────────────────────────────────
        #  Baseline  (no boundary metrics — single agent, explicit skip)
        # ─────────────────────────────────────────────────
        elif args.method == "baseline":
            messages = build_prompt_latent("planner", question, args)
            prompt_text = model_wrapper.render_chat(
                messages, add_generation_prompt=True,
            )
            if args.think:
                prompt_text = f"{prompt_text}<think>"

            input_ids, attn_mask = _tokenize_prompt(
                model_wrapper, prompt_text, device,
            )
            (metrics, decoded_text, n_decoded,
             start_hidden, start_log_probs, start_probs,
             end_hidden, end_log_probs,
             ppl_values) = \
                run_decode_agent_with_metrics(
                    model_wrapper, input_ids, attn_mask,
                    max_new_tokens=args.max_new_tokens,
                    n_metric_steps=args.n_metric_steps,
                    past_kv=None,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            final_text = decoded_text
            ag_type = "decode"
            ag_info = {
                "name": "Baseline", "role": "baseline",
                "type": ag_type, "exec_idx": 0,
                "decoded_tokens": n_decoded, **metrics,
            }
            agent_records.append(ag_info)
            # Baseline: no boundary metrics — single agent, explicit skip

            sampled_ppl = _sample_perplexity_steps(ppl_values, max_points=80)
            perplexity_rows.append({
                "case_idx":          case_idx,
                "exec_idx":          0,
                "agent_role":        "baseline",
                "agent_type":        "decode",
                "perplexity_values": sampled_ppl,
            })

            for mk in METRIC_KEYS:
                data_rows.append({
                    "case_idx":   case_idx,
                    "metric":     mk,
                    "agent_role": "baseline",
                    "exec_idx":   0,
                    "agent_type": ag_type,
                    "values":     metrics[mk],
                })

        # ── Evaluate answer ──
        pred, gold, correct = evaluate_answer(final_text, item, args.task)
        if correct:
            n_correct += 1

        cases_meta.append({
            "case_idx":   case_idx,
            "prediction": pred,
            "gold":       str(gold),
            "correct":    correct,
        })

        # ── Write TXT (first max_txt_cases only) ──
        if case_idx < args.max_txt_cases:
            write_txt_case(
                txt_file, case_idx, question, agent_records,
                boundary_rows, final_text, pred, str(gold), correct,
            )

        torch.cuda.empty_cache()

    # ═════════════════════════════════════════════════════════════════
    # Deferred top-token analysis  (latent_mas only)
    # ═════════════════════════════════════════════════════════════════
    if args.method == "latent_mas" and all_cases_hidden_records:
        print(f"\n[Top-token analysis] Processing {len(all_cases_hidden_records)} cases ...")
        top_tokens_path = os.path.join(
            args.out_dir, f"{prefix}_latent_top_tokens.json",
        )
        batch_project_hidden_states(
            model_wrapper, all_cases_hidden_records,
            out_path=top_tokens_path,
        )

    # ── Summary ──
    total = len(dataset)
    accuracy = n_correct / total if total > 0 else 0.0

    write_txt_summary(txt_file, args, total, n_correct, accuracy)
    print(f"\n[Saved] {txt_path}")

    # ── Save JSON ──
    json_path = save_json_results(
        args, prefix, agents, cases_meta,
        data_rows, boundary_rows, perplexity_rows,
        total, n_correct, accuracy,
    )

    # ── Plots ──
    run_all_plots(
        data_rows, boundary_rows, perplexity_rows,
        args.out_dir, prefix, args.method,
        bin_size=args.bin_size,
    )

    print(f"\n[Result] Accuracy = {accuracy:.4f} ({n_correct}/{total})")
    print(f"[Done] All outputs saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
