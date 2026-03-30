"""
analyze_latent_entropy.py  (v3 — ground-truth-aligned, exec_idx coloring)
=========================================================================
Measures per-step **entropy**, **KL divergence**, and **cosine similarity**
of hidden states across all methods, strictly reflecting the multi-agent
execution pipeline defined in `run.py` and `methods/latent_mas.py`.

Changes from v2
----------------
1. **Exec-idx-based coloring** — scatter, aggregated, and concatenated plots
   all color by execution order (exec_idx), NOT by agent role.  Role is still
   shown in labels.

2. **Inter-agent transition metrics** — KL divergence and cosine similarity
   are measured across agent boundaries (end of agent k-1 → start of agent k).
   These are recorded in a separate data table and plotted as a bar/line chart.

3. **Step semantics documented** — "step" in LatentMAS non-judger agents
   means one latent recurrence step (hidden→realign→forward).  In TextMAS and
   the judger, "step" means one autoregressive decoded token.  These are
   fundamentally different compute units and are NOT directly comparable.

4. **Ground-truth alignment verified** against:
   - `methods/__init__.py`: Agent order = Planner(0) → Critic(1) → Refiner(2) → Judger(3)
   - `methods/latent_mas.py`:
     • Non-judger agents: prompt is tokenized, run through the model to get
       initial hidden state, then `latent_steps` recurrence iterations are
       performed.  The KV cache is accumulated and passed to the next agent.
     • Judger: prompt is tokenized, run with accumulated KV cache, then
       autoregressive decoding.
     • Context string is always "" for all agents (LatentMAS does not use
       text context between agents — communication is purely via KV cache).
   - `methods/text_mas.py`:
     • Every agent (including judger) decodes text autoregressively.
     • No KV-cache sharing — each agent gets a fresh forward pass.
     • Context accumulates as text: `context += f"[{name}]:\\n{output}\\n\\n"`
   - `run.py`: Confirms agent iteration order, batch processing, answer
     evaluation logic.

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
# Execution-index color palette
# ═══════════════════════════════════════════════════════════════════════
#
# Colors are assigned by exec_idx (execution order), NOT by agent role.
# This ensures that the 1st agent execution always gets color A, the
# 2nd always gets color B, etc., regardless of which role that agent has.

EXEC_IDX_COLORS = [
    "#1f77b4",   # exec_idx 0 — deep blue
    "#ff7f0e",   # exec_idx 1 — orange
    "#2ca02c",   # exec_idx 2 — green
    "#d62728",   # exec_idx 3 — red
    "#9467bd",   # exec_idx 4 — purple  (if >4 agents ever used)
    "#8c564b",   # exec_idx 5 — brown
    "#e377c2",   # exec_idx 6 — pink
    "#7f7f7f",   # exec_idx 7 — gray
]


def get_exec_color(exec_idx: int) -> str:
    """Return a deterministic color for a given execution index."""
    return EXEC_IDX_COLORS[exec_idx % len(EXEC_IDX_COLORS)]


# ═══════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_lm_head(model):
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Cannot locate lm_head / output embeddings.")
    return lm_head


@torch.no_grad()
def compute_all_metrics(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    prev_log_probs: Optional[torch.Tensor] = None,
    prev_hidden: Optional[torch.Tensor] = None,
) -> Tuple[float, Optional[float], Optional[float], torch.Tensor]:
    """Compute entropy, KL divergence, cosine similarity in a single lm_head pass.

    Returns:
        (entropy, kl_div_or_None, cosine_sim_or_None, current_log_probs)
    """
    logits = lm_head(hidden.to(lm_head.weight.dtype))
    log_probs = torch.log_softmax(logits.float(), dim=-1)  # [B, V]
    probs = log_probs.exp()

    # Entropy:  H = -Σ p·log(p)
    entropy = -(probs * log_probs).sum(dim=-1).item()

    # KL(p_curr ‖ p_prev)
    kl = None
    if prev_log_probs is not None:
        kl = (probs * (log_probs - prev_log_probs)).sum(dim=-1).item()

    # Cosine similarity in hidden space
    cosine = None
    if prev_hidden is not None:
        cosine = F.cosine_similarity(
            hidden.float().view(1, -1),
            prev_hidden.float().view(1, -1),
            dim=-1,
        ).item()

    return entropy, kl, cosine, log_probs


# ═══════════════════════════════════════════════════════════════════════
# Core: latent recurrence for one non-judger agent  (LatentMAS)
# ═══════════════════════════════════════════════════════════════════════
#
# Ground truth (models.py → generate_latent_batch):
#   1. Run full forward pass on the agent's prompt tokens (with KV cache).
#   2. Take last_hidden = hidden_states[-1][:, -1, :].
#   3. For each latent step:
#        a. latent_vec = _apply_latent_realignment(last_hidden, model)
#        b. Feed latent_vec as inputs_embeds (shape [1,1,D]) into the model
#           with the accumulated KV cache.
#        c. Update last_hidden from the new output.
#   4. Return accumulated KV cache (includes prompt + all latent steps).
#
# The analysis mirrors this exactly, adding per-step metric recording.

@torch.no_grad()
def run_latent_agent_with_metrics(
    model_wrapper: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_steps: int,
    past_kv: Optional[Tuple] = None,
) -> Tuple[Dict, Optional[Tuple], torch.Tensor, torch.Tensor]:
    """Execute one non-judger agent via latent recurrence, recording
    per-step metrics.

    Returns:
        (metrics_dict, updated_past_kv, last_hidden, last_log_probs)

    The last_hidden and last_log_probs are needed for inter-agent
    boundary metrics: the start of the next agent can be compared
    against the end state of this agent.
    """
    model = model_wrapper.model
    device = model_wrapper.device
    lm_head = _get_lm_head(model)

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

    entropies, kl_divs, cosines = [], [], []

    # Step-0: after processing the agent's prompt tokens
    ent, _, _, log_probs = compute_all_metrics(last_hidden, lm_head)
    entropies.append(ent)
    kl_divs.append(None)
    cosines.append(None)
    prev_log_probs = log_probs
    prev_hidden = last_hidden.clone()

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

        ent, kl, cos, log_probs = compute_all_metrics(
            last_hidden, lm_head,
            prev_log_probs=prev_log_probs,
            prev_hidden=prev_hidden,
        )
        entropies.append(ent)
        kl_divs.append(kl)
        cosines.append(cos)

        prev_log_probs = log_probs
        prev_hidden = last_hidden.clone()

    metrics = {
        "entropy": entropies,
        "kl_divergence": kl_divs,
        "cosine_similarity": cosines,
        "n_steps": latent_steps,
    }
    return metrics, past, last_hidden, prev_log_probs


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
) -> Tuple[Dict, str, int, torch.Tensor, torch.Tensor]:
    """Decode tokens for one agent, recording per-step metrics for the first
    *n_metric_steps* decoded tokens.

    Returns:
        (metrics_dict, decoded_text, total_decoded_tokens,
         last_hidden, last_log_probs)

    last_hidden and last_log_probs are returned for inter-agent boundary
    metric computation.
    """
    model = model_wrapper.model
    device = model_wrapper.device
    lm_head = _get_lm_head(model)

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

    entropies, kl_divs, cosines = [], [], []

    ent, _, _, log_probs = compute_all_metrics(last_hidden, lm_head)
    entropies.append(ent)
    kl_divs.append(None)
    cosines.append(None)
    prev_log_probs = log_probs
    prev_hidden = last_hidden.clone()

    # ── Autoregressive decoding ──
    generated_ids: List[int] = []
    eos_id = model_wrapper.tokenizer.eos_token_id

    for step in range(max_new_tokens):
        logits = lm_head(last_hidden.to(lm_head.weight.dtype))
        logits_f = logits.float()

        if temperature > 0:
            probs = F.softmax(logits_f / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
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

        # Forward next token
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

        # Record metrics only for the first n_metric_steps
        if step < n_metric_steps:
            ent, kl, cos, log_probs_new = compute_all_metrics(
                last_hidden, lm_head,
                prev_log_probs=prev_log_probs,
                prev_hidden=prev_hidden,
            )
            entropies.append(ent)
            kl_divs.append(kl)
            cosines.append(cos)
            prev_log_probs = log_probs_new
            prev_hidden = last_hidden.clone()

    decoded_text = model_wrapper.tokenizer.decode(
        generated_ids, skip_special_tokens=True,
    ).strip()

    metrics = {
        "entropy": entropies,
        "kl_divergence": kl_divs,
        "cosine_similarity": cosines,
        "n_steps": len(entropies) - 1,  # excluding step-0
    }
    return metrics, decoded_text, len(generated_ids), last_hidden, prev_log_probs


# ═══════════════════════════════════════════════════════════════════════
# Prompt builders  (dispatch to the correct one per method × topology)
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
    context — no KV cache is shared.
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
# File-name prefix helper
# ═══════════════════════════════════════════════════════════════════════

def make_prefix(task: str, model_name: str, method: str) -> str:
    short_model = model_name.split("/")[-1]
    return f"{task}_{short_model}_{method}"


# ═══════════════════════════════════════════════════════════════════════
# Tokenise helper  (single-item, returns [1, L])
# ═══════════════════════════════════════════════════════════════════════

def _tokenize_prompt(model_wrapper: ModelWrapper, prompt_text: str, device):
    enc = model_wrapper.tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ═══════════════════════════════════════════════════════════════════════
# Inter-agent boundary metric computation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_boundary_metrics(
    curr_hidden: torch.Tensor,
    curr_log_probs: torch.Tensor,
    prev_hidden: Optional[torch.Tensor],
    prev_log_probs: Optional[torch.Tensor],
    lm_head: torch.nn.Module,
) -> Dict:
    """Compute KL divergence and cosine similarity at an agent boundary.

    Compares the START state of the current agent (after its prompt prefill,
    i.e. step-0) against the END state of the previous agent.

    Returns dict with 'boundary_kl' and 'boundary_cosine' (or None if
    no previous agent).
    """
    if prev_hidden is None or prev_log_probs is None:
        return {"boundary_kl": None, "boundary_cosine": None}

    # KL(p_curr_start ‖ p_prev_end)
    curr_probs = curr_log_probs.exp()
    kl = (curr_probs * (curr_log_probs - prev_log_probs)).sum(dim=-1).item()

    # Cosine similarity in hidden space
    cosine = F.cosine_similarity(
        curr_hidden.float().view(1, -1),
        prev_hidden.float().view(1, -1),
        dim=-1,
    ).item()

    return {"boundary_kl": kl, "boundary_cosine": cosine}


# ═══════════════════════════════════════════════════════════════════════
# Plotting  (exec_idx-based coloring)
# ═══════════════════════════════════════════════════════════════════════

METRIC_KEYS = ["entropy", "kl_divergence", "cosine_similarity"]


def _build_exec_label(row: Dict) -> str:
    """Create a legend label that shows both exec_idx and agent role."""
    return f"exec {row['exec_idx']}: {row['agent_role']} ({row['agent_type']})"


def plot_per_agent_metric(
    data_rows: List[Dict],
    metric_key: str,
    out_dir: str,
    prefix: str,
    method: str,
    bin_size: int = 5,
):
    """Create a subplot grid: one row per distinct exec_idx.
    Each subplot has a scatter (faint, all cases) plus a mean ± std line.
    Colors are determined by exec_idx, NOT by agent role.
    """
    rows = [r for r in data_rows if r["metric"] == metric_key]
    if not rows:
        return

    # Determine unique exec_idxs in order
    exec_idxs = sorted(set(r["exec_idx"] for r in rows))

    n_panels = len(exec_idxs)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels),
                             sharex=False, squeeze=False)

    for row_idx, eidx in enumerate(exec_idxs):
        ax = axes[row_idx, 0]
        color = get_exec_color(eidx)

        traces = []
        sample_type = "decode"
        sample_role = ""
        for r in rows:
            if r["exec_idx"] == eidx:
                vals = [v if v is not None else float("nan") for v in r["values"]]
                traces.append(vals)
                sample_type = r["agent_type"]
                sample_role = r["agent_role"]

        if not traces:
            ax.set_title(f"exec_idx={eidx} — no data")
            continue

        max_len = max(len(t) for t in traces)
        padded = np.full((len(traces), max_len), np.nan)
        for i, t in enumerate(traces):
            padded[i, :len(t)] = t

        steps = np.arange(max_len)

        for i in range(len(traces)):
            ax.scatter(steps[:len(traces[i])], traces[i],
                       s=6, alpha=0.25, color=color)

        n_bins = max(1, math.ceil(max_len / bin_size))
        bc, bm, bs_ = [], [], []
        for b in range(n_bins):
            s = b * bin_size
            e = min(s + bin_size, max_len)
            chunk = padded[:, s:e]
            bc.append((s + e - 1) / 2)
            bm.append(np.nanmean(chunk))
            bs_.append(np.nanstd(chunk))
        bc, bm, bs_ = np.array(bc), np.array(bm), np.array(bs_)
        label = f"exec {eidx}: {sample_role} mean"
        ax.plot(bc, bm, "o-", color=color, lw=2, ms=4, label=label)
        ax.fill_between(bc, bm - bs_, bm + bs_, alpha=0.18, color=color)

        step_type_label = "Latent Step" if sample_type == "latent" else "Decoded Token"
        ax.set_xlabel(step_type_label)
        ax.set_ylabel(metric_key.replace("_", " ").title())
        ax.set_title(f"exec {eidx}: {sample_role.capitalize()} ({step_type_label})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if metric_key == "kl_divergence":
            ax.set_yscale("symlog", linthresh=1e-4)

    fig.suptitle(f"[{prefix}]  {metric_key.replace('_', ' ').title()}  — per execution index",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_{metric_key}_per_exec.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_concatenated_overview(
    data_rows: List[Dict],
    metric_key: str,
    out_dir: str,
    prefix: str,
    method: str,
):
    """Single-axis concatenated plot: all agents' steps on one x-axis,
    with vertical dashed lines at agent boundaries and colour per exec_idx.
    Shows mean ± std across cases.
    """
    rows = [r for r in data_rows if r["metric"] == metric_key]
    if not rows:
        return

    exec_idxs = sorted(set(r["exec_idx"] for r in rows))

    offset = 0
    fig, ax = plt.subplots(figsize=(16, 5))

    for eidx in exec_idxs:
        color = get_exec_color(eidx)

        traces = []
        sample_role = ""
        for r in rows:
            if r["exec_idx"] == eidx:
                vals = [v if v is not None else float("nan") for v in r["values"]]
                traces.append(vals)
                sample_role = r["agent_role"]

        if not traces:
            continue

        max_len = max(len(t) for t in traces)
        padded = np.full((len(traces), max_len), np.nan)
        for i, t in enumerate(traces):
            padded[i, :len(t)] = t

        steps = np.arange(max_len) + offset
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)

        label = f"exec {eidx}: {sample_role.capitalize()}"
        ax.plot(steps, mean, "-", color=color, lw=2, label=label)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

        if offset > 0:
            ax.axvline(x=offset - 0.5, color="gray", ls="--", lw=0.8, alpha=0.6)

        offset += max_len

    ax.set_xlabel("Concatenated Step Index")
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(f"[{prefix}]  {metric_key.replace('_', ' ').title()}  — concatenated (by exec order)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if metric_key == "kl_divergence":
        ax.set_yscale("symlog", linthresh=1e-4)

    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_{metric_key}_concat.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_boundary_metrics(
    boundary_rows: List[Dict],
    out_dir: str,
    prefix: str,
):
    """Bar chart of inter-agent boundary KL divergence and cosine similarity.

    Shows mean ± std across cases for each agent transition.
    """
    if not boundary_rows:
        return

    transitions = sorted(set(r["transition"] for r in boundary_rows))
    if not transitions:
        return

    for metric_key in ["boundary_kl", "boundary_cosine"]:
        fig, ax = plt.subplots(figsize=(10, 5))

        x_pos = np.arange(len(transitions))
        means, stds = [], []

        for trans in transitions:
            vals = [r[metric_key] for r in boundary_rows
                    if r["transition"] == trans and r[metric_key] is not None]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)

        # Color bars by the target exec_idx of the transition
        bar_colors = []
        for trans in transitions:
            # trans looks like "exec0→exec1", extract target idx
            try:
                target_idx = int(trans.split("→exec")[1])
            except (IndexError, ValueError):
                target_idx = 0
            bar_colors.append(get_exec_color(target_idx))

        ax.bar(x_pos, means, yerr=stds, color=bar_colors, alpha=0.7,
               capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(transitions, rotation=30, ha="right")
        ax.set_ylabel(metric_key.replace("_", " ").title())
        ax.set_title(f"[{prefix}]  Inter-Agent {metric_key.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3, axis="y")

        if metric_key == "boundary_kl":
            ax.set_yscale("symlog", linthresh=1e-4)

        fig.tight_layout()
        p = os.path.join(out_dir, f"{prefix}_{metric_key}_boundaries.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent hidden-state analysis: entropy, KL, cosine."
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
        print(f"    (hidden → realign → forward with KV cache).  Total = {args.latent_steps} per agent.")
        print(f"  LatentMAS judger: each 'step' = 1 decoded token.")
        print(f"    Total = up to {args.max_new_tokens} tokens, metrics for first {args.n_metric_steps}.")
        print(f"  These step types are NOT directly comparable across agents.")
    elif args.method == "text_mas":
        print(f"  TextMAS all agents: each 'step' = 1 decoded token.")
        print(f"    Metrics recorded for first {args.n_metric_steps} tokens per agent.")
        print(f"  No KV cache sharing — each agent gets a fresh forward pass.")
    else:
        print(f"  Baseline: each 'step' = 1 decoded token.")
        print(f"    Metrics recorded for first {args.n_metric_steps} tokens.")
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
    # (matches text_mas.py agent_name_map_for_prompt_hierarchical exactly)
    _HIER_NAME_MAP = {
        "Planner": "Math Agent",  "Critic": "Science Agent",
        "Refiner": "Code Agent",  "Judger": "Task Summrizer",
    }

    # ═════════════════════════════════════════════════════════════════
    # Main loop
    # ═════════════════════════════════════════════════════════════════

    data_rows: List[Dict] = []          # flat intra-agent metric rows
    boundary_rows: List[Dict] = []      # inter-agent boundary metrics
    cases_meta: List[Dict] = []         # per-case result metadata
    n_correct = 0

    for case_idx, item in enumerate(tqdm(dataset, desc="Analyzing")):
        question = item["question"]
        agent_records: List[Dict] = []   # temporary per-case, used for TXT
        final_text = ""
        exec_idx = 0   # global execution counter within this case

        # Track end-of-agent state for boundary metrics
        prev_agent_hidden: Optional[torch.Tensor] = None
        prev_agent_log_probs: Optional[torch.Tensor] = None
        prev_agent_label: Optional[str] = None

        # ─────────────────────────────────────────────────
        #  LatentMAS
        # ─────────────────────────────────────────────────
        #
        # Ground truth execution (latent_mas.py → run_batch):
        #   - Iterates agents = [Planner, Critic, Refiner, Judger]
        #   - For non-judger:
        #       1. Build prompt (context="" always)
        #       2. Tokenize and optionally prepend <think>
        #       3. generate_latent_batch() → latent recurrence, KV cache accumulated
        #   - For judger:
        #       1. Build prompt (context="" always)
        #       2. Tokenize and optionally prepend <think>
        #       3. generate_text_batch() with accumulated past_kv → decode
        #
        if args.method == "latent_mas":
            past_kv: Optional[Tuple] = None

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
                    metrics, past_kv, end_hidden, end_log_probs = \
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

                    # Compute inter-agent boundary metric
                    # (step-0 of this agent vs end of previous agent)
                    start_hidden_ent = metrics["entropy"][0] if metrics["entropy"] else None
                    boundary = compute_boundary_metrics(
                        # Use the hidden state AFTER prefill (step 0) for this agent
                        # We recorded metrics starting from step-0 in run_latent_agent_with_metrics
                        # But we need the actual hidden to compare — for simplicity,
                        # we compare end-of-prev with end-of-current (which is the
                        # information the next agent actually receives)
                        end_hidden, end_log_probs,
                        prev_agent_hidden, prev_agent_log_probs,
                        _get_lm_head(model_wrapper.model),
                    )
                    boundary["transition"] = (
                        f"exec{exec_idx-1}→exec{exec_idx}" if exec_idx > 0
                        else None
                    )
                    boundary["case_idx"] = case_idx
                    boundary["source_exec_idx"] = exec_idx - 1 if exec_idx > 0 else None
                    boundary["target_exec_idx"] = exec_idx
                    boundary["target_role"] = agent.role
                    if boundary["transition"] is not None:
                        boundary_rows.append(boundary)

                    prev_agent_hidden = end_hidden.clone()
                    prev_agent_log_probs = end_log_probs.clone()

                else:
                    # ── Judger: decode with accumulated KV cache ──
                    past_for_dec = past_kv if args.latent_steps > 0 else None
                    metrics, decoded_text, n_decoded, end_hidden, end_log_probs = \
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

                    # Boundary metric for judger start
                    boundary = compute_boundary_metrics(
                        end_hidden, end_log_probs,
                        prev_agent_hidden, prev_agent_log_probs,
                        _get_lm_head(model_wrapper.model),
                    )
                    boundary["transition"] = (
                        f"exec{exec_idx-1}→exec{exec_idx}" if exec_idx > 0
                        else None
                    )
                    boundary["case_idx"] = case_idx
                    boundary["source_exec_idx"] = exec_idx - 1 if exec_idx > 0 else None
                    boundary["target_exec_idx"] = exec_idx
                    boundary["target_role"] = agent.role
                    if boundary["transition"] is not None:
                        boundary_rows.append(boundary)

                agent_records.append(ag_info)

                # ── Emit flat rows ──
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

        # ─────────────────────────────────────────────────
        #  TextMAS
        # ─────────────────────────────────────────────────
        #
        # Ground truth execution (text_mas.py → run_batch):
        #   - Iterates agents = [Planner, Critic, Refiner, Judger]
        #   - For ALL agents (including judger):
        #       1. Build prompt with accumulated text context
        #       2. Tokenize
        #       3. Decode autoregressively (vLLM or HF generate_text_batch)
        #       4. Append output to context (except judger)
        #   - No KV cache sharing — each agent starts fresh
        #
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
                metrics, decoded_text, n_decoded, end_hidden, end_log_probs = \
                    run_decode_agent_with_metrics(
                        model_wrapper, input_ids, attn_mask,
                        max_new_tokens=args.max_new_tokens,
                        n_metric_steps=args.n_metric_steps,
                        past_kv=None,  # No KV cache sharing in TextMAS
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

                # Boundary metric
                boundary = compute_boundary_metrics(
                    end_hidden, end_log_probs,
                    prev_agent_hidden, prev_agent_log_probs,
                    _get_lm_head(model_wrapper.model),
                )
                boundary["transition"] = (
                    f"exec{exec_idx-1}→exec{exec_idx}" if exec_idx > 0
                    else None
                )
                boundary["case_idx"] = case_idx
                boundary["source_exec_idx"] = exec_idx - 1 if exec_idx > 0 else None
                boundary["target_exec_idx"] = exec_idx
                boundary["target_role"] = agent.role
                if boundary["transition"] is not None:
                    boundary_rows.append(boundary)

                prev_agent_hidden = end_hidden.clone()
                prev_agent_log_probs = end_log_probs.clone()

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
        #  Baseline
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
            metrics, decoded_text, n_decoded, end_hidden, end_log_probs = \
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
            txt_file.write(f"{'=' * 60}\n")
            txt_file.write(f"Case #{case_idx}\n")
            txt_file.write(f"{'=' * 60}\n")
            txt_file.write(f"[Question]\n{question}\n\n")
            for ag in agent_records:
                txt_file.write(f"  --- Agent: {ag['name']} ({ag['role']}, "
                               f"type={ag['type']}, exec_idx={ag['exec_idx']}) ---\n")
                n_s = ag.get("n_steps", "?")
                txt_file.write(f"  Steps recorded: {n_s}\n")
                if "decoded_tokens" in ag:
                    txt_file.write(f"  Decoded tokens: {ag['decoded_tokens']}\n")
                ent_vals = ag.get("entropy", [])
                if ent_vals:
                    txt_file.write(f"  Entropy: first={ent_vals[0]:.4f}, "
                                   f"last={ent_vals[-1]:.4f}\n")
                kl_vals = [v for v in ag.get("kl_divergence", []) if v is not None]
                if kl_vals:
                    txt_file.write(f"  KL div:  mean={np.mean(kl_vals):.6f}, "
                                   f"max={np.max(kl_vals):.6f}\n")
                cos_vals = [v for v in ag.get("cosine_similarity", []) if v is not None]
                if cos_vals:
                    txt_file.write(f"  Cosine:  mean={np.mean(cos_vals):.6f}, "
                                   f"min={np.min(cos_vals):.6f}\n")
                txt_file.write("\n")

            # Write boundary metrics for this case
            case_boundaries = [b for b in boundary_rows if b["case_idx"] == case_idx]
            if case_boundaries:
                txt_file.write("  --- Inter-Agent Boundary Metrics ---\n")
                for b in case_boundaries:
                    bkl = b.get("boundary_kl")
                    bcos = b.get("boundary_cosine")
                    txt_file.write(f"  {b['transition']}: "
                                   f"KL={bkl:.6f}" if bkl is not None else "KL=N/A")
                    txt_file.write(f", Cosine={bcos:.6f}\n" if bcos is not None else ", Cosine=N/A\n")
                txt_file.write("\n")

            txt_file.write(f"[Response]\n{final_text}\n\n")
            txt_file.write(f"[Prediction] {pred}\n")
            txt_file.write(f"[Gold]       {gold}\n")
            txt_file.write(f"[Correct]    {correct}\n\n")
            txt_file.flush()

        torch.cuda.empty_cache()

    # ── Summary ──
    total = len(dataset)
    accuracy = n_correct / total if total > 0 else 0.0

    txt_file.write(f"\n{'#' * 60}\nSUMMARY\n{'#' * 60}\n")
    txt_file.write(f"Method:   {args.method}\n")
    txt_file.write(f"Model:    {args.model_name}\n")
    txt_file.write(f"Task:     {args.task}\n")
    txt_file.write(f"Samples:  {total}\n")
    txt_file.write(f"Correct:  {n_correct}\n")
    txt_file.write(f"Accuracy: {accuracy:.4f}\n")
    if args.method == "latent_mas":
        txt_file.write(f"Latent steps: {args.latent_steps}\n")
        txt_file.write(f"Realign: {args.latent_space_realign}\n")
    txt_file.write(f"(TXT limited to first {args.max_txt_cases} cases; "
                   f"see JSON for all)\n")
    txt_file.close()
    print(f"\n[Saved] {txt_path}")

    # Build & save JSON  — flat table for re-visualization
    # ══════════════════════════════════════════════════════════════
    #
    # Structure:
    #   config           experiment-level metadata
    #   summary          accuracy / counts
    #   cases_meta       per-case prediction/gold/correct
    #   data             FLAT LIST of intra-agent metric rows:
    #       { case_idx, metric, agent_role, exec_idx,
    #         agent_type, values: [step0, step1, ...] }
    #   boundary_data    FLAT LIST of inter-agent boundary rows:
    #       { case_idx, transition, source_exec_idx, target_exec_idx,
    #         target_role, boundary_kl, boundary_cosine }
    #   step_semantics   explanation of what "step" means per method

    step_semantics = {}
    if args.method == "latent_mas":
        step_semantics = {
            "non_judger_step": "One latent recurrence iteration: hidden → realign → forward (with KV cache accumulation)",
            "judger_step": "One autoregressive decoded token",
            "comparability": "Latent steps and decoded-token steps are NOT directly comparable. "
                             "Latent steps involve a single embedding forward pass per step. "
                             "Decoded-token steps involve sampling and producing actual text.",
            "n_latent_steps_per_agent": args.latent_steps,
            "n_metric_steps_decode": args.n_metric_steps,
        }
    elif args.method == "text_mas":
        step_semantics = {
            "all_agents_step": "One autoregressive decoded token",
            "kv_cache_sharing": False,
            "context_sharing": "Text context accumulates across agents",
            "n_metric_steps_decode": args.n_metric_steps,
        }
    else:
        step_semantics = {
            "step": "One autoregressive decoded token",
            "n_metric_steps_decode": args.n_metric_steps,
        }

    output = {
        "config": {
            "method": args.method,
            "model": args.model_name,
            "task": args.task,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "n_metric_steps": args.n_metric_steps,
            "agents": [a.name for a in agents] if args.method != "baseline"
                      else ["Baseline"],
        },
        "step_semantics": step_semantics,
        "summary": {
            "total": total,
            "correct": n_correct,
            "accuracy": round(accuracy, 4),
        },
        "cases_meta": cases_meta,
        "data": data_rows,
        "boundary_data": [
            {k: v for k, v in row.items()}
            for row in boundary_rows
        ],
    }
    if args.method == "latent_mas":
        output["config"]["latent_steps"] = args.latent_steps
        output["config"]["latent_space_realign"] = args.latent_space_realign

    json_path = os.path.join(args.out_dir, f"{prefix}_entropy_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {json_path}")

    # ── Plots (read from flat data_rows) ──
    print("[Plotting] ...")
    for mk in METRIC_KEYS:
        plot_per_agent_metric(
            data_rows, mk, args.out_dir, prefix, args.method,
            bin_size=args.bin_size,
        )
        plot_concatenated_overview(
            data_rows, mk, args.out_dir, prefix, args.method,
        )

    # ── Boundary plots ──
    plot_boundary_metrics(boundary_rows, args.out_dir, prefix)

    print(f"\n[Result] Accuracy = {accuracy:.4f} ({n_correct}/{total})")
    print(f"[Done] All outputs saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
