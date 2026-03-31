"""
scripts.py
==========
Extracted visualization, JSON/TXT logging, and latent-state token probability
analysis utilities for analyze_latent_entropy.py.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Execution-index color palette
# ═══════════════════════════════════════════════════════════════════════

EXEC_IDX_COLORS = [
    "#1f77b4",   # exec_idx 0 — deep blue
    "#ff7f0e",   # exec_idx 1 — orange
    "#2ca02c",   # exec_idx 2 — green
    "#d62728",   # exec_idx 3 — red
    "#9467bd",   # exec_idx 4 — purple
    "#8c564b",   # exec_idx 5 — brown
    "#e377c2",   # exec_idx 6 — pink
    "#7f7f7f",   # exec_idx 7 — gray
]

METRIC_KEYS = ["entropy", "kl_divergence", "cosine_similarity"]


def get_exec_color(exec_idx: int) -> str:
    """Return a deterministic color for a given execution index."""
    return EXEC_IDX_COLORS[exec_idx % len(EXEC_IDX_COLORS)]


def _build_exec_label(row: Dict) -> str:
    """Create a legend label that shows both exec_idx and agent role."""
    return f"exec {row['exec_idx']}: {row['agent_role']} ({row['agent_type']})"


# ═══════════════════════════════════════════════════════════════════════
# File-name prefix helper
# ═══════════════════════════════════════════════════════════════════════

def make_prefix(task: str, model_name: str, method: str) -> str:
    short_model = model_name.split("/")[-1]
    return f"{task}_{short_model}_{method}"


# ═══════════════════════════════════════════════════════════════════════
# Plotting  (exec_idx-based coloring)
# ═══════════════════════════════════════════════════════════════════════

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

        bar_colors = []
        for trans in transitions:
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
# TXT case-study writer
# ═══════════════════════════════════════════════════════════════════════

def write_txt_case(
    txt_file,
    case_idx: int,
    question: str,
    agent_records: List[Dict],
    boundary_rows: List[Dict],
    final_text: str,
    pred: str,
    gold: str,
    correct: bool,
):
    """Write a single case entry to the TXT case-study file."""
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
            kl_str = f"KL={bkl:.6f}" if bkl is not None else "KL=N/A"
            cos_str = f"Cosine={bcos:.6f}" if bcos is not None else "Cosine=N/A"
            txt_file.write(f"  {b['transition']}: {kl_str}, {cos_str}\n")
        txt_file.write("\n")

    txt_file.write(f"[Response]\n{final_text}\n\n")
    txt_file.write(f"[Prediction] {pred}\n")
    txt_file.write(f"[Gold]       {gold}\n")
    txt_file.write(f"[Correct]    {correct}\n\n")
    txt_file.flush()


def write_txt_summary(
    txt_file,
    args,
    total: int,
    n_correct: int,
    accuracy: float,
):
    """Write the summary block at the end of the TXT file."""
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


# ═══════════════════════════════════════════════════════════════════════
# JSON result builder & saver
# ═══════════════════════════════════════════════════════════════════════

def build_step_semantics(args) -> Dict:
    """Build step_semantics dict for the JSON output."""
    if args.method == "latent_mas":
        return {
            "non_judger_step": "One latent recurrence iteration: hidden → realign → forward (with KV cache accumulation)",
            "judger_step": "One autoregressive decoded token",
            "comparability": (
                "Latent steps and decoded-token steps are NOT directly comparable. "
                "Latent steps involve a single embedding forward pass per step. "
                "Decoded-token steps involve sampling and producing actual text."
            ),
            "n_latent_steps_per_agent": args.latent_steps,
            "n_metric_steps_decode": args.n_metric_steps,
        }
    elif args.method == "text_mas":
        return {
            "all_agents_step": "One autoregressive decoded token",
            "kv_cache_sharing": False,
            "context_sharing": "Text context accumulates across agents",
            "n_metric_steps_decode": args.n_metric_steps,
        }
    else:
        return {
            "step": "One autoregressive decoded token",
            "n_metric_steps_decode": args.n_metric_steps,
        }


def save_json_results(
    args,
    prefix: str,
    agents,
    cases_meta: List[Dict],
    data_rows: List[Dict],
    boundary_rows: List[Dict],
    total: int,
    n_correct: int,
    accuracy: float,
) -> str:
    """Build and save the JSON results file. Returns the saved path."""
    step_semantics = build_step_semantics(args)

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
    return json_path


def run_all_plots(
    data_rows: List[Dict],
    boundary_rows: List[Dict],
    out_dir: str,
    prefix: str,
    method: str,
    bin_size: int = 5,
):
    """Run all standard plots (per-agent, concatenated, boundary)."""
    print("[Plotting] ...")
    for mk in METRIC_KEYS:
        plot_per_agent_metric(
            data_rows, mk, out_dir, prefix, method,
            bin_size=bin_size,
        )
        plot_concatenated_overview(
            data_rows, mk, out_dir, prefix, method,
        )
    plot_boundary_metrics(boundary_rows, out_dir, prefix)


# ═══════════════════════════════════════════════════════════════════════
# Top-token probabilities from latent hidden states
# (Gaussian ±1.5σ cumulative cutoff)
# ═══════════════════════════════════════════════════════════════════════
#
# For each latent step, project the last hidden state through lm_head to
# get the full vocabulary distribution, then collect top tokens in
# descending probability order until the cumulative probability exceeds
# the ±1.5σ threshold of a Gaussian (≈ 86.64%).
#
# Gaussian CDF:  P(|X| ≤ 1.5σ) = erf(1.5/√2) ≈ 0.8664
#
# Output format per step:
#   {"token_1": 0.35, "token_2": 0.22, ..., "cumulative": 0.87}
#
# This shows how "concentrated" or "diffuse" the model's belief is at
# each latent recurrence step — a complementary view to entropy.

GAUSSIAN_1_5_SIGMA_MASS = 0.8663855  # erf(1.5 / sqrt(2))


@torch.no_grad()
def get_top_tokens_at_hidden(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    tokenizer,
    sigma: float = 1.5,
) -> Dict:
    """Given a single hidden state [1, D] or [D], project through lm_head
    and return top tokens until cumulative probability reaches the
    Gaussian ±σ mass threshold.

    Args:
        hidden:    last hidden state, shape [1, D] or [D].
        lm_head:   the model's output projection layer.
        tokenizer: tokenizer for id → string conversion.
        sigma:     number of standard deviations for the Gaussian cutoff
                   (default 1.5 → ~86.64% mass).

    Returns:
        dict with:
          "tokens": {token_str: prob_float_2dp, ...}
          "cumulative": float  (total probability of listed tokens)
          "n_tokens": int
    """
    import scipy.special as sp

    threshold = float(sp.erf(sigma / (2 ** 0.5)))

    h = hidden.view(1, -1).to(lm_head.weight.dtype)
    logits = lm_head(h).float().squeeze(0)          # [V]
    probs = torch.softmax(logits, dim=-1)            # [V]

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    sorted_probs = sorted_probs.cpu()
    sorted_indices = sorted_indices.cpu()

    cumsum = 0.0
    token_probs = {}
    for i in range(sorted_probs.shape[0]):
        p = sorted_probs[i].item()
        tok_id = sorted_indices[i].item()
        tok_str = tokenizer.decode([tok_id])
        token_probs[tok_str] = round(p, 2)
        cumsum += p
        if cumsum >= threshold:
            break

    return {
        "tokens": token_probs,
        "cumulative": round(cumsum, 4),
        "n_tokens": len(token_probs),
        "sigma": sigma,
        "threshold": round(threshold, 4),
    }


@torch.no_grad()
def collect_latent_top_tokens(
    model_wrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_steps: int,
    past_kv=None,
    sigma: float = 1.5,
) -> List[Dict]:
    """Run latent recurrence for one agent and collect top-token probability
    lists at each latent step (including step-0 after prompt prefill).

    Returns:
        List of dicts, one per step (length = latent_steps + 1).
        Each dict has keys: tokens, cumulative, n_tokens, sigma, threshold.
    """
    from models import _past_length

    model = model_wrapper.model
    device = model_wrapper.device
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Cannot locate lm_head / output embeddings.")

    # Extend attention mask for existing KV cache
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

    results = []

    # Step-0: after prompt prefill
    step0 = get_top_tokens_at_hidden(
        last_hidden, lm_head, model_wrapper.tokenizer, sigma=sigma,
    )
    step0["step"] = 0
    results.append(step0)

    # Latent recurrence
    for s in range(latent_steps):
        latent_vec = model_wrapper._apply_latent_realignment(last_hidden, model)
        latent_embed = latent_vec.unsqueeze(1)

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

        step_info = get_top_tokens_at_hidden(
            last_hidden, lm_head, model_wrapper.tokenizer, sigma=sigma,
        )
        step_info["step"] = s + 1
        results.append(step_info)

    return results, past


def format_top_tokens_table(step_results: List[Dict]) -> str:
    """Format top-token results into a readable string table.

    Args:
        step_results: list from collect_latent_top_tokens().

    Returns:
        Multi-line string suitable for logging / printing.
    """
    lines = []
    lines.append(f"Top-token probabilities (Gaussian ±{step_results[0].get('sigma', 1.5)}σ "
                 f"cutoff ≈ {step_results[0].get('threshold', 0.8664) * 100:.2f}%)")
    lines.append("=" * 72)

    for entry in step_results:
        step = entry["step"]
        n = entry["n_tokens"]
        cum = entry["cumulative"]
        tok_str = ", ".join(
            f"{tok}: {prob:.2f}" for tok, prob in entry["tokens"].items()
        )
        lines.append(f"  Step {step:>3d}  ({n:>4d} tokens, cum={cum:.4f}):  {tok_str}")

    lines.append("=" * 72)
    return "\n".join(lines)


def save_top_tokens_json(
    step_results: List[Dict],
    out_path: str,
    metadata: Optional[Dict] = None,
):
    """Save top-token results to a JSON file.

    Args:
        step_results: list from collect_latent_top_tokens().
        out_path:     output file path.
        metadata:     optional dict of extra info (case_idx, agent, etc.).
    """
    output = {
        "metadata": metadata or {},
        "steps": step_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_path}")
