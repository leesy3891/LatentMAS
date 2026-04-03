"""
scripts.py
==========
Visualization, JSON/TXT logging, and latent-state token probability
analysis utilities for analyze_latent_entropy.py.

Refactored metric taxonomy
--------------------------
- **Confidence metrics** (decision-time):
    normalized_entropy ∈ [0, 1]
    js_divergence      ∈ [0, 1]
- **Stability / drift metrics** (post-update for decode, latent_step for latent):
    js_divergence           ∈ [0, 1]
    cosine_similarity       ∈ [-1, 1]
    angular_distance        ∈ [0, 1]
- **Boundary metrics** (inter-agent transfer):
    boundary_js_divergence  ∈ [0, 1]
    boundary_cosine_similarity ∈ [-1, 1]
    boundary_angular_distance  ∈ [0, 1]
- **Top-token analysis**:
    Fixed top-5 tokens per step, Jaccard overlap across adjacent steps.
- **Perplexity**:
    Per-step perplexity = exp(entropy), plotted for error propagation analysis.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

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

# ── New metric key set ────────────────────────────────────────────────
# These are the metrics plotted per-exec and concatenated.
# They replace the old ["entropy", "kl_divergence", "cosine_similarity"].
METRIC_KEYS = [
    "normalized_entropy",
    "js_divergence",
    "cosine_similarity",
    "angular_distance",
]

# Y-axis bounds per metric for consistent plotting
METRIC_YLIM: Dict[str, Optional[Tuple[float, float]]] = {
    "normalized_entropy": (0.0, 1.0),
    "js_divergence":      (0.0, 1.0),
    "cosine_similarity":  (-1.0, 1.0),
    "angular_distance":   (0.0, 1.0),
}


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

    ylim = METRIC_YLIM.get(metric_key)

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

        if ylim is not None:
            ax.set_ylim(ylim)

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
    ylim = METRIC_YLIM.get(metric_key)

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
    if ylim is not None:
        ax.set_ylim(ylim)

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
    """Bar chart of inter-agent boundary JS divergence, cosine similarity,
    and angular distance.

    Shows mean ± std across cases for each agent transition.
    """
    if not boundary_rows:
        return

    transitions = sorted(set(r["transition"] for r in boundary_rows))
    if not transitions:
        return

    boundary_metric_keys = [
        "boundary_js_divergence",
        "boundary_cosine_similarity",
        "boundary_angular_distance",
    ]

    for bm_key in boundary_metric_keys:
        fig, ax = plt.subplots(figsize=(10, 5))

        x_pos = np.arange(len(transitions))
        means, stds = [], []

        for trans in transitions:
            vals = [r[bm_key] for r in boundary_rows
                    if r["transition"] == trans and r.get(bm_key) is not None]
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
        bm_label = bm_key.replace("_", " ").title()
        ax.set_ylabel(bm_label)
        ax.set_title(f"[{prefix}]  Inter-Agent {bm_label}")
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        p = os.path.join(out_dir, f"{prefix}_{bm_key}_boundaries.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════
# Perplexity plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_perplexity(
    perplexity_data: List[Dict],
    out_dir: str,
    prefix: str,
    method: str,
):
    """Plot aggregated perplexity curves (mean ± std across cases).

    For latent_mas: x-axis = latent step index (per agent exec_idx),
                    concatenated across agents.
    For text_mas:   x-axis = concatenated uniform-step axis (sampled
                    from each agent's decoded tokens).

    Interpretation rule (documented in output):
        A monotonically increasing or consistently upward-trending
        perplexity curve MUST be interpreted as evidence of error
        propagation.  A flat or decreasing curve indicates stable
        inference.
    """
    if not perplexity_data:
        return

    exec_idxs = sorted(set(r["exec_idx"] for r in perplexity_data))

    offset = 0
    fig, ax = plt.subplots(figsize=(16, 5))

    for eidx in exec_idxs:
        color = get_exec_color(eidx)

        traces = []
        sample_role = ""
        sample_type = ""
        for r in perplexity_data:
            if r["exec_idx"] == eidx:
                vals = r["perplexity_values"]
                traces.append(vals)
                sample_role = r["agent_role"]
                sample_type = r["agent_type"]

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

    step_label = ("Latent Step Index" if method == "latent_mas"
                  else "Concatenated Decoded Step Index")
    ax.set_xlabel(step_label)
    ax.set_ylabel("Perplexity")
    ax.set_title(f"[{prefix}]  Perplexity  — concatenated (by exec order)\n"
                 f"(Upward trend → error propagation; flat/decreasing → stable)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_perplexity_concat.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════
# Master plot runner
# ═══════════════════════════════════════════════════════════════════════

def run_all_plots(
    data_rows: List[Dict],
    boundary_rows: List[Dict],
    perplexity_data: List[Dict],
    out_dir: str,
    prefix: str,
    method: str,
    bin_size: int = 5,
):
    """Run all standard plots (per-agent, concatenated, boundary, perplexity)."""
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
    plot_perplexity(perplexity_data, out_dir, prefix, method)


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
    """Write a single case entry to the TXT case-study file.

    Uses the new metric names:
      normalized_entropy, js_divergence, cosine_similarity, angular_distance.
    Perplexity is NOT included in TXT case summaries (per spec).
    """
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

        # Normalized entropy
        ent_vals = ag.get("normalized_entropy", [])
        if ent_vals:
            valid = [v for v in ent_vals if v is not None]
            if valid:
                txt_file.write(f"  Norm. entropy: first={valid[0]:.4f}, "
                               f"last={valid[-1]:.4f}\n")

        # JS divergence
        js_vals = [v for v in ag.get("js_divergence", []) if v is not None]
        if js_vals:
            txt_file.write(f"  JS div:  mean={np.mean(js_vals):.6f}, "
                           f"max={np.max(js_vals):.6f}\n")

        # Cosine similarity
        cos_vals = [v for v in ag.get("cosine_similarity", []) if v is not None]
        if cos_vals:
            txt_file.write(f"  Cosine:  mean={np.mean(cos_vals):.6f}, "
                           f"min={np.min(cos_vals):.6f}\n")

        # Angular distance
        ang_vals = [v for v in ag.get("angular_distance", []) if v is not None]
        if ang_vals:
            txt_file.write(f"  Angular: mean={np.mean(ang_vals):.6f}, "
                           f"max={np.max(ang_vals):.6f}\n")
        txt_file.write("\n")

    # Write boundary metrics for this case
    case_boundaries = [b for b in boundary_rows if b["case_idx"] == case_idx]
    if case_boundaries:
        txt_file.write("  --- Inter-Agent Boundary Metrics ---\n")
        for b in case_boundaries:
            bjs  = b.get("boundary_js_divergence")
            bcos = b.get("boundary_cosine_similarity")
            bang = b.get("boundary_angular_distance")
            js_s  = f"JS={bjs:.6f}"   if bjs  is not None else "JS=N/A"
            cos_s = f"Cos={bcos:.6f}"  if bcos is not None else "Cos=N/A"
            ang_s = f"Ang={bang:.6f}"  if bang is not None else "Ang=N/A"
            btype = b.get("boundary_type", "?")
            txt_file.write(f"  {b['transition']} [{btype}]: {js_s}, {cos_s}, {ang_s}\n")
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
            "non_judger_step": (
                "One latent recurrence iteration: hidden → realign → forward "
                "(with KV cache accumulation)"
            ),
            "judger_step": "One autoregressive decoded token",
            "comparability": (
                "Latent steps and decoded-token steps are NOT directly comparable. "
                "Latent steps involve a single embedding forward pass per step. "
                "Decoded-token steps involve sampling and producing actual text."
            ),
            "metric_phases": {
                "latent_agents": (
                    "Metrics are recorded under 'latent_step' category per recurrence "
                    "step.  This includes confidence (normalized_entropy, js_divergence) "
                    "and hidden drift (cosine_similarity, angular_distance).  There is "
                    "no token sampling stage so no decision_time/post_update split."
                ),
                "judger_agent": (
                    "Metrics are split into 'decision_time' (before sampling) and "
                    "'post_update' (after feeding sampled token).  decision_time captures "
                    "confidence; post_update captures state drift."
                ),
            },
            "n_latent_steps_per_agent": args.latent_steps,
            "n_metric_steps_decode": args.n_metric_steps,
        }
    elif args.method == "text_mas":
        return {
            "all_agents_step": "One autoregressive decoded token",
            "kv_cache_sharing": False,
            "context_sharing": "Text context accumulates across agents",
            "metric_phases": (
                "Metrics are split into 'decision_time' (before sampling) and "
                "'post_update' (after feeding sampled token).  decision_time captures "
                "confidence; post_update captures state drift."
            ),
            "n_metric_steps_decode": args.n_metric_steps,
        }
    else:
        return {
            "step": "One autoregressive decoded token",
            "metric_phases": (
                "Metrics are split into 'decision_time' (before sampling) and "
                "'post_update' (after feeding sampled token)."
            ),
            "n_metric_steps_decode": args.n_metric_steps,
        }


def save_json_results(
    args,
    prefix: str,
    agents,
    cases_meta: List[Dict],
    data_rows: List[Dict],
    boundary_rows: List[Dict],
    perplexity_data: List[Dict],
    total: int,
    n_correct: int,
    accuracy: float,
) -> str:
    """Build and save the JSON results file. Returns the saved path.

    Uses '_stability_results.json' suffix (replacing old '_entropy_results').
    Includes perplexity data and interpretation rule.
    """
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
        "perplexity_interpretation": (
            "A monotonically increasing or consistently upward-trending "
            "perplexity curve MUST be interpreted as evidence of error "
            "propagation.  A flat or decreasing perplexity curve indicates "
            "stable inference."
        ),
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
        "perplexity_data": perplexity_data,
    }
    if args.method == "latent_mas":
        output["config"]["latent_steps"] = args.latent_steps
        output["config"]["latent_space_realign"] = args.latent_space_realign

    # Renamed from _entropy_results to _stability_results
    json_path = os.path.join(args.out_dir, f"{prefix}_stability_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {json_path}")
    return json_path


# ═══════════════════════════════════════════════════════════════════════
# Top-5 token probabilities from stored latent hidden states
# (fixed top-5 extraction with Jaccard overlap analysis)
# ═══════════════════════════════════════════════════════════════════════
#
# Hidden states are stored on CPU during inference, then batch-projected
# through lm_head after all cases are done.  This avoids interleaving
# lm_head calls inside the latent recurrence loop.
#
# For each hidden state, we project through lm_head → softmax, then
# extract exactly the top 5 tokens.
#
# Jaccard overlap: |top5_t ∩ top5_{t-1}| / |top5_t ∪ top5_{t-1}|
# computed over token IDs between adjacent steps within each agent.

TOP_K = 5


@torch.no_grad()
def batch_project_hidden_states(
    model_wrapper,
    all_cases_hidden_records: list,
    out_path: str,
    gpu_batch_size: int = 128,
    max_txt_cases: int = 30,
):
    """Batch-project all stored hidden states through lm_head and save
    top-5 token probability lists plus Jaccard overlap to JSON + TXT.

    Args:
        model_wrapper:  ModelWrapper with .model and .tokenizer.
        all_cases_hidden_records:  list of dicts built during the main loop:
            [{ "case_idx": int,
               "agents": [{ "agent_name": str, "agent_role": str,
                            "exec_idx": int,
                            "hiddens": [tensor [1,D] on CPU, ...] }, ...] }, ...]
        out_path:         JSON output path (TXT uses same stem + _top_tokens.txt).
        gpu_batch_size:   how many hiddens to project through lm_head at once.
        max_txt_cases:    how many cases to write to the TXT file.

    Output JSON structure:
        { "top_k": 5,
          "cases": [{ "case_idx": 0,
                      "agents": [{ "agent_name": "Planner", ...,
                                   "steps": [{ "step": 0,
                                               "tokens": [{"token": "The",
                                                           "token_id": 123,
                                                           "probability": 0.35,
                                                           "rank": 0}, ...],
                                               "jaccard_overlap": null }, ...] }, ...] }, ...] }
    """
    if not all_cases_hidden_records:
        return

    model = model_wrapper.model
    device = next(model.parameters()).device
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Cannot locate lm_head / output embeddings.")
    tokenizer = model_wrapper.tokenizer

    # ── 1. Flatten all hidden states into [total_N, D] ──
    flat_hiddens = []
    index_map = []  # (case_list_idx, agent_list_idx, step_idx)

    for ci, case_rec in enumerate(all_cases_hidden_records):
        for ai, agent_rec in enumerate(case_rec["agents"]):
            for si, h in enumerate(agent_rec["hiddens"]):
                flat_hiddens.append(h.view(-1))  # [D]
                index_map.append((ci, ai, si))

    total_N = len(flat_hiddens)
    if total_N == 0:
        return

    flat_tensor = torch.stack(flat_hiddens, dim=0)  # [total_N, D]
    print(f"[Top-Token] Projecting {total_N} hidden states through lm_head "
          f"(fixed top-{TOP_K}) ...")

    # ── 2. Batched lm_head projection ──
    all_sorted_probs = []
    all_sorted_indices = []

    for start in range(0, total_N, gpu_batch_size):
        end = min(start + gpu_batch_size, total_N)
        chunk = flat_tensor[start:end].to(device=device, dtype=lm_head.weight.dtype)
        logits = lm_head(chunk).float()                # [B, V]
        probs = torch.softmax(logits, dim=-1)          # [B, V]
        sp, si = torch.sort(probs, descending=True, dim=-1)
        # Only keep top-K to save memory
        all_sorted_probs.append(sp[:, :TOP_K].cpu())
        all_sorted_indices.append(si[:, :TOP_K].cpu())
        del chunk, logits, probs, sp, si
        torch.cuda.empty_cache()

    all_sorted_probs = torch.cat(all_sorted_probs, dim=0)     # [total_N, K]
    all_sorted_indices = torch.cat(all_sorted_indices, dim=0)  # [total_N, K]

    # ── 3. Extract top-5 tokens per hidden state ──
    print(f"[Top-Token] Extracting top-{TOP_K} tokens for {total_N} states ...")

    # Pre-build output structure
    output_cases = []
    for case_rec in all_cases_hidden_records:
        case_out = {"case_idx": case_rec["case_idx"], "agents": []}
        for agent_rec in case_rec["agents"]:
            case_out["agents"].append({
                "agent_name": agent_rec["agent_name"],
                "agent_role": agent_rec["agent_role"],
                "exec_idx":   agent_rec["exec_idx"],
                "steps":      [None] * len(agent_rec["hiddens"]),
            })
        output_cases.append(case_out)

    for flat_idx, (ci, ai, si_idx) in enumerate(index_map):
        sorted_p = all_sorted_probs[flat_idx]     # [K]
        sorted_i = all_sorted_indices[flat_idx]    # [K]

        tokens_list = []
        for rank in range(TOP_K):
            tok_id = sorted_i[rank].item()
            tok_str = tokenizer.decode([tok_id])
            prob = sorted_p[rank].item()
            tokens_list.append({
                "token": tok_str,
                "token_id": tok_id,
                "probability": round(prob, 6),
                "rank": rank,
            })

        output_cases[ci]["agents"][ai]["steps"][si_idx] = {
            "step":       si_idx,
            "tokens":     tokens_list,
            "jaccard_overlap": None,  # filled in next pass
        }

    # ── 4. Compute Jaccard overlap between adjacent steps ──
    for case_out in output_cases:
        for agent_out in case_out["agents"]:
            steps = agent_out["steps"]
            for s_idx in range(len(steps)):
                if steps[s_idx] is None:
                    continue
                if s_idx == 0:
                    steps[s_idx]["jaccard_overlap"] = None
                    continue
                if steps[s_idx - 1] is None:
                    steps[s_idx]["jaccard_overlap"] = None
                    continue

                prev_ids = set(t["token_id"] for t in steps[s_idx - 1]["tokens"])
                curr_ids = set(t["token_id"] for t in steps[s_idx]["tokens"])
                union = prev_ids | curr_ids
                intersection = prev_ids & curr_ids
                jaccard = len(intersection) / len(union) if union else 0.0
                steps[s_idx]["jaccard_overlap"] = round(jaccard, 4)

    # ── 5. Save JSON ──
    result = {
        "description": (
            f"Top-{TOP_K} token probabilities at each hidden state. "
            f"Jaccard overlap measures step-to-step top-{TOP_K} token set "
            f"stability (1.0 = identical set, 0.0 = fully disjoint).  "
            f"High Jaccard with low cosine similarity may indicate hidden-space "
            f"movement beneath similar lexical projections."
        ),
        "top_k": TOP_K,
        "cases": output_cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_path}")
    print(f"  → {len(output_cases)} cases, top-{TOP_K}")

    # ── 6. Save TXT (human-readable) ──
    txt_path = out_path.replace(".json", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Latent hidden-state top-{TOP_K} token probabilities\n")
        f.write(f"With step-to-step Jaccard overlap analysis\n")
        f.write("=" * 80 + "\n\n")

        for case_rec in output_cases[:max_txt_cases]:
            case_idx = case_rec["case_idx"]
            f.write(f"{'─' * 60}\nCase #{case_idx}\n{'─' * 60}\n")

            for agent_rec in case_rec["agents"]:
                name = agent_rec["agent_name"]
                role = agent_rec["agent_role"]
                eidx = agent_rec["exec_idx"]
                f.write(f"\n  Agent: {name} ({role}, exec_idx={eidx})\n")

                for step_info in agent_rec["steps"]:
                    if step_info is None:
                        continue
                    step = step_info["step"]
                    jac = step_info["jaccard_overlap"]
                    jac_str = f"J={jac:.4f}" if jac is not None else "J=N/A  "
                    tok_str = ", ".join(
                        f"'{t['token']}'({t['probability']:.4f})"
                        for t in step_info["tokens"]
                    )
                    f.write(f"    Step {step:>3d}  [{jac_str}]:  {tok_str}\n")
            f.write("\n")

        if len(output_cases) > max_txt_cases:
            f.write(f"\n(Showing first {max_txt_cases} of {len(output_cases)} "
                    f"cases; see JSON for all)\n")

    print(f"[Saved] {txt_path}")
