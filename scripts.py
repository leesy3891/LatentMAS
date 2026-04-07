"""
scripts.py  (v5)
=================
Visualization, JSON/TXT logging, and latent-state token probability
analysis utilities for analyze_latent_entropy.py.

Changes from v4 (scripts v3):
- Removed write_txt_case / write_txt_summary (case_study.txt disabled)
- batch_project_hidden_states: TXT only, no JSON; invisible tokens displayed as '\\n'
- normalized_entropy removed from JSON; raw_entropy + entropy_logV kept
- JSON config trimmed to vocab_size / hidden_size
- All JSON floats rounded to 6 decimal places
- Added: entropy vs JS scatter plots (Fig 3a style)
- Added: logit-lens overlap analysis (Fig 3b style)
- Added: PCA hidden-state vs embedding visualization (Fig 2 style)
"""

import csv
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

METRIC_KEYS = [
    "normalized_entropy",
    "js_divergence",
    "cosine_similarity",
    "angular_distance",
]

METRIC_YLIM: Dict[str, Optional[Tuple[float, float]]] = {
    "normalized_entropy": (0.0, 1.0),
    "js_divergence":      (0.0, 1.0),
    "cosine_similarity":  (-1.0, 1.0),
    "angular_distance":   (0.0, 1.0),
}


def get_exec_color(exec_idx: int) -> str:
    return EXEC_IDX_COLORS[exec_idx % len(EXEC_IDX_COLORS)]


def _build_exec_label(row: Dict) -> str:
    return f"exec {row['exec_idx']}: {row['agent_role']} ({row['agent_type']})"


# ═══════════════════════════════════════════════════════════════════════
# File-name prefix helper
# ═══════════════════════════════════════════════════════════════════════

def make_prefix(task: str, model_name: str, method: str) -> str:
    short_model = model_name.split("/")[-1]
    return f"{task}_{short_model}_{method}"


# ═══════════════════════════════════════════════════════════════════════
# Recursive float rounding for JSON serialization
# ═══════════════════════════════════════════════════════════════════════

def _round_floats(obj, ndigits=6):
    """Recursively round all floats in a nested structure to ndigits."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        rounded = [_round_floats(x, ndigits) for x in obj]
        return type(obj)(rounded) if isinstance(obj, tuple) else rounded
    return obj


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
# TXT case-study writer — DISABLED (kept as no-ops for import compat)
# ═══════════════════════════════════════════════════════════════════════

def write_txt_case(*args, **kwargs):
    """Disabled: case_study.txt no longer generated."""
    pass


def write_txt_summary(*args, **kwargs):
    """Disabled: case_study.txt no longer generated."""
    pass


# ═══════════════════════════════════════════════════════════════════════
# JSON result builder & saver
# ═══════════════════════════════════════════════════════════════════════

def build_step_semantics(args) -> Dict:
    if args.method == "latent_mas":
        return {
            "non_judger_step": (
                "One latent recurrence iteration: hidden → realign → forward "
                "(with KV cache accumulation)"
            ),
            "judger_step": "One autoregressive decoded token",
            "comparability": (
                "Latent steps and decoded-token steps are NOT directly comparable."
            ),
            "metric_phases": {
                "latent_agents": (
                    "Metrics recorded under 'latent_step' category per recurrence step."
                ),
                "judger_agent": (
                    "Metrics split into 'decision_time' and 'post_update'."
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
    perplexity_data: List[Dict],
    total: int,
    n_correct: int,
    accuracy: float,
    vocab_size: int = 0,
    hidden_size: int = 0,
) -> str:
    """Build and save JSON results.

    Changes from v3:
    - config contains only vocab_size and hidden_size (model-space metadata)
    - normalized_entropy removed from stored data
    - raw_entropy and entropy_logV kept
    - All floats rounded to 6 decimal places
    """
    step_semantics = build_step_semantics(args)

    output = {
        "config": {
            "method": args.method,
            "model": args.model_name,
            "task": args.task,
            "seed": args.seed,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
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
            "accuracy": round(accuracy, 6),
        },
        "cases_meta": cases_meta,
        # Remove normalized_entropy from JSON; keep raw_entropy and entropy_logV
        "data": [r for r in data_rows if r.get("metric") != "normalized_entropy"],
        "boundary_data": [
            {k: v for k, v in row.items()}
            for row in boundary_rows
        ],
        "perplexity_data": perplexity_data,
    }
    if args.method == "latent_mas":
        output["config"]["latent_steps"] = args.latent_steps
        output["config"]["latent_space_realign"] = args.latent_space_realign

    # Round all floats to 6 decimal places
    output = _round_floats(output, ndigits=6)

    json_path = os.path.join(args.out_dir, f"{prefix}_stability_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {json_path}")
    return json_path


# ═══════════════════════════════════════════════════════════════════════
# Top-5 token probabilities — TXT only (JSON removed)
# ═══════════════════════════════════════════════════════════════════════

TOP_K = 5


def _sanitize_token_display(tok_str: str) -> str:
    """Replace visually empty / invisible tokens with '\\n' for display."""
    if not tok_str or tok_str.isspace() or tok_str in ("\n", "\r", "\r\n", "\t"):
        return "\\n"
    # Check for common invisible unicode characters
    stripped = tok_str.strip()
    if not stripped:
        return "\\n"
    return tok_str


@torch.no_grad()
def batch_project_hidden_states(
    model_wrapper,
    all_cases_hidden_records: list,
    out_path: str,
    gpu_batch_size: int = 128,
    max_txt_cases: int = 30,
):
    """Batch-project all stored hidden states through lm_head and save
    top-5 token probability lists plus Jaccard overlap to TXT only.

    JSON output is no longer generated (removed per spec).
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

    # 1. Flatten all hidden states into [total_N, D]
    flat_hiddens = []
    index_map = []  # (case_list_idx, agent_list_idx, step_idx)

    for ci, case_rec in enumerate(all_cases_hidden_records):
        for ai, agent_rec in enumerate(case_rec["agents"]):
            for si, h in enumerate(agent_rec["hiddens"]):
                flat_hiddens.append(h.view(-1))
                index_map.append((ci, ai, si))

    total_N = len(flat_hiddens)
    if total_N == 0:
        return

    flat_tensor = torch.stack(flat_hiddens, dim=0)
    print(f"[Top-Token] Projecting {total_N} hidden states through lm_head "
          f"(fixed top-{TOP_K}) ...")

    # 2. Batched lm_head projection
    all_sorted_probs = []
    all_sorted_indices = []

    for start in range(0, total_N, gpu_batch_size):
        end = min(start + gpu_batch_size, total_N)
        chunk = flat_tensor[start:end].to(device=device, dtype=lm_head.weight.dtype)
        logits = lm_head(chunk).float()
        probs = torch.softmax(logits, dim=-1)
        sp, si = torch.sort(probs, descending=True, dim=-1)
        all_sorted_probs.append(sp[:, :TOP_K].cpu())
        all_sorted_indices.append(si[:, :TOP_K].cpu())
        del chunk, logits, probs, sp, si
        torch.cuda.empty_cache()

    all_sorted_probs = torch.cat(all_sorted_probs, dim=0)
    all_sorted_indices = torch.cat(all_sorted_indices, dim=0)

    # 3. Build output structure
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
        sorted_p = all_sorted_probs[flat_idx]
        sorted_i = all_sorted_indices[flat_idx]

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
            "jaccard_overlap": None,
        }

    # 4. Compute Jaccard overlap
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

    # 5. Save TXT only (JSON removed)
    # Use out_path stem but force .txt extension
    txt_path = out_path.replace(".json", ".txt") if out_path.endswith(".json") else out_path
    if not txt_path.endswith(".txt"):
        txt_path = txt_path + ".txt"

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
                    # Sanitize token display: replace invisible tokens with '\\n'
                    tok_str = ", ".join(
                        f"'{_sanitize_token_display(t['token'])}'({t['probability']:.4f})"
                        for t in step_info["tokens"]
                    )
                    f.write(f"    Step {step:>3d}  [{jac_str}]:  {tok_str}\n")
            f.write("\n")

        if len(output_cases) > max_txt_cases:
            f.write(f"\n(Showing first {max_txt_cases} of {len(output_cases)} "
                    f"cases)\n")

    print(f"[Saved] {txt_path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3(a): Entropy vs JS divergence (soft vs top-1, soft vs top-2)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """JS divergence between two probability distributions, normalized to [0,1]."""
    p = p.float().view(-1).clamp_min(1e-12)
    q = q.float().view(-1).clamp_min(1e-12)
    m = 0.5 * (p + q)
    m = m.clamp_min(1e-12)
    kl_pm = (p * (p.log() - m.log())).sum()
    kl_qm = (q * (q.log() - m.log())).sum()
    js = (0.5 * kl_pm + 0.5 * kl_qm) / math.log(2)
    return js.clamp(0.0, 1.0).item()


@torch.no_grad()
def run_entropy_js_analysis(
    model_wrapper,
    all_cases_hidden_records: list,
    out_dir: str,
    prefix: str,
    target_steps: Optional[List[int]] = None,
    gpu_batch_size: int = 64,
):
    """Figure-3(a)-style analysis: entropy vs JS divergence.

    For each selected latent step, computes:
    - P_soft: next-token distribution from the actual latent hidden state
    - P1: distribution from a forward pass using the top-1 token embedding
    - P2: distribution from a forward pass using the top-2 token embedding
    Then computes JS(P_soft, P1) and JS(P_soft, P2).

    Step selection: if target_steps is None, uses all steps.
    """
    if not all_cases_hidden_records:
        print("[Entropy-JS] No hidden records, skipping.")
        return

    model = model_wrapper.model
    device = next(model.parameters()).device
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Cannot locate lm_head.")
    tokenizer = model_wrapper.tokenizer
    input_embeds_layer = model.get_input_embeddings()
    vocab_size = lm_head.weight.shape[0]
    log_V = math.log(vocab_size) if vocab_size > 1 else 1.0

    csv_rows = []

    for case_rec in all_cases_hidden_records:
        case_idx = case_rec["case_idx"]
        for agent_rec in case_rec["agents"]:
            exec_idx = agent_rec["exec_idx"]
            hiddens = agent_rec["hiddens"]  # list of [1, D] tensors on CPU

            for step_idx, h in enumerate(hiddens):
                if target_steps is not None and step_idx not in target_steps:
                    continue

                h_gpu = h.to(device=device, dtype=lm_head.weight.dtype)

                # P_soft from the actual hidden state
                logits_soft = lm_head(h_gpu).float()
                probs_soft = torch.softmax(logits_soft, dim=-1).view(-1)

                # Raw and normalized entropy
                log_probs_soft = torch.log_softmax(logits_soft, dim=-1).view(-1)
                raw_ent = -(probs_soft * log_probs_soft).sum().item()
                norm_ent = raw_ent / log_V

                # Top-1 and top-2 tokens
                sorted_probs, sorted_idx = torch.sort(probs_soft, descending=True)
                top1_id = sorted_idx[0].item()
                top1_prob = sorted_probs[0].item()
                top1_text = _sanitize_token_display(tokenizer.decode([top1_id]))
                top2_id = sorted_idx[1].item()
                top2_prob = sorted_probs[1].item()
                top2_text = _sanitize_token_display(tokenizer.decode([top2_id]))

                # P1: forward pass with top-1 token embedding
                tok1_emb = input_embeds_layer(torch.tensor([[top1_id]], device=device))
                out1 = model(inputs_embeds=tok1_emb, output_hidden_states=True, return_dict=True)
                h1_last = out1.hidden_states[-1][:, -1, :]
                logits1 = lm_head(h1_last.to(lm_head.weight.dtype)).float()
                probs1 = torch.softmax(logits1, dim=-1).view(-1)

                # P2: forward pass with top-2 token embedding
                tok2_emb = input_embeds_layer(torch.tensor([[top2_id]], device=device))
                out2 = model(inputs_embeds=tok2_emb, output_hidden_states=True, return_dict=True)
                h2_last = out2.hidden_states[-1][:, -1, :]
                logits2 = lm_head(h2_last.to(lm_head.weight.dtype)).float()
                probs2 = torch.softmax(logits2, dim=-1).view(-1)

                js_soft_top1 = _js_divergence(probs_soft, probs1)
                js_soft_top2 = _js_divergence(probs_soft, probs2)

                csv_rows.append({
                    "case_idx": case_idx,
                    "exec_idx": exec_idx,
                    "step": step_idx,
                    "entropy_normalized": round(norm_ent, 6),
                    "entropy_raw": round(raw_ent, 6),
                    "js_soft_top1": round(js_soft_top1, 6),
                    "js_soft_top2": round(js_soft_top2, 6),
                    "top1_token_id": top1_id,
                    "top1_token_text": top1_text,
                    "top1_prob": round(top1_prob, 6),
                    "top2_token_id": top2_id,
                    "top2_token_text": top2_text,
                    "top2_prob": round(top2_prob, 6),
                })

                del h_gpu, logits_soft, probs_soft, tok1_emb, out1, logits1, probs1
                del tok2_emb, out2, logits2, probs2
                torch.cuda.empty_cache()

    if not csv_rows:
        print("[Entropy-JS] No data collected, skipping plots.")
        return

    # Save CSV
    csv_path = os.path.join(out_dir, f"{prefix}_entropy_js_analysis.csv")
    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[Saved] {csv_path}")

    # Plot: normalized entropy vs JS
    _plot_entropy_vs_js(
        csv_rows, "entropy_normalized", out_dir, prefix,
        suffix="normalized", xlabel="Normalized Entropy", ylim=None,
    )
    # Plot: raw entropy vs JS
    _plot_entropy_vs_js(
        csv_rows, "entropy_raw", out_dir, prefix,
        suffix="raw", xlabel="Raw Entropy (nats)", ylim=(0.2, 1.2),
    )


def _plot_entropy_vs_js(
    csv_rows: List[Dict],
    entropy_key: str,
    out_dir: str,
    prefix: str,
    suffix: str,
    xlabel: str,
    ylim: Optional[Tuple[float, float]] = None,
):
    """Scatter plot of entropy vs JS divergence for soft-vs-top1 and soft-vs-top2."""
    ent_vals = [r[entropy_key] for r in csv_rows]
    js1_vals = [r["js_soft_top1"] for r in csv_rows]
    js2_vals = [r["js_soft_top2"] for r in csv_rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ent_vals, js1_vals, s=12, alpha=0.5, label="1st_VS_Soft", color="#1f77b4")
    ax.scatter(ent_vals, js2_vals, s=12, alpha=0.5, label="2nd_VS_Soft", color="#ff7f0e")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("JS Divergence")
    ax.set_title(f"[{prefix}] Entropy vs JS Divergence ({suffix})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_entropy_vs_js_{suffix}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3(b): Logit-lens overlap across layers
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_logit_lens_overlap(
    model_wrapper,
    all_cases_hidden_records: list,
    out_dir: str,
    prefix: str,
    top_k_overlap: int = 10,
    target_steps: Optional[List[int]] = None,
):
    """Figure-3(b)-style analysis: logit-lens overlap across transformer layers.

    For selected latent steps, runs three forward paths:
    - soft: uses the actual latent hidden state
    - top-1 token: uses the embedding of the top-1 predicted token
    - top-2 token: uses the embedding of the top-2 predicted token

    At each transformer layer, applies logit lens (project intermediate hidden
    through lm_head) to get top-k tokens, then computes overlap proportion
    with the soft path's top-k.

    Logit lens implementation: For Qwen-style models, we access
    model.model.layers[i] sequentially. The logit lens projects each layer's
    output through the final lm_head without the final layernorm — this is the
    standard "logit lens" approach. If the model has a final layernorm
    (model.model.norm), we apply it before projecting for more accurate results.

    Step selection: if target_steps is None, uses steps [0, mid, last] per agent.
    """
    if not all_cases_hidden_records:
        print("[Logit-Lens] No hidden records, skipping.")
        return

    model = model_wrapper.model
    device = next(model.parameters()).device
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Cannot locate lm_head.")
    tokenizer = model_wrapper.tokenizer
    input_embeds_layer = model.get_input_embeddings()

    # Locate transformer layers and final norm
    # Supports: model.model.layers (Qwen, Llama, Mistral, etc.)
    inner_model = getattr(model, "model", None)
    if inner_model is None or not hasattr(inner_model, "layers"):
        print("[Logit-Lens] Cannot locate model.model.layers, skipping.")
        return
    layers = inner_model.layers
    n_layers = len(layers)
    final_norm = getattr(inner_model, "norm", None)

    csv_rows = []

    for case_rec in all_cases_hidden_records:
        case_idx = case_rec["case_idx"]
        for agent_rec in case_rec["agents"]:
            exec_idx = agent_rec["exec_idx"]
            hiddens = agent_rec["hiddens"]

            # Default step selection: first, middle, last
            if target_steps is not None:
                steps_to_analyze = [s for s in target_steps if s < len(hiddens)]
            else:
                n_h = len(hiddens)
                steps_to_analyze = sorted(set([0, n_h // 2, n_h - 1]))
                steps_to_analyze = [s for s in steps_to_analyze if s < n_h]

            for step_idx in steps_to_analyze:
                h = hiddens[step_idx].to(device=device, dtype=lm_head.weight.dtype)

                # Get top-1 and top-2 token IDs from the hidden state
                logits_h = lm_head(h).float().view(-1)
                probs_h = torch.softmax(logits_h, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs_h, descending=True)
                top1_id = sorted_idx[0].item()
                top2_id = sorted_idx[1].item()

                # Run 3 paths through all layers, collecting per-layer hidden states
                # soft path: use the hidden state directly as "input" to each layer
                # For logit-lens we need to do a single-token forward pass and capture
                # intermediate hidden states.

                # --- Soft path: full forward from the hidden embedding ---
                # We'll use the stored hidden state with output_hidden_states
                # Actually for logit lens, we need the intermediate states from a
                # full forward pass. The stored hidden is the LAST layer output.
                # We need to re-run the forward pass to get per-layer states.
                # Use the latent embedding (realigned hidden) as input.

                # For soft path: use inputs_embeds = hidden state unsqueezed
                soft_emb = h.unsqueeze(0)  # [1, 1, D]
                out_soft = model(inputs_embeds=soft_emb, output_hidden_states=True, return_dict=True)
                soft_layer_hiddens = out_soft.hidden_states  # tuple of (n_layers+1) x [1, 1, D]

                # top-1 token path
                tok1_emb = input_embeds_layer(torch.tensor([[top1_id]], device=device))
                out1 = model(inputs_embeds=tok1_emb, output_hidden_states=True, return_dict=True)
                top1_layer_hiddens = out1.hidden_states

                # top-2 token path
                tok2_emb = input_embeds_layer(torch.tensor([[top2_id]], device=device))
                out2 = model(inputs_embeds=tok2_emb, output_hidden_states=True, return_dict=True)
                top2_layer_hiddens = out2.hidden_states

                # For each layer, apply logit lens and compute overlap
                # hidden_states[0] = input embeddings, hidden_states[i] = output of layer i-1
                # So hidden_states[i+1] = output of layer i (0-indexed)
                for layer_idx in range(n_layers):
                    # Layer output is at index layer_idx + 1
                    hs_idx = layer_idx + 1

                    soft_h = soft_layer_hiddens[hs_idx][:, -1, :]
                    top1_h = top1_layer_hiddens[hs_idx][:, -1, :]
                    top2_h = top2_layer_hiddens[hs_idx][:, -1, :]

                    # Apply final norm if available (standard logit lens improvement)
                    if final_norm is not None:
                        soft_h = final_norm(soft_h)
                        top1_h = final_norm(top1_h)
                        top2_h = final_norm(top2_h)

                    # Project through lm_head
                    soft_logits = lm_head(soft_h.to(lm_head.weight.dtype)).float().view(-1)
                    top1_logits = lm_head(top1_h.to(lm_head.weight.dtype)).float().view(-1)
                    top2_logits = lm_head(top2_h.to(lm_head.weight.dtype)).float().view(-1)

                    # Get top-k token sets
                    soft_topk = set(torch.topk(soft_logits, top_k_overlap).indices.tolist())
                    top1_topk = set(torch.topk(top1_logits, top_k_overlap).indices.tolist())
                    top2_topk = set(torch.topk(top2_logits, top_k_overlap).indices.tolist())

                    overlap1 = len(soft_topk & top1_topk) / len(soft_topk) if soft_topk else 0.0
                    overlap2 = len(soft_topk & top2_topk) / len(soft_topk) if soft_topk else 0.0

                    csv_rows.append({
                        "case_idx": case_idx,
                        "exec_idx": exec_idx,
                        "step": step_idx,
                        "layer": layer_idx,
                        "overlap_top1": round(overlap1, 6),
                        "overlap_top2": round(overlap2, 6),
                    })

                del out_soft, out1, out2
                torch.cuda.empty_cache()

    if not csv_rows:
        print("[Logit-Lens] No data collected, skipping.")
        return

    # Save CSV
    csv_path = os.path.join(out_dir, f"{prefix}_logit_lens_overlap.csv")
    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[Saved] {csv_path}")

    # Plot: aggregate overlap across all cases/steps, mean ± std per layer
    layers_range = sorted(set(r["layer"] for r in csv_rows))
    overlap1_by_layer = {l: [] for l in layers_range}
    overlap2_by_layer = {l: [] for l in layers_range}
    for r in csv_rows:
        overlap1_by_layer[r["layer"]].append(r["overlap_top1"])
        overlap2_by_layer[r["layer"]].append(r["overlap_top2"])

    x = np.array(layers_range)
    mean1 = np.array([np.mean(overlap1_by_layer[l]) for l in layers_range])
    std1 = np.array([np.std(overlap1_by_layer[l]) for l in layers_range])
    mean2 = np.array([np.mean(overlap2_by_layer[l]) for l in layers_range])
    std2 = np.array([np.std(overlap2_by_layer[l]) for l in layers_range])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mean1, "o-", color="#1f77b4", lw=2, ms=3, label="1st token")
    ax.fill_between(x, mean1 - std1, mean1 + std1, alpha=0.15, color="#1f77b4")
    ax.plot(x, mean2, "s-", color="#ff7f0e", lw=2, ms=3, label="2nd token")
    ax.fill_between(x, mean2 - std2, mean2 + std2, alpha=0.15, color="#ff7f0e")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel(f"Top-{top_k_overlap} Overlap Proportion")
    ax.set_title(f"[{prefix}] Logit-Lens Overlap: Soft vs Discrete Tokens")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_logit_lens_overlap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════
# PCA: Hidden states vs token embeddings (Figure 2 style)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_pca_hidden_vs_embedding(
    model_wrapper,
    all_cases_hidden_records: list,
    out_dir: str,
    prefix: str,
    max_hidden_samples: int = 2000,
    max_embed_samples: int = 5000,
):
    """PCA visualization of hidden states vs token embedding vectors.

    Sampling strategy:
    - Hidden states: uniformly sample up to max_hidden_samples from all
      stored latent hidden states across all cases/agents/steps.
    - Token embeddings: uniformly sample up to max_embed_samples from the
      full vocabulary embedding matrix.

    Statistics reported on plot:
    - Hidden mean L2, Embedding mean L2
    - FID (Fréchet Inception Distance, using Gaussian assumption)
    - MMD² (RBF kernel, bandwidth = median heuristic)
    - Cosine similarity (mean of random pairs)
    """
    if not all_cases_hidden_records:
        print("[PCA] No hidden records, skipping.")
        return

    model = model_wrapper.model
    device = next(model.parameters()).device
    input_embeds_layer = model.get_input_embeddings()
    embed_weight = input_embeds_layer.weight.detach().float().cpu()  # [V, D]

    # Collect hidden states
    all_h = []
    for case_rec in all_cases_hidden_records:
        for agent_rec in case_rec["agents"]:
            for h in agent_rec["hiddens"]:
                all_h.append(h.view(-1).float())

    if len(all_h) == 0:
        print("[PCA] No hidden states, skipping.")
        return

    all_h = torch.stack(all_h, dim=0)  # [N_h, D]

    # Sample hidden states
    if all_h.shape[0] > max_hidden_samples:
        idx = torch.randperm(all_h.shape[0])[:max_hidden_samples]
        all_h = all_h[idx]

    # Sample token embeddings
    if embed_weight.shape[0] > max_embed_samples:
        idx = torch.randperm(embed_weight.shape[0])[:max_embed_samples]
        embed_sample = embed_weight[idx]
    else:
        embed_sample = embed_weight

    # Statistics
    h_l2 = all_h.norm(dim=1).mean().item()
    e_l2 = embed_sample.norm(dim=1).mean().item()

    # FID (Gaussian assumption)
    mu_h = all_h.mean(dim=0)
    mu_e = embed_sample.mean(dim=0)
    diff = mu_h - mu_e
    # Use diagonal covariance for efficiency
    var_h = all_h.var(dim=0)
    var_e = embed_sample.var(dim=0)
    fid = (diff ** 2).sum().item() + var_h.sum().item() + var_e.sum().item() - 2 * (var_h * var_e).sqrt().sum().item()

    # MMD² with RBF kernel (median heuristic for bandwidth)
    n_mmd_samples = min(500, all_h.shape[0], embed_sample.shape[0])
    h_sub = all_h[:n_mmd_samples]
    e_sub = embed_sample[:n_mmd_samples]
    # Compute pairwise distances for bandwidth selection
    dists_he = torch.cdist(h_sub, e_sub).view(-1)
    sigma = dists_he.median().item()
    if sigma < 1e-6:
        sigma = 1.0
    gamma = 1.0 / (2 * sigma ** 2)

    def rbf_kernel(x, y):
        d = torch.cdist(x, y)
        return torch.exp(-gamma * d ** 2)

    kxx = rbf_kernel(h_sub, h_sub).mean().item()
    kyy = rbf_kernel(e_sub, e_sub).mean().item()
    kxy = rbf_kernel(h_sub, e_sub).mean().item()
    mmd2 = kxx + kyy - 2 * kxy

    # Cosine similarity (random pairs)
    n_cos = min(1000, all_h.shape[0], embed_sample.shape[0])
    h_cos = all_h[:n_cos]
    e_cos = embed_sample[:n_cos]
    cos_sim = F.cosine_similarity(h_cos, e_cos, dim=1).mean().item()

    # PCA (2D) on combined data
    combined = torch.cat([all_h, embed_sample], dim=0)  # [N_h + N_e, D]
    combined_centered = combined - combined.mean(dim=0, keepdim=True)

    # Use SVD for PCA (more numerically stable for high-dim)
    # Limit to reasonable size
    if combined_centered.shape[0] > 10000:
        pca_idx = torch.randperm(combined_centered.shape[0])[:10000]
        pca_data = combined_centered[pca_idx]
    else:
        pca_data = combined_centered

    U, S, V = torch.svd_lowrank(pca_data, q=2)
    # Project all data
    proj = combined_centered @ V  # [N, 2]
    proj_h = proj[:all_h.shape[0]].numpy()
    proj_e = proj[all_h.shape[0]:].numpy()

    # Plot: global + zoomed
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Global view
    ax = axes[0]
    ax.scatter(proj_e[:, 0], proj_e[:, 1], s=4, alpha=0.3, label="Token Embeddings", color="#2ca02c")
    ax.scatter(proj_h[:, 0], proj_h[:, 1], s=4, alpha=0.3, label="Hidden States", color="#d62728")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Global View")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Zoomed view around token embeddings
    ax = axes[1]
    e_x_range = np.percentile(proj_e[:, 0], [5, 95])
    e_y_range = np.percentile(proj_e[:, 1], [5, 95])
    margin_x = (e_x_range[1] - e_x_range[0]) * 0.3
    margin_y = (e_y_range[1] - e_y_range[0]) * 0.3
    xlim = (e_x_range[0] - margin_x, e_x_range[1] + margin_x)
    ylim_zoom = (e_y_range[0] - margin_y, e_y_range[1] + margin_y)

    ax.scatter(proj_e[:, 0], proj_e[:, 1], s=4, alpha=0.3, label="Token Embeddings", color="#2ca02c")
    ax.scatter(proj_h[:, 0], proj_h[:, 1], s=4, alpha=0.3, label="Hidden States", color="#d62728")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim_zoom)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Zoomed (Token-Centric)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Add statistics text
    stats_text = (
        f"Hidden mean L2: {h_l2:.2f}\n"
        f"Embed mean L2: {e_l2:.2f}\n"
        f"FID: {fid:.2f}\n"
        f"MMD² (RBF): {mmd2:.6f}\n"
        f"Cosine sim: {cos_sim:.4f}"
    )
    fig.text(0.5, -0.02, stats_text, ha="center", fontsize=9,
             family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"[{prefix}] Hidden States vs Token Embeddings (PCA)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_hidden_vs_embedding_pca.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")
