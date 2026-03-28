"""
analyze_latent_entropy.py
=========================
Measures per-step entropy & perplexity of the last-layer hidden state
during LatentMAS latent recurrence (steps 0 .. latent_steps).

Place this file in the LatentMAS repo root (next to run.py) and run:

  CUDA_VISIBLE_DEVICES=3 python analyze_latent_entropy.py \
      --model_name Qwen/Qwen3-4B \
      --task aime2024 \
      --latent_steps 80 \
      --max_samples -1 \
      --prompt sequential

Outputs (saved to example_logs/):
  - latent_entropy_results.json
  - entropy_scatter.png   / entropy_line.png
  - perplexity_scatter.png / perplexity_line.png
"""

import argparse
import json
import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from models import ModelWrapper, _past_length
from data import (
    load_aime2024,
    load_aime2025,
    load_gsm8k,
    load_gpqa_diamond,
    load_arc_easy,
    load_arc_challenge,
    load_mbppplus,
    load_humanevalplus,
    load_medqa,
)
from utils import set_seed, auto_device

try:
    from prompts import build_agent_message_sequential_latent_mas
    _HAS_PROMPTS = True
except ImportError:
    _HAS_PROMPTS = False


# ── Dataset loader dispatch ────────────────────────────────────────────────

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


# ── Metric helpers ──────────────────────────────────────────────────────────

@torch.no_grad()
def compute_step_metrics(
    hidden_state: torch.Tensor,
    lm_head: torch.nn.Module,
) -> Tuple[List[float], List[float]]:
    """Entropy & perplexity from a last-layer hidden state via lm_head."""
    logits = lm_head(hidden_state.to(lm_head.weight.dtype))  # [B, V]
    log_probs = torch.log_softmax(logits.float(), dim=-1)     # [B, V]
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)                # [B]
    perplexity = entropy.exp()                                # [B]
    return entropy.cpu().tolist(), perplexity.cpu().tolist()


# ── Core analysis loop ──────────────────────────────────────────────────────

@torch.no_grad()
def run_latent_steps_with_metrics(
    model_wrapper: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_steps: int = 80,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Run latent recurrence for *latent_steps*, recording metrics per step.

    Returns:
        (all_entropies, all_perplexities)
        Each list has length (latent_steps + 1).
        Index 0 = initial hidden state (before any recurrence).
    """
    model = model_wrapper.model
    device = model_wrapper.device

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Cannot locate lm_head / output embeddings.")

    # ── Step 0: initial forward pass ──
    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]

    all_entropies: List[List[float]] = []
    all_perplexities: List[List[float]] = []

    ent, ppl = compute_step_metrics(last_hidden, lm_head)
    all_entropies.append(ent)
    all_perplexities.append(ppl)

    # ── Latent recurrence: steps 1 .. latent_steps ──
    for step in range(latent_steps):
        source_model = model
        latent_vec = model_wrapper._apply_latent_realignment(last_hidden, source_model)
        latent_embed = latent_vec.unsqueeze(1)  # [B, 1, D]

        past_len = _past_length(past)
        latent_mask = torch.ones(
            (latent_embed.shape[0], past_len + 1),
            dtype=torch.long,
            device=device,
        )

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

        ent, ppl = compute_step_metrics(last_hidden, lm_head)
        all_entropies.append(ent)
        all_perplexities.append(ppl)

    return all_entropies, all_perplexities


# ── Prompt builder ──────────────────────────────────────────────────────────

def build_prompt_messages(question: str, role: str, args) -> List[Dict]:
    if _HAS_PROMPTS:
        return build_agent_message_sequential_latent_mas(
            role=role,
            question=question,
            context="",
            method="latent_mas",
            args=args,
        )
    return [
        {"role": "system", "content": "You are a helpful math assistant. Think step by step."},
        {"role": "user", "content": question},
    ]


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_metrics(
    results: Dict,
    metric_key: str,
    out_dir: str,
    bin_size: int = 5,
    latent_steps: int = 80,
):
    dataset_key = list(results.keys())[0]
    cases = results[dataset_key]

    n_points = latent_steps + 1
    steps = np.arange(n_points)

    all_traces = []
    for case_key in sorted(cases.keys(), key=lambda k: int(k.replace("case", ""))):
        vals = cases[case_key][metric_key]
        all_traces.append(vals[:n_points])
    all_traces = np.array(all_traces)

    # ── 1. Scatter plot ──
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, trace in enumerate(all_traces):
        ax.scatter(steps, trace, s=8, alpha=0.45, label=f"case{i}" if i < 10 else None)
    ax.set_xlabel("Latent Step")
    ax.set_ylabel(metric_key.capitalize())
    ax.set_title(f"Per-case {metric_key} across latent steps (scatter)")
    if len(all_traces) <= 10:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    scatter_path = os.path.join(out_dir, f"{metric_key}_scatter.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {scatter_path}")

    # ── 2. Binned line plot (mean +/- std) ──
    n_bins = math.ceil(n_points / bin_size)
    bin_centers, bin_means, bin_stds = [], [], []
    for b in range(n_bins):
        s = b * bin_size
        e = min(s + bin_size, n_points)
        chunk = all_traces[:, s:e]
        bin_centers.append((s + e - 1) / 2)
        bin_means.append(chunk.mean())
        bin_stds.append(chunk.std())

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(bin_centers, bin_means, "o-", color="tab:blue", linewidth=2, markersize=5, label="mean")
    ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                     alpha=0.2, color="tab:blue", label="±1 std")
    ax.set_xlabel(f"Latent Step (bin size={bin_size})")
    ax.set_ylabel(metric_key.capitalize())
    ax.set_title(f"Average {metric_key} across latent steps (line, bin={bin_size})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    line_path = os.path.join(out_dir, f"{metric_key}_line.png")
    fig.savefig(line_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {line_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measure hidden-state entropy/perplexity during LatentMAS latent recurrence."
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--task", type=str, default="aime2024",
                        choices=list(TASK_LOADERS.keys()),
                        help="Dataset/task to evaluate.")
    parser.add_argument("--latent_steps", type=int, default=80,
                        help="Number of latent recurrence steps (0..latent_steps)")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max dataset cases (-1 = all)")
    parser.add_argument("--prompt", type=str, default="sequential",
                        choices=["sequential", "hierarchical"])
    parser.add_argument("--agent_role", type=str, default="planner",
                        help="Agent role prompt for initial forward pass")
    parser.add_argument("--device", type=str, default="cuda",
                        help="CUDA device, e.g. 'cuda', 'cuda:0', 'cuda:3'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--bin_size", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="example_logs",
                        help="Output directory for JSON and plots")
    # Args needed by ModelWrapper / prompt builders
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--method", type=str, default="latent_mas")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--text_mas_context_length", type=int, default=-1)

    args = parser.parse_args()

    set_seed(args.seed)
    device = auto_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Config] model={args.model_name}, latent_steps={args.latent_steps}, "
          f"task={args.task}, realign={args.latent_space_realign}, device={device}")

    # ── Load model (HF backend only) ──
    model_wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)

    # ── Load dataset ──
    loader_fn = TASK_LOADERS[args.task]
    dataset = list(loader_fn())
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
    print(f"[Data] {args.task}: {len(dataset)} cases")

    # ── Run analysis ──
    results: Dict = {}
    for case_idx, item in enumerate(tqdm(dataset, desc="Analyzing")):
        question = item["question"]

        messages = build_prompt_messages(question, role=args.agent_role, args=args)
        prompt_text = model_wrapper.render_chat(messages, add_generation_prompt=True)

        if args.think:
            prompt_text = f"{prompt_text}<think>"

        encoded = model_wrapper.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        ent_steps, ppl_steps = run_latent_steps_with_metrics(
            model_wrapper, input_ids, attention_mask, latent_steps=args.latent_steps,
        )

        # B=1, so flatten
        results[f"case{case_idx}"] = {
            "question": question,
            "entropy": [e[0] for e in ent_steps],
            "perplexity": [p[0] for p in ppl_steps],
        }

        torch.cuda.empty_cache()

    # ── Wrap & save ──
    output = {args.task: results}

    json_path = os.path.join(args.out_dir, "latent_entropy_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {json_path}")

    print("[Plotting] ...")
    plot_metrics(output, "entropy", args.out_dir,
                 bin_size=args.bin_size, latent_steps=args.latent_steps)
    plot_metrics(output, "perplexity", args.out_dir,
                 bin_size=args.bin_size, latent_steps=args.latent_steps)

    print("\n[Done] All outputs saved to:", args.out_dir)


if __name__ == "__main__":
    main()
