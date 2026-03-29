"""
analyze_latent_entropy.py
=========================
Measures per-step entropy & perplexity of hidden states across all methods:

  - latent_mas : latent recurrence steps 0..latent_steps
  - baseline / text_mas : first 80 decoded tokens' hidden states

Also records full case-study logs, per-case predictions, and final accuracy.

Usage:
  # latent_mas
  CUDA_VISIBLE_DEVICES=3 python analyze_latent_entropy.py \
      --method latent_mas --model_name Qwen/Qwen3-4B --task aime2024 \
      --latent_steps 80 --prompt sequential --max_samples -1

  # baseline
  CUDA_VISIBLE_DEVICES=3 python analyze_latent_entropy.py \
      --method baseline --model_name Qwen/Qwen3-4B --task aime2024 \
      --prompt sequential --max_samples -1

  # text_mas
  CUDA_VISIBLE_DEVICES=3 python analyze_latent_entropy.py \
      --method text_mas --model_name Qwen/Qwen3-4B --task aime2024 \
      --prompt sequential --max_samples -1

Outputs (all in example_logs/, prefixed with {task}_{model}_{method}_):
  - {prefix}_entropy_results.json
  - {prefix}_entropy_scatter.png  /  {prefix}_entropy_line.png
  - {prefix}_perplexity_scatter.png  /  {prefix}_perplexity_line.png
  - {prefix}_case_study.txt
"""

import argparse
import json
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
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

try:
    from prompts import build_agent_message_sequential_latent_mas
    _HAS_PROMPTS = True
except ImportError:
    _HAS_PROMPTS = False


# ═══════════════════════════════════════════════════════════════════════════
# Dataset loaders
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_step_metrics(
    hidden_state: torch.Tensor,
    lm_head: torch.nn.Module,
) -> Tuple[List[float], List[float]]:
    """Entropy & perplexity from a hidden state via lm_head.
    hidden_state: [B, D]  →  returns (entropy[B], perplexity[B])
    """
    logits = lm_head(hidden_state.to(lm_head.weight.dtype))
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    perplexity = entropy.exp()
    return entropy.cpu().tolist(), perplexity.cpu().tolist()


def _get_lm_head(model):
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Cannot locate lm_head / output embeddings.")
    return lm_head


# ═══════════════════════════════════════════════════════════════════════════
# Core analysis: latent_mas mode
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_latent_steps_with_metrics(
    model_wrapper: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_steps: int = 80,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Latent recurrence for latent_steps, recording metrics per step.
    Returns (entropies, perplexities), each length (latent_steps + 1).
    Index 0 = initial hidden (before recurrence).
    """
    model = model_wrapper.model
    device = model_wrapper.device
    lm_head = _get_lm_head(model)

    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1, :]

    all_ent: List[List[float]] = []
    all_ppl: List[List[float]] = []

    ent, ppl = compute_step_metrics(last_hidden, lm_head)
    all_ent.append(ent)
    all_ppl.append(ppl)

    for _ in range(latent_steps):
        source_model = model
        latent_vec = model_wrapper._apply_latent_realignment(last_hidden, source_model)
        latent_embed = latent_vec.unsqueeze(1)

        past_len = _past_length(past)
        latent_mask = torch.ones(
            (latent_embed.shape[0], past_len + 1),
            dtype=torch.long, device=device,
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
        all_ent.append(ent)
        all_ppl.append(ppl)

    return all_ent, all_ppl


# ═══════════════════════════════════════════════════════════════════════════
# Core analysis: baseline / text_mas mode  (first N decoded tokens)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_decode_with_metrics(
    model_wrapper: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 2048,
    n_metric_steps: int = 80,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> Tuple[List[List[float]], List[List[float]], str, int]:
    """Decode tokens and record hidden-state metrics for the first
    n_metric_steps decoded tokens.

    Returns:
        (entropies, perplexities, decoded_text, total_decoded_tokens)
        entropies/perplexities: each length min(n_metric_steps+1, total+1)
            index 0 = hidden state at end of prompt (before any decoding)
    """
    model = model_wrapper.model
    device = model_wrapper.device
    lm_head = _get_lm_head(model)

    # Prefill
    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]

    all_ent: List[List[float]] = []
    all_ppl: List[List[float]] = []

    ent, ppl = compute_step_metrics(last_hidden, lm_head)
    all_ent.append(ent)
    all_ppl.append(ppl)

    # Autoregressive decoding
    generated_ids: List[int] = []
    eos_id = model_wrapper.tokenizer.eos_token_id

    for step in range(max_new_tokens):
        logits = lm_head(last_hidden.to(lm_head.weight.dtype))  # [B, V]
        logits_f = logits.float()

        # Sample next token
        if temperature > 0:
            probs = F.softmax(logits_f / temperature, dim=-1)
            # top-p filtering
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = sorted_probs.cumsum(dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
        else:
            next_token = logits_f.argmax(dim=-1, keepdim=True)

        next_token = next_token.squeeze(-1)  # [B]
        tok_id = next_token[0].item()
        generated_ids.append(tok_id)

        if tok_id == eos_id:
            break

        # Forward with next token
        next_input = next_token.unsqueeze(-1)  # [B, 1]
        past_len = _past_length(past)
        new_mask = torch.ones(
            (next_input.shape[0], past_len + 1),
            dtype=torch.long, device=device,
        )
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

        # Record metrics for the first n_metric_steps decoded tokens
        if step < n_metric_steps:
            ent, ppl = compute_step_metrics(last_hidden, lm_head)
            all_ent.append(ent)
            all_ppl.append(ppl)

    decoded_text = model_wrapper.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    total_decoded = len(generated_ids)

    return all_ent, all_ppl, decoded_text, total_decoded


# ═══════════════════════════════════════════════════════════════════════════
# Latent_mas: latent recurrence + final decode
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_latent_mas_full(
    model_wrapper: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_steps: int = 80,
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> Tuple[List[List[float]], List[List[float]], str, int, int]:
    """Latent recurrence + greedy/sampled decode for final answer.

    Returns:
        (entropies, perplexities, decoded_text, total_decoded_tokens, latent_steps_used)
    """
    model = model_wrapper.model
    device = model_wrapper.device
    lm_head = _get_lm_head(model)

    # ── Latent recurrence with metrics ──
    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1, :]

    all_ent: List[List[float]] = []
    all_ppl: List[List[float]] = []

    ent, ppl = compute_step_metrics(last_hidden, lm_head)
    all_ent.append(ent)
    all_ppl.append(ppl)

    for _ in range(latent_steps):
        source_model = model
        latent_vec = model_wrapper._apply_latent_realignment(last_hidden, source_model)
        latent_embed = latent_vec.unsqueeze(1)

        past_len = _past_length(past)
        latent_mask = torch.ones(
            (latent_embed.shape[0], past_len + 1),
            dtype=torch.long, device=device,
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
        all_ent.append(ent)
        all_ppl.append(ppl)

    # ── Decode from KV-cache after latent recurrence ──
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

        next_input = next_token.unsqueeze(-1)
        past_len = _past_length(past)
        new_mask = torch.ones(
            (next_input.shape[0], past_len + 1),
            dtype=torch.long, device=device,
        )
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

    decoded_text = model_wrapper.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    total_decoded = len(generated_ids)

    return all_ent, all_ppl, decoded_text, total_decoded, latent_steps


# ═══════════════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════════════

def build_prompt_messages(question: str, role: str, args) -> List[Dict]:
    if _HAS_PROMPTS:
        return build_agent_message_sequential_latent_mas(
            role=role, question=question, context="",
            method="latent_mas", args=args,
        )
    return [
        {"role": "system", "content": "You are a helpful assistant. Think step by step."},
        {"role": "user", "content": question},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Answer evaluation (mirroring run.py logic)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_answer(final_text: str, item: Dict, task: str) -> Tuple[str, str, bool]:
    """Returns (prediction, gold, correct)."""
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


# ═══════════════════════════════════════════════════════════════════════════
# File-name prefix helper
# ═══════════════════════════════════════════════════════════════════════════

def make_prefix(task: str, model_name: str, method: str) -> str:
    """e.g. 'aime2024_Qwen3-4B_latent_mas'"""
    short_model = model_name.split("/")[-1]  # Qwen/Qwen3-4B → Qwen3-4B
    return f"{task}_{short_model}_{method}"


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_metrics(
    cases: Dict,
    metric_key: str,
    out_dir: str,
    prefix: str,
    bin_size: int = 5,
    n_steps: int = 80,
    step_label: str = "Latent Step",
):
    """Scatter + binned line plot for a metric."""
    n_points = n_steps + 1
    steps = np.arange(n_points)

    all_traces = []
    for case_key in sorted(cases.keys(), key=lambda k: int(k.replace("case", ""))):
        vals = cases[case_key][metric_key]
        # Pad if trace is shorter (e.g. decode ended early)
        padded = vals[:n_points]
        if len(padded) < n_points:
            padded = padded + [padded[-1]] * (n_points - len(padded))
        all_traces.append(padded)
    all_traces = np.array(all_traces)

    # ── Scatter ──
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, trace in enumerate(all_traces):
        ax.scatter(steps, trace, s=8, alpha=0.45,
                   label=f"case{i}" if i < 10 else None)
    ax.set_xlabel(step_label)
    ax.set_ylabel(metric_key.capitalize())
    ax.set_title(f"[{prefix}] Per-case {metric_key} ({step_label})")
    if len(all_traces) <= 10:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_{metric_key}_scatter.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")

    # ── Binned line ──
    n_bins = math.ceil(n_points / bin_size)
    bc, bm, bs_ = [], [], []
    for b in range(n_bins):
        s = b * bin_size
        e = min(s + bin_size, n_points)
        chunk = all_traces[:, s:e]
        bc.append((s + e - 1) / 2)
        bm.append(chunk.mean())
        bs_.append(chunk.std())
    bc, bm, bs_ = np.array(bc), np.array(bm), np.array(bs_)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(bc, bm, "o-", color="tab:blue", lw=2, ms=5, label="mean")
    ax.fill_between(bc, bm - bs_, bm + bs_, alpha=0.2, color="tab:blue", label="±1 std")
    ax.set_xlabel(f"{step_label} (bin size={bin_size})")
    ax.set_ylabel(metric_key.capitalize())
    ax.set_title(f"[{prefix}] Avg {metric_key} ({step_label}, bin={bin_size})")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_{metric_key}_line.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Measure hidden-state entropy/perplexity and record case study."
    )
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "text_mas", "latent_mas"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--task", type=str, default="aime2024",
                        choices=list(TASK_LOADERS.keys()))
    parser.add_argument("--latent_steps", type=int, default=80,
                        help="Latent recurrence steps (latent_mas only; "
                             "baseline/text_mas always measure first 80 decoded tokens)")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="sequential",
                        choices=["sequential", "hierarchical"])
    parser.add_argument("--agent_role", type=str, default="planner")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--bin_size", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="example_logs")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    # Args required by ModelWrapper / prompt builders
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--text_mas_context_length", type=int, default=-1)

    args = parser.parse_args()

    N_METRIC_STEPS = 80  # number of steps to measure for baseline/text_mas

    set_seed(args.seed)
    device = torch.device(args.device)
    prefix = make_prefix(args.task, args.model_name, args.method)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Config] method={args.method}, model={args.model_name}, "
          f"task={args.task}, device={device}")
    if args.method == "latent_mas":
        print(f"         latent_steps={args.latent_steps}, "
              f"realign={args.latent_space_realign}")
    else:
        print(f"         measuring first {N_METRIC_STEPS} decoded tokens")

    # ── Load model ──
    model_wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)

    # ── Load dataset ──
    dataset = list(TASK_LOADERS[args.task]())
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
    print(f"[Data] {args.task}: {len(dataset)} cases\n")

    # ── Prepare case-study txt file ──
    txt_path = os.path.join(args.out_dir, f"{prefix}_case_study.txt")
    txt_file = open(txt_path, "w", encoding="utf-8")

    # ── Run ──
    cases_json: Dict = {}
    n_correct = 0

    for case_idx, item in enumerate(tqdm(dataset, desc="Analyzing")):
        question = item["question"]

        messages = build_prompt_messages(question, role=args.agent_role, args=args)
        prompt_text = model_wrapper.render_chat(messages, add_generation_prompt=True)
        if args.think:
            prompt_text = f"{prompt_text}<think>"

        encoded = model_wrapper.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # ── Method-specific analysis ──
        if args.method == "latent_mas":
            ent_steps, ppl_steps, decoded_text, n_decoded, n_latent = \
                run_latent_mas_full(
                    model_wrapper, input_ids, attention_mask,
                    latent_steps=args.latent_steps,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            n_steps_used = args.latent_steps
        else:
            # baseline / text_mas
            ent_steps, ppl_steps, decoded_text, n_decoded = \
                run_decode_with_metrics(
                    model_wrapper, input_ids, attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    n_metric_steps=N_METRIC_STEPS,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            n_latent = 0
            n_steps_used = N_METRIC_STEPS

        # ── Evaluate answer ──
        pred, gold, correct = evaluate_answer(decoded_text, item, args.task)
        if correct:
            n_correct += 1

        # ── Store JSON record ──
        case_record = {
            "question": question,
            "entropy": [e[0] for e in ent_steps],
            "perplexity": [p[0] for p in ppl_steps],
            "prediction": pred,
            "gold": str(gold),
            "correct": correct,
            "decoded_tokens": n_decoded if args.method != "latent_mas" else n_decoded,
            "raw_response": decoded_text,
        }
        if args.method == "latent_mas":
            case_record["latent_steps"] = n_latent

        cases_json[f"case{case_idx}"] = case_record

        # ── Write to case-study txt ──
        txt_file.write(f"{'='*60}\n")
        txt_file.write(f"Case #{case_idx}\n")
        txt_file.write(f"{'='*60}\n")
        txt_file.write(f"[Question]\n{question}\n\n")
        txt_file.write(f"[Response]\n{decoded_text}\n\n")
        txt_file.write(f"[Prediction] {pred}\n")
        txt_file.write(f"[Gold]       {gold}\n")
        txt_file.write(f"[Correct]    {correct}\n")
        txt_file.write(f"[Decoded tokens] {n_decoded if args.method != 'latent_mas' else n_decoded}")
        if args.method == "latent_mas":
            txt_file.write(f"  |  [Latent steps] {n_latent}")
        txt_file.write(f"\n\n")
        txt_file.flush()

        torch.cuda.empty_cache()

    # ── Final accuracy ──
    total = len(dataset)
    accuracy = n_correct / total if total > 0 else 0.0

    # Write summary to txt
    txt_file.write(f"\n{'#'*60}\n")
    txt_file.write(f"SUMMARY\n")
    txt_file.write(f"{'#'*60}\n")
    txt_file.write(f"Method:   {args.method}\n")
    txt_file.write(f"Model:    {args.model_name}\n")
    txt_file.write(f"Task:     {args.task}\n")
    txt_file.write(f"Samples:  {total}\n")
    txt_file.write(f"Correct:  {n_correct}\n")
    txt_file.write(f"Accuracy: {accuracy:.4f}\n")
    if args.method == "latent_mas":
        txt_file.write(f"Latent steps: {args.latent_steps}\n")
        txt_file.write(f"Realign: {args.latent_space_realign}\n")
    txt_file.close()
    print(f"\n[Saved] {txt_path}")

    # ── Build & save JSON ──
    output = {
        "config": {
            "method": args.method,
            "model": args.model_name,
            "task": args.task,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": {
            "total": total,
            "correct": n_correct,
            "accuracy": round(accuracy, 4),
        },
        "cases": cases_json,
    }
    if args.method == "latent_mas":
        output["config"]["latent_steps"] = args.latent_steps
        output["config"]["latent_space_realign"] = args.latent_space_realign
        output["summary"]["latent_steps"] = args.latent_steps

    json_path = os.path.join(args.out_dir, f"{prefix}_entropy_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {json_path}")

    # ── Plots ──
    print("[Plotting] ...")
    step_label = "Latent Step" if args.method == "latent_mas" else "Decoded Token"
    n_steps_for_plot = args.latent_steps if args.method == "latent_mas" else N_METRIC_STEPS

    plot_metrics(cases_json, "entropy", args.out_dir, prefix,
                 bin_size=args.bin_size, n_steps=n_steps_for_plot,
                 step_label=step_label)
    plot_metrics(cases_json, "perplexity", args.out_dir, prefix,
                 bin_size=args.bin_size, n_steps=n_steps_for_plot,
                 step_label=step_label)

    print(f"\n[Result] Accuracy = {accuracy:.4f} ({n_correct}/{total})")
    print(f"[Done] All outputs saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
