"""
analyze_layerwise_hidden_pca.py
-------------------------------
Runs the full multi-agent inference pipeline (including judger decode),
collects layer-wise hidden states, evaluates execution accuracy,
and produces PCA scatter plots.

Usage:
  CUDA_VISIBLE_DEVICES=3 python analyze_layerwise_hidden_pca.py \
    --method latent_mas --model_name Qwen/Qwen3-14B --task aime2025 \
    --prompt sequential --latent_steps 40 --max_samples 30 \
    --latent_space_realign --max_new_tokens 10000
"""

import argparse
import json
import os
import sys
import time

import torch

from data import (
    load_aime2024, load_aime2025, load_arc_easy, load_arc_challenge,
    load_gsm8k, load_gpqa_diamond, load_mbppplus, load_humanevalplus, load_medqa,
)
from methods import default_agents
from models import ModelWrapper
from utils import auto_device, set_seed
from layerwise_hidden_analysis_utils import (
    LayerwiseHiddenCollector, sample_collector,
    collect_latent_mas_states, collect_text_mas_states,
    run_pca_and_plot,
)


def build_parser():
    p = argparse.ArgumentParser(description="Layer-wise hidden-state PCA analysis")

    # Core args (same interface as run.py)
    p.add_argument("--method", choices=["text_mas", "latent_mas"], required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--task", choices=[
        "gsm8k", "aime2024", "aime2025", "gpqa",
        "arc_easy", "arc_challenge", "mbppplus", "humanevalplus", "medqa",
    ], default="gsm8k")
    p.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--latent_steps", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent_space_realign", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--think", action="store_true")

    # text_mas-specific args  ← FIX: was missing, caused AttributeError
    p.add_argument("--text_mas_context_length", type=int, default=-1,
                   help="TextMAS context length limit in characters (-1 = no limit)")

    # vLLM compat (parsed but unused; analysis uses HF backend)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--enable_prefix_caching", action="store_true")
    p.add_argument("--use_second_HF_model", action="store_true")
    p.add_argument("--device2", type=str, default="cuda:1")
    p.add_argument("--generate_bs", type=int, default=1)

    # Analysis-specific args
    p.add_argument("--out_dir", type=str, default="example_logs")
    p.add_argument("--max_decode_analysis_steps", type=int, default=80,
                   help="Max number of decode steps to collect hidden states at (uniformly sampled)")
    p.add_argument("--max_hidden_samples_per_layer", type=int, default=2000)
    p.add_argument("--max_hidden_samples_last_layer", type=int, default=8000)
    p.add_argument("--pca_chunk_size", type=int, default=16)
    p.add_argument("--save_hidden_cache", action="store_true")

    return p


def load_dataset(task, split):
    loaders = {
        "gsm8k": lambda: load_gsm8k(split=split),
        "aime2024": lambda: load_aime2024(split="train"),
        "aime2025": lambda: load_aime2025(split="train"),
        "gpqa": lambda: load_gpqa_diamond(split="test"),
        "arc_easy": lambda: load_arc_easy(split="test"),
        "arc_challenge": lambda: load_arc_challenge(split="test"),
        "mbppplus": lambda: load_mbppplus(split="test"),
        "humanevalplus": lambda: load_humanevalplus(split="test"),
        "medqa": lambda: load_medqa(split="test"),
    }
    return loaders[task]()


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = auto_device(args.device)

    # Load dataset
    dataset = list(load_dataset(args.task, args.split))
    if args.max_samples == -1:
        args.max_samples = len(dataset)
    else:
        dataset = dataset[: args.max_samples]

    if len(dataset) > 200:
        print(f"WARNING: {len(dataset)} samples. Analysis on HF backend will be very slow. Exiting.")
        sys.exit(1)

    print(f"Analysis: method={args.method}  model={args.model_name}  task={args.task}  "
          f"samples={len(dataset)}  latent_steps={args.latent_steps}  "
          f"max_new_tokens={args.max_new_tokens}")

    # Load model (HF backend only)
    model = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    num_layers = model.model.config.num_hidden_layers + 1
    print(f"Model loaded. {num_layers} layers (incl. embedding output).")

    # Collect hidden states + run full inference
    collector = LayerwiseHiddenCollector(num_layers)
    agents = default_agents()
    t0 = time.time()

    if args.method == "latent_mas":
        eval_results = collect_latent_mas_states(
            model, agents, dataset, args, collector,
            max_decode_analysis_steps=args.max_decode_analysis_steps)
    else:
        eval_results = collect_text_mas_states(
            model, agents, dataset, args, collector,
            max_decode_analysis_steps=args.max_decode_analysis_steps)

    elapsed = time.time() - t0

    # Report accuracy
    total = len(eval_results)
    correct = sum(1 for r in eval_results if r["correct"])
    acc = correct / total if total > 0 else 0.0
    print(f"\nCollection done in {elapsed:.1f}s")
    print(f"Total records (all layers): {collector.total_records()}")
    print(json.dumps({
        "method": args.method, "model": args.model_name, "task": args.task,
        "seed": args.seed, "max_samples": total,
        "accuracy": round(acc, 4), "correct": correct, "total": total,
        "total_time_sec": round(elapsed, 2),
        "time_per_sample_sec": round(elapsed / total, 2) if total > 0 else 0,
    }, ensure_ascii=False))

    # Optionally save raw hidden cache
    if args.save_hidden_cache:
        os.makedirs(args.out_dir, exist_ok=True)
        short_model = args.model_name.split("/")[-1]
        cache_path = os.path.join(args.out_dir,
                                  f"{args.task}_{short_model}_{args.method}_hidden_cache.pt")
        torch.save({li: collector.buffers[li] for li in range(num_layers)}, cache_path)
        print(f"Hidden cache saved: {cache_path}")

    # Sample + PCA + Plot
    sampled = sample_collector(
        collector, args.max_hidden_samples_per_layer,
        args.max_hidden_samples_last_layer, args.seed)

    saved_files, meta_path = run_pca_and_plot(
        sampled=sampled, num_layers=num_layers,
        method=args.method, model_name=args.model_name, task=args.task,
        seed=args.seed, latent_steps=args.latent_steps,
        max_samples=len(dataset), out_dir=args.out_dir,
        pca_chunk_size=args.pca_chunk_size)

    print(f"\nDone. {len(saved_files)} figure(s) + metadata -> {args.out_dir}/")
    print(f"Accuracy: {correct}/{total} = {acc:.4f}")


if __name__ == "__main__":
    main()
