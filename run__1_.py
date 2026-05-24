import argparse
import json
import os
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
import time


# ── Result logger ──────────────────────────────────────────────────
FIELD_SEP = "\t"
RECORD_SEP = "\n" + "=" * 80 + "\n"

def _init_log_file(path: str, args) -> None:
    """Write header to the log file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# method={args.method}  model={args.model_name}  task={args.task}  seed={args.seed}\n")
        f.write(f"# Format: task_number{FIELD_SEP}raw_answer{FIELD_SEP}prediction{FIELD_SEP}gold{FIELD_SEP}correct\n")
        f.write(RECORD_SEP)


def _append_result(path: str, task_number: int, result: Dict) -> None:
    """Append one result to the log file in a parseable format.

    Each record looks like:
        ================================================================================
        [TASK_NUMBER]<TAB>42
        [PREDICTION]<TAB>12
        [GOLD]<TAB>12
        [CORRECT]<TAB>True
        [RAW_ANSWER_START]
        <the full CoT text, may span multiple lines>
        [RAW_ANSWER_END]
        [AGENT_TRACES_START]
        --- agent: Planner (planner) ---
        <agent output text>
        --- agent: Critic (critic) ---
        ...
        [AGENT_TRACES_END]
        ================================================================================
    """
    raw = result.get("raw_prediction", "")
    pred = result.get("prediction", "")
    gold = result.get("gold", "")
    correct = result.get("correct", False)

    lines: List[str] = []
    lines.append(f"[TASK_NUMBER]{FIELD_SEP}{task_number}")
    lines.append(f"[PREDICTION]{FIELD_SEP}{pred}")
    lines.append(f"[GOLD]{FIELD_SEP}{gold}")
    lines.append(f"[CORRECT]{FIELD_SEP}{correct}")
    lines.append("[RAW_ANSWER_START]")
    lines.append(raw)
    lines.append("[RAW_ANSWER_END]")

    # Agent traces (CoT from each agent)
    agents = result.get("agents", [])
    if agents:
        lines.append("[AGENT_TRACES_START]")
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            lines.append(f"--- agent: {name} ({role}) ---")
            output = a.get("output", "").rstrip()
            if output:
                lines.append(output)
            else:
                latent_steps = a.get("latent_steps", None)
                if latent_steps is not None:
                    lines.append(f"(latent_steps={latent_steps}, no text output)")
        lines.append("[AGENT_TRACES_END]")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write(RECORD_SEP)


# ── Evaluation ─────────────────────────────────────────────────────
def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


# ── Batch processing ───────────────────────────────────────────────
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
    log_path: str,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

        # ── Write to log file ──
        _append_result(log_path, problem_idx, res)

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"], required=True,
                        help="Which multi-agent method to run: 'baseline', 'text_mas', or 'latent_mas'.")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"],
                        help="Model choices to use for experiments (e.g. 'Qwen/Qwen3-8B').")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0, help="Number of latent steps for LatentMAS method")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    # Memory optimization
    parser.add_argument("--memory_opt", action="store_true",
                        help="Enable memory optimizations (lower vLLM util, eager mode, aggressive cache flush). "
                             "Keeps peak VRAM under ~49 GB for Qwen3-8B dual-load or Qwen3-14B single-load.")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Max model sequence length for vLLM when --memory_opt is on")
    parser.add_argument("--swap_space", type=int, default=8,
                        help="CPU swap space (GB) for vLLM when --memory_opt is on")

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save CoT answer logs")

    args = parser.parse_args()
    
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
    
    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    
    start_time = time.time()

    # ── Prepare log file ──
    os.makedirs(args.log_dir, exist_ok=True)
    model_short = args.model_name.replace("/", "_")
    log_filename = f"{args.method}_{model_short}_{args.task}_seed{args.seed}.txt"
    log_path = os.path.join(args.log_dir, log_filename)
    _init_log_file(log_path, args)
    print(f"[LOG] Writing CoT answers to: {log_path}")

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
    # dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)  
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
                args,
                log_path,
            )
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
            args=args,
            log_path=log_path,
        )
    progress.close()
    
    total_time = time.time() - start_time

    acc, correct = evaluate(preds)
    
    # Append summary to log
    summary = {
        "method": args.method,
        "model": args.model_name,
        "split": args.split,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "accuracy": acc,
        "correct": correct,
        "total_time_sec": round(total_time, 4),
        "time_per_sample_sec": round(total_time / args.max_samples, 4),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n[SUMMARY]\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n")

    # Print JSON summary to stdout
    print(json.dumps(summary, ensure_ascii=False))
    print(f"[LOG] Results saved to: {log_path}")



if __name__ == "__main__":
    main()
