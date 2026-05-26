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


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


# ──────────────────────────────────────────────────────────────
# CoT Reasoning Logger — 배치 단위로 즉시 파일에 기록 (streaming)
# ──────────────────────────────────────────────────────────────
class CoTLogger:
    """
    문제별 CoT reasoning을 텍스트 파일에 streaming 기록한다.
    배치 처리 직후 flush하여 메모리 부담 없이 동작.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        # 파일을 새로 생성 (기존 내용 초기화)
        with open(filepath, "w", encoding="utf-8") as f:
            pass

    def write_result(self, task_number: int, res: Dict) -> None:
        """단일 task 결과를 파일에 append 한다."""
        lines: List[str] = []
        lines.append(f"[TASK_NUMBER] {task_number}")
        lines.append(f"[PREDICTION] {res.get('prediction', '')}")
        lines.append(f"[GOLD] {res.get('gold', '')}")
        lines.append(f"[CORRECT] {res.get('correct', False)}")

        # raw answer
        lines.append("[RAW_ANSWER_START]")
        lines.append(res.get("raw_prediction", "").rstrip())
        lines.append("[RAW_ANSWER_END]")

        # agent traces
        agents = res.get("agents", [])
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
                        lines.append(f"(latent thinking: {latent_steps} steps, no text output)")
            lines.append("[AGENT_TRACES_END]")

        lines.append("")  # 빈 줄로 task 구분

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def write_batch(self, batch_start: int, results: List[Dict]) -> None:
        """배치 내 모든 결과를 순서대로 기록."""
        for offset, res in enumerate(results):
            self.write_result(batch_start + offset, res)


def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
    cot_logger: "CoTLogger | None" = None,
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

    # ── CoT 로깅: 즉시 파일에 flush ──
    if cot_logger is not None:
        cot_logger.write_batch(batch_start, results)

    for offset, res in enumerate(results):
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

        # ── 메모리 최적화: 무거운 필드를 preds 에 넣지 않음 ──
        # raw_prediction, agents 내 input/input_ids/input_tokens 는
        # 이미 CoT 파일과 stdout에 기록되었으므로 제거
        lightweight_res = {
            "question": res.get("question", ""),
            "gold": res.get("gold", ""),
            "prediction": res.get("prediction", ""),
            "correct": res.get("correct", False),
        }
        preds.append(lightweight_res)

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
                        help="Model choices to use for experiments (e.g. 'Qwen/Qwen3-14B').")
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

    # logit lens
    parser.add_argument("--logit_lens", action="store_true",
                        help="Enable logit lens: decode top-5 tokens per layer and save to CSV")
    parser.add_argument("--logit_lens_dir", type=str, default="resource",
                        help="Directory to save logit lens CSV (default: resource/)")

    # memory optimization
    parser.add_argument("--max_gpu_mem_gb", type=float, default=0,
                        help="GPU memory cap in GiB (e.g. 46). 0 = no cap. "
                             "Sets device_map='auto' + max_memory for HF model loading.")

    # ===== NEW: CoT reasoning log =====
    parser.add_argument("--cot_log", action="store_true",
                        help="Enable CoT reasoning log: save per-task reasoning traces to a text file")
    parser.add_argument("--cot_log_dir", type=str, default="cot_logs",
                        help="Directory to save CoT log text files (default: cot_logs/)")

    args = parser.parse_args()
    
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
    
    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    
    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

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

    # ── CoT Logger 초기화 ──
    cot_logger = None
    if args.cot_log:
        model_short = args.model_name.replace("/", "-")
        cot_filename = f"{args.method}_{model_short}_{args.task}.txt"
        cot_filepath = os.path.join(args.cot_log_dir, cot_filename)
        cot_logger = CoTLogger(cot_filepath)
        print(f"[CoT] Logging to: {cot_filepath}")

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
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
                cot_logger=cot_logger,
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
            cot_logger=cot_logger,
        )
    progress.close()
    
    total_time = time.time() - start_time

    acc, correct = evaluate(preds)

    # ===== logit lens CSV flush =====
    if args.logit_lens and model.logit_lens is not None:
        model_short = args.model_name.replace("/", "-")
        csv_name = f"{args.method}_{model_short}_{args.task}.csv"
        csv_path = model.logit_lens.flush_csv(csv_name)
        print(f"[LogitLens] CSV saved to: {csv_path}")
    
    # ===== CoT log summary =====
    if cot_logger is not None:
        print(f"[CoT] Reasoning log saved to: {cot_logger.filepath}")

    print(
        json.dumps(
            {
                "method": args.method,
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "accuracy": acc,
                "correct": correct,
                "total_time_sec": round(total_time, 4),
                "time_per_sample_sec": round(total_time / args.max_samples, 4),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
