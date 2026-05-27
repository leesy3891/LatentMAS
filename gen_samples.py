"""
gen_samples.py
--------------
Generates 3 paraphrased versions (version1-3) of each question in a dataset
using Qwen3-14B (via vLLM or HF transformers), and saves as gzipped JSONL.

Output format per line (JSON):
{
  "task_id": <int>,
  "question": <original question>,
  "solution": <original solution>,
  "gold": <original gold>,
  "version1": <paraphrased v1 - different language>,
  "version2": <paraphrased v2 - word replacement>,
  "version3": <paraphrased v3 - free paraphrase>
}

Usage:
  python gen_samples.py --task gsm8k --max_samples 10
  python gen_samples.py --task aime2025 --use_vllm --tensor_parallel_size 2
  python gen_samples.py --task gpqa --max_samples 50 --resume

Saved to: /example_logs/<task>_paraphrased.jsonl.gz
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Minimal dataset loaders (mirrors data.py)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    if s.endswith("."):
        s = s[:-1]
    s = s.replace(",", "")
    return s.strip()


def extract_gold(solution: str) -> str:
    if "####" in solution:
        return solution.split("####")[-1].strip()
    return solution.strip()


def load_gsm8k(split="test", cache_dir=None):
    ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    for item in ds:
        yield {"question": item["question"].strip(), "solution": item["answer"],
               "gold": normalize_answer(extract_gold(item["answer"]))}

def load_aime2025(split="train", cache_dir=None):
    ds = load_dataset("yentinglin/aime_2025", split=split, cache_dir=cache_dir)
    for item in ds:
        a = str(item["answer"]).strip()
        yield {"question": item["problem"].strip(), "solution": a, "gold": normalize_answer(a)}

def load_aime2024(split="train", cache_dir=None):
    ds = load_dataset("HuggingFaceH4/aime_2024", split=split, cache_dir=cache_dir)
    for item in ds:
        a = str(item["answer"]).strip()
        yield {"question": item["problem"].strip(), "solution": a, "gold": normalize_answer(a)}

def load_gpqa_diamond(split="test", cache_dir=None):
    ds = load_dataset("fingertap/GPQA-Diamond", split=split, cache_dir=cache_dir)
    for item in ds:
        yield {"question": item["question"].strip(), "solution": item["answer"].strip(),
               "gold": normalize_answer(item["answer"].strip())}

def load_arc_easy(split="test", cache_dir=None):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split, cache_dir=cache_dir)
    lm = {"1":"a","2":"b","3":"c","4":"d"}
    for item in ds:
        stem = item["question"].strip()
        labels, texts = item["choices"]["label"], item["choices"]["text"]
        mapped = [(lm.get(str(l).strip(), str(l).strip().lower()), t.strip()) for l, t in zip(labels, texts)]
        q = stem + "\n" + "\n".join(f"{l}: {t}" for l, t in mapped)
        raw = item.get("answerKey","").strip()
        ans = lm.get(raw, raw.lower())
        yield {"question": q, "solution": ans, "gold": normalize_answer(ans)}

def load_arc_challenge(split="test", cache_dir=None):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, cache_dir=cache_dir)
    lm = {"1":"a","2":"b","3":"c","4":"d"}
    for item in ds:
        stem = item["question"].strip()
        labels, texts = item["choices"]["label"], item["choices"]["text"]
        mapped = [(lm.get(str(l).strip(), str(l).strip().lower()), t.strip()) for l, t in zip(labels, texts)]
        q = stem + "\n" + "\n".join(f"{l}: {t}" for l, t in mapped)
        raw = item.get("answerKey","").strip()
        ans = lm.get(raw, raw.lower())
        yield {"question": q, "solution": ans, "gold": normalize_answer(ans)}

def load_mbppplus(split="test", cache_dir=None):
    ds = load_dataset("evalplus/mbppplus", split=split, cache_dir=cache_dir)
    for item in ds:
        q = (f"Please provide a self-contained Python script that solves the "
             f"following problem in a markdown code block:\n```python\n"
             f"YOUR_PYTHON_CODE\n```:\n{item['prompt']}\n"
             f"Your answer will be tested on test cases like:\n"
             f"{item['test_list'][0]}\n{item['test_list'][1]}\n{item['test_list'][2]}")
        a = str(item["test"])
        yield {"question": q, "solution": a, "gold": a}

def load_humanevalplus(split="test", cache_dir=None):
    ds = load_dataset("evalplus/humanevalplus", split=split, cache_dir=cache_dir)
    for item in ds:
        q = (f"Please provide a self-contained Python script that solves the "
             f"following problem in a markdown code block:\n```python\n"
             f"YOUR_PYTHON_CODE\n```:\n{item['prompt']}")
        raw = str(item["test"])
        a = raw.replace("candidate", item["entry_point"]) + f"\n\ncheck({item['entry_point']})"
        yield {"question": q, "solution": a, "gold": a}

def load_medqa(split=None, cache_dir=None):
    ds = load_dataset("json", data_files="./data/medqa.json", split="train")
    cm = {"0":"A","1":"B","2":"C","3":"D"}
    for item in ds:
        raw = str(item["answer"])
        ans = ""
        for idx, op in enumerate(item["options"]):
            if raw in op:
                ans = cm[str(idx)].lower(); break
        yield {"question": item["query"], "solution": ans, "gold": normalize_answer(ans)}


LOADERS = {
    "gsm8k": load_gsm8k, "aime2024": load_aime2024, "aime2025": load_aime2025,
    "gpqa": load_gpqa_diamond, "arc_easy": load_arc_easy, "arc_challenge": load_arc_challenge,
    "mbppplus": load_mbppplus, "humanevalplus": load_humanevalplus, "medqa": load_medqa,
}
MC_TASKS = {"arc_easy", "arc_challenge", "gpqa", "medqa"}
NUMERIC_TASKS = {"gsm8k", "aime2024", "aime2025"}
CODE_TASKS = {"mbppplus", "humanevalplus"}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_paraphrase_prompt(question: str, task: str) -> str:
    if task in MC_TASKS:
        return f"""Paraphrase this question in 3 different ways, including the answer candidates in each version. Do not change the A-D ordering because it will change the GOLD of the original question. 
Each of 3 versions should be version1: write the question in a different language (maybe chinese), version 2: replace words, version 3: paraphrase the question in any different way except ver1, 2.
Target Question: {question}
Write your answer in: 
[VERSION1] 
version 1
---
[VERSION2]
version 2
---
[VERSION3]
version 3"""

    elif task in NUMERIC_TASKS:
        return f"""Paraphrase this math question in 3 different ways. The numerical answer must remain exactly the same. Keep all numbers, equations, and constraints identical so the answer does not change. Only reword/restructure the problem statement.
Each of 3 versions should be version1: write the question in a different language (maybe chinese), version 2: replace words while keeping all numbers and math intact, version 3: paraphrase the question in any different way except ver1, 2 while keeping the answer the same.
Target Question: {question}
Write your answer in: 
[VERSION1] 
version 1
---
[VERSION2]
version 2
---
[VERSION3]
version 3"""

    elif task in CODE_TASKS:
        return f"""Paraphrase this coding question in 3 different ways. The function signature and test cases must remain the same so that the expected code solution does not change. Only reword the problem description.
Each of 3 versions should be version1: write the question description in a different language (maybe chinese) but keep code/function names in English, version 2: replace descriptive words while keeping function names and test cases intact, version 3: paraphrase the question in any different way except ver1, 2 while keeping the specification identical.
Target Question: {question}
Write your answer in: 
[VERSION1] 
version 1
---
[VERSION2]
version 2
---
[VERSION3]
version 3"""

    else:
        return f"""Paraphrase this question in 3 different ways. The answer must remain the same.
Each of 3 versions should be version1: write the question in a different language (maybe chinese), version 2: replace words, version 3: paraphrase the question in any different way except ver1, 2.
Target Question: {question}
Write your answer in: 
[VERSION1] 
version 1
---
[VERSION2]
version 2
---
[VERSION3]
version 3"""


# ---------------------------------------------------------------------------
# Parse 3-version output
# ---------------------------------------------------------------------------

def parse_versions(text: str) -> Dict[str, str]:
    result = {"version1": "", "version2": "", "version3": ""}
    patterns = [
        (r"\[VERSION1\]\s*\n?(.*?)(?=\n\s*-{2,}\s*\n|\[VERSION2\])", "version1"),
        (r"\[VERSION2\]\s*\n?(.*?)(?=\n\s*-{2,}\s*\n|\[VERSION3\])", "version2"),
        (r"\[VERSION3\]\s*\n?(.*?)$", "version3"),
    ]
    for pat, key in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            result[key] = m.group(1).strip().strip("-").strip()

    # Fallback: split by ---
    if not result["version1"] and "---" in text:
        parts = re.split(r"\n-{2,}\n", text)
        for i, key in enumerate(["version1", "version2", "version3"]):
            if i < len(parts):
                result[key] = re.sub(r"\[VERSION\d\]\s*", "", parts[i]).strip()
    return result


# ---------------------------------------------------------------------------
# Model wrapper (lightweight, for generation only)
# ---------------------------------------------------------------------------

class ParaphraseGenerator:
    """Wraps Qwen3-14B for paraphrase generation via vLLM or HF transformers."""

    def __init__(self, model_name: str, use_vllm: bool, args):
        import torch
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.device = args.device

        if use_vllm:
            from vllm import LLM, SamplingParams
            tp = getattr(args, "tensor_parallel_size", 1)
            gpu_util = getattr(args, "gpu_memory_utilization", 0.9)
            print(f"[vLLM] Loading {model_name} (tp={tp}, gpu_util={gpu_util})")
            self.engine = LLM(model=model_name, tensor_parallel_size=tp,
                              gpu_memory_utilization=gpu_util)
            self.tokenizer = self.engine.get_tokenizer()
            self.sampling_params = SamplingParams(
                temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_new_tokens,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[HF] Loading {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            ).to(self.device).eval()

    def _render_chat(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Follow the user's formatting instructions exactly."},
            {"role": "user", "content": user_prompt},
        ]
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        # fallback
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m['content']}\n</|{m['role']}|>")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def generate_batch(self, user_prompts: List[str]) -> List[str]:
        """Generate responses for a batch of user prompts."""
        rendered = [self._render_chat(p) for p in user_prompts]

        if self.use_vllm:
            outputs = self.engine.generate(rendered, self.sampling_params)
            return [o.outputs[0].text.strip() for o in outputs]
        else:
            import torch
            encoded = self.tokenizer(
                rendered, return_tensors="pt", padding=True, add_special_tokens=False
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            prompt_lens = attention_mask.sum(dim=1).tolist()

            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=4096,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            results = []
            for idx, plen in enumerate(prompt_lens):
                gen_ids = out[idx, int(plen):]
                results.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
            return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate 3 paraphrased versions of dataset questions using Qwen3-14B")
    parser.add_argument("--task", required=True, choices=list(LOADERS.keys()))
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 = all")
    parser.add_argument("--output_dir", type=str, default="/example_logs")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing partial output")
    # Model args
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of questions per generation batch")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.task}_paraphrased.jsonl.gz")

    # Resume support
    existing_ids = set()
    existing_records: List[str] = []
    if args.resume and os.path.exists(out_path):
        with gzip.open(out_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing_ids.add(rec["task_id"])
                    existing_records.append(line)
        print(f"Resuming: {len(existing_ids)} records already done.")

    # Load dataset
    split_defaults = {
        "gsm8k": "test", "aime2024": "train", "aime2025": "train", "gpqa": "test",
        "arc_easy": "test", "arc_challenge": "test", "mbppplus": "test",
        "humanevalplus": "test", "medqa": None,
    }
    split = args.split or split_defaults.get(args.task, "test")
    loader_fn = LOADERS[args.task]
    items = list(loader_fn(split=split) if split else loader_fn())

    if args.max_samples > 0:
        items = items[:args.max_samples]
    total = len(items)
    print(f"Task: {args.task} | Samples: {total} | Output: {out_path}")

    # Load model
    generator = ParaphraseGenerator(args.model_name, args.use_vllm, args)

    # Filter out already-done items
    todo = [(tid, item) for tid, item in enumerate(items) if tid not in existing_ids]
    print(f"To generate: {len(todo)} (skipping {len(existing_ids)} existing)")

    # Process in batches
    with gzip.open(out_path, "wt", encoding="utf-8") as fout:
        # Write existing records first
        for line in existing_records:
            fout.write(line + "\n")

        for batch_start in range(0, len(todo), args.batch_size):
            batch = todo[batch_start : batch_start + args.batch_size]
            prompts = [build_paraphrase_prompt(item["question"], args.task)
                       for _, item in batch]

            try:
                outputs = generator.generate_batch(prompts)
            except Exception as e:
                print(f"  [ERROR] batch starting at {batch_start}: {e}", file=sys.stderr)
                # Write empty-version records so progress is saved
                for tid, item in batch:
                    record = {
                        "task_id": tid,
                        "question": item["question"],
                        "solution": item["solution"],
                        "gold": item["gold"],
                        "version1": "", "version2": "", "version3": "",
                        "parse_error": str(e),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            for (tid, item), raw_output in zip(batch, outputs):
                versions = parse_versions(raw_output)
                record = {
                    "task_id": tid,
                    "question": item["question"],
                    "solution": item["solution"],
                    "gold": item["gold"],
                    "version1": versions["version1"],
                    "version2": versions["version2"],
                    "version3": versions["version3"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            fout.flush()
            done = min(batch_start + args.batch_size, len(todo))
            print(f"  [{done}/{len(todo)}] generated")

    # Summary
    count = 0
    sample = None
    with gzip.open(out_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                if count == 1:
                    sample = json.loads(line)
    print(f"\nDone. Total records: {count} | Saved to {out_path}")
    if sample:
        print(f"Sample keys: {list(sample.keys())}")
        print(f"  task_id={sample['task_id']}, gold={sample['gold']}")
        print(f"  version1 (first 100 chars): {sample.get('version1','')[:100]}")


if __name__ == "__main__":
    main()
