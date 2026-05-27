"""
gen_samples.py
--------------
Generates 3 paraphrased versions (version1-3) of each question in a dataset
using Qwen3-14B (via vLLM or HF transformers), and saves as gzipped JSONL.

Usage:
  python gen_samples.py --task gpqa --max_samples 50 --device cuda --batch_size 4
  python gen_samples.py --task aime2025 --use_vllm --tensor_parallel_size 2
  python gen_samples.py --task gpqa --resume

Saved to: ./example_logs/<task>_paraphrased.jsonl.gz
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
# KEY FIX: Use clear XML-style delimiters and explicit placeholder markers
# so Qwen3 does not confuse the template with a completed answer.

def build_paraphrase_prompt(question: str, task: str) -> str:
    if task in MC_TASKS:
        return f"""Paraphrase the following question in 3 different ways, including the answer candidates in each version. Do not change the A-D ordering because it will change the GOLD of the original question.

- VERSION 1: Rewrite the entire question in Chinese (中文).
- VERSION 2: Replace key words and rephrase sentences while keeping the same language (English).
- VERSION 3: Paraphrase the question in a completely different way (English), different from version 2.

Target Question:
{question}

Now write your 3 versions. Use exactly this format (replace the placeholder text with your actual paraphrases):

<VERSION1>
(your Chinese paraphrase here, including A/B/C/D choices in Chinese)
</VERSION1>

<VERSION2>
(your English word-replaced paraphrase here, including A/B/C/D choices)
</VERSION2>

<VERSION3>
(your English free paraphrase here, including A/B/C/D choices)
</VERSION3>"""

    elif task in NUMERIC_TASKS:
        return f"""Paraphrase the following math question in 3 different ways. The numerical answer must remain exactly the same. Keep all numbers, equations, and constraints identical — only reword the problem statement.

- VERSION 1: Rewrite the entire question in Chinese (中文). Keep all math notation intact.
- VERSION 2: Replace key words and rephrase sentences while keeping the same language (English) and all numbers/math intact.
- VERSION 3: Paraphrase the question in a completely different way (English), different from version 2. Keep the answer the same.

Target Question:
{question}

Now write your 3 versions. Use exactly this format (replace the placeholder text with your actual paraphrases):

<VERSION1>
(your Chinese paraphrase here)
</VERSION1>

<VERSION2>
(your English word-replaced paraphrase here)
</VERSION2>

<VERSION3>
(your English free paraphrase here)
</VERSION3>"""

    elif task in CODE_TASKS:
        return f"""Paraphrase the following coding question in 3 different ways. The function signature and test cases must remain exactly the same — only reword the problem description.

- VERSION 1: Rewrite the description in Chinese (中文), but keep function names, code, and test cases in English.
- VERSION 2: Replace descriptive words while keeping function names and test cases intact (English).
- VERSION 3: Paraphrase the question in a completely different way (English), different from version 2.

Target Question:
{question}

Now write your 3 versions. Use exactly this format (replace the placeholder text with your actual paraphrases):

<VERSION1>
(your Chinese paraphrase here)
</VERSION1>

<VERSION2>
(your English word-replaced paraphrase here)
</VERSION2>

<VERSION3>
(your English free paraphrase here)
</VERSION3>"""

    else:
        return f"""Paraphrase the following question in 3 different ways. The answer must remain the same.

- VERSION 1: Rewrite the entire question in Chinese (中文).
- VERSION 2: Replace key words and rephrase sentences (English).
- VERSION 3: Paraphrase the question in a completely different way (English), different from version 2.

Target Question:
{question}

Now write your 3 versions. Use exactly this format (replace the placeholder text with your actual paraphrases):

<VERSION1>
(your Chinese paraphrase here)
</VERSION1>

<VERSION2>
(your English word-replaced paraphrase here)
</VERSION2>

<VERSION3>
(your English free paraphrase here)
</VERSION3>"""


# ---------------------------------------------------------------------------
# Parse 3-version output (supports both XML-style and [VERSION] style)
# ---------------------------------------------------------------------------

def strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 output."""
    # Remove complete think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # If think block was never closed (truncated), remove from <think> onwards
    # Actually we want the part AFTER </think>, so if there's an unclosed <think>,
    # the real content may not exist — but let's try removing partial too
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def parse_versions(raw_text: str) -> Dict[str, str]:
    """Parse VERSION1-3 from model output, handling <think> blocks."""
    # Step 1: strip thinking
    text = strip_think_block(raw_text)

    result = {"version1": "", "version2": "", "version3": ""}

    # Try XML-style <VERSION1>...</VERSION1> first
    for key, tag in [("version1", "VERSION1"), ("version2", "VERSION2"), ("version3", "VERSION3")]:
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
        if m:
            result[key] = m.group(1).strip()

    if result["version1"] and result["version2"] and result["version3"]:
        return result

    # Fallback: try [VERSION1] ... --- [VERSION2] ... style
    for key, pat in [
        ("version1", r"\[VERSION1\]\s*\n?(.*?)(?=\n\s*-{2,}\s*\n|\[VERSION2\]|<VERSION2>)"),
        ("version2", r"\[VERSION2\]\s*\n?(.*?)(?=\n\s*-{2,}\s*\n|\[VERSION3\]|<VERSION3>)"),
        ("version3", r"\[VERSION3\]\s*\n?(.*?)$"),
    ]:
        if not result[key]:
            m = re.search(pat, text, re.DOTALL)
            if m:
                result[key] = m.group(1).strip().strip("-").strip()

    # Last fallback: split by ---
    if not any(result.values()) and "---" in text:
        parts = re.split(r"\n-{2,}\n", text)
        for i, key in enumerate(["version1", "version2", "version3"]):
            if i < len(parts):
                result[key] = re.sub(r"</?VERSION\d>|\[VERSION\d\]\s*", "", parts[i]).strip()

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
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens

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
        """Render chat template for Qwen3. Uses /no_think to disable thinking mode."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Follow the user's formatting instructions exactly. Output only the requested versions, nothing else."},
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
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            results = []
            for idx, plen in enumerate(prompt_lens):
                gen_ids = out[idx, int(plen):]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                results.append(text)
            return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate 3 paraphrased versions of dataset questions using Qwen3-14B")
    parser.add_argument("--task", required=True, choices=list(LOADERS.keys()))
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 = all")
    parser.add_argument("--output_dir", type=str, default="./example_logs")
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
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=4,
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

                # Validation: warn if any version is empty or suspiciously short
                for vk in ["version1", "version2", "version3"]:
                    v = versions[vk]
                    if not v or len(v) < 20:
                        print(f"  [WARN] task_id={tid} {vk} is empty or very short ({len(v)} chars)",
                              file=sys.stderr)

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
    empty_count = 0
    sample = None
    with gzip.open(out_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                rec = json.loads(line)
                if not rec.get("version1"):
                    empty_count += 1
                if count == 1:
                    sample = rec
    print(f"\nDone. Total records: {count} | Empty versions: {empty_count} | Saved to {out_path}")
    if sample:
        print(f"Sample keys: {list(sample.keys())}")
        print(f"  task_id={sample['task_id']}, gold={sample['gold']}")
        print(f"  version1 (first 120 chars): {sample.get('version1','')[:120]}")


if __name__ == "__main__":
    main()
