"""
gen_samples.py
--------------
Generates 3 paraphrased versions (version1-3) of each question in a dataset
using the Anthropic API, and saves them as gzipped JSONL for minimal storage.

Output format per line (JSON):
{
  "task_id": <int>,
  "question": <original question>,
  "solution": <original solution>,
  "gold": <original gold>,
  "version1": <paraphrased question v1 - different language>,
  "version2": <paraphrased question v2 - word replacement>,
  "version3": <paraphrased question v3 - free paraphrase>
}

Usage:
  python gen_samples.py --task gsm8k --max_samples 10
  python gen_samples.py --task aime2025 --max_samples -1
  python gen_samples.py --task gpqa --max_samples 50

Saved to: /example_logs/<task>_paraphrased.jsonl.gz
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Minimal dataset loaders (mirrors data.py but without torch/utils dependency)
# ---------------------------------------------------------------------------
from datasets import load_dataset


def normalize_answer(s: str) -> str:
    """Lightweight normalize (mirrors utils.normalize_answer)."""
    s = s.strip().lower()
    # remove trailing period
    if s.endswith("."):
        s = s[:-1]
    # remove commas in numbers
    s = s.replace(",", "")
    return s.strip()


def extract_gold(solution: str) -> str:
    """Extract the numeric answer after #### in GSM8K solutions."""
    if "####" in solution:
        return solution.split("####")[-1].strip()
    return solution.strip()


def load_gsm8k(split="test", cache_dir=None):
    ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        yield {"question": question, "solution": solution, "gold": gold}


def load_aime2025(split="train", cache_dir=None):
    ds = load_dataset("yentinglin/aime_2025", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {"question": problem, "solution": answer, "gold": gold}


def load_aime2024(split="train", cache_dir=None):
    ds = load_dataset("HuggingFaceH4/aime_2024", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {"question": problem, "solution": answer, "gold": gold}


def load_gpqa_diamond(split="test", cache_dir=None):
    ds = load_dataset("fingertap/GPQA-Diamond", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        answer = item["answer"].strip()
        gold = normalize_answer(answer)
        yield {"question": question, "solution": answer, "gold": gold}


def load_arc_easy(split="test", cache_dir=None):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split, cache_dir=cache_dir)
    label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels, texts = choices["label"], choices["text"]
        mapped = []
        for l, t in zip(labels, texts):
            ml = label_map.get(str(l).strip(), str(l).strip().lower())
            mapped.append((ml, t.strip()))
        question = stem + "\n" + "\n".join(f"{l}: {t}" for l, t in mapped)
        raw = item.get("answerKey", "").strip()
        ans = label_map.get(raw, raw.lower())
        gold = normalize_answer(ans)
        yield {"question": question, "solution": ans, "gold": gold}


def load_arc_challenge(split="test", cache_dir=None):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, cache_dir=cache_dir)
    label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels, texts = choices["label"], choices["text"]
        mapped = []
        for l, t in zip(labels, texts):
            ml = label_map.get(str(l).strip(), str(l).strip().lower())
            mapped.append((ml, t.strip()))
        question = stem + "\n" + "\n".join(f"{l}: {t}" for l, t in mapped)
        raw = item.get("answerKey", "").strip()
        ans = label_map.get(raw, raw.lower())
        gold = normalize_answer(ans)
        yield {"question": question, "solution": ans, "gold": gold}


def load_mbppplus(split="test", cache_dir=None):
    ds = load_dataset("evalplus/mbppplus", split=split, cache_dir=cache_dir)
    for item in ds:
        question = (
            f"Please provide a self-contained Python script that solves the "
            f"following problem in a markdown code block:\n```python\n"
            f"YOUR_PYTHON_CODE\n```:\n{item['prompt']}\n"
            f"Your answer will be tested on test cases like:\n"
            f"{item['test_list'][0]}\n{item['test_list'][1]}\n{item['test_list'][2]}"
        )
        answer = str(item["test"])
        yield {"question": question, "solution": answer, "gold": answer}


def load_humanevalplus(split="test", cache_dir=None):
    ds = load_dataset("evalplus/humanevalplus", split=split, cache_dir=cache_dir)
    for item in ds:
        question = (
            f"Please provide a self-contained Python script that solves the "
            f"following problem in a markdown code block:\n```python\n"
            f"YOUR_PYTHON_CODE\n```:\n{item['prompt']}"
        )
        raw_answer = str(item["test"])
        answer = raw_answer.replace("candidate", item["entry_point"])
        answer += f"\n\ncheck({item['entry_point']})"
        yield {"question": question, "solution": answer, "gold": answer}


def load_medqa(split=None, cache_dir=None):
    ds = load_dataset("json", data_files="./data/medqa.json", split="train")
    choice_map = {"0": "A", "1": "B", "2": "C", "3": "D"}
    for item in ds:
        question = item["query"]
        raw_answer = str(item["answer"])
        answer = ""
        for idx, op in enumerate(item["options"]):
            if raw_answer in op:
                answer = choice_map[str(idx)].lower()
                break
        gold = normalize_answer(answer)
        yield {"question": question, "solution": answer, "gold": gold}


LOADERS = {
    "gsm8k": load_gsm8k,
    "aime2024": load_aime2024,
    "aime2025": load_aime2025,
    "gpqa": load_gpqa_diamond,
    "arc_easy": load_arc_easy,
    "arc_challenge": load_arc_challenge,
    "mbppplus": load_mbppplus,
    "humanevalplus": load_humanevalplus,
    "medqa": load_medqa,
}

# Tasks whose answers are A-D multiple choice
MC_TASKS = {"arc_easy", "arc_challenge", "gpqa", "medqa"}
# Tasks whose answers are numeric (no A-D choices)
NUMERIC_TASKS = {"gsm8k", "aime2024", "aime2025"}
# Tasks whose answers are code
CODE_TASKS = {"mbppplus", "humanevalplus"}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_paraphrase_prompt(question: str, task: str) -> str:
    """Build the paraphrase prompt based on task type."""

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
        return f"""Paraphrase this math question in 3 different ways. The numerical answer must remain the same. Keep all numbers, equations, and constraints identical so the answer does not change. Only reword/restructure the problem statement.
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
        # fallback / winogrande etc.
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
# Parsing the 3-version output
# ---------------------------------------------------------------------------

def parse_versions(text: str) -> Dict[str, str]:
    """Parse [VERSION1] ... --- [VERSION2] ... --- [VERSION3] ... from model output."""
    result = {"version1": "", "version2": "", "version3": ""}

    # Try regex-based extraction first
    patterns = [
        (r"\[VERSION1\]\s*\n?(.*?)(?=\n---|\[VERSION2\])", "version1"),
        (r"\[VERSION2\]\s*\n?(.*?)(?=\n---|\[VERSION3\])", "version2"),
        (r"\[VERSION3\]\s*\n?(.*?)$", "version3"),
    ]
    for pat, key in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            result[key] = m.group(1).strip().strip("-").strip()

    # Fallback: split by --- if regex missed
    if not result["version1"] and "---" in text:
        parts = re.split(r"\n-{2,}\n", text)
        for i, key in enumerate(["version1", "version2", "version3"]):
            if i < len(parts):
                clean = re.sub(r"\[VERSION\d\]\s*", "", parts[i]).strip()
                result[key] = clean

    return result


# ---------------------------------------------------------------------------
# Anthropic API call with retry
# ---------------------------------------------------------------------------

def call_anthropic_api(prompt: str, max_retries: int = 5) -> str:
    """Call Anthropic Messages API with retries."""
    import urllib.request
    import urllib.error

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    url = "https://api.anthropic.com/v1/messages"

    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text_parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
                return "\n".join(text_parts)
        except (urllib.error.HTTPError, urllib.error.URLError, Exception) as e:
            wait = min(2 ** attempt * 2, 60)
            print(f"  [API retry {attempt+1}/{max_retries}] {e}  -- waiting {wait}s", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(f"Anthropic API failed after {max_retries} retries")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate 3 paraphrased versions of dataset questions via Claude API")
    parser.add_argument("--task", required=True, choices=list(LOADERS.keys()),
                        help="Dataset/task to process")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples to process (-1 = all)")
    parser.add_argument("--output_dir", type=str, default="/example_logs",
                        help="Output directory")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split override (default: task-specific)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial output file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.task}_paraphrased.jsonl.gz")

    # Load existing records if resuming
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
    loader_fn = LOADERS[args.task]
    split_overrides = {
        "gsm8k": "test",
        "aime2024": "train",
        "aime2025": "train",
        "gpqa": "test",
        "arc_easy": "test",
        "arc_challenge": "test",
        "mbppplus": "test",
        "humanevalplus": "test",
        "medqa": None,
    }
    split = args.split or split_overrides.get(args.task, "test")

    if split is not None:
        items = list(loader_fn(split=split))
    else:
        items = list(loader_fn())

    if args.max_samples > 0:
        items = items[: args.max_samples]

    total = len(items)
    print(f"Task: {args.task} | Total samples: {total} | Output: {out_path}")

    # Open output in write mode (rewrite with existing + new)
    with gzip.open(out_path, "wt", encoding="utf-8") as fout:
        # Write back existing records first
        for line in existing_records:
            fout.write(line + "\n")

        for task_id, item in enumerate(items):
            if task_id in existing_ids:
                continue

            question = item["question"]
            prompt = build_paraphrase_prompt(question, args.task)

            try:
                raw_output = call_anthropic_api(prompt)
            except RuntimeError as e:
                print(f"  [SKIP] task_id={task_id}: {e}", file=sys.stderr)
                # Write record with empty versions so we don't lose the original data
                record = {
                    "task_id": task_id,
                    "question": item["question"],
                    "solution": item["solution"],
                    "gold": item["gold"],
                    "version1": "",
                    "version2": "",
                    "version3": "",
                    "parse_error": str(e),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            versions = parse_versions(raw_output)

            record = {
                "task_id": task_id,
                "question": item["question"],
                "solution": item["solution"],
                "gold": item["gold"],
                "version1": versions["version1"],
                "version2": versions["version2"],
                "version3": versions["version3"],
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            # Progress
            done = task_id + 1
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] processed")

    print(f"Done. Saved to {out_path}")

    # Print a quick sanity check
    count = 0
    sample = None
    with gzip.open(out_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                if count == 1:
                    sample = json.loads(line)
    print(f"Total records: {count}")
    if sample:
        print("Sample record keys:", list(sample.keys()))
        print(f"  task_id: {sample['task_id']}")
        print(f"  gold: {sample['gold']}")
        v1_preview = sample.get("version1", "")[:100]
        print(f"  version1 (first 100 chars): {v1_preview}")


if __name__ == "__main__":
    main()
