"""
decompress_jsonl.py
-------------------
Decompress .jsonl.gz files to .jsonl in the same directory.

Usage:
  python decompress_jsonl.py ./example_logs/gpqa_paraphrased.jsonl.gz
  python decompress_jsonl.py ./example_logs/*.jsonl.gz
"""

import gzip
import glob
import sys
import os


def decompress(path: str):
    if not path.endswith(".jsonl.gz"):
        print(f"[SKIP] Not a .jsonl.gz file: {path}")
        return
    out_path = path[:-3]  # strip .gz
    with gzip.open(path, "rt", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line)
    in_size = os.path.getsize(path)
    out_size = os.path.getsize(out_path)
    print(f"{path} ({in_size/1024:.1f}KB) -> {out_path} ({out_size/1024:.1f}KB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python decompress_jsonl.py <file.jsonl.gz> [...]")
        sys.exit(1)
    paths = []
    for arg in sys.argv[1:]:
        paths.extend(glob.glob(arg))
    if not paths:
        print("No matching files found.")
        sys.exit(1)
    for p in paths:
        decompress(p)
