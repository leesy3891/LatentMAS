"""
Paraphrase similarity CSV analysis script.

Reads *_record.csv files from ./plots/, computes per-layer and per-decode_step
statistics (mean, std) for cosine similarity and JS divergence across 6 version
columns, and writes a parseable .txt analysis file to ./paraphrase_data/.

Usage:
    python analyze_paraphrase.py                          # process all *_record.csv in ./plots/
    python analyze_paraphrase.py path/to/file_record.csv  # process a single file
"""

import os
import sys
import glob
import pandas as pd
import numpy as np


# The 6 version prefixes to analyse
VERSION_PREFIXES = ["version1", "version2", "version3", "ver1to2", "ver1to3", "ver2to3"]


def analyse_file(csv_path: str, output_dir: str = "./paraphrase_data"):
    """Analyse a single *_record.csv and write results to output_dir."""
    basename = os.path.basename(csv_path)
    is_prefill = "prefill" in basename.lower()

    # Build output filename: replace 'record' with 'analysis', change ext to .txt
    out_name = basename.replace("record", "analysis")
    out_name = os.path.splitext(out_name)[0] + ".txt"
    out_path = os.path.join(output_dir, out_name)

    # ── Auto-detect separator and read CSV ──
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if "\t" in first_line:
        sep = "\t"
    elif "," in first_line:
        sep = ","
    else:
        sep = None  # let pandas sniff

    df = pd.read_csv(csv_path, sep=sep, engine="python" if sep is None else "c")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Debug: print detected columns
    print(f"  separator: {repr(sep)}")
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")

    # Identify available columns
    cos_cols = {p: f"{p}_cos" for p in VERSION_PREFIXES if f"{p}_cos" in df.columns}
    js_cols  = {p: f"{p}_js"  for p in VERSION_PREFIXES if f"{p}_js"  in df.columns}

    if not cos_cols and not js_cols:
        print(f"  [WARNING] No matching version columns found! Skipping.")
        return

    has_decode_step = "decode_step" in df.columns and not is_prefill

    lines: list[str] = []

    def _header(title: str):
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)

    def _write_group_stats(group_col: str, metric_cols: dict[str, str], metric_label: str):
        """Compute mean/std grouped by group_col for every version prefix."""
        _header(f"{metric_label} by {group_col}")
        lines.append(f"{group_col}\tversion\tmean\tstd")
        for prefix, col in sorted(metric_cols.items()):
            grouped = df.groupby(group_col)[col].agg(["mean", "std"]).reset_index()
            grouped["std"] = grouped["std"].fillna(0.0)
            for _, row in grouped.iterrows():
                lines.append(
                    f"{int(row[group_col])}\t{prefix}\t{row['mean']:.6f}\t{row['std']:.6f}"
                )
        lines.append("")

    # 1. Per-layer cosine similarity
    _write_group_stats("layer", cos_cols, "cosine_similarity")

    # 2. Per-decode_step cosine similarity (skip for prefill)
    if has_decode_step:
        _write_group_stats("decode_step", cos_cols, "cosine_similarity")

    # 3. Per-layer JS divergence
    _write_group_stats("layer", js_cols, "js_divergence")

    # 4. Per-decode_step JS divergence (skip for prefill)
    if has_decode_step:
        _write_group_stats("decode_step", js_cols, "js_divergence")

    # ── Write output ──
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[done] {csv_path}  →  {out_path}")


def main():
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = sorted(glob.glob("./plots/*_record.csv"))
        if not targets:
            print("No *_record.csv files found in ./plots/")
            return

    for path in targets:
        analyse_file(path)


if __name__ == "__main__":
    main()
