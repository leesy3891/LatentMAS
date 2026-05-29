"""
Plot paraphrase analysis results from ./paraphrase_data/*_analysis.txt

For each analysis file, generates:
  - cosine_by_layer:       3x2 subplots (6 versions), x=layer
  - js_by_layer:           3x2 subplots (6 versions), x=layer
  - cosine_by_decode_step: 3x2 subplots (6 versions), x=decode_step  (skip if prefill)
  - js_by_decode_step:     3x2 subplots (6 versions), x=decode_step  (skip if prefill)

All 6 subplots within a figure share identical y-axis scale and ticks.
Standard deviation is shown as a shaded band around the mean line.

Usage:
    python plot_paraphrase.py                                        # all *_analysis.txt
    python plot_paraphrase.py ./paraphrase_data/some_analysis.txt    # single file
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERSION_ORDER = ["version1", "version2", "version3", "ver1to2", "ver1to3", "ver2to3"]
VERSION_LABELS = {
    "version1": "Version 1",
    "version2": "Version 2",
    "version3": "Version 3",
    "ver1to2":  "Ver 1→2",
    "ver1to3":  "Ver 1→3",
    "ver2to3":  "Ver 2→3",
}
COLORS = {
    "version1": "#1f77b4",
    "version2": "#ff7f0e",
    "version3": "#2ca02c",
    "ver1to2":  "#d62728",
    "ver1to3":  "#9467bd",
    "ver2to3":  "#8c564b",
}


def parse_analysis(filepath: str) -> dict:
    """
    Parse an analysis .txt file into a nested dict:
        result[section_key] = { version: { x_val: (mean, std) } }
    where section_key is e.g. 'cosine_similarity by layer'
    """
    result = {}
    current_section = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("=" * 10):
                continue
            if line.startswith("cosine_similarity") or line.startswith("js_divergence"):
                current_section = line.strip()
                result[current_section] = {}
                continue
            if current_section is None:
                continue
            if line == "" or line.startswith("layer\t") or line.startswith("decode_step\t"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            x_val, version, mean, std = parts
            x_val = int(x_val)
            mean = float(mean)
            std = float(std)
            if version not in result[current_section]:
                result[current_section][version] = {}
            result[current_section][version][x_val] = (mean, std)

    return result


def plot_section(data_section: dict, x_label: str, y_label: str, suptitle: str,
                 save_path: str):
    """
    Draw a 3x2 figure. Each subplot = one version.
    data_section: { version: { x_val: (mean, std) } }
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=True)
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.98)

    # Determine global y range across all versions for uniform scale
    all_lo, all_hi = [], []
    for ver in VERSION_ORDER:
        if ver not in data_section:
            continue
        for x_val, (mean, std) in data_section[ver].items():
            all_lo.append(mean - std)
            all_hi.append(mean + std)

    if not all_lo:
        plt.close(fig)
        return

    y_min = min(all_lo)
    y_max = max(all_hi)
    y_margin = (y_max - y_min) * 0.08
    y_min = max(0, y_min - y_margin)
    y_max = y_max + y_margin

    # Compute nice tick step
    y_range = y_max - y_min
    if y_range <= 0.15:
        tick_step = 0.02
    elif y_range <= 0.4:
        tick_step = 0.05
    elif y_range <= 1.0:
        tick_step = 0.1
    else:
        tick_step = 0.2

    tick_min = np.floor(y_min / tick_step) * tick_step
    tick_max = np.ceil(y_max / tick_step) * tick_step
    y_ticks = np.arange(tick_min, tick_max + tick_step / 2, tick_step)

    for idx, ver in enumerate(VERSION_ORDER):
        row, col = divmod(idx, 2)
        ax = axes[row][col]

        if ver not in data_section or not data_section[ver]:
            ax.set_title(VERSION_LABELS.get(ver, ver), fontsize=12)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray")
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(y_ticks)
            continue

        xs = sorted(data_section[ver].keys())
        means = np.array([data_section[ver][x][0] for x in xs])
        stds = np.array([data_section[ver][x][1] for x in xs])

        color = COLORS[ver]
        ax.plot(xs, means, color=color, linewidth=1.5, label="mean")
        ax.fill_between(xs, means - stds, means + stds,
                        color=color, alpha=0.2, label="±1 std")

        ax.set_title(VERSION_LABELS.get(ver, ver), fontsize=12, fontweight="bold")
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=8, loc="lower left")

        if row == 2:
            ax.set_xlabel(x_label, fontsize=10)
        if col == 0:
            ax.set_ylabel(y_label, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


def process_file(txt_path: str):
    """Process one analysis .txt file and generate plot PNGs."""
    basename = os.path.basename(txt_path)
    out_dir = os.path.dirname(txt_path)
    is_prefill = "prefill" in basename.lower()
    stem = os.path.splitext(basename)[0]  # e.g. text_mas_gpqa_Qwen3-8B_decode_analysis

    data = parse_analysis(txt_path)
    if not data:
        print(f"  [WARNING] No data parsed from {txt_path}")
        return

    print(f"Processing: {txt_path}")
    print(f"  Sections found: {list(data.keys())}")

    # Map section key → (x_label, y_label, filename_suffix)
    section_configs = {
        "cosine_similarity by layer":       ("Layer", "Cosine Similarity", "cos_layer"),
        "cosine_similarity by decode_step": ("Decode Step", "Cosine Similarity", "cos_decode_step"),
        "js_divergence by layer":           ("Layer", "JS Divergence", "js_layer"),
        "js_divergence by decode_step":     ("Decode Step", "JS Divergence", "js_decode_step"),
    }

    for section_key, (x_lab, y_lab, suffix) in section_configs.items():
        if section_key not in data:
            continue
        if is_prefill and "decode_step" in section_key:
            continue

        title = f"{stem}\n{y_lab} by {x_lab}"
        save_path = os.path.join(out_dir, f"{stem}_{suffix}.png")
        plot_section(data[section_key], x_lab, y_lab, title, save_path)


def main():
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = sorted(glob.glob("./paraphrase_data/*_analysis.txt"))
        if not targets:
            print("No *_analysis.txt files found in ./paraphrase_data/")
            return

    for path in targets:
        process_file(path)


if __name__ == "__main__":
    main()
