#!/usr/bin/env python3
"""
analyze_metric_csv.py
=====================
Post-process the CSV outputs of analyze_representations.py into interpretable
analysis tables and figures under ./analysis (no raw tensors required).

Design rules (enforced throughout):
  * NO weighted / composite / z-score-sum "sensitivity score" is ever produced.
  * mean_pairwise_cosine, maxmatch_cosine, cka_linear, mmd_rbf, l2_mean are
    NEVER mixed into one number. Each is analyzed on its own scale & direction.
  * cosine-family: similarity, higher == more similar.
  * mmd_rbf / l2_mean: distance, lower == more similar.
  * cka_linear: only valid rows (cka_status in {matched, aligned_resampled}).
  * MMD absolute scale is component-local; cross-component absolute comparison
    is avoided (only correlations across components, never raw merges).
  * l2_mean is compared only within the same layer (activation scale differs
    across layers).

Plot color convention (consistency requirement):
  * the SAME pair always gets the SAME color in every figure;
  * components are distinguished by line style, not color;
  * each metric always uses the same colormap (cosine->viridis, mmd->magma,
    l2->cividis, cka->viridis).
"""
import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy.stats import pearsonr, spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# =====================================================================
# Constants: fixed color / style maps for cross-figure consistency
# =====================================================================
PAIR_ORDER = [
    "original_vs_version1", "original_vs_version2", "original_vs_version3",
    "version1_vs_version2", "version1_vs_version3", "version2_vs_version3",
]
_TAB = plt.get_cmap("tab10").colors
PAIR_COLORS = {p: _TAB[i % 10] for i, p in enumerate(PAIR_ORDER)}

COMPONENT_LS = {"q": "-", "k": "--", "v": ":",
                "head_output": "-", "residual": "-"}

METRIC_CMAP = {
    "maxmatch_cosine": "viridis", "mean_pairwise_cosine": "viridis",
    "head_output_maxmatch_cosine": "viridis", "residual_maxmatch_cosine": "viridis",
    "residual_cka_linear": "viridis", "cka_linear": "viridis",
    "mmd_rbf": "magma", "head_output_mmd_rbf": "magma",
    "l2_mean": "cividis",
}

METRIC_DIRECTION = {
    "mean_pairwise_cosine": "higher = more similar",
    "maxmatch_cosine": "higher = more similar",
    "cka_linear": "higher = more structurally similar; valid only",
    "mmd_rbf": "lower = more similar",
    "l2_mean": "lower = more similar; same-layer interpretation",
}

VALID_CKA_STATUS = {"matched", "aligned_resampled"}


def pair_color(p):
    return PAIR_COLORS.get(p, (0.4, 0.4, 0.4))


# =====================================================================
# IO helpers
# =====================================================================
def find_one(input_dir: str, suffix: str) -> Optional[str]:
    hits = sorted(glob.glob(os.path.join(input_dir, f"*{suffix}")))
    return hits[0] if hits else None


def load_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # ensure numeric metric columns are numeric (coerce stray strings to NaN)
    for c in ["mean_pairwise_cosine", "maxmatch_cosine", "cka_linear",
              "mmd_rbf", "l2_mean", "value", "n_tokens_a", "n_tokens_b",
              "prefix_skip", "layer", "head", "task_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_dirs(out_dir: str):
    for sub in ["summary", "layerwise", "headwise", "heatmaps",
                "validity", "correlation", "figures"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def save_csv(df: pd.DataFrame, out_dir: str, sub: str, name: str):
    path = os.path.join(out_dir, sub, name)
    df.to_csv(path, index=False)
    print(f"  CSV -> {os.path.join(sub, name)}  ({len(df)} rows)")


# =====================================================================
# Stat helpers
# =====================================================================
def add_cka_valid_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cka_status" not in df.columns:
        df["cka_status"] = "matched"
    df["cka_valid"] = df["cka_status"].isin(VALID_CKA_STATUS) & df["cka_linear"].notna()
    return df


def agg_metrics(df: pd.DataFrame, keys: List[str],
                metrics: List[str], with_cka: bool = True) -> pd.DataFrame:
    """Group by `keys`, return mean/std (and median where useful) per metric.
    CKA uses valid-only rows; cka_valid_rate computed over the full group."""
    g = df.groupby(keys, dropna=False)
    out = g.size().rename("n_rows").reset_index()

    def merge(col_df):
        nonlocal out
        out = out.merge(col_df, on=keys, how="left")

    if "task_id" in df.columns:
        merge(g["task_id"].nunique().rename("n_tasks").reset_index())
    if "layer" in df.columns:
        merge(g["layer"].nunique().rename("n_layers").reset_index())
    if "head" in df.columns:
        merge(g["head"].nunique().rename("n_heads").reset_index())

    for m in metrics:
        if m not in df.columns:
            continue
        stat = g[m].agg(["mean", "std", "median"]).reset_index()
        stat = stat.rename(columns={"mean": f"{m}_mean",
                                    "std": f"{m}_std",
                                    "median": f"{m}_median"})
        merge(stat)

    if with_cka and "cka_linear" in df.columns:
        valid = df[df["cka_valid"]]
        cnt_total = g.size().rename("cka_total_count").reset_index()
        cnt_valid = (df.groupby(keys, dropna=False)["cka_valid"].sum()
                     .rename("cka_valid_count").reset_index())
        merge(cnt_total)
        merge(cnt_valid)
        out["cka_valid_rate"] = (out["cka_valid_count"] /
                                 out["cka_total_count"].replace(0, np.nan))
        if len(valid):
            vg = valid.groupby(keys, dropna=False)["cka_linear"]
            vstat = vg.agg(["mean", "std", "median"]).reset_index()
            vstat = vstat.rename(columns={
                "mean": "cka_linear_mean_valid_only",
                "std": "cka_linear_std_valid_only",
                "median": "cka_linear_median_valid_only"})
            merge(vstat)
        else:
            out["cka_linear_mean_valid_only"] = np.nan
            out["cka_linear_std_valid_only"] = np.nan
            out["cka_linear_median_valid_only"] = np.nan
    return out


def round_df(df: pd.DataFrame, ndigits: int = 6) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).columns
    df = df.copy()
    df[num] = df[num].round(ndigits)
    return df


# =====================================================================
# Generic line-plot (x = layer), colored by pair, styled by component
# =====================================================================
def lineplot_layerwise(df, metric_col, title, ylabel, out_path,
                       component_col=None, cka_valid_col=None,
                       min_valid_rate=0.5):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    comps = [None] if component_col is None else sorted(df[component_col].unique())
    legend_pairs, legend_comps = [], []
    for pair in PAIR_ORDER:
        sub_pair = df[df["pair"] == pair]
        if sub_pair.empty:
            continue
        legend_pairs.append(pair)
        for comp in comps:
            sub = sub_pair if comp is None else sub_pair[sub_pair[component_col] == comp]
            if sub.empty:
                continue
            sub = sub.sort_values("layer")
            ls = COMPONENT_LS.get(comp, "-") if comp is not None else "-"
            alpha, lw = 0.9, 1.6
            if cka_valid_col is not None and cka_valid_col in sub.columns:
                vr = sub[cka_valid_col].mean()
                if pd.notna(vr) and vr < min_valid_rate:
                    alpha, ls = 0.35, ":"
            ax.plot(sub["layer"], sub[metric_col], color=pair_color(pair),
                    linestyle=ls, alpha=alpha, linewidth=lw)
            if comp is not None and comp not in legend_comps:
                legend_comps.append(comp)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    handles = [Line2D([0], [0], color=pair_color(p), lw=2) for p in legend_pairs]
    leg1 = ax.legend(handles, legend_pairs, title="pair", fontsize=7,
                     loc="best", framealpha=0.85)
    ax.add_artist(leg1)
    if legend_comps:
        chandles = [Line2D([0], [0], color="black",
                           linestyle=COMPONENT_LS.get(c, "-"), lw=2)
                    for c in legend_comps]
        ax.legend(chandles, legend_comps, title="component", fontsize=7,
                  loc="lower left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig -> {os.path.basename(out_path)}")


# =====================================================================
# Ranking helper (no scores: pure per-metric sort)
# =====================================================================
def ranking(table: pd.DataFrame, group_keys: List[str], sort_col: str,
            ascending: bool, top_k: int, require_cka_valid: bool = False) -> pd.DataFrame:
    df = table.copy()
    if require_cka_valid:
        df = df[df[sort_col].notna()]
        if "cka_valid_rate" in df.columns:
            df = df[df["cka_valid_rate"] > 0]
    df = df[df[sort_col].notna()]
    if df.empty:
        return df
    out = (df.sort_values(sort_col, ascending=ascending)
             .groupby(group_keys, dropna=False, group_keys=False)
             .head(top_k))
    return out.sort_values(group_keys + [sort_col], ascending=[True] * len(group_keys) + [ascending])


# =====================================================================
# Section 1: Q/K/V head-wise
# =====================================================================
def analyze_qkv(df, out_dir, top_k, do_plots):
    df = add_cka_valid_flag(df)
    metrics = ["mean_pairwise_cosine", "maxmatch_cosine", "mmd_rbf"]

    # A: pair-level summary (per pair, component)
    summ = agg_metrics(df, ["pair", "component"], metrics)
    save_csv(round_df(summ), out_dir, "summary", "qkv_pair_component_summary.csv")

    # B: layer-wise trends
    lw = agg_metrics(df, ["pair", "component", "layer"], metrics)
    keep = ["pair", "component", "layer", "n_rows", "n_tasks", "n_heads",
            "mean_pairwise_cosine_mean", "mean_pairwise_cosine_std",
            "maxmatch_cosine_mean", "maxmatch_cosine_std",
            "mmd_rbf_mean", "mmd_rbf_std",
            "cka_valid_rate", "cka_linear_mean_valid_only", "cka_linear_std_valid_only"]
    lw = lw[[c for c in keep if c in lw.columns]]
    save_csv(round_df(lw), out_dir, "layerwise", "qkv_layerwise_trends.csv")

    # C: head-wise table (task mean per pair, component, layer, head)
    ht = agg_metrics(df, ["pair", "component", "layer", "head"], metrics)
    # attach head_type (constant within component)
    htype = df.groupby(["pair", "component", "layer", "head"], dropna=False)["head_type"] \
              .first().reset_index()
    ht = ht.merge(htype, on=["pair", "component", "layer", "head"], how="left")
    cols = ["pair", "component", "layer", "head", "head_type", "n_tasks",
            "mean_pairwise_cosine_mean", "mean_pairwise_cosine_std",
            "maxmatch_cosine_mean", "maxmatch_cosine_std",
            "mmd_rbf_mean", "mmd_rbf_std",
            "cka_valid_rate", "cka_linear_mean_valid_only", "cka_linear_std_valid_only"]
    ht = ht[[c for c in cols if c in ht.columns]]
    save_csv(round_df(ht), out_dir, "headwise", "qkv_headwise_metric_table.csv")

    # D: per-metric rankings (no composite)
    save_csv(round_df(ranking(ht, ["pair", "component"], "maxmatch_cosine_mean",
             True, top_k)), out_dir, "headwise", "qkv_lowest_maxmatch_heads.csv")
    save_csv(round_df(ranking(ht, ["pair", "component"], "mean_pairwise_cosine_mean",
             True, top_k)), out_dir, "headwise", "qkv_lowest_mean_pairwise_heads.csv")
    save_csv(round_df(ranking(ht, ["pair", "component"], "mmd_rbf_mean",
             False, top_k)), out_dir, "headwise", "qkv_highest_mmd_heads.csv")
    save_csv(round_df(ranking(ht, ["pair", "component"], "cka_linear_mean_valid_only",
             True, top_k, require_cka_valid=True)), out_dir, "headwise",
             "qkv_lowest_cka_valid_heads.csv")

    if do_plots:
        fdir = os.path.join(out_dir, "figures")
        lineplot_layerwise(lw, "maxmatch_cosine_mean",
                           "QKV layerwise maxmatch_cosine (higher = more similar)",
                           "maxmatch_cosine", os.path.join(fdir, "qkv_layerwise_maxmatch_cosine.png"),
                           component_col="component")
        lineplot_layerwise(lw, "mean_pairwise_cosine_mean",
                           "QKV layerwise mean_pairwise_cosine (higher = more similar)",
                           "mean_pairwise_cosine", os.path.join(fdir, "qkv_layerwise_mean_pairwise_cosine.png"),
                           component_col="component")
        lineplot_layerwise(lw, "mmd_rbf_mean",
                           "QKV layerwise mmd_rbf (higher = more different) | "
                           "MMD values are component-local; avoid cross-component absolute comparison",
                           "mmd_rbf", os.path.join(fdir, "qkv_layerwise_mmd_rbf.png"),
                           component_col="component")
        lineplot_layerwise(lw, "cka_linear_mean_valid_only",
                           "QKV layerwise CKA (valid only; higher = more structurally similar)",
                           "cka_linear (valid only)",
                           os.path.join(fdir, "qkv_layerwise_cka_valid_only.png"),
                           component_col="component", cka_valid_col="cka_valid_rate")
    return df


# =====================================================================
# Section 2: head_output
# =====================================================================
def analyze_head_output(df, out_dir, top_k, do_plots):
    df = add_cka_valid_flag(df)
    metrics = ["mean_pairwise_cosine", "maxmatch_cosine", "mmd_rbf", "l2_mean"]

    # A: pair-level summary
    summ = agg_metrics(df, ["pair"], metrics)
    cols = ["pair", "n_rows", "n_tasks", "n_layers", "n_heads",
            "mean_pairwise_cosine_mean", "mean_pairwise_cosine_std",
            "maxmatch_cosine_mean", "maxmatch_cosine_std",
            "mmd_rbf_mean", "mmd_rbf_std",
            "l2_mean_mean", "l2_mean_std",
            "cka_valid_rate", "cka_linear_mean_valid_only", "cka_linear_std_valid_only"]
    summ = summ[[c for c in cols if c in summ.columns]]
    save_csv(round_df(summ), out_dir, "summary", "head_output_pair_summary.csv")

    # B: layer-wise trends
    lw = agg_metrics(df, ["pair", "layer"], metrics)
    save_csv(round_df(lw), out_dir, "layerwise", "head_output_layerwise_trends.csv")

    # C: same-layer l2 pair table
    l2tab = agg_metrics(df, ["layer", "head", "pair"], ["l2_mean"], with_cka=False)
    l2tab = l2tab[["layer", "head", "pair", "n_rows", "l2_mean_mean", "l2_mean_std"]]
    save_csv(round_df(l2tab), out_dir, "headwise", "head_output_l2_same_layer_pair_table.csv")

    # D: head-wise table
    ht = agg_metrics(df, ["pair", "layer", "head"], metrics)
    cols = ["pair", "layer", "head", "n_tasks",
            "mean_pairwise_cosine_mean", "mean_pairwise_cosine_std",
            "maxmatch_cosine_mean", "maxmatch_cosine_std",
            "mmd_rbf_mean", "mmd_rbf_std",
            "l2_mean_mean", "l2_mean_std",
            "cka_valid_rate", "cka_linear_mean_valid_only", "cka_linear_std_valid_only"]
    ht = ht[[c for c in cols if c in ht.columns]]
    save_csv(round_df(ht), out_dir, "headwise", "head_output_headwise_metric_table.csv")

    # E: rankings
    save_csv(round_df(ranking(ht, ["pair"], "maxmatch_cosine_mean", True, top_k)),
             out_dir, "headwise", "head_output_lowest_maxmatch_heads.csv")
    save_csv(round_df(ranking(ht, ["pair"], "mmd_rbf_mean", False, top_k)),
             out_dir, "headwise", "head_output_highest_mmd_heads.csv")
    save_csv(round_df(ranking(ht, ["pair"], "l2_mean_mean", False, top_k)),
             out_dir, "headwise", "head_output_highest_l2_heads.csv")
    save_csv(round_df(ranking(ht, ["pair"], "cka_linear_mean_valid_only", True,
             top_k, require_cka_valid=True)), out_dir, "headwise",
             "head_output_lowest_cka_valid_heads.csv")

    if do_plots:
        fdir = os.path.join(out_dir, "figures")
        lineplot_layerwise(lw, "maxmatch_cosine_mean",
                           "head_output layerwise maxmatch_cosine (higher = more similar)",
                           "maxmatch_cosine", os.path.join(fdir, "head_output_layerwise_maxmatch_cosine.png"))
        lineplot_layerwise(lw, "mmd_rbf_mean",
                           "head_output layerwise mmd_rbf (higher = more different) | component-local scale",
                           "mmd_rbf", os.path.join(fdir, "head_output_layerwise_mmd_rbf.png"))
        lineplot_layerwise(lw, "l2_mean_mean",
                           "head_output layerwise l2_mean | same-layer pair comparison only; "
                           "do not compare absolute scale across layers",
                           "l2_mean", os.path.join(fdir, "head_output_layerwise_l2_mean.png"))
        lineplot_layerwise(lw, "cka_linear_mean_valid_only",
                           "head_output layerwise CKA (valid only; higher = more structurally similar)",
                           "cka_linear (valid only)",
                           os.path.join(fdir, "head_output_layerwise_cka_valid_only.png"),
                           cka_valid_col="cka_valid_rate")
        # same-layer l2 pair comparison (line per pair; read vertically at fixed layer)
        _plot_l2_same_layer(lw, os.path.join(fdir, "head_output_l2_same_layer_pair_boxplot.png"))
    return df


def _plot_l2_same_layer(lw, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for pair in PAIR_ORDER:
        sub = lw[lw["pair"] == pair].sort_values("layer")
        if sub.empty:
            continue
        ax.plot(sub["layer"], sub["l2_mean_mean"], color=pair_color(pair),
                marker="o", ms=3, lw=1.5, label=pair)
        if "l2_mean_std" in sub.columns:
            ax.fill_between(sub["layer"],
                            sub["l2_mean_mean"] - sub["l2_mean_std"].fillna(0),
                            sub["l2_mean_mean"] + sub["l2_mean_std"].fillna(0),
                            color=pair_color(pair), alpha=0.10)
    ax.set_xlabel("layer")
    ax.set_ylabel("l2_mean (per-layer scale)")
    ax.set_title("head_output l2_mean per layer — SAME-LAYER pair comparison only; "
                 "absolute scale differs across layers", fontsize=9)
    ax.legend(title="pair", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig -> {os.path.basename(out_path)}")


# =====================================================================
# Section 3: residual stream
# =====================================================================
def analyze_residual(df, out_dir, do_plots):
    df = add_cka_valid_flag(df)
    metrics = ["mean_pairwise_cosine", "maxmatch_cosine", "mmd_rbf"]

    summ = agg_metrics(df, ["pair"], metrics)
    save_csv(round_df(summ), out_dir, "summary", "residual_pair_summary.csv")

    lw = agg_metrics(df, ["pair", "layer"], metrics)
    save_csv(round_df(lw), out_dir, "layerwise", "residual_layerwise_trends.csv")

    # C: transition (delta across consecutive layers); CKA excluded by default
    trans_rows = []
    for pair in sorted(lw["pair"].unique()):
        sub = lw[lw["pair"] == pair].sort_values("layer").reset_index(drop=True)
        for i in range(len(sub) - 1):
            trans_rows.append({
                "pair": pair,
                "layer_from": int(sub.loc[i, "layer"]),
                "layer_to": int(sub.loc[i + 1, "layer"]),
                "delta_maxmatch_cosine": (sub.loc[i + 1, "maxmatch_cosine_mean"]
                                          - sub.loc[i, "maxmatch_cosine_mean"]),
                "delta_mmd_rbf": (sub.loc[i + 1, "mmd_rbf_mean"]
                                  - sub.loc[i, "mmd_rbf_mean"]),
                "cka_valid_rate_min": float(np.nanmin([
                    sub.loc[i, "cka_valid_rate"], sub.loc[i + 1, "cka_valid_rate"]]))
                if "cka_valid_rate" in sub.columns else np.nan,
            })
    trans = pd.DataFrame(trans_rows)
    save_csv(round_df(trans), out_dir, "layerwise", "residual_layer_transition_trends.csv")

    if do_plots:
        fdir = os.path.join(out_dir, "figures")
        note = " | layer 0 = embedding output, layer l+1 = block l output"
        lineplot_layerwise(lw, "maxmatch_cosine_mean",
                           "residual layerwise maxmatch_cosine (higher = more similar)" + note,
                           "maxmatch_cosine", os.path.join(fdir, "residual_layerwise_maxmatch_cosine.png"))
        lineplot_layerwise(lw, "mmd_rbf_mean",
                           "residual layerwise mmd_rbf (higher = more different)" + note,
                           "mmd_rbf", os.path.join(fdir, "residual_layerwise_mmd_rbf.png"))
        lineplot_layerwise(lw, "cka_linear_mean_valid_only",
                           "residual layerwise CKA (valid only; higher = more structurally similar)" + note,
                           "cka_linear (valid only)",
                           os.path.join(fdir, "residual_layerwise_cka_valid_only.png"),
                           cka_valid_col="cka_valid_rate")
    return df


# =====================================================================
# Section 4: CKA validity
# =====================================================================
def analyze_cka_validity(stages: Dict[str, pd.DataFrame], out_dir, do_plots):
    frames = []
    for name, df in stages.items():
        if df is None:
            continue
        d = add_cka_valid_flag(df)
        d = d.copy()
        d["stage"] = name
        if "component" not in d.columns:
            d["component"] = name
        frames.append(d)
    if not frames:
        return
    alld = pd.concat(frames, ignore_index=True)

    # status summary
    status = (alld.groupby(["stage", "component", "pair", "cka_status"], dropna=False)
              .size().rename("count").reset_index())
    save_csv(status, out_dir, "validity", "cka_status_summary.csv")

    # valid rate by pair (and component)
    by_pair = (alld.groupby(["stage", "component", "pair"], dropna=False)["cka_valid"]
               .agg(["mean", "sum", "count"]).reset_index()
               .rename(columns={"mean": "cka_valid_rate", "sum": "cka_valid_count",
                                "count": "cka_total_count"}))
    # approximate (aligned_resampled) count, reported separately
    approx = (alld.assign(is_approx=alld["cka_status"] == "aligned_resampled")
              .groupby(["stage", "component", "pair"], dropna=False)["is_approx"]
              .sum().rename("aligned_resampled_count").reset_index())
    by_pair = by_pair.merge(approx, on=["stage", "component", "pair"], how="left")
    save_csv(round_df(by_pair), out_dir, "validity", "cka_valid_rate_by_pair.csv")

    # valid rate by layer
    by_layer = (alld.groupby(["stage", "component", "pair", "layer"], dropna=False)["cka_valid"]
                .mean().rename("cka_valid_rate").reset_index())
    save_csv(round_df(by_layer), out_dir, "validity", "cka_valid_rate_by_layer.csv")

    # token length summary (dedup to one row per task/pair/stage)
    tok_cols = ["n_tokens_a", "n_tokens_b"]
    if all(c in alld.columns for c in tok_cols):
        dd = alld.drop_duplicates(subset=["stage", "task_id", "pair"]).copy()
        dd["abs_token_len_diff"] = (dd["n_tokens_a"] - dd["n_tokens_b"]).abs()
        tl = dd.groupby(["stage", "pair"], dropna=False).agg(
            n_tokens_a_mean=("n_tokens_a", "mean"),
            n_tokens_a_std=("n_tokens_a", "std"),
            n_tokens_a_median=("n_tokens_a", "median"),
            n_tokens_b_mean=("n_tokens_b", "mean"),
            n_tokens_b_std=("n_tokens_b", "std"),
            n_tokens_b_median=("n_tokens_b", "median"),
            abs_token_len_diff_mean=("abs_token_len_diff", "mean"),
        ).reset_index()
        save_csv(round_df(tl), out_dir, "validity", "token_length_summary.csv")
    else:
        tl = pd.DataFrame()

    if do_plots:
        fdir = os.path.join(out_dir, "figures")
        # valid rate by pair (bar, colored by pair) — averaged over stages/components
        vr = by_pair.groupby("pair")["cka_valid_rate"].mean().reindex(
            [p for p in PAIR_ORDER if p in by_pair["pair"].unique()])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(range(len(vr)), vr.values,
               color=[pair_color(p) for p in vr.index])
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_xticks(range(len(vr)))
        ax.set_xticklabels(vr.index, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("CKA valid rate")
        ax.set_ylim(0, 1)
        ax.set_title("CKA valid rate by pair (low rate => CKA unreliable; "
                     "original_vs_version1 often lowest)", fontsize=9)
        for i, p in enumerate(vr.index):
            if pd.notna(vr.values[i]) and vr.values[i] < 0.5:
                ax.text(i, vr.values[i] + 0.02, "LOW", ha="center", color="red", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, "cka_valid_rate_by_pair.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("  fig -> cka_valid_rate_by_pair.png")

        if not tl.empty:
            td = tl.groupby("pair")["abs_token_len_diff_mean"].mean().reindex(
                [p for p in PAIR_ORDER if p in tl["pair"].unique()])
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(range(len(td)), td.values, color=[pair_color(p) for p in td.index])
            ax.set_xticks(range(len(td)))
            ax.set_xticklabels(td.index, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("mean |n_tokens_a - n_tokens_b|")
            ax.set_title("token length difference by pair (drives CKA length-mismatch skips)", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(fdir, "token_length_diff_by_pair.png"), dpi=140, bbox_inches="tight")
            plt.close(fig)
            print("  fig -> token_length_diff_by_pair.png")


# =====================================================================
# Section 5: pair metric profile (no ranking score)
# =====================================================================
def build_pair_profile(stages: Dict[str, pd.DataFrame], out_dir):
    notes = {
        "mean_pairwise_cosine": "cosine: higher means more similar",
        "maxmatch_cosine": "cosine: higher means more similar",
        "mmd_rbf": "mmd_rbf: lower means more similar",
        "l2_mean": "l2_mean: lower means more similar; same-layer comparison preferred",
        "cka_linear_valid_only": "cka: higher means more structurally similar; valid only",
        "cka_valid_rate": "fraction of rows with a valid CKA comparison",
    }
    metric_cols = {
        "mean_pairwise_cosine": "mean_pairwise_cosine",
        "maxmatch_cosine": "maxmatch_cosine",
        "mmd_rbf": "mmd_rbf",
        "l2_mean": "l2_mean",
    }
    rows = []
    for stage, df in stages.items():
        if df is None:
            continue
        d = add_cka_valid_flag(df)
        has_comp = "component" in d.columns
        comps = sorted(d["component"].unique()) if has_comp else [stage]
        for pair in sorted(d["pair"].unique()):
            for comp in comps:
                sub = d[d["pair"] == pair]
                if has_comp:
                    sub = sub[sub["component"] == comp]
                if sub.empty:
                    continue
                for mname, col in metric_cols.items():
                    if col not in sub.columns or sub[col].notna().sum() == 0:
                        continue
                    s = sub[col].dropna()
                    rows.append({
                        "pair": pair, "source_file_or_stage": stage, "component": comp,
                        "metric": mname, "mean": s.mean(), "std": s.std(),
                        "median": s.median(), "min": s.min(), "max": s.max(),
                        "n_rows": int(s.shape[0]), "note": notes[mname]})
                # cka valid-only
                v = sub[sub["cka_valid"]]["cka_linear"].dropna()
                if len(v):
                    rows.append({
                        "pair": pair, "source_file_or_stage": stage, "component": comp,
                        "metric": "cka_linear_valid_only", "mean": v.mean(), "std": v.std(),
                        "median": v.median(), "min": v.min(), "max": v.max(),
                        "n_rows": int(v.shape[0]), "note": notes["cka_linear_valid_only"]})
                # cka valid rate
                rows.append({
                    "pair": pair, "source_file_or_stage": stage, "component": comp,
                    "metric": "cka_valid_rate", "mean": float(sub["cka_valid"].mean()),
                    "std": np.nan, "median": np.nan, "min": np.nan, "max": np.nan,
                    "n_rows": int(sub.shape[0]), "note": notes["cka_valid_rate"]})
    prof = pd.DataFrame(rows)
    save_csv(round_df(prof), out_dir, "summary", "pair_metric_profile.csv")
    return prof


# =====================================================================
# Section 6: heatmap long
# =====================================================================
def analyze_heatmaps(hm, out_dir, normalize_for_plot, do_plots):
    keys = ["pair", "metric", "component", "layer", "head"]
    means = hm.groupby(keys, dropna=False)["value"].mean().rename("value_mean").reset_index()
    stds = hm.groupby(keys, dropna=False)["value"].std().rename("value_std").reset_index()
    save_csv(round_df(means), out_dir, "heatmaps", "heatmap_metric_means.csv")
    save_csv(round_df(stds), out_dir, "heatmaps", "heatmap_metric_stds.csv")

    if not do_plots:
        return

    fdir = os.path.join(out_dir, "figures")
    for (metric, component), grp in means.groupby(["metric", "component"]):
        cmap = METRIC_CMAP.get(metric, "viridis")
        is_distance = "mmd" in metric
        cbar_label = "higher = more different" if is_distance else (
            "higher = more similar" if "cosine" in metric else
            "higher = more structurally similar" if "cka" in metric else "value")
        # global normalization across all pairs of this (metric, component)
        gvmin = gvmax = None
        if normalize_for_plot == "global":
            finite = grp["value_mean"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite):
                gvmin, gvmax = float(finite.min()), float(finite.max())

        for pair, sub in grp.groupby("pair"):
            is_residual = component == "residual" or (sub["head"] < 0).all()
            if is_residual:
                arr = sub.sort_values("layer")["value_mean"].values.reshape(-1, 1)
                grid = arr
                xt = ["residual"]
            else:
                n_layer = int(sub["layer"].max()) + 1
                n_head = int(sub["head"].max()) + 1
                grid = np.full((n_layer, n_head), np.nan)
                for _, r in sub.iterrows():
                    grid[int(r["layer"]), int(r["head"])] = r["value_mean"]
                xt = None
            fig, ax = plt.subplots(figsize=(max(5, (grid.shape[1]) * 0.45), max(5, grid.shape[0] * 0.28)))
            im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest",
                           vmin=gvmin, vmax=gvmax)
            ax.set_title(f"{pair} | {component} | {metric}", fontsize=9)
            ax.set_ylabel("layer")
            ax.set_xlabel("head" if not is_residual else "")
            if xt is not None:
                ax.set_xticks([0]); ax.set_xticklabels(xt)
            fig.colorbar(im, ax=ax, shrink=0.7, label=cbar_label)
            fig.tight_layout()
            fname = f"heatmap_{pair}_{component}_{metric}.png"
            fig.savefig(os.path.join(fdir, fname), dpi=130, bbox_inches="tight")
            plt.close(fig)
    print("  fig -> heatmap_*.png (per pair/metric/component)")


# =====================================================================
# Section 7: correlations (within-metric only, never merge raw across components)
# =====================================================================
def _corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan"), n
    if _HAS_SCIPY:
        pr = pearsonr(x, y)[0]; sr = spearmanr(x, y)[0]
    else:
        pr = float(np.corrcoef(x, y)[0, 1]); sr = float("nan")
    return float(pr), float(sr), n


def _taskmean(df, keys, col):
    return df.groupby(keys, dropna=False)[col].mean().reset_index()


def corr_qkv_to_head_output(qkv, ho, group_size, out_dir, metric):
    """metric in {maxmatch_cosine, mmd_rbf}. Q maps head->head; K/V map via GQA."""
    ho_m = _taskmean(ho, ["pair", "layer", "head"], metric).rename(columns={metric: "ho_val"})
    files = {"q": f"q_to_head_output_{_short(metric)}_corr.csv",
             "k": f"k_to_head_output_{_short(metric)}_corr.csv",
             "v": f"v_to_head_output_{_short(metric)}_corr.csv"}
    for comp in ["q", "k", "v"]:
        sub = qkv[qkv["component"] == comp]
        if sub.empty:
            continue
        cm = _taskmean(sub, ["pair", "layer", "head"], metric).rename(columns={metric: "comp_val"})
        rows = []
        for pair in sorted(ho_m["pair"].unique()):
            hp = ho_m[ho_m["pair"] == pair]
            cp = cm[cm["pair"] == pair].copy()
            if comp == "q":
                cp["q_head"] = cp["head"]
            else:
                # map every query head to its kv head, then attach kv value
                hp_heads = hp[["layer", "head"]].copy()
                hp_heads["kv_head"] = hp_heads["head"] // max(1, group_size)
                cp = cp.rename(columns={"head": "kv_head"})
                cp = hp_heads.merge(cp[["layer", "kv_head", "comp_val"]],
                                    on=["layer", "kv_head"], how="left")
                cp["q_head"] = cp["head"]
            merged = hp.merge(
                cp[["layer", "q_head", "comp_val"]].rename(columns={"q_head": "head"}),
                on=["layer", "head"], how="inner")
            pr, sr, n = _corr(merged["comp_val"], merged["ho_val"])
            rows.append({"pair": pair, "component": comp, "metric": metric,
                         "pearson": pr, "spearman": sr, "n": n,
                         "note": f"{comp}->head_output same-metric corr; "
                                 + ("GQA kv->q mapped" if comp != "q" else "head aligned")})
        save_csv(round_df(pd.DataFrame(rows)), out_dir, "correlation", files[comp])


def corr_head_output_to_residual(ho, res, out_dir, min_cka):
    """Compress head_output layer l by head mean, match residual layer l+1."""
    for metric in ["maxmatch_cosine", "mmd_rbf"]:
        ho_l = _taskmean(ho, ["pair", "layer"], metric).rename(columns={metric: "ho_val"})
        res_l = _taskmean(res, ["pair", "layer"], metric).rename(columns={metric: "res_val"})
        rows = []
        for pair in sorted(ho_l["pair"].unique()):
            hp = ho_l[ho_l["pair"] == pair].copy()
            rp = res_l[res_l["pair"] == pair].copy()
            hp["match_layer"] = hp["layer"] + 1   # head_output[l] <-> residual[l+1]
            merged = hp.merge(rp.rename(columns={"layer": "match_layer"}),
                              on="match_layer", how="inner")
            pr, sr, n = _corr(merged["ho_val"], merged["res_val"])
            rows.append({"pair": pair, "component": "head_output->residual",
                         "metric": metric, "pearson": pr, "spearman": sr, "n": n,
                         "note": "head_output layer l (head-mean) vs residual layer l+1"})
        save_csv(round_df(pd.DataFrame(rows)), out_dir, "correlation",
                 f"head_output_to_residual_{_short(metric)}_corr.csv")


def _short(metric):
    return "maxmatch" if "maxmatch" in metric else ("mmd" if "mmd" in metric else metric)


# =====================================================================
# Section 8: markdown report
# =====================================================================
def write_report(out_dir, input_files, meta, sizes, stages, profile):
    lines = []
    A = lines.append
    A("# Representation-sensitivity analysis report\n")
    A("## 1. Input files\n")
    for k, v in input_files.items():
        A(f"- **{k}**: `{v}`" if v else f"- **{k}**: (not found)")
    A("")
    A("## 2. Data size\n")
    for k, v in sizes.items():
        A(f"- {k}: {v} rows")
    if meta:
        A(f"- model: `{meta.get('model_name')}`, task: `{meta.get('task')}`, "
          f"layers: {meta.get('num_layers')}, q_heads: {meta.get('num_attention_heads')}, "
          f"kv_heads: {meta.get('num_key_value_heads')}")
    A("")
    A("## 3. Token length difference by pair\n")
    tl_path = os.path.join(out_dir, "validity", "token_length_summary.csv")
    if os.path.exists(tl_path):
        tl = pd.read_csv(tl_path)
        agg = tl.groupby("pair")["abs_token_len_diff_mean"].mean().sort_values(ascending=False)
        for p, v in agg.items():
            A(f"- {p}: mean |Δtokens| = {v:.1f}")
    A("")
    A("## 4. CKA valid rate\n")
    vr_path = os.path.join(out_dir, "validity", "cka_valid_rate_by_pair.csv")
    if os.path.exists(vr_path):
        vr = pd.read_csv(vr_path).groupby("pair")["cka_valid_rate"].mean().sort_values()
        for p, v in vr.items():
            flag = "  **(LOW — CKA unreliable)**" if v < 0.5 else ""
            A(f"- {p}: valid rate = {v:.2f}{flag}")
    A("")
    A("## 5. maxmatch_cosine trend by pair (higher = more similar)\n")
    _report_metric_means(A, stages, "maxmatch_cosine")
    A("\n## 6. MMD trend by pair (lower = more similar; component-local scale)\n")
    _report_metric_means(A, stages, "mmd_rbf")
    A("")
    A("## 7. head_output l2_mean caveat\n")
    A("- l2_mean should not be used for direct cross-layer comparison; it is a "
      "same-layer pair comparison metric only (activation scale differs across layers).")
    A("")
    A("## 8. Residual stream layer trend\n")
    A("- Residual layer indexing: **layer 0 = embedding output; layer l+1 = transformer block l output.**")
    res = stages.get("residual")
    if res is not None:
        rl = add_cka_valid_flag(res).groupby("layer")["maxmatch_cosine"].mean()
        if len(rl):
            A(f"- maxmatch_cosine spans {rl.min():.3f} (layer {int(rl.idxmin())}) "
              f"to {rl.max():.3f} (layer {int(rl.idxmax())}) across layers.")
    A("")
    A("## 9. Sensitive-head rankings\n")
    A("- Rankings are provided **per metric only** (lowest maxmatch / highest MMD / "
      "highest L2 / lowest valid-CKA). No composite ranking across metrics is produced.")
    A("")
    A("## 10. Interpretation caveats\n")
    A("> No weighted or composite sensitivity score is computed. Metrics are analyzed "
      "separately because cosine, CKA, MMD, and L2 have different scales, directions, "
      "and validity conditions.\n")
    A("> CKA is interpreted only when cka_status indicates a valid comparison. "
      "Length-mismatched or resampled CKA values should not be treated as primary evidence.\n")
    A("> MMD is a distribution distance. Its absolute scale depends on the component and "
      "the median-heuristic bandwidth, so cross-component absolute comparisons are avoided.\n")
    path = os.path.join(out_dir, "analysis_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  report -> analysis_report.md")


def _report_metric_means(A, stages, metric):
    for stage in ["qkv", "head_output", "residual"]:
        df = stages.get(stage)
        if df is None or metric not in df.columns:
            continue
        if stage == "qkv":
            g = df.groupby(["pair", "component"])[metric].mean()
            for (p, c), v in g.items():
                A(f"- [{stage}/{c}] {p}: {v:.4f}")
        else:
            g = df.groupby("pair")[metric].mean()
            for p, v in g.items():
                A(f"- [{stage}] {p}: {v:.4f}")


# =====================================================================
# Main
# =====================================================================
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--input_dir", required=True)
    pa.add_argument("--out_dir", default="./analysis")
    pa.add_argument("--top_k", type=int, default=20)
    pa.add_argument("--normalize_for_plot", choices=["none", "global"], default="none")
    pa.add_argument("--min_cka_valid_rate", type=float, default=0.5)
    pa.add_argument("--make_plots", action="store_true", default=True)
    pa.add_argument("--no_plots", action="store_true", default=False)
    args = pa.parse_args()

    do_plots = args.make_plots and not args.no_plots
    ensure_dirs(args.out_dir)

    files = {
        "qkv": find_one(args.input_dir, "_prefill_qkv_headwise_metrics.csv"),
        "head_output": find_one(args.input_dir, "_prefill_head_output_metrics.csv"),
        "residual": find_one(args.input_dir, "_prefill_residual_stream_metrics.csv"),
        "heatmap_long": find_one(args.input_dir, "_heatmap_long.csv"),
        "metadata": find_one(args.input_dir, "_run_metadata.json"),
    }
    print("Discovered input files:")
    for k, v in files.items():
        print(f"  {k}: {v}")

    meta = {}
    if files["metadata"]:
        with open(files["metadata"], encoding="utf-8") as f:
            meta = json.load(f)
    n_q = int(meta.get("num_attention_heads", 0)) if meta else 0
    n_kv = int(meta.get("num_key_value_heads", 0)) if meta else 0
    group_size = (n_q // n_kv) if (n_q and n_kv) else 1

    qkv = load_csv(files["qkv"])
    ho = load_csv(files["head_output"])
    res = load_csv(files["residual"])
    hm = load_csv(files["heatmap_long"])

    stages = {}
    sizes = {}

    print("\n[1] Q/K/V head-wise analysis")
    if qkv is not None:
        stages["qkv"] = analyze_qkv(qkv, args.out_dir, args.top_k, do_plots)
        sizes["qkv"] = len(qkv)
    else:
        print("  (qkv CSV not found — skipped)")

    print("\n[2] head_output analysis")
    if ho is not None:
        stages["head_output"] = analyze_head_output(ho, args.out_dir, args.top_k, do_plots)
        sizes["head_output"] = len(ho)
    else:
        print("  (head_output CSV not found — skipped)")

    print("\n[3] residual stream analysis")
    if res is not None:
        stages["residual"] = analyze_residual(res, args.out_dir, do_plots)
        sizes["residual"] = len(res)
    else:
        print("  (residual CSV not found — skipped)")

    print("\n[4] CKA validity analysis")
    analyze_cka_validity({"qkv": stages.get("qkv"),
                          "head_output": stages.get("head_output"),
                          "residual": stages.get("residual")},
                         args.out_dir, do_plots)

    print("\n[5] pair metric profile")
    profile = build_pair_profile({"qkv": stages.get("qkv"),
                                  "head_output": stages.get("head_output"),
                                  "residual": stages.get("residual")}, args.out_dir)

    print("\n[6] heatmap long analysis")
    if hm is not None:
        analyze_heatmaps(hm, args.out_dir, args.normalize_for_plot, do_plots)
    else:
        print("  (heatmap_long CSV not found — skipped)")

    print("\n[7] correlation analysis (within-metric only)")
    if qkv is not None and ho is not None:
        corr_qkv_to_head_output(stages["qkv"], stages["head_output"], group_size,
                                args.out_dir, "maxmatch_cosine")
        corr_qkv_to_head_output(stages["qkv"], stages["head_output"], group_size,
                                args.out_dir, "mmd_rbf")
    if ho is not None and res is not None:
        corr_head_output_to_residual(stages["head_output"], stages["residual"],
                                     args.out_dir, args.min_cka_valid_rate)

    print("\n[8] report")
    write_report(args.out_dir, files, meta, sizes, stages, profile)

    print(f"\nAll done. Outputs under {args.out_dir}/")


if __name__ == "__main__":
    main()
