#!/usr/bin/env python3
"""
analyze_metric_csv.py  (v2 — multi-source, CKA removed)
=======================================================
Post-process the CSVs from analyze_representations.py (v5 multi-source pooled
distribution comparison) into interpretable tables and figures under ./analysis.

Sources are (dataset, version) groups, e.g. gsm8k_orgn / gsm8k_ver1 /
gpqa_orgn / gpqa_ver1, and each pair is automatically classified into:
  * diff_semantic               : cross-domain   (e.g. gsm8k_* vs gpqa_*)
  * same_semantic_diff_surface  : paraphrase     (e.g. *_orgn vs *_ver1)
plus a language tag (same_lang / cross_lang).

Rules:
  * Metrics are NEVER mixed into a composite/weighted score.
  * cosine-family: similarity, higher == more similar.
  * mmd_rbf / l2_mean: distance, lower == more similar.
  * MMD scale is component-local: no cross-component absolute comparison.
  * l2_mean: same-layer comparison only.
  * CKA is fully removed (undefined for unpaired cross-domain clouds).

Plot color convention: the SAME pair is the SAME color across every figure;
components are distinguished by line style; each metric uses a fixed colormap.
"""
import argparse
import glob
import json
import os
from typing import Dict, List, Optional

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
# Colors / styles (consistent across figures)
# =====================================================================
_TAB = plt.get_cmap("tab10").colors
PAIR_COLORS: Dict[str, tuple] = {}
PAIR_ORDER: List[str] = []
COMPONENT_LS = {"q": "-", "k": "--", "v": ":", "head_output": "-", "residual": "-"}
METRIC_CMAP = {"maxmatch_cosine": "viridis", "mean_pairwise_cosine": "viridis",
               "head_output_maxmatch_cosine": "viridis", "residual_maxmatch_cosine": "viridis",
               "mmd_rbf": "magma", "head_output_mmd_rbf": "magma", "residual_mmd_rbf": "magma",
               "l2_mean": "cividis"}
_LANG = {"orgn": "en", "ver1": "zh", "ver2": "en", "ver3": "en"}


def pair_color(p):
    return PAIR_COLORS.get(p, (0.4, 0.4, 0.4))


def resolve_pairs(dfs) -> List[str]:
    present = set()
    for df in dfs:
        if df is not None and "pair" in df.columns:
            present |= set(df["pair"].dropna().unique())
    ordered = sorted(present)
    for i, p in enumerate(ordered):
        PAIR_COLORS[p] = _TAB[i % 10]
    return ordered


def classify_pair(pair: str) -> dict:
    """Parse 'gsm8k_orgn_vs_gpqa_ver1' into task/version/category fields."""
    try:
        a, b = pair.split("_vs_")
        ta, va = a.rsplit("_", 1)
        tb, vb = b.rsplit("_", 1)
    except ValueError:
        return dict(task_a="?", task_b="?", version_a="?", version_b="?",
                    same_task=False, same_version=False,
                    pair_category="unknown", language_pair="unknown")
    same_task, same_ver = ta == tb, va == vb
    if not same_task:
        cat = "diff_semantic"
    elif not same_ver:
        cat = "same_semantic_diff_surface"
    else:
        cat = "same"
    la, lb = _LANG.get(va, "?"), _LANG.get(vb, "?")
    return dict(task_a=ta, task_b=tb, version_a=va, version_b=vb,
                same_task=same_task, same_version=same_ver,
                pair_category=cat,
                language_pair="same_lang" if la == lb else "cross_lang")


def attach_category(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or "pair" not in df.columns:
        return df
    cats = df["pair"].map(classify_pair).apply(pd.Series)
    return pd.concat([df, cats[["pair_category", "language_pair",
                                "same_task", "same_version"]]], axis=1)


# =====================================================================
# IO
# =====================================================================
def find_one(input_dir, suffix, prefix=""):
    pat = f"{prefix}*{suffix}" if prefix else f"*{suffix}"
    hits = sorted(glob.glob(os.path.join(input_dir, pat)))
    return hits[0] if hits else None


def load_csv(path):
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for c in ["mean_pairwise_cosine", "maxmatch_cosine", "mmd_rbf", "l2_mean",
              "value", "n_tokens_a", "n_tokens_b", "n_items_a", "n_items_b",
              "layer", "head"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_dirs(out_dir):
    for sub in ["summary", "layerwise", "headwise", "heatmaps",
                "correlation", "figures"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def save_csv(df, out_dir, sub, name):
    path = os.path.join(out_dir, sub, name)
    df.to_csv(path, index=False)
    print(f"  CSV -> {os.path.join(sub, name)}  ({len(df)} rows)")


def round_df(df, n=6):
    num = df.select_dtypes(include=[np.number]).columns
    df = df.copy(); df[num] = df[num].round(n)
    return df


# =====================================================================
# Aggregation (no CKA)
# =====================================================================
def agg_metrics(df, keys, metrics):
    g = df.groupby(keys, dropna=False)
    out = g.size().rename("n_rows").reset_index()
    if "layer" in df.columns:
        out = out.merge(g["layer"].nunique().rename("n_layers").reset_index(), on=keys, how="left")
    if "head" in df.columns:
        out = out.merge(g["head"].nunique().rename("n_heads").reset_index(), on=keys, how="left")
    for m in metrics:
        if m not in df.columns:
            continue
        stat = g[m].agg(["mean", "std", "median"]).reset_index().rename(
            columns={"mean": f"{m}_mean", "std": f"{m}_std", "median": f"{m}_median"})
        out = out.merge(stat, on=keys, how="left")
    return out


def ranking(table, group_keys, sort_col, ascending, top_k):
    df = table[table[sort_col].notna()].copy()
    if df.empty:
        return df
    out = (df.sort_values(sort_col, ascending=ascending)
             .groupby(group_keys, dropna=False, group_keys=False).head(top_k))
    return out.sort_values(group_keys + [sort_col],
                           ascending=[True] * len(group_keys) + [ascending])


# =====================================================================
# Plot: layerwise lines (color = pair, fixed)
# =====================================================================
def lineplot_layerwise(df, metric_col, title, ylabel, out_path, component_col=None):
    comps = [None] if component_col is None else sorted(df[component_col].unique())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    legend_pairs, legend_comps = [], []
    for pair in PAIR_ORDER:
        dp = df[df["pair"] == pair]
        if dp.empty:
            continue
        legend_pairs.append(pair)
        for comp in comps:
            sub = dp if comp is None else dp[dp[component_col] == comp]
            if sub.empty:
                continue
            sub = sub.sort_values("layer")
            ls = COMPONENT_LS.get(comp, "-") if comp is not None else "-"
            ax.plot(sub["layer"], sub[metric_col], color=pair_color(pair),
                    linestyle=ls, lw=1.7, alpha=0.9)
            if comp is not None and comp not in legend_comps:
                legend_comps.append(comp)
    ax.set_xlabel("layer"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    h1 = [Line2D([0], [0], color=pair_color(p), lw=2) for p in legend_pairs]
    leg1 = ax.legend(h1, legend_pairs, title="pair", fontsize=7, loc="best")
    ax.add_artist(leg1)
    if legend_comps:
        h2 = [Line2D([0], [0], color="black", linestyle=COMPONENT_LS.get(c, "-"), lw=2)
              for c in legend_comps]
        ax.legend(h2, legend_comps, title="component", fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  fig -> {os.path.basename(out_path)}")


def qkv_layerwise_3panel(lw, metric_col, suptitle, ylabel, out_path, sharey=True):
    comps = ["q", "k", "v"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=sharey)
    legend_pairs = []
    for ax, comp in zip(axes, comps):
        cc = lw[lw["component"] == comp]
        for pair in PAIR_ORDER:
            sub = cc[cc["pair"] == pair]
            if sub.empty:
                continue
            sub = sub.sort_values("layer")
            ax.plot(sub["layer"], sub[metric_col], color=pair_color(pair),
                    lw=1.8, marker="o", ms=2.5)
            if pair not in legend_pairs:
                legend_pairs.append(pair)
        ax.set_title(f"component = {comp}", fontsize=10)
        ax.set_xlabel("layer"); ax.grid(True, alpha=0.2)
    axes[0].set_ylabel(ylabel)
    h = [Line2D([0], [0], color=pair_color(p), lw=2) for p in legend_pairs]
    if h:
        fig.legend(h, legend_pairs, title="pair", fontsize=8, loc="upper center",
                   ncol=min(len(legend_pairs), 6), bbox_to_anchor=(0.5, 1.07))
    fig.suptitle(suptitle, fontsize=10, y=1.14)
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  fig -> {os.path.basename(out_path)}")


# =====================================================================
# Section 1: Q/K/V
# =====================================================================
def analyze_qkv(df, out_dir, top_k, do_plots):
    metrics = ["mean_pairwise_cosine", "maxmatch_cosine", "mmd_rbf"]
    summ = attach_category(agg_metrics(df, ["pair", "component"], metrics))
    save_csv(round_df(summ), out_dir, "summary", "qkv_pair_component_summary.csv")

    lw = agg_metrics(df, ["pair", "component", "layer"], metrics)
    save_csv(round_df(lw), out_dir, "layerwise", "qkv_layerwise_trends.csv")

    ht = agg_metrics(df, ["pair", "component", "layer", "head"], metrics)
    htype = df.groupby(["pair", "component", "layer", "head"], dropna=False)["head_type"].first().reset_index()
    ht = ht.merge(htype, on=["pair", "component", "layer", "head"], how="left")
    save_csv(round_df(ht), out_dir, "headwise", "qkv_headwise_metric_table.csv")

    save_csv(round_df(ranking(ht, ["pair", "component"], "maxmatch_cosine_mean", True, top_k)),
             out_dir, "headwise", "qkv_lowest_maxmatch_heads.csv")
    save_csv(round_df(ranking(ht, ["pair", "component"], "mean_pairwise_cosine_mean", True, top_k)),
             out_dir, "headwise", "qkv_lowest_mean_pairwise_heads.csv")
    save_csv(round_df(ranking(ht, ["pair", "component"], "mmd_rbf_mean", False, top_k)),
             out_dir, "headwise", "qkv_highest_mmd_heads.csv")

    if do_plots:
        fdir = os.path.join(out_dir, "figures")
        qkv_layerwise_3panel(lw, "maxmatch_cosine_mean",
            "QKV layerwise maxmatch_cosine (higher = more similar)",
            "maxmatch_cosine", os.path.join(fdir, "qkv_layerwise_maxmatch_cosine.png"))
        qkv_layerwise_3panel(lw, "mean_pairwise_cosine_mean",
            "QKV layerwise mean_pairwise_cosine (higher = more similar)",
            "mean_pairwise_cosine", os.path.join(fdir, "qkv_layerwise_mean_pairwise_cosine.png"))
        qkv_layerwise_3panel(lw, "mmd_rbf_mean",
            "QKV layerwise mmd_rbf (higher = more different) | MMD is component-local",
            "mmd_rbf", os.path.join(fdir, "qkv_layerwise_mmd_rbf.png"), sharey=False)
    return df


# =====================================================================
# Section 2: head_output
# =====================================================================
def analyze_head_output(df, out_dir, top_k, do_plots):
    metrics = ["mean_pairwise_cosine", "maxmatch_cosine", "mmd_rbf", "l2_mean"]
    summ = attach_category(agg_metrics(df, ["pair"], metrics))
    save_csv(round_df(summ), out_dir, "summary", "head_output_pair_summary.csv")

    lw = agg_metrics(df, ["pair", "layer"], metrics)
    save_csv(round_df(lw), out_dir, "layerwise", "head_output_layerwise_trends.csv")

    l2tab = agg_metrics(df, ["layer", "head", "pair"], ["l2_mean"])
    save_csv(round_df(l2tab), out_dir, "headwise", "head_output_l2_same_layer_pair_table.csv")

    ht = agg_metrics(df, ["pair", "layer", "head"], metrics)
    save_csv(round_df(ht), out_dir, "headwise", "head_output_headwise_metric_table.csv")

    save_csv(round_df(ranking(ht, ["pair"], "maxmatch_cosine_mean", True, top_k)),
             out_dir, "headwise", "head_output_lowest_maxmatch_heads.csv")
    save_csv(round_df(ranking(ht, ["pair"], "mmd_rbf_mean", False, top_k)),
             out_dir, "headwise", "head_output_highest_mmd_heads.csv")
    save_csv(round_df(ranking(ht, ["pair"], "l2_mean_mean", False, top_k)),
             out_dir, "headwise", "head_output_highest_l2_heads.csv")

    if do_plots:
        fdir = os.path.join(out_dir, "figures")
        lineplot_layerwise(lw, "maxmatch_cosine_mean",
            "head_output layerwise maxmatch_cosine (higher = more similar)",
            "maxmatch_cosine", os.path.join(fdir, "head_output_layerwise_maxmatch_cosine.png"))
        lineplot_layerwise(lw, "mmd_rbf_mean",
            "head_output layerwise mmd_rbf (higher = more different) | component-local",
            "mmd_rbf", os.path.join(fdir, "head_output_layerwise_mmd_rbf.png"))
        lineplot_layerwise(lw, "l2_mean_mean",
            "head_output layerwise l2_mean | same-layer comparison only; do not compare across layers",
            "l2_mean", os.path.join(fdir, "head_output_layerwise_l2_mean.png"))
        _plot_l2_same_layer(lw, os.path.join(fdir, "head_output_l2_same_layer_pair.png"))
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
    ax.set_xlabel("layer"); ax.set_ylabel("l2_mean (per-layer scale)")
    ax.set_title("head_output l2_mean per layer — SAME-LAYER comparison only; "
                 "absolute scale differs across layers", fontsize=9)
    ax.legend(title="pair", fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  fig -> {os.path.basename(out_path)}")


# =====================================================================
# Section 3: residual
# =====================================================================
def analyze_residual(df, out_dir, do_plots):
    metrics = ["mean_pairwise_cosine", "maxmatch_cosine", "mmd_rbf"]
    summ = attach_category(agg_metrics(df, ["pair"], metrics))
    save_csv(round_df(summ), out_dir, "summary", "residual_pair_summary.csv")

    lw = agg_metrics(df, ["pair", "layer"], metrics)
    save_csv(round_df(lw), out_dir, "layerwise", "residual_layerwise_trends.csv")

    trans = []
    for pair in sorted(lw["pair"].unique()):
        sub = lw[lw["pair"] == pair].sort_values("layer").reset_index(drop=True)
        for i in range(len(sub) - 1):
            trans.append(dict(pair=pair,
                layer_from=int(sub.loc[i, "layer"]), layer_to=int(sub.loc[i + 1, "layer"]),
                delta_maxmatch_cosine=sub.loc[i + 1, "maxmatch_cosine_mean"] - sub.loc[i, "maxmatch_cosine_mean"],
                delta_mmd_rbf=sub.loc[i + 1, "mmd_rbf_mean"] - sub.loc[i, "mmd_rbf_mean"]))
    save_csv(round_df(pd.DataFrame(trans)), out_dir, "layerwise", "residual_layer_transition_trends.csv")

    if do_plots:
        fdir = os.path.join(out_dir, "figures")
        note = " | layer 0 = embedding, l+1 = block l output"
        lineplot_layerwise(lw, "maxmatch_cosine_mean",
            "residual layerwise maxmatch_cosine (higher = more similar)" + note,
            "maxmatch_cosine", os.path.join(fdir, "residual_layerwise_maxmatch_cosine.png"))
        lineplot_layerwise(lw, "mmd_rbf_mean",
            "residual layerwise mmd_rbf (higher = more different)" + note,
            "mmd_rbf", os.path.join(fdir, "residual_layerwise_mmd_rbf.png"))
    return df


# =====================================================================
# Section 4: pair-category contrast (the headline cross-domain vs paraphrase table)
# =====================================================================
def pair_category_summary(stages, out_dir, do_plots):
    rows = []
    for stage, df in stages.items():
        if df is None:
            continue
        d = attach_category(df)
        has_comp = "component" in d.columns
        comps = sorted(d["component"].unique()) if has_comp else [stage]
        for cat in sorted(d["pair_category"].unique()):
            for comp in comps:
                sub = d[d["pair_category"] == cat]
                if has_comp:
                    sub = sub[sub["component"] == comp]
                if sub.empty:
                    continue
                for m in ["maxmatch_cosine", "mean_pairwise_cosine", "mmd_rbf", "l2_mean"]:
                    if m not in sub.columns or sub[m].notna().sum() == 0:
                        continue
                    s = sub[m].dropna()
                    rows.append(dict(stage=stage, component=comp, pair_category=cat,
                        metric=m, mean=s.mean(), std=s.std(), median=s.median(),
                        n_pairs=sub["pair"].nunique(), n_rows=len(s)))
    prof = pd.DataFrame(rows)
    save_csv(round_df(prof), out_dir, "summary", "pair_category_summary.csv")

    if do_plots and not prof.empty:
        fdir = os.path.join(out_dir, "figures")
        for metric, direction in [("maxmatch_cosine", "higher = more similar"),
                                  ("mmd_rbf", "higher = more different")]:
            sub = prof[(prof.metric == metric)]
            if sub.empty:
                continue
            piv = sub.pivot_table(index="component", columns="pair_category",
                                  values="mean", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(piv.index)); w = 0.8 / max(1, len(piv.columns))
            cat_color = {"diff_semantic": "tab:red",
                         "same_semantic_diff_surface": "tab:blue", "same": "gray"}
            for j, cat in enumerate(piv.columns):
                ax.bar(x + j * w, piv[cat].values, width=w, label=cat,
                       color=cat_color.get(cat, None))
            ax.set_xticks(x + w * (len(piv.columns) - 1) / 2)
            ax.set_xticklabels(piv.index, rotation=20, ha="right")
            ax.set_ylabel(f"{metric} (mean)")
            ax.set_title(f"{metric} by pair_category and component ({direction})", fontsize=9)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(fdir, f"pair_category_{metric}.png"), dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"  fig -> pair_category_{metric}.png")
    return prof


# =====================================================================
# Section 5: token length summary (EN vs ZH source token counts)
# =====================================================================
def token_length_summary(stages, out_dir, do_plots):
    frames = []
    for stage, df in stages.items():
        if df is None or "n_tokens_a" not in df.columns:
            continue
        d = df.drop_duplicates(subset=["pair"])[["pair", "n_tokens_a", "n_tokens_b",
                                                 "n_items_a", "n_items_b"]].copy()
        d["stage"] = stage
        frames.append(d)
    if not frames:
        return
    tl = pd.concat(frames, ignore_index=True)
    tl["pooled_token_diff"] = (tl["n_tokens_a"] - tl["n_tokens_b"]).abs()
    tl = attach_category(tl)
    save_csv(round_df(tl), out_dir, "summary", "token_length_summary.csv")

    if do_plots:
        agg = tl.groupby("pair")["pooled_token_diff"].mean().reindex(
            [p for p in PAIR_ORDER if p in tl["pair"].unique()])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(range(len(agg)), agg.values, color=[pair_color(p) for p in agg.index])
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg.index, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("|pooled tokens A - B|")
        ax.set_title("Pooled token-count difference by pair (EN vs ZH differs most)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figures", "token_length_diff_by_pair.png"),
                    dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("  fig -> token_length_diff_by_pair.png")


# =====================================================================
# Section 6: pair metric profile (no composite score)
# =====================================================================
def build_pair_profile(stages, out_dir):
    notes = {"mean_pairwise_cosine": "cosine: higher means more similar",
             "maxmatch_cosine": "cosine: higher means more similar",
             "mmd_rbf": "mmd_rbf: lower means more similar; component-local scale",
             "l2_mean": "l2_mean: lower means more similar; same-layer comparison preferred"}
    rows = []
    for stage, df in stages.items():
        if df is None:
            continue
        d = attach_category(df)
        has_comp = "component" in d.columns
        comps = sorted(d["component"].unique()) if has_comp else [stage]
        for pair in sorted(d["pair"].unique()):
            cat = classify_pair(pair)["pair_category"]
            for comp in comps:
                sub = d[d["pair"] == pair]
                if has_comp:
                    sub = sub[sub["component"] == comp]
                if sub.empty:
                    continue
                for m, note in notes.items():
                    if m not in sub.columns or sub[m].notna().sum() == 0:
                        continue
                    s = sub[m].dropna()
                    rows.append(dict(pair=pair, pair_category=cat,
                        source_file_or_stage=stage, component=comp, metric=m,
                        mean=s.mean(), std=s.std(), median=s.median(),
                        min=s.min(), max=s.max(), n_rows=len(s), note=note))
    save_csv(round_df(pd.DataFrame(rows)), out_dir, "summary", "pair_metric_profile.csv")


# =====================================================================
# Section 7: heatmap long
# =====================================================================
def analyze_heatmaps(hm, out_dir, normalize, do_plots):
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
        is_dist = "mmd" in metric
        cbar = "higher = more different" if is_dist else "higher = more similar"
        gvmin = gvmax = None
        if normalize == "global":
            fin = grp["value_mean"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(fin):
                gvmin, gvmax = float(fin.min()), float(fin.max())
        for pair, sub in grp.groupby("pair"):
            is_res = component == "residual" or (sub["head"] < 0).all()
            if is_res:
                grid = sub.sort_values("layer")["value_mean"].values.reshape(-1, 1)
            else:
                nl = int(sub["layer"].max()) + 1; nh = int(sub["head"].max()) + 1
                grid = np.full((nl, nh), np.nan)
                for _, r in sub.iterrows():
                    grid[int(r["layer"]), int(r["head"])] = r["value_mean"]
            fig, ax = plt.subplots(figsize=(max(5, grid.shape[1] * 0.45), max(5, grid.shape[0] * 0.28)))
            im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest",
                           vmin=gvmin, vmax=gvmax)
            ax.set_title(f"{pair} | {component} | {metric}", fontsize=8)
            ax.set_ylabel("layer"); ax.set_xlabel("head" if not is_res else "")
            if is_res:
                ax.set_xticks([0]); ax.set_xticklabels(["residual"])
            fig.colorbar(im, ax=ax, shrink=0.7, label=cbar)
            fig.tight_layout()
            fig.savefig(os.path.join(fdir, f"heatmap_{pair}_{component}_{metric}.png"),
                        dpi=130, bbox_inches="tight")
            plt.close(fig)
    print("  fig -> heatmap_*.png (per pair/metric/component)")


# =====================================================================
# Section 8: correlations (within-metric only)
# =====================================================================
def _corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    n = len(x)
    if n < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan"), n
    if _HAS_SCIPY:
        return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0]), n
    return float(np.corrcoef(x, y)[0, 1]), float("nan"), n


def _short(metric):
    return "maxmatch" if "maxmatch" in metric else ("mmd" if "mmd" in metric else metric)


def corr_qkv_to_head_output(qkv, ho, group_size, out_dir, metric):
    ho_m = ho.groupby(["pair", "layer", "head"], dropna=False)[metric].mean().rename("ho_val").reset_index()
    files = {"q": f"q_to_head_output_{_short(metric)}_corr.csv",
             "k": f"k_to_head_output_{_short(metric)}_corr.csv",
             "v": f"v_to_head_output_{_short(metric)}_corr.csv"}
    for comp in ["q", "k", "v"]:
        sub = qkv[qkv["component"] == comp]
        if sub.empty:
            continue
        cm = sub.groupby(["pair", "layer", "head"], dropna=False)[metric].mean().rename("comp_val").reset_index()
        rows = []
        for pair in sorted(ho_m["pair"].unique()):
            hp = ho_m[ho_m["pair"] == pair]
            cp = cm[cm["pair"] == pair].copy()
            if comp == "q":
                merged = hp.merge(cp[["layer", "head", "comp_val"]], on=["layer", "head"], how="inner")
            else:
                hp2 = hp.copy(); hp2["kv_head"] = hp2["head"] // max(1, group_size)
                cp = cp.rename(columns={"head": "kv_head"})
                merged = hp2.merge(cp[["layer", "kv_head", "comp_val"]],
                                   on=["layer", "kv_head"], how="inner")
            pr, sr, n = _corr(merged["comp_val"], merged["ho_val"])
            rows.append(dict(pair=pair, component=comp, metric=metric,
                pair_category=classify_pair(pair)["pair_category"],
                pearson=pr, spearman=sr, n=n,
                note=("GQA kv->q mapped" if comp != "q" else "head aligned")))
        save_csv(round_df(pd.DataFrame(rows)), out_dir, "correlation", files[comp])


def corr_head_output_to_residual(ho, res, out_dir):
    for metric in ["maxmatch_cosine", "mmd_rbf"]:
        ho_l = ho.groupby(["pair", "layer"], dropna=False)[metric].mean().rename("ho_val").reset_index()
        res_l = res.groupby(["pair", "layer"], dropna=False)[metric].mean().rename("res_val").reset_index()
        rows = []
        for pair in sorted(ho_l["pair"].unique()):
            hp = ho_l[ho_l["pair"] == pair].copy(); hp["match_layer"] = hp["layer"] + 1
            rp = res_l[res_l["pair"] == pair].rename(columns={"layer": "match_layer"})
            merged = hp.merge(rp, on="match_layer", how="inner")
            pr, sr, n = _corr(merged["ho_val"], merged["res_val"])
            rows.append(dict(pair=pair, component="head_output->residual", metric=metric,
                pair_category=classify_pair(pair)["pair_category"],
                pearson=pr, spearman=sr, n=n,
                note="head_output layer l (head-mean) vs residual layer l+1"))
        save_csv(round_df(pd.DataFrame(rows)), out_dir, "correlation",
                 f"head_output_to_residual_{_short(metric)}_corr.csv")


# =====================================================================
# Section 9: report
# =====================================================================
def write_report(out_dir, files, meta, sizes, stages, cat_prof):
    L = []
    A = L.append
    A("# Multi-source representation comparison report\n")
    A("## 1. Input files\n")
    for k, v in files.items():
        A(f"- **{k}**: `{v}`" if v else f"- **{k}**: (not found)")
    A("\n## 2. Sources & data size\n")
    if meta and "sources" in meta:
        for s in meta["sources"]:
            A(f"- **{s['tag']}**: task={s['task']}, version={s['version']} "
              f"({s.get('language','?')}), items={s.get('n_items')}, "
              f"pooled_tokens={s.get('n_pooled_tokens')}")
    for k, v in sizes.items():
        A(f"- {k}: {v} metric rows")
    A(f"- comparison mode: `{meta.get('comparison_mode','?')}`" if meta else "")
    A("\n## 3. Pair categories\n")
    for p in PAIR_ORDER:
        c = classify_pair(p)
        A(f"- {p}: **{c['pair_category']}**, {c['language_pair']}")
    A("\n## 4. Headline contrast: cross-domain (diff_semantic) vs paraphrase "
      "(same_semantic_diff_surface)\n")
    if cat_prof is not None and not cat_prof.empty:
        for metric in ["maxmatch_cosine", "mmd_rbf"]:
            sub = cat_prof[cat_prof.metric == metric]
            if sub.empty:
                continue
            dirn = "higher=more similar" if "cosine" in metric else "higher=more different"
            A(f"\n**{metric}** ({dirn}), mean by category × component:")
            piv = sub.pivot_table(index="component", columns="pair_category",
                                  values="mean", aggfunc="mean")
            for comp in piv.index:
                parts = [f"{cat}={piv.loc[comp, cat]:.3f}" for cat in piv.columns
                         if pd.notna(piv.loc[comp, cat])]
                A(f"- {comp}: " + ", ".join(parts))
    A("\n## 5. Residual layer indexing\n")
    A("- layer 0 = embedding output; layer l+1 = transformer block l output.")
    A("\n## 6. Interpretation caveats\n")
    A("> No weighted or composite sensitivity score is computed. Metrics are analyzed "
      "separately because cosine, MMD, and L2 have different scales and directions.\n")
    A("> Comparison is distribution-level over pooled, template-stripped question tokens; "
      "there is no per-item pairing (required, since cross-domain sources do not correspond).\n")
    A("> MMD is a distribution distance. Its absolute scale depends on the component and the "
      "median-heuristic bandwidth, so cross-component absolute comparisons are avoided.\n")
    A("> CKA was removed: it requires paired equal-count samples, which are undefined for "
      "unpaired cross-domain token clouds.\n")
    A("> l2_mean is a same-layer comparison metric only; activation scale differs across layers.\n")
    with open(os.path.join(out_dir, "analysis_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("  report -> analysis_report.md")


# =====================================================================
# Main
# =====================================================================
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--input_dir", required=True)
    pa.add_argument("--out_dir", default="./analysis")
    pa.add_argument("--prefix", default="", help="Disambiguate when multiple runs share input_dir.")
    pa.add_argument("--top_k", type=int, default=20)
    pa.add_argument("--normalize_for_plot", choices=["none", "global"], default="none")
    pa.add_argument("--make_plots", action="store_true", default=True)
    pa.add_argument("--no_plots", action="store_true", default=False)
    args = pa.parse_args()

    do_plots = args.make_plots and not args.no_plots
    ensure_dirs(args.out_dir)

    files = {
        "qkv": find_one(args.input_dir, "_prefill_qkv_headwise_metrics.csv", args.prefix),
        "head_output": find_one(args.input_dir, "_prefill_head_output_metrics.csv", args.prefix),
        "residual": find_one(args.input_dir, "_prefill_residual_stream_metrics.csv", args.prefix),
        "heatmap_long": find_one(args.input_dir, "_heatmap_long.csv", args.prefix),
        "metadata": find_one(args.input_dir, "_run_metadata.json", args.prefix),
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

    qkv = load_csv(files["qkv"]); ho = load_csv(files["head_output"])
    res = load_csv(files["residual"]); hm = load_csv(files["heatmap_long"])

    global PAIR_ORDER
    PAIR_ORDER = resolve_pairs([qkv, ho, res, hm])
    print(f"Active pairs: {PAIR_ORDER}")
    for p in PAIR_ORDER:
        c = classify_pair(p)
        print(f"  {p}: {c['pair_category']} / {c['language_pair']}")

    stages, sizes = {}, {}
    print("\n[1] Q/K/V")
    if qkv is not None:
        stages["qkv"] = analyze_qkv(qkv, args.out_dir, args.top_k, do_plots); sizes["qkv"] = len(qkv)
    print("\n[2] head_output")
    if ho is not None:
        stages["head_output"] = analyze_head_output(ho, args.out_dir, args.top_k, do_plots); sizes["head_output"] = len(ho)
    print("\n[3] residual")
    if res is not None:
        stages["residual"] = analyze_residual(res, args.out_dir, do_plots); sizes["residual"] = len(res)

    print("\n[4] pair-category contrast")
    cat_prof = pair_category_summary(stages, args.out_dir, do_plots)
    print("\n[5] token length summary")
    token_length_summary(stages, args.out_dir, do_plots)
    print("\n[6] pair metric profile")
    build_pair_profile(stages, args.out_dir)
    print("\n[7] heatmaps")
    if hm is not None:
        analyze_heatmaps(hm, args.out_dir, args.normalize_for_plot, do_plots)
    print("\n[8] correlations")
    if qkv is not None and ho is not None:
        corr_qkv_to_head_output(qkv, ho, group_size, args.out_dir, "maxmatch_cosine")
        corr_qkv_to_head_output(qkv, ho, group_size, args.out_dir, "mmd_rbf")
    if ho is not None and res is not None:
        corr_head_output_to_residual(ho, res, args.out_dir)
    print("\n[9] report")
    write_report(args.out_dir, files, meta, sizes, stages, cat_prof)

    print(f"\nAll done. Outputs under {args.out_dir}/")


if __name__ == "__main__":
    main()
