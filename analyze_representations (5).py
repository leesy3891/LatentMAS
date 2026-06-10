#!/usr/bin/env python3
"""
analyze_representations.py  (v5 — multi-source pooled distribution comparison)
=============================================================================
Compare the PREFILL representations of several *source groups*, where each
source is a (dataset, version) combination, e.g.

    gsm8k_orgn, gsm8k_ver1, gpqa_orgn, gpqa_ver1

The 6 source-pairs span two contrasts:
  * different semantics   : gsm8k_* vs gpqa_*           (cross-domain)
  * same semantics, diff  : *_orgn vs *_ver1            (paraphrase / EN<->ZH)
    surface / language

Because cross-domain sources have NO per-item correspondence, the comparison is
done at the **distribution level**: for each source we pool the question-specific
token representations (template stripped) per layer/head, then compare two
sources' pooled clouds with unpaired metrics:

    mean_pairwise_cosine, maxmatch_cosine, mmd_rbf   (+ l2_mean for head_output)

CKA is intentionally NOT computed (it requires paired equal-count samples, which
do not exist across domains). latent_mas is fully removed; text_mas is kept only
as a formal stub.

Measurement points (per prefill forward pass, via read-only hooks):
  * Q/K/V        : attention input subspace  (Q/K after RoPE, V raw)
  * head_output  : attn_probs @ V, before o_proj
  * residual     : hidden_states[l]; index 0 = embedding, l+1 = block l output
"""
import argparse
import csv
import gc
import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ModelWrapper
from prompts import build_agent_messages_single_agent

_VER_SHORT = {"original": "orgn", "version1": "ver1",
              "version2": "ver2", "version3": "ver3"}


# =====================================================================
# Metric primitives (unpaired; operate on [n_tokens, dim] clouds). No CKA.
# =====================================================================
def rowwise_l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def mean_pairwise_cosine(X: np.ndarray, Y: np.ndarray) -> float:
    if X.size == 0 or Y.size == 0:
        return float("nan")
    Xn, Yn = rowwise_l2_normalize(X), rowwise_l2_normalize(Y)
    return float((Xn @ Yn.T).mean())


def maxmatch_cosine(X: np.ndarray, Y: np.ndarray) -> float:
    if X.size == 0 or Y.size == 0:
        return float("nan")
    Xn, Yn = rowwise_l2_normalize(X), rowwise_l2_normalize(Y)
    C = Xn @ Yn.T
    return float(0.5 * (C.max(axis=1).mean() + C.max(axis=0).mean()))


def l2_mean(X: np.ndarray, Y: np.ndarray) -> float:
    if X.size == 0 or Y.size == 0:
        return float("nan")
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    xx = (X * X).sum(1)[:, None]; yy = (Y * Y).sum(1)[None, :]
    d2 = np.clip(xx + yy - 2.0 * (X @ Y.T), 0.0, None)
    return float(np.sqrt(d2).mean())


def rbf_mmd2(X: np.ndarray, Y: np.ndarray, max_tokens: int = 256,
             rng: Optional[np.random.Generator] = None) -> float:
    """Biased RBF-MMD^2 with median-heuristic bandwidth. Distance: smaller==closer."""
    if X.size == 0 or Y.size == 0:
        return float("nan")
    if rng is None:
        rng = np.random.default_rng(0)
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    if X.shape[0] > max_tokens:
        X = X[np.sort(rng.choice(X.shape[0], max_tokens, replace=False))]
    if Y.shape[0] > max_tokens:
        Y = Y[np.sort(rng.choice(Y.shape[0], max_tokens, replace=False))]

    def sq(A, B):
        aa = (A * A).sum(1)[:, None]; bb = (B * B).sum(1)[None, :]
        return np.clip(aa + bb - 2.0 * (A @ B.T), 0.0, None)

    dxx, dyy, dxy = sq(X, X), sq(Y, Y), sq(X, Y)
    pooled = np.concatenate([dxx[np.triu_indices_from(dxx, 1)],
                             dyy[np.triu_indices_from(dyy, 1)], dxy.ravel()])
    med = np.median(pooled[pooled > 0]) if np.any(pooled > 0) else 1.0
    med = med if med > 1e-12 else 1.0
    g = 1.0 / med
    mmd2 = np.exp(-g * dxx).mean() + np.exp(-g * dyy).mean() - 2.0 * np.exp(-g * dxy).mean()
    return float(max(mmd2, 0.0))


def subsample_tokens(arr: np.ndarray, axis: int, cap: int,
                     rng: np.random.Generator) -> np.ndarray:
    n = arr.shape[axis]
    if n <= cap:
        return arr
    idx = np.sort(rng.choice(n, cap, replace=False))
    return np.take(arr, idx, axis=axis)


# =====================================================================
# RoPE re-application
# =====================================================================
def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope(t, cos, sin):
    cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
    return t * cos + _rotate_half(t) * sin


# =====================================================================
# Template-stripping helpers (isolate question tokens)
# =====================================================================
def _ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def common_prefix_len(tokenizer, a: str, b: str) -> int:
    ia, ib = _ids(tokenizer, a), _ids(tokenizer, b)
    n = min(len(ia), len(ib))
    for i in range(n):
        if ia[i] != ib[i]:
            return i
    return n


def common_suffix_len(tokenizer, a: str, b: str) -> int:
    ia, ib = _ids(tokenizer, a), _ids(tokenizer, b)
    n = min(len(ia), len(ib))
    for i in range(n):
        if ia[-1 - i] != ib[-1 - i]:
            return i
    return n


# =====================================================================
# Locate Qwen submodules
# =====================================================================
def _locate_modules(model):
    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("Could not find `model.model` (decoder backbone).")
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Could not find `model.model.layers`.")
    rotary = getattr(base, "rotary_emb", None)
    if rotary is None:
        raise RuntimeError("Could not find `model.model.rotary_emb` (needed for RoPE recovery).")
    sa = getattr(layers[0], "self_attn", None)
    if sa is None:
        raise RuntimeError("Could not find `model.model.layers[0].self_attn`.")
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if getattr(sa, name, None) is None:
            raise RuntimeError(f"Could not find `self_attn.{name}` in {type(sa).__name__}.")
    return layers, rotary


# =====================================================================
# Per-prompt prefill activation collection (read-only hooks)
# =====================================================================
@torch.no_grad()
def collect_prefill_activations(wrapper, question: str, args) -> Dict:
    model = wrapper.model
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_q)

    layers, rotary = _locate_modules(model)
    msgs = build_agent_messages_single_agent(question=question, args=args)
    prompt_text, input_ids, attention_mask, _ = wrapper.prepare_chat_input(msgs)

    raw_q, raw_k, raw_v, raw_ho, rope = {}, {}, {}, {}, {}
    handles = []

    def proj_hook(store, li):
        def h(_m, _i, out):
            store[li] = out.detach()
        return h

    def oproj_pre(li):
        def h(_m, inp):
            raw_ho[li] = inp[0].detach()
            return None
        return h

    def rope_hook(_m, _i, out):
        rope["cos"], rope["sin"] = out[0].detach(), out[1].detach()

    try:
        handles.append(rotary.register_forward_hook(rope_hook))
        for li, layer in enumerate(layers):
            sa = layer.self_attn
            handles.append(sa.q_proj.register_forward_hook(proj_hook(raw_q, li)))
            handles.append(sa.k_proj.register_forward_hook(proj_hook(raw_k, li)))
            handles.append(sa.v_proj.register_forward_hook(proj_hook(raw_v, li)))
            handles.append(sa.o_proj.register_forward_pre_hook(oproj_pre(li)))
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states
    finally:
        for h in handles:
            h.remove()

    if "cos" not in rope:
        raise RuntimeError("RoPE cos/sin not captured (rotary_emb hook did not fire).")
    cos = rope["cos"].to(torch.float32); sin = rope["sin"].to(torch.float32)
    seq = input_ids.shape[1]

    q_out, k_out, v_out, ho_out, hs_out = {}, {}, {}, {}, {}
    for li, layer in enumerate(layers):
        sa = layer.self_attn
        q = raw_q[li].view(1, seq, n_q, head_dim)
        if getattr(sa, "q_norm", None) is not None:
            q = sa.q_norm(q)
        q = _apply_rope(q.transpose(1, 2).to(torch.float32), cos, sin)
        q_out[li] = q[0].cpu().numpy()

        k = raw_k[li].view(1, seq, n_kv, head_dim)
        if getattr(sa, "k_norm", None) is not None:
            k = sa.k_norm(k)
        k = _apply_rope(k.transpose(1, 2).to(torch.float32), cos, sin)
        k_out[li] = k[0].cpu().numpy()

        v = raw_v[li].view(1, seq, n_kv, head_dim).transpose(1, 2).to(torch.float32)
        v_out[li] = v[0].cpu().numpy()

        ho = raw_ho[li].view(1, seq, n_q, head_dim).transpose(1, 2).to(torch.float32)
        ho_out[li] = ho[0].cpu().numpy()

        raw_q[li] = raw_k[li] = raw_v[li] = raw_ho[li] = None
        del q, k, v, ho

    for idx, hs in enumerate(hidden_states):
        hs_out[idx] = hs[0].to(torch.float32).cpu().numpy()

    del raw_q, raw_k, raw_v, raw_ho, out, hidden_states, cos, sin
    gc.collect(); torch.cuda.empty_cache()

    return {"prompt_text": prompt_text, "seq": seq,
            "q": q_out, "k": k_out, "v": v_out,
            "head_output": ho_out, "hidden_states": hs_out,
            "n_q": n_q, "n_kv": n_kv, "head_dim": head_dim}


# =====================================================================
# Build a pooled, template-stripped representation cloud for one source
# =====================================================================
def build_source_pool(wrapper, items, source, args, rng) -> Dict:
    """Pool question-specific token representations across items for one source.

    Returns per-(component, layer, head) fp16 arrays capped at max_pool_tokens."""
    args.task = source["task"]          # ensure correct prompt template
    version = source["version"]
    # template prompt with empty question -> strip shared head/tail tokens
    msgs_tmpl = build_agent_messages_single_agent(question="", args=args)
    template_str = wrapper.render_chat(msgs_tmpl)

    acc = {c: defaultdict(list) for c in ["q", "k", "v", "head_output"]}
    acc_h = defaultdict(list)
    n_items = 0
    n_q = n_kv = head_dim = num_layers = None

    n_use = len(items) if args.max_tasks <= 0 else min(len(items), args.max_tasks)
    for it in items[:n_use]:
        question = it["question"] if version == "original" else it.get(version, "")
        if not question:
            continue
        a = collect_prefill_activations(wrapper, question, args)
        n_q, n_kv, head_dim = a["n_q"], a["n_kv"], a["head_dim"]
        num_layers = len(a["hidden_states"]) - 1
        seq = a["seq"]
        pre = common_prefix_len(wrapper.tokenizer, template_str, a["prompt_text"])
        suf = common_suffix_len(wrapper.tokenizer, template_str, a["prompt_text"])
        lo, hi = pre, min(seq, max(pre + 1, seq - suf))
        if hi <= lo:
            lo, hi = 0, seq
        for li in a["q"]:
            acc["q"][li].append(a["q"][li][:, lo:hi, :].astype(np.float16))
            acc["k"][li].append(a["k"][li][:, lo:hi, :].astype(np.float16))
            acc["v"][li].append(a["v"][li][:, lo:hi, :].astype(np.float16))
            acc["head_output"][li].append(a["head_output"][li][:, lo:hi, :].astype(np.float16))
        for idx in a["hidden_states"]:
            acc_h[idx].append(a["hidden_states"][idx][lo:hi, :].astype(np.float16))
        n_items += 1
        del a
        gc.collect(); torch.cuda.empty_cache()

    cap = args.max_pool_tokens
    pool = {"q": {}, "k": {}, "v": {}, "head_output": {}, "hidden": {},
            "n_items": n_items, "n_q": n_q, "n_kv": n_kv,
            "head_dim": head_dim, "num_layers": num_layers}
    for c in ["q", "k", "v", "head_output"]:
        for li, chunks in acc[c].items():
            arr = np.concatenate(chunks, axis=1)         # [heads, T, dim]
            pool[c][li] = subsample_tokens(arr, axis=1, cap=cap, rng=rng)
    for idx, chunks in acc_h.items():
        arr = np.concatenate(chunks, axis=0)             # [T, dim]
        pool["hidden"][idx] = subsample_tokens(arr, axis=0, cap=cap, rng=rng)
    any_layer = next(iter(pool["q"].values())) if pool["q"] else np.zeros((1, 0, 1))
    pool["n_tokens"] = int(any_layer.shape[1])
    del acc, acc_h
    gc.collect()
    return pool


# =====================================================================
# Pairwise metrics over pooled clouds
# =====================================================================
def compare_sources(poolA, poolB, pair, args, rng):
    qkv_rows, ho_rows, res_rows, hm_rows = [], [], [], []
    num_layers = poolA["num_layers"]
    na_items, nb_items = poolA["n_items"], poolB["n_items"]
    na_tok, nb_tok = poolA["n_tokens"], poolB["n_tokens"]

    for comp, htype in [("q", "q_head"), ("k", "kv_head"), ("v", "kv_head")]:
        for li in range(num_layers):
            A, B = poolA[comp].get(li), poolB[comp].get(li)
            if A is None or B is None:
                continue
            for h in range(A.shape[0]):
                X, Y = A[h], B[h]
                mpc = mean_pairwise_cosine(X, Y)
                mmc = maxmatch_cosine(X, Y)
                mmd = rbf_mmd2(X, Y, max_tokens=args.max_mmd_tokens, rng=rng)
                qkv_rows.append(dict(pair=pair, component=comp, layer=li, head=h,
                    head_type=htype, mean_pairwise_cosine=round(mpc, 6),
                    maxmatch_cosine=round(mmc, 6), mmd_rbf=round(mmd, 6),
                    n_tokens_a=na_tok, n_tokens_b=nb_tok,
                    n_items_a=na_items, n_items_b=nb_items))
                hm_rows.append(dict(pair=pair, metric="maxmatch_cosine", component=comp,
                                    layer=li, head=h, value=round(mmc, 6)))
                hm_rows.append(dict(pair=pair, metric="mmd_rbf", component=comp,
                                    layer=li, head=h, value=round(mmd, 6)))

    for li in range(num_layers):
        A, B = poolA["head_output"].get(li), poolB["head_output"].get(li)
        if A is None or B is None:
            continue
        for h in range(A.shape[0]):
            X, Y = A[h], B[h]
            mpc = mean_pairwise_cosine(X, Y); mmc = maxmatch_cosine(X, Y)
            l2m = l2_mean(X, Y); mmd = rbf_mmd2(X, Y, max_tokens=args.max_mmd_tokens, rng=rng)
            ho_rows.append(dict(pair=pair, layer=li, head=h,
                mean_pairwise_cosine=round(mpc, 6), maxmatch_cosine=round(mmc, 6),
                l2_mean=round(l2m, 6), mmd_rbf=round(mmd, 6),
                n_tokens_a=na_tok, n_tokens_b=nb_tok,
                n_items_a=na_items, n_items_b=nb_items))
            hm_rows.append(dict(pair=pair, metric="head_output_maxmatch_cosine",
                                component="head_output", layer=li, head=h, value=round(mmc, 6)))
            hm_rows.append(dict(pair=pair, metric="head_output_mmd_rbf",
                                component="head_output", layer=li, head=h, value=round(mmd, 6)))

    for idx in sorted(poolA["hidden"].keys()):
        A, B = poolA["hidden"].get(idx), poolB["hidden"].get(idx)
        if A is None or B is None:
            continue
        mpc = mean_pairwise_cosine(A, B); mmc = maxmatch_cosine(A, B)
        mmd = rbf_mmd2(A, B, max_tokens=args.max_mmd_tokens, rng=rng)
        res_rows.append(dict(pair=pair, layer=idx,
            mean_pairwise_cosine=round(mpc, 6), maxmatch_cosine=round(mmc, 6),
            mmd_rbf=round(mmd, 6), n_tokens_a=na_tok, n_tokens_b=nb_tok,
            n_items_a=na_items, n_items_b=nb_items))
        hm_rows.append(dict(pair=pair, metric="residual_maxmatch_cosine",
                            component="residual", layer=idx, head=-1, value=round(mmc, 6)))
        hm_rows.append(dict(pair=pair, metric="residual_mmd_rbf",
                            component="residual", layer=idx, head=-1, value=round(mmd, 6)))
    return qkv_rows, ho_rows, res_rows, hm_rows


# =====================================================================
# Output helpers
# =====================================================================
def write_csv(path, rows):
    if not rows:
        print(f"  (no rows for {os.path.basename(path)})")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  CSV -> {os.path.basename(path)}  ({len(rows)} rows)")


def plot_raw_heatmap(hm_rows, metric, component, out_path, num_layers, normalize="raw"):
    cells = defaultdict(list); max_head = -1
    for r in hm_rows:
        if r["metric"] != metric or r["component"] != component:
            continue
        v = r["value"]
        if v != v:
            continue
        cells[(r["layer"], r["head"])].append(v); max_head = max(max_head, r["head"])
    if not cells:
        return
    is_res = component == "residual" or max_head < 0
    rows_n = (num_layers + 1) if is_res else num_layers
    n_head = 1 if is_res else max_head + 1
    grid = np.full((rows_n, n_head), np.nan)
    for (l, h), vals in cells.items():
        col = 0 if is_res else h
        if 0 <= l < rows_n and 0 <= col < n_head:
            grid[l, col] = float(np.mean(vals))
    if normalize == "global":
        fin = grid[np.isfinite(grid)]
        if fin.size and fin.max() - fin.min() > 1e-12:
            grid = (grid - fin.min()) / (fin.max() - fin.min())
    cmap = "magma" if "mmd" in metric else "viridis"
    fig, ax = plt.subplots(figsize=(max(5, n_head * 0.45), max(5, rows_n * 0.28)))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(f"{component} · {metric} ({normalize})", fontsize=10)
    ax.set_xlabel("head" if not is_res else "(residual)"); ax.set_ylabel("layer")
    lbl = "higher = more different" if "mmd" in metric else "higher = more similar"
    fig.colorbar(im, ax=ax, shrink=0.7, label=lbl if normalize == "raw" else "global-norm")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"    -> {os.path.basename(out_path)}")


# =====================================================================
# Main multi-source analysis
# =====================================================================
def run_analysis(wrapper, sources, datasets, args, model_tag):
    rng = np.random.default_rng(args.seed)
    cfg = wrapper.model.config
    num_layers = cfg.num_hidden_layers
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_q)

    tasks = sorted({s["task"] for s in sources})
    versions = sorted({s["version"] for s in sources})
    run_tag = "-".join(tasks) + "_" + "-".join(_VER_SHORT.get(v, v) for v in versions)
    prefix = f"baseline_{run_tag}_{model_tag}"
    plots_dir = args.plots_dir
    os.makedirs(plots_dir, exist_ok=True)

    # 1) build pooled clouds for every source
    pools = {}
    for s in sources:
        print(f"  pooling source: {s['tag']} (task={s['task']}, version={s['version']})")
        pools[s["tag"]] = build_source_pool(wrapper, datasets[s["dataset_key"]], s, args, rng)
        print(f"    -> {pools[s['tag']]['n_items']} items, "
              f"{pools[s['tag']]['n_tokens']} pooled tokens/head")

    # 2) pairwise comparison
    qkv_rows, ho_rows, res_rows, hm_rows = [], [], [], []
    tags = [s["tag"] for s in sources]
    for a, b in combinations(tags, 2):
        pair = f"{a}_vs_{b}"
        print(f"  comparing {pair}")
        r1, r2, r3, hm = compare_sources(pools[a], pools[b], pair, args, rng)
        qkv_rows += r1; ho_rows += r2; res_rows += r3; hm_rows += hm

    write_csv(os.path.join(plots_dir, f"{prefix}_prefill_qkv_headwise_metrics.csv"), qkv_rows)
    write_csv(os.path.join(plots_dir, f"{prefix}_prefill_head_output_metrics.csv"), ho_rows)
    write_csv(os.path.join(plots_dir, f"{prefix}_prefill_residual_stream_metrics.csv"), res_rows)
    write_csv(os.path.join(plots_dir, f"{prefix}_heatmap_long.csv"), hm_rows)

    if args.make_heatmaps:
        for metric, comp, tag in [
            ("maxmatch_cosine", "q", "q_maxmatch_cosine"),
            ("maxmatch_cosine", "k", "k_maxmatch_cosine"),
            ("maxmatch_cosine", "v", "v_maxmatch_cosine"),
            ("head_output_maxmatch_cosine", "head_output", "head_output_maxmatch_cosine"),
            ("head_output_mmd_rbf", "head_output", "head_output_mmd_rbf"),
            ("residual_maxmatch_cosine", "residual", "residual_maxmatch_cosine"),
        ]:
            for norm in ("raw", "global"):
                sfx = "raw" if norm == "raw" else "globalnorm"
                for a, b in combinations(tags, 2):
                    pair = f"{a}_vs_{b}"
                    sub = [r for r in hm_rows if r["pair"] == pair]
                    plot_raw_heatmap(sub, metric, comp,
                        os.path.join(plots_dir, f"{prefix}_heatmap_{pair}_{tag}_{sfx}.png"),
                        num_layers, normalize=norm)

    meta = {
        "model_name": args.model_name, "method": "baseline",
        "primary_analysis_method": args.primary_analysis_method,
        "seed": args.seed,
        "comparison_mode": "multi_source_pooled_distribution",
        "sources": [{"tag": s["tag"], "task": s["task"], "version": s["version"],
                     "language": ("zh" if s["version"] == "version1" else "en"),
                     "data_path": s["data_path"], "n_items": pools[s["tag"]]["n_items"],
                     "n_pooled_tokens": pools[s["tag"]]["n_tokens"]} for s in sources],
        "pairs": [f"{a}_vs_{b}" for a, b in combinations(tags, 2)],
        "num_layers": num_layers, "num_attention_heads": n_q,
        "num_key_value_heads": n_kv, "head_dim": head_dim,
        "max_pool_tokens": args.max_pool_tokens, "max_mmd_tokens": args.max_mmd_tokens,
        "analysis_phase": "prefill",
        "removed_metrics": ["js_divergence", "tsne", "cka_linear"],
        "removed_methods": ["latent_mas"],
        "metrics": {
            "qkv": ["mean_pairwise_cosine", "maxmatch_cosine", "mmd_rbf"],
            "head_output": ["mean_pairwise_cosine", "maxmatch_cosine", "l2_mean", "mmd_rbf"],
            "residual_stream": ["mean_pairwise_cosine", "maxmatch_cosine", "mmd_rbf"],
        },
        "note": {
            "comparison": "Sources are compared at the DISTRIBUTION level over pooled, "
                          "template-stripped question tokens (no per-item pairing), because "
                          "cross-domain sources have no item correspondence.",
            "qkv": "Q/K/V measured before attention output (Q/K after RoPE, V raw).",
            "head_output": "measured after attn_probs @ V and before o_proj.",
            "residual_stream": "hidden_states[l]; index 0 = embedding, l+1 = block l output.",
            "mmd": "MMD is a distance: smaller means more similar; scale is component-local.",
            "cka_removed": "CKA removed: needs paired equal-count samples, undefined across domains.",
            "tsne_js_removed": "t-SNE / JS removed as before.",
        },
    }
    with open(os.path.join(plots_dir, f"{prefix}_run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  metadata -> {prefix}_run_metadata.json")


# =====================================================================
# text_mas: kept only as a formal stub (no head-wise analysis)
# =====================================================================
def analyze_text_mas(*_a, **_k):  # pragma: no cover
    raise NotImplementedError(
        "text_mas is retained only as a formal stub; the multi-source pooled "
        "analysis runs for --primary_analysis_method baseline only.")


# =====================================================================
# Entry
# =====================================================================
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_paths", nargs="+", required=True,
                    help="One jsonl per dataset, e.g. gsm8k_paraphrased.jsonl gpqa_paraphrased.jsonl")
    pa.add_argument("--tasks", nargs="+", required=True,
                    choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy",
                             "arc_challenge", "mbppplus", "humanevalplus", "medqa"],
                    help="Task name per data path (same order/length as --data_paths). "
                         "Controls the baseline prompt template per source.")
    pa.add_argument("--versions", nargs="+", default=["original", "version1"],
                    choices=["original", "version1", "version2", "version3"],
                    help="Versions to turn into sources. Sources = datasets x versions.")
    pa.add_argument("--model_name", default="Qwen/Qwen3-8B")
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--methods", nargs="+", default=["baseline"],
                    choices=["baseline", "text_mas"],
                    help="latent_mas is removed; text_mas is a formal stub.")
    pa.add_argument("--primary_analysis_method", default="baseline",
                    choices=["baseline", "text_mas"])
    pa.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    pa.add_argument("--max_tasks", type=int, default=50,
                    help="Items per source to pool (<=0 = all).")
    pa.add_argument("--max_pool_tokens", type=int, default=256,
                    help="Cap on pooled question tokens per source per (layer, head).")
    pa.add_argument("--max_mmd_tokens", type=int, default=256)
    pa.add_argument("--make_heatmaps", action="store_true", default=True)
    pa.add_argument("--no_heatmaps", dest="make_heatmaps", action="store_false")
    pa.add_argument("--plots_dir", default="./plots_multi")
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    if len(args.data_paths) != len(args.tasks):
        pa.error("--data_paths and --tasks must have the same length/order.")

    # ModelWrapper expectations (HF backend, no vLLM, no latent options)
    args.use_vllm = False
    args.use_second_HF_model = False
    args.enable_prefix_caching = False
    args.latent_space_realign = False
    args.device2 = "cpu"
    args.tensor_parallel_size = 1
    args.gpu_memory_utilization = 0.9
    args.method = "baseline"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load datasets (dataset_key = data path basename)
    datasets = {}
    for path, task in zip(args.data_paths, args.tasks):
        key = os.path.splitext(os.path.basename(path))[0]
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        datasets[key] = rows
        print(f"Loaded {len(rows)} items from {path} (task={task}, key={key})")

    # sources = datasets x versions ; tag = {task}_{vshort}
    sources = []
    for path, task in zip(args.data_paths, args.tasks):
        key = os.path.splitext(os.path.basename(path))[0]
        for v in args.versions:
            sources.append({"tag": f"{task}_{_VER_SHORT.get(v, v)}",
                            "task": task, "version": v,
                            "dataset_key": key, "data_path": path})
    print("Sources:", [s["tag"] for s in sources])
    if len(sources) < 2:
        pa.error("Need at least 2 sources to form a pair.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    model_tag = args.model_name.split("/")[-1]

    for method in args.methods:
        print(f"\n{'=' * 60}\n  Method: {method}\n{'=' * 60}")
        args.method = method
        if method == args.primary_analysis_method == "baseline":
            run_analysis(wrapper, sources, datasets, args, model_tag)
        else:
            print(f"  [skip] '{method}' retained for dispatch only; pooled analysis "
                  f"runs for primary_analysis_method='baseline'.")

    print(f"\nDone. Outputs under {args.plots_dir}/")


if __name__ == "__main__":
    main()
