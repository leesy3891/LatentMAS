#!/usr/bin/env python3
"""
analyze_representations.py  (v4 — head-wise prefill sensitivity)
================================================================
Purpose
-------
Measure, in the *prefill* stage of a baseline single-agent forward pass, how the
internal representations of a Qwen model change across paraphrased question
variants (original / version1 / version2 / version3).

Unlike the previous K-cache-centric pairwise-mean analysis, this version keeps
the **head dimension** and reports, per layer and per head:

  * Q / K / V                  — attention *input* subspace (after RoPE for Q/K)
  * head_output                — attention *result*  (attn_probs @ V, before o_proj)
  * residual stream hidden     — transformer *block output* representation

Metrics per head / layer:
  * mean_pairwise_cosine
  * maxmatch_cosine
  * linear CKA   (token-count matched only; NaN otherwise)
  * RBF MMD^2    (median-heuristic bandwidth; distance-like, smaller == closer)
  * head_output also reports l2_mean

Removed vs older versions:
  * JS divergence  (Q/K/V/hidden are not probability distributions)
  * t-SNE          (visualization-only, unstable for quantitative claims)
  * head-flattened cosine as a core metric  (replaced by head-wise metrics)

TextMAS / LatentMAS dispatch and CLI are kept for compatibility, but the new
head-wise analysis and the final CSV/heatmap deliverables are produced for the
*baseline* method only (--primary_analysis_method baseline).
"""
import argparse
import csv
import gc
import json
import math
import os
import sys
from collections import namedtuple, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ModelWrapper, _past_length
from prompts import (
    build_agent_messages_single_agent,
    build_agent_messages_sequential_text_mas,
    build_agent_messages_hierarchical_text_mas,
    build_agent_message_sequential_latent_mas,
    build_agent_message_hierarchical_latent_mas,
)

# ── agents (kept for text_mas / latent_mas dispatch) ─────────────────
Agent = namedtuple("Agent", ["name", "role"])
AGENTS = [Agent("Planner", "planner"), Agent("Critic", "critic"),
          Agent("Refiner", "refiner"), Agent("Judger", "judger")]
HIER_NAME = {"Planner": "Math Agent", "Critic": "Science Agent",
             "Refiner": "Code Agent", "Judger": "Task Summrizer"}

VER_LABELS = ["original", "version1", "version2", "version3"]
PAIRS = [
    ("original", "version1"), ("original", "version2"), ("original", "version3"),
    ("version1", "version2"), ("version1", "version3"), ("version2", "version3"),
]


# =====================================================================
# Metric primitives  (all operate on [n_tokens, dim] head/layer matrices)
# =====================================================================
def rowwise_l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def mean_pairwise_cosine(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean over the full [n_a, n_b] cosine matrix of L2-normalized rows."""
    if X.size == 0 or Y.size == 0:
        return float("nan")
    Xn, Yn = rowwise_l2_normalize(X), rowwise_l2_normalize(Y)
    return float((Xn @ Yn.T).mean())


def maxmatch_cosine(X: np.ndarray, Y: np.ndarray) -> float:
    """Average of (row-wise max over Y) and (column-wise max over X).

    More robust than mean pairwise cosine when token alignment differs
    (e.g. a Chinese paraphrase with a different tokenization)."""
    if X.size == 0 or Y.size == 0:
        return float("nan")
    Xn, Yn = rowwise_l2_normalize(X), rowwise_l2_normalize(Y)
    C = Xn @ Yn.T                      # [n_a, n_b]
    a2b = C.max(axis=1).mean()         # each original token -> best version token
    b2a = C.max(axis=0).mean()         # each version token  -> best original token
    return float(0.5 * (a2b + b2a))


def l2_mean(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean of the full [n_a, n_b] Euclidean-distance matrix."""
    if X.size == 0 or Y.size == 0:
        return float("nan")
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    xx = (X * X).sum(axis=1)[:, None]
    yy = (Y * Y).sum(axis=1)[None, :]
    d2 = np.clip(xx + yy - 2.0 * (X @ Y.T), 0.0, None)
    return float(np.sqrt(d2).mean())


def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two equal-sample representations [n, dx], [n, dy].

    Caller MUST guarantee X.shape[0] == Y.shape[0]; CKA is undefined for
    mismatched sample counts (token alignment problem)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0] or X.shape[0] < 2:
        return float("nan")
    Kx = _center_gram(X @ X.T)
    Ky = _center_gram(Y @ Y.T)
    hsic_xy = float((Kx * Ky).sum())
    hsic_xx = float((Kx * Kx).sum())
    hsic_yy = float((Ky * Ky).sum())
    denom = math.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return float("nan")
    return hsic_xy / denom


def _uniform_resample(X: np.ndarray, n_out: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    if n == n_out:
        return X
    if n > n_out:
        idx = np.linspace(0, n - 1, n_out).round().astype(int)
        return X[idx]
    # n < n_out should not happen for resampling-down; pad by resampling with replacement
    idx = rng.integers(0, n, size=n_out)
    return X[idx]


def rbf_mmd2(X: np.ndarray, Y: np.ndarray, max_tokens: int = 256,
             rng: Optional[np.random.Generator] = None) -> float:
    """Biased RBF-kernel MMD^2 with median-heuristic bandwidth.

    Works for n_a != n_b. Long sequences are uniformly subsampled to
    `max_tokens` per side. Smaller value == more similar distributions."""
    if X.size == 0 or Y.size == 0:
        return float("nan")
    if rng is None:
        rng = np.random.default_rng(0)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] > max_tokens:
        X = X[np.sort(rng.choice(X.shape[0], max_tokens, replace=False))]
    if Y.shape[0] > max_tokens:
        Y = Y[np.sort(rng.choice(Y.shape[0], max_tokens, replace=False))]

    def sq_dists(A, B):
        aa = (A * A).sum(axis=1)[:, None]
        bb = (B * B).sum(axis=1)[None, :]
        return np.clip(aa + bb - 2.0 * (A @ B.T), 0.0, None)

    d_xx = sq_dists(X, X)
    d_yy = sq_dists(Y, Y)
    d_xy = sq_dists(X, Y)

    # median heuristic over pooled pairwise distances
    pooled = np.concatenate([d_xx[np.triu_indices_from(d_xx, k=1)],
                             d_yy[np.triu_indices_from(d_yy, k=1)],
                             d_xy.ravel()])
    med = np.median(pooled[pooled > 0]) if np.any(pooled > 0) else 1.0
    if med < 1e-12:
        med = 1.0
    gamma = 1.0 / med            # = 1 / (2 * sigma^2) with sigma^2 = med/2

    k_xx = np.exp(-gamma * d_xx)
    k_yy = np.exp(-gamma * d_yy)
    k_xy = np.exp(-gamma * d_xy)
    mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return float(max(mmd2, 0.0))


def cka_for_pair(X: np.ndarray, Y: np.ndarray, align: bool,
                 rng: np.random.Generator) -> Tuple[float, str]:
    """Returns (cka_linear, cka_status)."""
    na, nb = X.shape[0], Y.shape[0]
    if na == nb:
        if na < 2:
            return float("nan"), "skipped_too_few_tokens"
        return linear_cka(X, Y), "matched"
    if not align:
        return float("nan"), "skipped_length_mismatch"
    m = min(na, nb)
    if m < 2:
        return float("nan"), "skipped_too_few_tokens"
    Xa = _uniform_resample(X, m, rng)
    Ya = _uniform_resample(Y, m, rng)
    return linear_cka(Xa, Ya), "aligned_resampled"


# =====================================================================
# RoPE re-application (to recover post-RoPE Q/K from pre-RoPE captures)
# =====================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """t: [B, heads, seq, head_dim];  cos/sin: [B, seq, head_dim]."""
    cos = cos.unsqueeze(1)             # [B, 1, seq, hd]
    sin = sin.unsqueeze(1)
    return t * cos + _rotate_half(t) * sin


# =====================================================================
# Common-prefix detection (strips shared system prompt / template head)
# =====================================================================
def common_prefix_len(tokenizer, prompt_a: str, prompt_b: str) -> int:
    ids_a = tokenizer(prompt_a, add_special_tokens=False)["input_ids"]
    ids_b = tokenizer(prompt_b, add_special_tokens=False)["input_ids"]
    n = min(len(ids_a), len(ids_b))
    for i in range(n):
        if ids_a[i] != ids_b[i]:
            return i
    return n


# =====================================================================
# Locate Qwen-style submodules for hooking
# =====================================================================
def _locate_modules(model):
    """Return (decoder_layers, rotary_emb_module) for a Qwen-style model.

    Raises RuntimeError naming the missing path on failure."""
    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError(
            "Could not find `model.model` (decoder backbone). "
            "Hooking only supports Qwen-style `AutoModelForCausalLM` layouts.")
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Could not find `model.model.layers` (decoder layer list).")
    rotary = getattr(base, "rotary_emb", None)
    if rotary is None:
        raise RuntimeError(
            "Could not find `model.model.rotary_emb`. Post-RoPE Q/K recovery "
            "requires the shared rotary embedding module.")
    # sanity-check the first attention block
    sa = getattr(layers[0], "self_attn", None)
    if sa is None:
        raise RuntimeError("Could not find `model.model.layers[0].self_attn`.")
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if getattr(sa, name, None) is None:
            raise RuntimeError(
                f"Could not find `self_attn.{name}` in attention module "
                f"of type {type(sa).__name__}. Unsupported architecture for head-wise hooks.")
    return layers, rotary


# =====================================================================
# Baseline prefill activation collection (single forward pass + hooks)
# =====================================================================
@torch.no_grad()
def collect_baseline_prefill_activations(wrapper, question: str, args) -> Dict:
    """Run ONE prefill forward pass on the rendered baseline prompt and collect
    head-wise Q/K/V, pre-o_proj head_output, and residual-stream hidden states.

    Returns dict described in the task spec.
    """
    model = wrapper.model
    cfg = model.config
    device = wrapper.device

    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_q)
    num_layers = cfg.num_hidden_layers

    layers, rotary = _locate_modules(model)

    msgs = build_agent_messages_single_agent(question=question, args=args)
    prompt_text, input_ids, attention_mask, tokens = wrapper.prepare_chat_input(msgs)

    raw_q: Dict[int, torch.Tensor] = {}
    raw_k: Dict[int, torch.Tensor] = {}
    raw_v: Dict[int, torch.Tensor] = {}
    raw_ho: Dict[int, torch.Tensor] = {}
    rope_cossin: Dict[str, torch.Tensor] = {}
    handles = []

    def make_proj_hook(store, lidx):
        def hook(_module, _inp, out):
            store[lidx] = out.detach()
        return hook

    def make_oproj_pre_hook(lidx):
        def hook(_module, inp):
            # inp[0]: [B, seq, n_q*head_dim] == per-head outputs before o_proj
            raw_ho[lidx] = inp[0].detach()
            return None  # do not modify -> model behavior unchanged
        return hook

    def rope_hook(_module, _inp, out):
        # out == (cos, sin), each [B, seq, head_dim]
        cos, sin = out
        rope_cossin["cos"] = cos.detach()
        rope_cossin["sin"] = sin.detach()

    try:
        handles.append(rotary.register_forward_hook(rope_hook))
        for li, layer in enumerate(layers):
            sa = layer.self_attn
            handles.append(sa.q_proj.register_forward_hook(make_proj_hook(raw_q, li)))
            handles.append(sa.k_proj.register_forward_hook(make_proj_hook(raw_k, li)))
            handles.append(sa.v_proj.register_forward_hook(make_proj_hook(raw_v, li)))
            handles.append(sa.o_proj.register_forward_pre_hook(make_oproj_pre_hook(li)))

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = out.hidden_states  # tuple len = num_layers + 1
    finally:
        for h in handles:
            h.remove()

    if "cos" not in rope_cossin:
        raise RuntimeError(
            "RoPE cos/sin were not captured — the rotary_emb forward hook did not "
            "fire. The model may compute rotary embeddings elsewhere.")

    cos = rope_cossin["cos"].to(torch.float32)
    sin = rope_cossin["sin"].to(torch.float32)

    q_out: Dict[int, np.ndarray] = {}
    k_out: Dict[int, np.ndarray] = {}
    v_out: Dict[int, np.ndarray] = {}
    ho_out: Dict[int, np.ndarray] = {}
    hs_out: Dict[int, np.ndarray] = {}

    seq = input_ids.shape[1]
    for li, layer in enumerate(layers):
        sa = layer.self_attn

        # ---- Q : pre-RoPE [B, seq, n_q*hd] -> q_norm -> RoPE ----
        q = raw_q[li].view(1, seq, n_q, head_dim)
        if getattr(sa, "q_norm", None) is not None:
            q = sa.q_norm(q)
        q = q.transpose(1, 2).to(torch.float32)            # [1, n_q, seq, hd]
        q = _apply_rope(q, cos, sin)
        q_out[li] = q[0].cpu().numpy()                     # [n_q, seq, hd]

        # ---- K : pre-RoPE [B, seq, n_kv*hd] -> k_norm -> RoPE ----
        k = raw_k[li].view(1, seq, n_kv, head_dim)
        if getattr(sa, "k_norm", None) is not None:
            k = sa.k_norm(k)
        k = k.transpose(1, 2).to(torch.float32)            # [1, n_kv, seq, hd]
        k = _apply_rope(k, cos, sin)
        k_out[li] = k[0].cpu().numpy()                     # [n_kv, seq, hd]

        # ---- V : no RoPE ----
        v = raw_v[li].view(1, seq, n_kv, head_dim).transpose(1, 2).to(torch.float32)
        v_out[li] = v[0].cpu().numpy()                     # [n_kv, seq, hd]

        # ---- head_output : pre-o_proj [B, seq, n_q*hd] ----
        ho = raw_ho[li].view(1, seq, n_q, head_dim).transpose(1, 2).to(torch.float32)
        ho_out[li] = ho[0].cpu().numpy()                   # [n_q, seq, hd]

        # free this layer's GPU captures immediately (lower peak GPU memory)
        raw_q[li] = raw_k[li] = raw_v[li] = raw_ho[li] = None
        del q, k, v, ho

    for idx, hs in enumerate(hidden_states):
        hs_out[idx] = hs[0].to(torch.float32).cpu().numpy()   # [seq, hidden]

    del raw_q, raw_k, raw_v, raw_ho, out, hidden_states, cos, sin
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "prompt_text": prompt_text,
        "input_ids": input_ids.detach().cpu(),
        "tokens": tokens,
        "q": q_out,
        "k": k_out,
        "v": v_out,
        "head_output": ho_out,
        "hidden_states": hs_out,
        "meta": {
            "num_layers": num_layers,
            "num_attention_heads": n_q,
            "num_key_value_heads": n_kv,
            "head_dim": head_dim,
        },
    }


# =====================================================================
# Per-pair metric computation
# =====================================================================
def compute_headwise_qkv_metrics(act_a, act_b, pair, tid, skip,
                                 num_layers, args, rng) -> Tuple[List[Dict], List[Dict]]:
    """Returns (metric_rows, heatmap_rows) for Q/K/V head-wise comparison."""
    rows, hm = [], []
    comp_spec = [("q", "q_head"), ("k", "kv_head"), ("v", "kv_head")]
    for comp, head_type in comp_spec:
        A, B = act_a[comp], act_b[comp]
        for l in range(num_layers):
            a, b = A.get(l), B.get(l)
            if a is None or b is None:
                continue
            n_heads = a.shape[0]
            aq = a[:, skip:, :]
            bq = b[:, skip:, :]
            na, nb = aq.shape[1], bq.shape[1]
            for h in range(n_heads):
                X, Y = aq[h], bq[h]            # [n, hd]
                mpc = mean_pairwise_cosine(X, Y)
                mmc = maxmatch_cosine(X, Y)
                cka, status = cka_for_pair(X, Y, args.align_tokens_for_cka, rng)
                mmd = rbf_mmd2(X, Y, max_tokens=args.max_mmd_tokens, rng=rng)
                rows.append({
                    "task_id": tid, "pair": pair, "component": comp,
                    "layer": l, "head": h, "head_type": head_type,
                    "mean_pairwise_cosine": round(mpc, 6),
                    "maxmatch_cosine": round(mmc, 6),
                    "cka_linear": (round(cka, 6) if cka == cka else float("nan")),
                    "cka_status": status,
                    "mmd_rbf": round(mmd, 6),
                    "n_tokens_a": na, "n_tokens_b": nb, "prefix_skip": skip,
                })
                hm.append({"task_id": tid, "pair": pair, "metric": "maxmatch_cosine",
                           "component": comp, "layer": l, "head": h, "value": round(mmc, 6)})
                hm.append({"task_id": tid, "pair": pair, "metric": "mmd_rbf",
                           "component": comp, "layer": l, "head": h, "value": round(mmd, 6)})
    return rows, hm


def compute_head_output_metrics(act_a, act_b, pair, tid, skip,
                                num_layers, args, rng) -> Tuple[List[Dict], List[Dict]]:
    rows, hm = [], []
    A, B = act_a["head_output"], act_b["head_output"]
    for l in range(num_layers):
        a, b = A.get(l), B.get(l)
        if a is None or b is None:
            continue
        n_heads = a.shape[0]
        aq, bq = a[:, skip:, :], b[:, skip:, :]
        na, nb = aq.shape[1], bq.shape[1]
        for h in range(n_heads):
            X, Y = aq[h], bq[h]
            mpc = mean_pairwise_cosine(X, Y)
            mmc = maxmatch_cosine(X, Y)
            l2m = l2_mean(X, Y)
            mmd = rbf_mmd2(X, Y, max_tokens=args.max_mmd_tokens, rng=rng)
            cka, status = cka_for_pair(X, Y, args.align_tokens_for_cka, rng)
            rows.append({
                "task_id": tid, "pair": pair, "layer": l, "head": h,
                "mean_pairwise_cosine": round(mpc, 6),
                "maxmatch_cosine": round(mmc, 6),
                "l2_mean": round(l2m, 6),
                "mmd_rbf": round(mmd, 6),
                "cka_linear": (round(cka, 6) if cka == cka else float("nan")),
                "cka_status": status,
                "n_tokens_a": na, "n_tokens_b": nb, "prefix_skip": skip,
            })
            hm.append({"task_id": tid, "pair": pair, "metric": "head_output_maxmatch_cosine",
                       "component": "head_output", "layer": l, "head": h, "value": round(mmc, 6)})
            hm.append({"task_id": tid, "pair": pair, "metric": "head_output_mmd_rbf",
                       "component": "head_output", "layer": l, "head": h, "value": round(mmd, 6)})
    return rows, hm


def compute_residual_stream_metrics(act_a, act_b, pair, tid, skip,
                                    num_layers, args, rng) -> Tuple[List[Dict], List[Dict]]:
    rows, hm = [], []
    A, B = act_a["hidden_states"], act_b["hidden_states"]
    # hidden_states[0] = embedding, hidden_states[l+1] = block l output
    for idx in sorted(A.keys()):
        a, b = A.get(idx), B.get(idx)
        if a is None or b is None:
            continue
        X, Y = a[skip:, :], b[skip:, :]
        na, nb = X.shape[0], Y.shape[0]
        mpc = mean_pairwise_cosine(X, Y)
        mmc = maxmatch_cosine(X, Y)
        cka, status = cka_for_pair(X, Y, args.align_tokens_for_cka, rng)
        mmd = rbf_mmd2(X, Y, max_tokens=args.max_mmd_tokens, rng=rng)
        rows.append({
            "task_id": tid, "pair": pair, "layer": idx,
            "mean_pairwise_cosine": round(mpc, 6),
            "maxmatch_cosine": round(mmc, 6),
            "cka_linear": (round(cka, 6) if cka == cka else float("nan")),
            "cka_status": status,
            "mmd_rbf": round(mmd, 6),
            "n_tokens_a": na, "n_tokens_b": nb, "prefix_skip": skip,
        })
        hm.append({"task_id": tid, "pair": pair, "metric": "residual_cka_linear",
                   "component": "residual", "layer": idx, "head": -1,
                   "value": (round(cka, 6) if cka == cka else float("nan"))})
        hm.append({"task_id": tid, "pair": pair, "metric": "residual_maxmatch_cosine",
                   "component": "residual", "layer": idx, "head": -1, "value": round(mmc, 6)})
    return rows, hm


# =====================================================================
# Output helpers
# =====================================================================
def write_csv(path: str, rows: List[Dict]):
    if not rows:
        print(f"  (no rows for {os.path.basename(path)})")
        return
    # union of keys preserves first-row order then appends any extra
    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}  ({len(rows)} rows)")


def save_heatmap_long_csv(path: str, rows: List[Dict]):
    write_csv(path, rows)


def plot_raw_heatmap(heatmap_rows: List[Dict], metric: str, component: str,
                     out_path: str, num_layers: int, normalize: str = "raw"):
    """Aggregate heatmap_rows (mean over tasks) into a [layer x head] grid and save PNG.

    normalize: 'raw' uses values as-is; 'global' min-max normalizes the whole grid.
    """
    cells = defaultdict(list)
    max_head = 0
    for r in heatmap_rows:
        if r["metric"] != metric or r["component"] != component:
            continue
        v = r["value"]
        if v != v:   # NaN
            continue
        cells[(r["layer"], r["head"])].append(v)
        max_head = max(max_head, r["head"])
    if not cells:
        return
    is_residual = (max_head < 0) or (component == "residual")
    n_head = 1 if is_residual else max_head + 1
    grid = np.full((num_layers, n_head), np.nan)
    for (l, h), vals in cells.items():
        col = 0 if is_residual else h
        if 0 <= l < num_layers and 0 <= col < n_head:
            grid[l, col] = float(np.mean(vals))

    plot_grid = grid.copy()
    if normalize == "global":
        finite = plot_grid[np.isfinite(plot_grid)]
        if finite.size:
            lo, hi = finite.min(), finite.max()
            if hi - lo > 1e-12:
                plot_grid = (plot_grid - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(max(6, n_head * 0.5), max(6, num_layers * 0.3)))
    im = ax.imshow(plot_grid, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title(f"{component} · {metric} ({normalize})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Head" if not is_residual else "(residual)")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(0, num_layers, max(1, num_layers // 12)))
    if not is_residual:
        ax.set_xticks(range(0, n_head, max(1, n_head // 16)))
    fig.colorbar(im, ax=ax, shrink=0.7,
                 label="value" if normalize == "raw" else "global-normalized")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out_path}")


def optional_pca_plot(act_by_ver: Dict[str, Dict], component: str,
                      num_layers: int, out_path: str):
    """Disabled by default. Minimal per-layer PCA scatter of head_output / Q tokens."""
    try:
        from sklearn.decomposition import PCA
    except Exception:
        print("  (PCA requested but sklearn unavailable — skipping)")
        return
    colors = {"original": "tab:red", "version1": "tab:purple",
              "version2": "tab:green", "version3": "tab:blue"}
    per_fig = 16
    n_fig = math.ceil(num_layers / per_fig)
    for fi in range(n_fig):
        s, e = fi * per_fig, min((fi + 1) * per_fig, num_layers)
        fig, axes = plt.subplots(4, 4, figsize=(18, 18))
        for pos in range(16):
            ax = axes[pos // 4][pos % 4]
            li = s + pos
            if li >= num_layers:
                ax.set_visible(False); continue
            mats, cols = [], []
            for ver in VER_LABELS:
                comp = act_by_ver.get(ver)
                if not comp:
                    continue
                arr = comp[component].get(li)
                if arr is None:
                    continue
                flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr
                mats.append(flat); cols += [ver] * flat.shape[0]
            if not mats:
                ax.set_visible(False); continue
            allm = np.vstack(mats)
            if allm.shape[0] < 3:
                ax.set_visible(False); continue
            try:
                proj = PCA(n_components=2).fit_transform(allm)
            except Exception:
                ax.set_visible(False); continue
            for ver in VER_LABELS:
                m = [i for i, c in enumerate(cols) if c == ver]
                if m:
                    ax.scatter(proj[m, 0], proj[m, 1], s=5, alpha=0.5,
                               c=colors[ver], label=ver)
            ax.set_title(f"L{li}", fontsize=8)
            ax.tick_params(labelsize=6)
        p = out_path.replace("{layers}", f"L{s:02d}-{e-1:02d}")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    → {p}")


# =====================================================================
# Optional decode analysis (deterministic by default, JS/t-SNE removed)
# =====================================================================
@torch.no_grad()
def analyze_decode_baseline(wrapper, qs, args, num_layers, tid, model_tag, plots_dir):
    """Light decode-stage K-cache cosine analysis. Optional, off by default.

    Generation mode is deterministic (greedy) when --deterministic is set, else
    sampling (recorded as decode_mode=free_generation in metadata). To stay
    version-robust we (1) generate token ids, then (2) re-run a single forward
    over prompt+generated tokens with use_cache=True and read the K cache for the
    decode positions, instead of relying on generate() returning past_key_values.
    """
    def gen_keys(question):
        msgs = build_agent_messages_single_agent(question=question, args=args)
        _, ids, mask, _ = wrapper.prepare_chat_input(msgs)
        do_sample = not args.deterministic
        gen = wrapper.model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=args.decode_analysis_steps,
            do_sample=do_sample,
            temperature=(args.temperature if do_sample else None),
            top_p=(args.top_p if do_sample else None),
            pad_token_id=wrapper.tokenizer.pad_token_id,
            use_cache=True,
        )
        seqs = gen[0] if isinstance(gen, (tuple, list)) else gen
        prompt_len = ids.shape[1]
        full = seqs[:, : prompt_len + args.decode_analysis_steps]
        full_mask = torch.ones_like(full)
        out = wrapper.model(input_ids=full, attention_mask=full_mask,
                            use_cache=True, return_dict=True)
        pkv = out.past_key_values
        keys = {}
        for l in range(num_layers):
            if hasattr(pkv, "key_cache"):
                kl = pkv.key_cache[l]
            else:
                kl = pkv[l][0] if isinstance(pkv[l], (tuple, list)) else pkv[l]
            keys[l] = kl[0, :, prompt_len:, :].detach().cpu().float().numpy()
        del out
        return keys

    key_by_ver = {}
    for ver in VER_LABELS:
        if qs.get(ver):
            try:
                key_by_ver[ver] = gen_keys(qs[ver])
            except Exception as ex:
                print(f"  decode gen failed for {ver}: {ex}")
    rows = []
    for (va, vb) in PAIRS:
        ka, kb = key_by_ver.get(va), key_by_ver.get(vb)
        if not ka or not kb:
            continue
        for l in range(num_layers):
            a, b = ka.get(l), kb.get(l)
            if a is None or b is None or a.size == 0 or b.size == 0:
                continue
            steps = min(a.shape[1], b.shape[1])
            for s in range(steps):
                X = a[:, s, :]; Y = b[:, s, :]
                rows.append({
                    "task_id": tid, "pair": f"{va}_vs_{vb}", "layer": l,
                    "decode_step": s,
                    "mean_pairwise_cosine": round(mean_pairwise_cosine(X, Y), 6),
                    "maxmatch_cosine": round(maxmatch_cosine(X, Y), 6),
                })
    return rows


# =====================================================================
# Main analysis (baseline = primary)
# =====================================================================
def analyze_baseline_headwise(wrapper, items, args, plots_dir, model_tag):
    cfg = wrapper.model.config
    num_layers = cfg.num_hidden_layers
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_q)
    print(f"  GQA: {n_q} Q-heads -> {n_kv} KV-heads | head_dim={head_dim} | layers={num_layers}")

    rng = np.random.default_rng(args.seed)
    prefix = f"baseline_{args.task}_{model_tag}"

    qkv_rows, ho_rows, res_rows, hm_rows, decode_rows = [], [], [], [], []
    # for optional PCA on the last task only (keeps memory bounded)
    last_acts: Dict[str, Dict] = {}

    for ti, item in enumerate(items):
        tid = item.get("task_id", ti)
        qs = {"original": item["question"]}
        for v in ["version1", "version2", "version3"]:
            qs[v] = item.get(v, "")
        print(f"  task {tid}  ({ti + 1}/{len(items)})")

        acts: Dict[str, Dict] = {}
        prompts: Dict[str, str] = {}
        for ver in VER_LABELS:
            q = qs.get(ver)
            if not q:
                continue
            a = collect_baseline_prefill_activations(wrapper, q, args)
            acts[ver] = a
            prompts[ver] = a["prompt_text"]
            if args.save_raw_tensors and ti < 1:
                np.savez_compressed(
                    os.path.join(plots_dir, f"{prefix}_raw_tensors_task{tid}_{ver}.npz"),
                    **{f"q_{l}": a["q"][l] for l in a["q"]},
                    **{f"k_{l}": a["k"][l] for l in a["k"]},
                    **{f"v_{l}": a["v"][l] for l in a["v"]},
                    **{f"ho_{l}": a["head_output"][l] for l in a["head_output"]},
                    **{f"hs_{l}": a["hidden_states"][l] for l in a["hidden_states"]},
                )
            gc.collect(); torch.cuda.empty_cache()

        # common prefix per pair
        for (va, vb) in PAIRS:
            if va not in acts or vb not in acts:
                continue
            skip = common_prefix_len(wrapper.tokenizer, prompts[va], prompts[vb])
            pair = f"{va}_vs_{vb}"
            if ti == 0 and va == "original" and vb == "version1":
                tot = next(iter(acts[va]["q"].values())).shape[1]
                print(f"    prefix-skip={skip}  total_tokens={tot}  question-only={tot - skip}")

            r1, h1 = compute_headwise_qkv_metrics(acts[va], acts[vb], pair, tid, skip,
                                                  num_layers, args, rng)
            r2, h2 = compute_head_output_metrics(acts[va], acts[vb], pair, tid, skip,
                                                 num_layers, args, rng)
            r3, h3 = compute_residual_stream_metrics(acts[va], acts[vb], pair, tid, skip,
                                                     num_layers, args, rng)
            qkv_rows += r1; ho_rows += r2; res_rows += r3
            hm_rows += h1 + h2 + h3

        if args.analyze_decode:
            decode_rows += analyze_decode_baseline(
                wrapper, qs, args, num_layers, tid, model_tag, plots_dir)

        if ti == len(items) - 1 and not args.disable_pca:
            last_acts = acts
        else:
            del acts; gc.collect(); torch.cuda.empty_cache()

    # ── write required CSVs ──
    write_csv(os.path.join(plots_dir, f"{prefix}_prefill_qkv_headwise_metrics.csv"), qkv_rows)
    write_csv(os.path.join(plots_dir, f"{prefix}_prefill_head_output_metrics.csv"), ho_rows)
    write_csv(os.path.join(plots_dir, f"{prefix}_prefill_residual_stream_metrics.csv"), res_rows)
    save_heatmap_long_csv(os.path.join(plots_dir, f"{prefix}_heatmap_long.csv"), hm_rows)
    if args.analyze_decode and decode_rows:
        write_csv(os.path.join(plots_dir, f"{prefix}_decode_key_metrics.csv"), decode_rows)

    # ── optional component heatmap raw CSVs ──
    def hm_subset(metric, component):
        return [r for r in hm_rows if r["metric"] == metric and r["component"] == component]
    write_csv(os.path.join(plots_dir, f"{prefix}_heatmap_q_cosine_raw.csv"),
              hm_subset("maxmatch_cosine", "q"))
    write_csv(os.path.join(plots_dir, f"{prefix}_heatmap_k_cosine_raw.csv"),
              hm_subset("maxmatch_cosine", "k"))
    write_csv(os.path.join(plots_dir, f"{prefix}_heatmap_v_cosine_raw.csv"),
              hm_subset("maxmatch_cosine", "v"))
    write_csv(os.path.join(plots_dir, f"{prefix}_heatmap_head_output_cosine_raw.csv"),
              hm_subset("head_output_maxmatch_cosine", "head_output"))
    write_csv(os.path.join(plots_dir, f"{prefix}_heatmap_residual_cka_raw.csv"),
              hm_subset("residual_cka_linear", "residual"))

    # ── optional PNG heatmaps (raw + global only) ──
    if args.make_heatmaps:
        specs = [
            ("maxmatch_cosine", "q", "q_maxmatch_cosine"),
            ("maxmatch_cosine", "k", "k_maxmatch_cosine"),
            ("maxmatch_cosine", "v", "v_maxmatch_cosine"),
            ("head_output_maxmatch_cosine", "head_output", "head_output_maxmatch_cosine"),
            ("head_output_mmd_rbf", "head_output", "head_output_mmd_rbf"),
            ("residual_maxmatch_cosine", "residual", "residual_maxmatch_cosine"),
        ]
        for metric, component, tag in specs:
            for norm in ("raw", "global"):
                suffix = "raw" if norm == "raw" else "globalnorm"
                plot_raw_heatmap(
                    hm_rows, metric, component,
                    os.path.join(plots_dir, f"{prefix}_heatmap_{tag}_{suffix}.png"),
                    num_layers, normalize=norm)

    # ── optional PCA (disabled by default) ──
    if not args.disable_pca and last_acts:
        optional_pca_plot(last_acts, "head_output", num_layers,
                          os.path.join(plots_dir, f"{prefix}_pca_head_output_{{layers}}.png"))

    # ── metadata ──
    meta = {
        "model_name": args.model_name,
        "task": args.task,
        "method": "baseline",
        "primary_analysis_method": args.primary_analysis_method,
        "seed": args.seed,
        "versions": VER_LABELS,
        "num_layers": num_layers,
        "num_attention_heads": n_q,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "analysis_phase": "prefill",
        "removed_metrics": ["js_divergence", "tsne"],
        "default_decode_analyzed": bool(args.analyze_decode),
        "decode_mode": ("deterministic" if args.deterministic else "free_generation"),
        "max_mmd_tokens": args.max_mmd_tokens,
        "align_tokens_for_cka": bool(args.align_tokens_for_cka),
        "num_tasks": len(items),
        "metrics": {
            "qkv": ["mean_pairwise_cosine", "maxmatch_cosine", "cka_linear", "mmd_rbf"],
            "head_output": ["mean_pairwise_cosine", "maxmatch_cosine", "l2_mean",
                            "mmd_rbf", "cka_linear"],
            "residual_stream": ["mean_pairwise_cosine", "maxmatch_cosine",
                                "cka_linear", "mmd_rbf"],
        },
        "note": {
            "qkv": "Q/K/V metrics are measured before attention output (Q/K after RoPE, V raw).",
            "head_output": "head_output metrics are measured after attention_probs @ V and before o_proj.",
            "residual_stream": "residual_stream metrics are measured after each transformer block output (hidden_states[l+1]); index 0 is the embedding output.",
            "mmd": "MMD is a distance: smaller means more similar.",
            "js_removed": "JS divergence removed because Q/K/V/hidden vectors are not probability distributions.",
            "tsne_removed": "t-SNE removed because it is visualization-only and unstable for quantitative interpretation.",
            "gqa": "Q stored per query-head; K/V stored per kv-head; head_output stored per query-head.",
        },
    }
    meta_path = os.path.join(plots_dir, f"{prefix}_run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  metadata → {meta_path}")


# =====================================================================
# Kept (compatibility) extractors for text_mas / latent_mas
# These remain importable/dispatchable but do NOT run the head-wise analysis.
# =====================================================================
@torch.no_grad()
def extract_text_mas(wrapper, question, args, num_layers):  # pragma: no cover
    raise NotImplementedError(
        "text_mas head-wise analysis is intentionally not implemented in v4. "
        "The pipeline effect mixes agent stages; use --primary_analysis_method baseline.")


@torch.no_grad()
def extract_latent_mas(wrapper, question, args, num_layers):  # pragma: no cover
    raise NotImplementedError(
        "latent_mas head-wise analysis is intentionally not implemented in v4. "
        "Use --primary_analysis_method baseline for the head-wise sensitivity study.")


def analyze_method(wrapper, method, items, args, plots_dir, model_tag):
    """Dispatch kept for compatibility. New head-wise analysis runs only for the
    primary analysis method (baseline)."""
    if method == args.primary_analysis_method == "baseline":
        analyze_baseline_headwise(wrapper, items, args, plots_dir, model_tag)
    else:
        print(f"  [skip] method '{method}' is retained for dispatch but the head-wise "
              f"prefill analysis only runs for primary_analysis_method="
              f"'{args.primary_analysis_method}'.")


# =====================================================================
# Entry
# =====================================================================
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_path", required=True)
    pa.add_argument("--task", default="gpqa",
                    choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy",
                             "arc_challenge", "mbppplus", "humanevalplus", "medqa"])
    pa.add_argument("--model_name", default="Qwen/Qwen3-8B")
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--methods", nargs="+", default=["baseline"],
                    choices=["baseline", "text_mas", "latent_mas"])
    pa.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    pa.add_argument("--max_tasks", type=int, default=3)
    pa.add_argument("--latent_steps", type=int, default=10)
    pa.add_argument("--temperature", type=float, default=0.6)
    pa.add_argument("--top_p", type=float, default=0.95)
    pa.add_argument("--max_new_tokens", type=int, default=4096)
    pa.add_argument("--decode_analysis_steps", type=int, default=64,
                    help="Decode tokens to record when --analyze_decode is set")
    pa.add_argument("--text_mas_context_length", type=int, default=-1)
    pa.add_argument("--think", action="store_true")
    pa.add_argument("--latent_space_realign", action="store_true")
    pa.add_argument("--plots_dir", default="./plots_headwise")
    pa.add_argument("--seed", type=int, default=42)

    # ── new head-wise analysis options ──
    pa.add_argument("--primary_analysis_method", default="baseline",
                    choices=["baseline", "text_mas", "latent_mas"],
                    help="Method on which the head-wise prefill analysis is run.")
    pa.add_argument("--analyze_decode", action="store_true", default=False,
                    help="Also run optional decode-stage K-cache cosine analysis.")
    pa.add_argument("--save_raw_tensors", action="store_true", default=False,
                    help="Save per-version Q/K/V/head_output/hidden as .npz (task 0).")
    pa.add_argument("--max_mmd_tokens", type=int, default=256)
    pa.add_argument("--align_tokens_for_cka", action="store_true", default=False,
                    help="If set, resample to min(n_a,n_b) so CKA can be computed for "
                         "mismatched token counts.")
    pa.add_argument("--make_heatmaps", action="store_true", default=True)
    pa.add_argument("--no_heatmaps", dest="make_heatmaps", action="store_false")
    pa.add_argument("--disable_pca", action="store_true", default=True)
    pa.add_argument("--enable_pca", dest="disable_pca", action="store_false")
    pa.add_argument("--deterministic", action="store_true", default=False,
                    help="Use do_sample=False (temperature ignored) for decode.")
    pa.add_argument("--do_sample", dest="deterministic", action="store_false")

    args = pa.parse_args()

    # fixed ModelWrapper expectations (HF backend, no vLLM for analysis)
    args.use_vllm = False
    args.use_second_HF_model = False
    args.enable_prefix_caching = False
    args.device2 = "cpu"
    args.tensor_parallel_size = 1
    args.gpu_memory_utilization = 0.9
    args.method = "baseline"
    if args.deterministic:
        args.temperature = 0.0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.plots_dir, exist_ok=True)

    items = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    if args.max_tasks > 0:
        items = items[:args.max_tasks]
    print(f"Loaded {len(items)} tasks from {args.data_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    model_tag = args.model_name.split("/")[-1]

    # NOTE: we deliberately keep the model's default attention implementation
    # (sdpa). The submodule hooks (q/k/v/o_proj) and the o_proj pre-hook fire
    # identically under sdpa/flash/eager, so forcing eager would only waste
    # memory by materializing the full [n_q, seq, seq] score matrix.

    for method in args.methods:
        print(f"\n{'=' * 60}\n  Method: {method}\n{'=' * 60}")
        args.method = method
        analyze_method(wrapper, method, items, args, args.plots_dir, model_tag)

    print(f"\nAll done. Outputs in {args.plots_dir}/")


if __name__ == "__main__":
    main()
