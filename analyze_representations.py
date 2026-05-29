#!/usr/bin/env python3
"""
analyze_representations.py  (v3)
================================
Uses original project code (models.py, prompts.py) directly.
No modifications to original files needed.

Fixes in v3:
 - GQA-aware: key cache is [kv_heads, seq, head_dim], heatmap labels explicit
 - Single-pass: prefill+decode extracted together (no double-prefill)
 - Decode CSV: per-step × per-layer rows (decode_step column)
 - Full generation via generate_text_batch, keys extracted from returned past
 - Activation heatmap: per-layer normalization for visibility
 - text_mas agent alpha: wider spread (1.0 / 0.55 / 0.25 / 0.08)
"""
import argparse, csv, gc, json, math, os, re, sys
from collections import namedtuple, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import jensenshannon

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

# ── agents ───────────────────────────────────────────────────────────
Agent = namedtuple("Agent", ["name", "role"])
AGENTS = [Agent("Planner","planner"), Agent("Critic","critic"),
          Agent("Refiner","refiner"), Agent("Judger","judger")]
HIER_NAME = {"Planner":"Math Agent","Critic":"Science Agent",
             "Refiner":"Code Agent","Judger":"Task Summrizer"}

# ── visual constants ─────────────────────────────────────────────────
AGENT_ALPHA = {"planner":1.0, "critic":0.55, "refiner":0.25, "judger":0.08}
VER_LABELS  = ["original","version1","version2","version3"]
VER_COLORS  = {"original":(1,0,0),"version1":(0.6,0,0.8),
               "version2":(0,0.7,0),"version3":(0,0.3,1)}
MAX_PCA_TOKENS = 200

# =====================================================================
# Helpers
# =====================================================================
def strip_thinking(text):
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()

def extract_keys_from_past(past, num_layers=None):
    """Full key extraction: dict[layer] → np [kv_heads, seq, head_dim]."""
    keys = {}
    if hasattr(past, "key_cache"):
        n = len(past.key_cache) if num_layers is None else num_layers
        for i in range(n):
            keys[i] = past.key_cache[i][0].detach().cpu().float().numpy()
    else:
        n = len(past) if num_layers is None else num_layers
        for i in range(n):
            kl = past[i][0] if isinstance(past[i], (tuple, list)) else past[i]
            keys[i] = kl[0].detach().cpu().float().numpy()
    return keys

def extract_keys_slice(past, start, end, num_layers):
    """Extract keys for positions [start:end] only — memory efficient."""
    keys = {}
    if hasattr(past, "key_cache"):
        for l in range(min(num_layers, len(past.key_cache))):
            keys[l] = past.key_cache[l][0, :, start:end, :].detach().cpu().float().numpy()
    else:
        for l in range(min(num_layers, len(past))):
            kl = past[l][0] if isinstance(past[l], (tuple, list)) else past[l]
            keys[l] = kl[0, :, start:end, :].detach().cpu().float().numpy()
    return keys

def softmax_dist(v):
    v = v - v.max(); e = np.exp(v)
    return e / (e.sum() + 1e-12)

def cosine_sim(a, b):
    af, bf = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / d) if d > 1e-12 else 0.0

def js_divergence(a, b):
    p, q = softmax_dist(a.flatten().astype(np.float64)), softmax_dist(b.flatten().astype(np.float64))
    return float(jensenshannon(p, q, base=2) ** 2)

# ── common-prefix detection (strips system prompt, <think>, etc.) ──
def common_prefix_len(tokenizer, prompt_a, prompt_b):
    """Number of leading tokens shared by two prompts (system msg, formatting, etc.)."""
    ids_a = tokenizer(prompt_a, add_special_tokens=False)["input_ids"]
    ids_b = tokenizer(prompt_b, add_special_tokens=False)["input_ids"]
    for i in range(min(len(ids_a), len(ids_b))):
        if ids_a[i] != ids_b[i]:
            return i
    return min(len(ids_a), len(ids_b))

# ── per-layer prefill metrics (pairwise cosine on question-portion keys) ──
def layer_metrics_prefill(ka, kb, num_layers, skip_a=0, skip_b=0):
    """Mean pairwise cosine & JS between question-specific token keys.
    ka/kb: dict[layer] → np [kv_heads, seq, head_dim].
    skip_a/skip_b: shared-prefix tokens to exclude from metrics.
    """
    res = []
    for l in range(num_layers):
        a, b = ka.get(l), kb.get(l)
        if a is None or b is None or a.size == 0 or b.size == 0:
            res.append((0., 1.)); continue
        # strip shared prefix → question-specific portion
        aq = a[:, skip_a:, :]   # [heads, n_a, dim]
        bq = b[:, skip_b:, :]   # [heads, n_b, dim]
        na, nb = aq.shape[1], bq.shape[1]
        if na == 0 or nb == 0:
            res.append((0., 1.)); continue
        # flatten heads → [n, heads*dim]
        af = aq.transpose(1, 0, 2).reshape(na, -1).astype(np.float64)
        bf = bq.transpose(1, 0, 2).reshape(nb, -1).astype(np.float64)
        # L2-normalise rows
        af /= (np.linalg.norm(af, axis=1, keepdims=True) + 1e-12)
        bf /= (np.linalg.norm(bf, axis=1, keepdims=True) + 1e-12)
        # mean pairwise cosine  (not cosine of means)
        cos_mat = af @ bf.T                       # [na, nb]
        cos_mean = float(cos_mat.mean())
        # JS on mean vectors of the question portion
        a_mean = aq.mean(axis=1).flatten()
        b_mean = bq.mean(axis=1).flatten()
        js = js_divergence(a_mean, b_mean)
        res.append((cos_mean, js))
    return res

# ── per-step metrics (for decode) ──
def layer_metrics_step(ka, kb, layer, step):
    """Single (cos, js) for one layer, one decode step."""
    a, b = ka.get(layer), kb.get(layer)
    if a is None or b is None:
        return (0., 1.)
    if a.shape[1] <= step or b.shape[1] <= step:
        return (0., 1.)
    av = a[:, step, :].flatten()
    bv = b[:, step, :].flatten()
    return (cosine_sim(av, bv), js_divergence(av, bv))

def activation_scores(keys, num_layers, num_kv_heads):
    """[num_layers, num_kv_heads] mean L2 norm of key vectors."""
    act = np.zeros((num_layers, num_kv_heads), dtype=np.float64)
    for l in range(num_layers):
        k = keys.get(l)
        if k is None or k.size == 0: continue
        norms = np.linalg.norm(k, axis=2)          # [heads, seq]
        act[l, :min(num_kv_heads, norms.shape[0])] = norms.mean(axis=1)[:num_kv_heads]
    return act

# =====================================================================
# Key extraction per method  (single-pass: generate → slice keys)
# =====================================================================
@torch.no_grad()
def _gen_and_extract(wrapper, ids, mask, max_new_tokens, n_analysis,
                     num_layers, past_kv=None, temperature=0.6, top_p=0.95):
    """Generate full response, then extract prefill+decode keys from returned past.
    Returns (prefill_keys, decode_keys, generated_text)."""
    acc_len = _past_length(past_kv) if past_kv else 0
    prompt_len = ids.shape[1]

    gens, gen_past = wrapper.generate_text_batch(
        ids, mask, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
        past_key_values=past_kv)

    gen_text = gens[0]
    prefill_keys = None
    decode_keys  = None

    if gen_past is not None:
        total_in_cache = _past_length(gen_past)
        decode_start = acc_len + prompt_len
        gen_len = total_in_cache - decode_start
        n_dec = min(n_analysis, gen_len)

        prefill_keys = extract_keys_slice(gen_past, 0, decode_start, num_layers)
        decode_keys  = extract_keys_slice(gen_past, decode_start,
                                           decode_start + n_dec, num_layers)
        del gen_past
    else:
        # Fallback: re-forward prompt + first n_analysis gen tokens
        gen_ids = wrapper.tokenize_text(gen_text)
        n_dec = min(n_analysis, gen_ids.shape[1])
        analysis_ids = torch.cat([ids, gen_ids[:, :n_dec]], dim=1)
        analysis_mask = torch.ones_like(analysis_ids)
        if past_kv is not None:
            pl = _past_length(past_kv)
            pm = torch.ones((1, pl), dtype=analysis_mask.dtype, device=wrapper.device)
            analysis_mask = torch.cat([pm, analysis_mask], dim=1)
            cpos = torch.arange(pl, pl + analysis_ids.shape[1],
                                dtype=torch.long, device=wrapper.device)
        else:
            cpos = None
        out = wrapper.model(analysis_ids, attention_mask=analysis_mask,
                            past_key_values=past_kv, use_cache=True,
                            cache_position=cpos, return_dict=True)
        decode_start = acc_len + prompt_len
        prefill_keys = extract_keys_slice(out.past_key_values, 0, decode_start, num_layers)
        decode_keys  = extract_keys_slice(out.past_key_values, decode_start,
                                           decode_start + n_dec, num_layers)
        del out

    torch.cuda.empty_cache()
    return prefill_keys, decode_keys, gen_text

# ── baseline ─────────────────────────────────────────────────────────
@torch.no_grad()
def extract_baseline(wrapper, question, args, num_layers):
    msgs = build_agent_messages_single_agent(question=question, args=args)
    _, ids, mask, _ = wrapper.prepare_chat_input(msgs)
    return _gen_and_extract(wrapper, ids, mask, args.max_new_tokens,
                            args.decode_analysis_steps, num_layers,
                            temperature=args.temperature, top_p=args.top_p)

# ── text_mas ─────────────────────────────────────────────────────────
@torch.no_grad()
def extract_text_mas(wrapper, question, args, num_layers):
    context = ""
    prefill_per_agent = {}

    for ag in AGENTS:
        if args.prompt == "hierarchical":
            msgs = build_agent_messages_hierarchical_text_mas(
                role=ag.role, question=question, context=context,
                method="text_mas", args=args)
        else:
            msgs = build_agent_messages_sequential_text_mas(
                role=ag.role, question=question, context=context,
                method="text_mas", args=args)
        _, ids, mask, _ = wrapper.prepare_chat_input(msgs)

        if ag.role != "judger":
            # intermediate agent: generate for context + extract prefill
            gens, ag_past = wrapper.generate_text_batch(
                ids, mask, max_new_tokens=args.gen_max_tokens,
                temperature=args.temperature, top_p=args.top_p)
            text_out = strip_thinking(gens[0])

            # extract prefill keys (prompt portion only)
            prompt_len = ids.shape[1]
            if ag_past is not None:
                prefill_per_agent[ag.role] = extract_keys_slice(
                    ag_past, 0, prompt_len, num_layers)
                del ag_past
            else:
                out = wrapper.model(ids, attention_mask=mask, use_cache=True, return_dict=True)
                prefill_per_agent[ag.role] = extract_keys_from_past(out.past_key_values, num_layers)
                del out

            if args.prompt == "hierarchical":
                context += f"[{HIER_NAME[ag.name]}]:\n{text_out}\n\n"
            else:
                context += f"[{ag.name}]:\n{text_out}\n\n"
            torch.cuda.empty_cache()
        else:
            pk, dk, _ = _gen_and_extract(
                wrapper, ids, mask, args.max_new_tokens,
                args.decode_analysis_steps, num_layers,
                temperature=args.temperature, top_p=args.top_p)
            prefill_per_agent[ag.role] = pk

    return prefill_per_agent, dk

# ── latent_mas ───────────────────────────────────────────────────────
@torch.no_grad()
def extract_latent_mas(wrapper, question, args, num_layers):
    accumulated_past = None
    device = wrapper.device

    for ag in AGENTS:
        if args.prompt == "sequential":
            msgs = build_agent_message_sequential_latent_mas(
                role=ag.role, question=question, context="",
                method="latent_mas", args=args)
        else:
            msgs = build_agent_message_hierarchical_latent_mas(
                role=ag.role, question=question, context="",
                method="latent_mas", args=args)
        prompt_text = wrapper.render_chat(msgs, add_generation_prompt=True)

        if ag.role != "judger":
            wrapped = f"{prompt_text}<think>" if args.think else prompt_text
            enc = wrapper.tokenizer(wrapped, return_tensors="pt", add_special_tokens=False)
            w_ids = enc["input_ids"].to(device)
            w_mask = enc["attention_mask"].to(device)
            accumulated_past = wrapper.generate_latent_batch(
                w_ids, w_mask, latent_steps=args.latent_steps,
                past_key_values=accumulated_past)
        else:
            judger_text = f"{prompt_text}<think>" if args.think else prompt_text
            enc = wrapper.tokenizer(judger_text, return_tensors="pt", add_special_tokens=False)
            j_ids = enc["input_ids"].to(device)
            j_mask = enc["attention_mask"].to(device)
            past_for_gen = accumulated_past if args.latent_steps > 0 else None
            pk, dk, _ = _gen_and_extract(
                wrapper, j_ids, j_mask, args.max_new_tokens,
                args.decode_analysis_steps, num_layers,
                past_kv=past_for_gen,
                temperature=args.temperature, top_p=args.top_p)

    del accumulated_past; torch.cuda.empty_cache()
    return pk, dk

# =====================================================================
# Plotting
# =====================================================================
def plot_scatter_grid(layer_data, num_layers, out_tpl, title_prefix, use_tsne):
    layers_per_fig = 16
    for fi in range(math.ceil(num_layers / layers_per_fig)):
        s, e = fi*layers_per_fig, min((fi+1)*layers_per_fig, num_layers)
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        tag = "t-SNE" if use_tsne else "PCA"
        fig.suptitle(f"{title_prefix} ({tag}) — Layers {s}–{e-1}", fontsize=14)
        for pos in range(16):
            ax = axes[pos//4][pos%4]; li = s+pos
            if li >= num_layers or li not in layer_data:
                ax.set_visible(False); continue
            ax.set_title(f"Layer {li}", fontsize=9)
            for pts, col, alpha, label in layer_data[li]:
                if len(pts): ax.scatter(pts[:,0], pts[:,1], c=[col], alpha=alpha, s=6, label=label)
            ax.tick_params(labelsize=6)
        seen = {}
        for pos in range(min(e-s,16)):
            for h, lb in zip(*axes[pos//4][pos%4].get_legend_handles_labels()):
                if lb not in seen: seen[lb] = h
        if seen: fig.legend(seen.values(), seen.keys(), loc="upper right", fontsize=7)
        plt.tight_layout(rect=[0,0,1,0.96])
        t = "tsne" if use_tsne else "pca"
        p = out_tpl.replace("{tag}",t).replace("{layers}",f"layers{s:02d}-{e-1:02d}")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"    → {p}")

def dim_reduce_plots(vecs, num_layers, out_tpl, title):
    for use_tsne in [False, True]:
        ld = {}
        for l in range(num_layers):
            items = vecs.get(l, [])
            all_pts = [v[0] for v in items if len(v[0])>0]
            if not all_pts: continue
            combined = np.vstack(all_pts)
            if combined.shape[0] < 4: continue
            try:
                if use_tsne:
                    proj = TSNE(n_components=2, perplexity=max(2,min(30,combined.shape[0]-1)),
                                random_state=42, max_iter=500).fit_transform(combined)
                else:
                    proj = PCA(n_components=min(2,*combined.shape)).fit_transform(combined)
            except Exception: continue
            entries = []; off = 0
            for pts_o,c,a,lb in items:
                n=len(pts_o)
                entries.append((proj[off:off+n] if n else np.empty((0,2)), c, a, lb)); off+=n
            ld[l] = entries
        plot_scatter_grid(ld, num_layers, out_tpl, title, use_tsne)

def plot_activation_diff_grid(act_dict, path, title_prefix, num_layers, num_kv):
    """2×3 grid of layer×head activation-difference heatmaps.
    Row 1: orig↔ver1, orig↔ver2, orig↔ver3
    Row 2: ver1↔ver2, ver1↔ver3, ver2↔ver3
    """
    pairs = [
        ("original","version1"), ("original","version2"), ("original","version3"),
        ("version1","version2"), ("version1","version3"), ("version2","version3"),
    ]
    labels = [
        "orig ↔ ver1", "orig ↔ ver2", "orig ↔ ver3",
        "ver1 ↔ ver2", "ver1 ↔ ver3", "ver2 ↔ ver3",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(max(14, num_kv*2.2), max(12, num_layers*0.35)))
    fig.suptitle(title_prefix, fontsize=14, y=1.01)

    # compute all diffs first to find global max for shared color scale
    diffs = []
    for va, vb in pairs:
        a = act_dict.get(va, np.zeros((num_layers, num_kv)))
        b = act_dict.get(vb, np.zeros((num_layers, num_kv)))
        diff = np.abs(a - b)
        # per-layer normalize for visibility
        for l in range(num_layers):
            rmax = diff[l].max()
            if rmax > 1e-12:
                diff[l] /= rmax
        diffs.append(diff)

    for idx, (diff, label) in enumerate(zip(diffs, labels)):
        ax = axes[idx // 3][idx % 3]
        im = ax.imshow(diff, aspect="auto", cmap="inferno", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("KV Head (GQA)", fontsize=8)
        ax.set_ylabel("Layer", fontsize=8)
        ax.set_yticks(range(0, num_layers, max(1, num_layers//12)))
        ax.set_xticks(range(num_kv))
        ax.tick_params(labelsize=6)

    fig.colorbar(im, ax=axes, shrink=0.6, label="|Δ activation| (per-layer norm)")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {path}")

# =====================================================================
# Main analysis loop
# =====================================================================
def analyze_method(wrapper, method, items, args, plots_dir, model_tag):
    num_layers = wrapper.model.config.num_hidden_layers
    num_kv = getattr(wrapper.model.config, "num_key_value_heads",
                     wrapper.model.config.num_attention_heads)
    n_q = wrapper.model.config.num_attention_heads
    print(f"  GQA: {n_q} Q-heads → {num_kv} KV-heads (ratio {n_q//num_kv}:1)")
    n_dec_steps = args.decode_analysis_steps

    # per-phase accumulators
    P, D = "prefill", "decode"
    csv_rows = {P:[], D:[]}
    act_acc  = {ph: {v: np.zeros((num_layers, num_kv)) for v in VER_LABELS} for ph in [P,D]}
    act_n    = {ph: {v:0 for v in VER_LABELS} for ph in [P,D]}
    pca_vecs = {ph: defaultdict(list) for ph in [P,D]}

    for ti, item in enumerate(items):
        tid = item["task_id"]
        qs = {v: item.get(v if v != "original" else "question", "")
              for v in VER_LABELS}
        qs["original"] = item["question"]
        print(f"  task {tid}  ({ti+1}/{len(items)})")

        all_pf, all_dk = {}, {}       # ver → keys
        for ver in VER_LABELS:
            q = qs[ver]
            if not q:
                all_pf[ver] = all_dk[ver] = None; continue
            if method == "baseline":
                pk, dk, _ = extract_baseline(wrapper, q, args, num_layers)
                all_pf[ver] = pk; all_dk[ver] = dk
            elif method == "text_mas":
                pka, dk = extract_text_mas(wrapper, q, args, num_layers)
                all_pf[ver] = pka; all_dk[ver] = dk
            elif method == "latent_mas":
                pk, dk = extract_latent_mas(wrapper, q, args, num_layers)
                all_pf[ver] = pk; all_dk[ver] = dk
            gc.collect(); torch.cuda.empty_cache()

        # ── helper: "final" keys for metrics ──
        def fk(ver, phase):
            src = all_pf if phase == P else all_dk
            k = src.get(ver)
            if k is None: return None
            if method == "text_mas" and phase == P and isinstance(k, dict):
                return k.get("judger")
            return k

        # ── helper: render prompt text for common-prefix detection ──
        def render_prompt(question):
            if method == "baseline":
                msgs = build_agent_messages_single_agent(question=question, args=args)
            elif method == "text_mas":
                # use judger prompt (context="" — prefix is the same for all versions)
                if args.prompt == "hierarchical":
                    msgs = build_agent_messages_hierarchical_text_mas(
                        role="judger", question=question, context="",
                        method="text_mas", args=args)
                else:
                    msgs = build_agent_messages_sequential_text_mas(
                        role="judger", question=question, context="",
                        method="text_mas", args=args)
            elif method == "latent_mas":
                if args.prompt == "sequential":
                    msgs = build_agent_message_sequential_latent_mas(
                        role="judger", question=question, context="",
                        method="latent_mas", args=args)
                else:
                    msgs = build_agent_message_hierarchical_latent_mas(
                        role="judger", question=question, context="",
                        method="latent_mas", args=args)
            txt = wrapper.render_chat(msgs, add_generation_prompt=True)
            if args.think: txt += "<think>"
            return txt

        # ====== PREFILL metrics (pairwise cosine on question-specific tokens) ======
        orig_p = fk("original", P)
        if orig_p:
            # pre-compute common prefix lengths per pair
            prompt_cache = {}
            for ver in VER_LABELS:
                q = qs[ver]
                if q: prompt_cache[ver] = render_prompt(q)

            pair_skip = {}  # (va, vb) → common prefix token count
            for va, vb in [("original","version1"),("original","version2"),
                           ("original","version3"),("version1","version2"),
                           ("version1","version3"),("version2","version3")]:
                pa, pb = prompt_cache.get(va), prompt_cache.get(vb)
                if pa and pb:
                    pair_skip[(va,vb)] = common_prefix_len(wrapper.tokenizer, pa, pb)
                else:
                    pair_skip[(va,vb)] = 0

            pm = {}
            for a,b in pair_skip:
                ka,kb = fk(a,P), fk(b,P)
                skip = pair_skip[(a,b)]
                if ka and kb:
                    pm[(a,b)] = layer_metrics_prefill(ka, kb, num_layers,
                                                       skip_a=skip, skip_b=skip)
                    if a == "original" and b == "version1" and ti == 0:
                        total_a = next(iter(ka.values())).shape[1] if ka else 0
                        print(f"    prefix-skip={skip}  total_tokens={total_a}  "
                              f"question-only={total_a - skip}")

            for l in range(num_layers):
                row = {"task_id":tid, "layer":l}
                for v in ["version1","version2","version3"]:
                    m=pm.get(("original",v))
                    row[f"{v}_cos"]=round(m[l][0],3) if m else float("nan")
                    row[f"{v}_js"] =round(m[l][1],3) if m else float("nan")
                for lbl,va,vb in [("ver1to2","version1","version2"),
                                  ("ver1to3","version1","version3"),
                                  ("ver2to3","version2","version3")]:
                    m=pm.get((va,vb))
                    row[f"{lbl}_cos"]=round(m[l][0],3) if m else float("nan")
                    row[f"{lbl}_js"] =round(m[l][1],3) if m else float("nan")
                csv_rows[P].append(row)

        # ====== DECODE metrics (per-step × per-layer) ======
        orig_d = fk("original", D)
        if orig_d:
            actual_steps = min(n_dec_steps,
                               min(orig_d[l].shape[1] for l in orig_d if orig_d[l].size > 0)
                               if orig_d else 0)
            for step in range(actual_steps):
                for l in range(num_layers):
                    row = {"task_id":tid, "decode_step":step, "layer":l}
                    for v in ["version1","version2","version3"]:
                        kv = fk(v, D)
                        if kv:
                            c,j = layer_metrics_step(orig_d, kv, l, step)
                        else:
                            c,j = float("nan"), float("nan")
                        row[f"{v}_cos"]=round(c,3); row[f"{v}_js"]=round(j,3)
                    for lbl,va,vb in [("ver1to2","version1","version2"),
                                      ("ver1to3","version1","version3"),
                                      ("ver2to3","version2","version3")]:
                        ka2, kb2 = fk(va,D), fk(vb,D)
                        if ka2 and kb2:
                            c,j = layer_metrics_step(ka2,kb2,l,step)
                        else:
                            c,j = float("nan"),float("nan")
                        row[f"{lbl}_cos"]=round(c,3); row[f"{lbl}_js"]=round(j,3)
                    csv_rows[D].append(row)

        # ====== Activation + PCA/t-SNE for BOTH phases ======
        for ph in [P, D]:
            for ver in VER_LABELS:
                k = fk(ver, ph)
                if k is None: continue
                # activation running mean
                a = activation_scores(k, num_layers, num_kv)
                n = act_n[ph][ver]
                act_acc[ph][ver] = (act_acc[ph][ver]*n + a)/(n+1)
                act_n[ph][ver] += 1

            # PCA/t-SNE collection
            if method == "text_mas" and ph == P:
                for ver in VER_LABELS:
                    k_agents = all_pf.get(ver)
                    if not isinstance(k_agents, dict): continue
                    for ag_role, ag_keys in k_agents.items():
                        alpha = AGENT_ALPHA.get(ag_role, 0.5)
                        col = VER_COLORS[ver]
                        for l in range(num_layers):
                            k=ag_keys.get(l)
                            if k is None or k.size==0: continue
                            seq=k.shape[1]
                            flat=k.transpose(1,0,2).reshape(seq,-1)
                            if flat.shape[0]>MAX_PCA_TOKENS:
                                flat=flat[np.random.choice(flat.shape[0],MAX_PCA_TOKENS,replace=False)]
                            pca_vecs[ph][l].append((flat, col, alpha, f"{ver}_{ag_role}"))
            else:
                for ver in VER_LABELS:
                    k=fk(ver, ph)
                    if k is None: continue
                    col=VER_COLORS[ver]
                    for l in range(num_layers):
                        kl=k.get(l)
                        if kl is None or kl.size==0: continue
                        seq=kl.shape[1]
                        flat=kl.transpose(1,0,2).reshape(seq,-1)
                        if flat.shape[0]>MAX_PCA_TOKENS:
                            flat=flat[np.random.choice(flat.shape[0],MAX_PCA_TOKENS,replace=False)]
                        pca_vecs[ph][l].append((flat, col, 0.6, ver))

        del all_pf, all_dk; gc.collect(); torch.cuda.empty_cache()

    # ── outputs per phase ──
    for ph in [P, D]:
        prefix = f"{method}_{args.task}_{model_tag}_{ph}"

        # CSV
        rows = csv_rows[ph]
        if rows:
            path = os.path.join(plots_dir, f"{prefix}_record.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            print(f"  CSV → {path}  ({len(rows)} rows)")

        # PCA / t-SNE
        tpl = os.path.join(plots_dir, f"{prefix}_{{tag}}_{{layers}}.png")
        dim_reduce_plots(pca_vecs[ph], num_layers, tpl, f"{method} {ph}")

        # Activation difference heatmap (2×3 grid: all pairwise diffs)
        p = os.path.join(plots_dir, f"{prefix}_activation_diff.png")
        plot_activation_diff_grid(act_acc[ph], p,
                                   f"{method} {ph} — pairwise |Δ activation|",
                                   num_layers, num_kv)

    del csv_rows, pca_vecs, act_acc; gc.collect()

    # ── diagnostic: mean cosine across versions for PCA overlap check ──
    print(f"\n  Diagnostic — mean cosine similarity (prefill, orig↔ver1/2/3):")
    # re-read CSV to compute
    csv_p = os.path.join(plots_dir, f"{method}_{args.task}_{model_tag}_prefill_record.csv")
    if os.path.exists(csv_p):
        import pandas as pd
        df = pd.read_csv(csv_p)
        for v in ["version1","version2","version3"]:
            col = f"{v}_cos"
            if col in df.columns:
                print(f"    {v}: {df[col].mean():.4f}  (min={df[col].min():.4f}  max={df[col].max():.4f})")
        print("  → High cosine (>0.95 everywhere) = genuine overlap, not a bug.")
        print("  → If decay in deeper layers, the model differentiates paraphrases there.\n")


# =====================================================================
# Entry
# =====================================================================
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_path", required=True)
    pa.add_argument("--task", default="gpqa",
                    choices=["gsm8k","aime2024","aime2025","gpqa","arc_easy",
                             "arc_challenge","mbppplus","humanevalplus","medqa"])
    pa.add_argument("--model_name", default="Qwen/Qwen3-8B")
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--methods", nargs="+", default=["baseline","text_mas","latent_mas"],
                    choices=["baseline","text_mas","latent_mas"])
    pa.add_argument("--prompt", choices=["sequential","hierarchical"], default="sequential")
    pa.add_argument("--max_tasks", type=int, default=3)
    pa.add_argument("--latent_steps", type=int, default=10)
    pa.add_argument("--temperature", type=float, default=0.6)
    pa.add_argument("--top_p", type=float, default=0.95)
    pa.add_argument("--max_new_tokens", type=int, default=4096,
                    help="Full generation length (for correct inference)")
    pa.add_argument("--gen_max_tokens", type=int, default=1024,
                    help="Max tokens for text_mas intermediate agents")
    pa.add_argument("--decode_analysis_steps", type=int, default=100,
                    help="How many decoded tokens to record in CSV/plots")
    pa.add_argument("--text_mas_context_length", type=int, default=-1)
    pa.add_argument("--think", action="store_true")
    pa.add_argument("--latent_space_realign", action="store_true")
    pa.add_argument("--plots_dir", default="./plots")
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    args.use_vllm = False
    args.use_second_HF_model = False
    args.enable_prefix_caching = False
    args.device2 = "cpu"
    args.tensor_parallel_size = 1
    args.gpu_memory_utilization = 0.9
    args.method = "baseline"

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.plots_dir, exist_ok=True)

    items = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip(): items.append(json.loads(line))
    if args.max_tasks > 0: items = items[:args.max_tasks]
    print(f"Loaded {len(items)} tasks from {args.data_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    model_tag = args.model_name.split("/")[-1]

    for method in args.methods:
        print(f"\n{'='*60}\n  Method: {method}\n{'='*60}")
        args.method = method
        analyze_method(wrapper, method, items, args, args.plots_dir, model_tag)

    print(f"\nAll done. Outputs in {args.plots_dir}/")


if __name__ == "__main__":
    main()
