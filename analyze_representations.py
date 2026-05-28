#!/usr/bin/env python3
"""
analyze_representations.py  (v2 — uses original project code directly)
=====================================================================
Place this file in the project root (alongside models.py, prompts.py).
It imports ModelWrapper and the exact prompt builders so representations
match the actual experiments.  No modifications to original files needed.

Outputs → ./plots/
  {method}_{task}_{model}_{prefill|decode}_{pca|tsne|activation|record}...

Usage:
  python analyze_representations.py \
      --data_path ./paraphrased_data/gpqa_paraphrased.jsonl \
      --task gpqa --model_name Qwen/Qwen3-8B --device cuda \
      --methods baseline text_mas latent_mas \
      --prompt sequential --latent_steps 10 --max_tasks 3
"""
import argparse, csv, gc, json, math, os, re, sys
from collections import namedtuple, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import jensenshannon

# ── import from the ORIGINAL project code ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ModelWrapper, _past_length
from prompts import (
    build_agent_messages_single_agent,
    build_agent_messages_sequential_text_mas,
    build_agent_messages_hierarchical_text_mas,
    build_agent_message_sequential_latent_mas,
    build_agent_message_hierarchical_latent_mas,
)

# ── agents (same as methods/__init__.py) ─────────────────────────────
Agent = namedtuple("Agent", ["name", "role"])
AGENTS = [Agent("Planner","planner"), Agent("Critic","critic"),
          Agent("Refiner","refiner"), Agent("Judger","judger")]
HIER_NAME = {"Planner":"Math Agent","Critic":"Science Agent",
             "Refiner":"Code Agent","Judger":"Task Summrizer"}

# ── visual constants ─────────────────────────────────────────────────
AGENT_ALPHA = {"planner":0.9,"critic":0.7,"refiner":0.5,"judger":0.3}
VER_LABELS  = ["original","version1","version2","version3"]
VER_COLORS  = {"original":(1,0,0),"version1":(0.6,0,0.8),
               "version2":(0,0.7,0),"version3":(0,0.3,1)}
MAX_PCA_TOKENS = 200
DECODE_STEPS   = 80

# =====================================================================
# Helpers
# =====================================================================
def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()

def extract_keys_from_past(past, to_numpy=True):
    """dict[layer] → np.array [kv_heads, seq_len, head_dim]"""
    keys = {}
    if hasattr(past, "key_cache"):            # DynamicCache
        for i, k in enumerate(past.key_cache):
            t = k[0].detach().cpu().float()
            keys[i] = t.numpy() if to_numpy else t
    else:                                      # legacy tuple
        for i, layer in enumerate(past):
            k = layer[0] if isinstance(layer, (tuple, list)) else layer
            t = k[0].detach().cpu().float()
            keys[i] = t.numpy() if to_numpy else t
    return keys

def extract_last_key(past, num_layers):
    """Extract only the last-position key per layer (one decode step)."""
    keys = {}
    if hasattr(past, "key_cache"):
        for l in range(num_layers):
            keys[l] = past.key_cache[l][0, :, -1:, :].detach().cpu().float().numpy()
    else:
        for l in range(num_layers):
            kl = past[l][0] if isinstance(past[l], (tuple,list)) else past[l]
            keys[l] = kl[0, :, -1:, :].detach().cpu().float().numpy()
    return keys

def softmax_dist(v):
    v = v - v.max()
    e = np.exp(v)
    return e / (e.sum() + 1e-12)

def cosine_sim(a, b):
    af, bf = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / d) if d > 1e-12 else 0.0

def js_divergence(a, b):
    p = softmax_dist(a.flatten().astype(np.float64))
    q = softmax_dist(b.flatten().astype(np.float64))
    return float(jensenshannon(p, q, base=2) ** 2)

def sample_next(logits, temperature=0.6, top_p=0.95):
    if temperature < 1e-8:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    mask = (torch.cumsum(sp, -1) - sp) > top_p
    sp[mask] = 0.0
    sp /= sp.sum()
    return si[torch.multinomial(sp, 1)].squeeze(-1)

# =====================================================================
# Step-by-step decode (extracts per-step keys)
# =====================================================================
@torch.no_grad()
def step_decode(wrapper: ModelWrapper, past, initial_logits: torch.Tensor,
                n_steps: int, num_layers: int, temperature=0.6, top_p=0.95):
    """Returns dict[layer] → np.array [kv_heads, n_decoded, head_dim]."""
    logits = initial_logits  # [vocab]
    decode_keys = {l: [] for l in range(num_layers)}
    device = wrapper.device

    for _ in range(n_steps):
        tok = sample_next(logits, temperature, top_p)
        if tok.item() == wrapper.tokenizer.eos_token_id:
            break
        pl = _past_length(past)
        dec_mask = torch.ones((1, pl + 1), dtype=torch.long, device=device)
        out = wrapper.model(tok.view(1, 1), attention_mask=dec_mask,
                            past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :]
        lk = extract_last_key(past, num_layers)
        for l in range(num_layers):
            decode_keys[l].append(lk[l])
        del lk

    result = {}
    for l in range(num_layers):
        if decode_keys[l]:
            result[l] = np.concatenate(decode_keys[l], axis=1)
        else:
            head_dim = wrapper.model.config.hidden_size // wrapper.model.config.num_attention_heads
            n_kv = getattr(wrapper.model.config,"num_key_value_heads",
                           wrapper.model.config.num_attention_heads)
            result[l] = np.zeros((n_kv, 0, head_dim), dtype=np.float32)
    del past; torch.cuda.empty_cache()
    return result

# =====================================================================
# Method-specific key extraction  (mirrors original method code exactly)
# =====================================================================

# ── baseline ─────────────────────────────────────────────────────────
@torch.no_grad()
def extract_baseline(wrapper, question, args, do_decode=True):
    msgs = build_agent_messages_single_agent(question=question, args=args)
    _, ids, mask, _ = wrapper.prepare_chat_input(msgs)
    num_layers = wrapper.model.config.num_hidden_layers

    # prefill
    out = wrapper.model(ids, attention_mask=mask, use_cache=True,
                        output_hidden_states=False, return_dict=True)
    prefill_keys = extract_keys_from_past(out.past_key_values)
    decode_keys = None
    if do_decode:
        decode_keys = step_decode(wrapper, out.past_key_values,
                                   out.logits[0, -1, :], DECODE_STEPS,
                                   num_layers, args.temperature, args.top_p)
    del out; torch.cuda.empty_cache()
    return prefill_keys, decode_keys

# ── text_mas ─────────────────────────────────────────────────────────
@torch.no_grad()
def extract_text_mas(wrapper, question, args, do_decode=True):
    context = ""
    prefill_per_agent = {}
    decode_keys = None
    num_layers = wrapper.model.config.num_hidden_layers

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

        # ---- prefill (every agent) ----
        out = wrapper.model(ids, attention_mask=mask, use_cache=True,
                            output_hidden_states=False, return_dict=True)
        prefill_per_agent[ag.role] = extract_keys_from_past(out.past_key_values)

        if ag.role != "judger":
            # generate text for context
            gen, _ = wrapper.generate_text_batch(
                ids, mask,
                max_new_tokens=args.gen_max_tokens,
                temperature=args.temperature, top_p=args.top_p)
            text_out = strip_thinking(gen[0])
            if args.prompt == "hierarchical":
                formatted = f"[{HIER_NAME[ag.name]}]:\n{text_out}\n\n"
            else:
                formatted = f"[{ag.name}]:\n{text_out}\n\n"
            context += formatted
            del out; torch.cuda.empty_cache()
        else:
            # ---- decode (judger only) ----
            if do_decode:
                decode_keys = step_decode(
                    wrapper, out.past_key_values, out.logits[0,-1,:],
                    DECODE_STEPS, num_layers, args.temperature, args.top_p)
            del out; torch.cuda.empty_cache()

    return prefill_per_agent, decode_keys

# ── latent_mas (mirrors LatentMASMethod.run_batch exactly) ───────────
@torch.no_grad()
def extract_latent_mas(wrapper, question, args, do_decode=True):
    accumulated_past = None
    num_layers = wrapper.model.config.num_hidden_layers
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
            # wrap with <think> if needed (same as original)
            wrapped = f"{prompt_text}<think>" if args.think else prompt_text
            enc = wrapper.tokenizer(wrapped, return_tensors="pt",
                                    add_special_tokens=False)
            w_ids = enc["input_ids"].to(device)
            w_mask = enc["attention_mask"].to(device)

            accumulated_past = wrapper.generate_latent_batch(
                w_ids, attention_mask=w_mask,
                latent_steps=args.latent_steps,
                past_key_values=accumulated_past)
        else:
            judger_text = f"{prompt_text}<think>" if args.think else prompt_text
            enc = wrapper.tokenizer(judger_text, return_tensors="pt",
                                    add_special_tokens=False)
            j_ids = enc["input_ids"].to(device)
            j_mask = enc["attention_mask"].to(device)

            past_for_dec = accumulated_past if args.latent_steps > 0 else None
            # build attention mask including accumulated past
            if past_for_dec is not None:
                pl = _past_length(past_for_dec)
                pm = torch.ones((1, pl), dtype=j_mask.dtype, device=device)
                full_mask = torch.cat([pm, j_mask], dim=1)
            else:
                full_mask = j_mask

            out = wrapper.model(j_ids, attention_mask=full_mask,
                                past_key_values=past_for_dec,
                                use_cache=True, output_hidden_states=False,
                                return_dict=True)
            prefill_keys = extract_keys_from_past(out.past_key_values)
            decode_keys = None
            if do_decode:
                decode_keys = step_decode(
                    wrapper, out.past_key_values, out.logits[0,-1,:],
                    DECODE_STEPS, num_layers, args.temperature, args.top_p)
            del out; torch.cuda.empty_cache()

    del accumulated_past; torch.cuda.empty_cache()
    return prefill_keys, decode_keys

# =====================================================================
# Metrics
# =====================================================================
def layer_metrics(ka, kb, num_layers):
    """Returns [(cos, js)] per layer."""
    res = []
    for l in range(num_layers):
        a, b = ka.get(l), kb.get(l)
        if a is None or b is None or a.size == 0 or b.size == 0:
            res.append((0.0, 1.0)); continue
        am = a.mean(axis=1).flatten(); bm = b.mean(axis=1).flatten()
        res.append((cosine_sim(am, bm), js_divergence(am, bm)))
    return res

def activation_scores(keys, num_layers, num_heads):
    act = np.zeros((num_layers, num_heads), dtype=np.float64)
    for l in range(num_layers):
        k = keys.get(l)
        if k is None or k.size == 0: continue
        norms = np.linalg.norm(k, axis=2)  # [heads, seq]
        act[l, :min(num_heads, norms.shape[0])] = norms.mean(axis=1)[:num_heads]
    return act

# =====================================================================
# Plotting
# =====================================================================
def plot_scatter_grid(layer_data, num_layers, out_tpl, title_prefix, use_tsne):
    layers_per_fig = 16
    for fig_i in range(math.ceil(num_layers / layers_per_fig)):
        s = fig_i * layers_per_fig
        e = min(s + layers_per_fig, num_layers)
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        tag = "t-SNE" if use_tsne else "PCA"
        fig.suptitle(f"{title_prefix} ({tag}) — Layers {s}–{e-1}", fontsize=14)
        for pos in range(16):
            ax = axes[pos//4][pos%4]
            li = s + pos
            if li >= num_layers or li not in layer_data:
                ax.set_visible(False); continue
            ax.set_title(f"Layer {li}", fontsize=9)
            for pts, col, alpha, label in layer_data[li]:
                if len(pts) == 0: continue
                ax.scatter(pts[:,0], pts[:,1], c=[col], alpha=alpha, s=6, label=label)
            ax.tick_params(labelsize=6)
        # deduplicated legend
        seen = {}
        for pos in range(min(e-s, 16)):
            ax = axes[pos//4][pos%4]
            for h, lb in zip(*ax.get_legend_handles_labels()):
                if lb not in seen: seen[lb] = h
        if seen:
            fig.legend(seen.values(), seen.keys(), loc="upper right", fontsize=7)
        plt.tight_layout(rect=[0,0,1,0.96])
        t = "tsne" if use_tsne else "pca"
        p = out_tpl.replace("{tag}", t).replace("{layers}", f"layers{s:02d}-{e-1:02d}")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"    → {p}")


def dim_reduce_plots(vecs, num_layers, out_tpl, title):
    for use_tsne in [False, True]:
        ld = {}
        for l in range(num_layers):
            items = vecs.get(l, [])
            all_pts = [v[0] for v in items if len(v[0]) > 0]
            if not all_pts: continue
            combined = np.vstack(all_pts)
            if combined.shape[0] < 4: continue
            try:
                if use_tsne:
                    proj = TSNE(n_components=2, perplexity=max(2, min(30, combined.shape[0]-1)),
                                random_state=42, max_iter=500).fit_transform(combined)
                else:
                    proj = PCA(n_components=min(2, *combined.shape)).fit_transform(combined)
            except Exception:
                continue
            entries = []; off = 0
            for pts_o, c, a, lb in items:
                n = len(pts_o)
                entries.append((proj[off:off+n] if n else np.empty((0,2)), c, a, lb))
                off += n
            ld[l] = entries
        plot_scatter_grid(ld, num_layers, out_tpl, title, use_tsne)


def plot_heatmap(act, path, title):
    fig, ax = plt.subplots(figsize=(max(6, act.shape[1]*0.6), max(8, act.shape[0]*0.3)))
    im = ax.imshow(act, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer"); ax.set_title(title)
    ax.set_yticks(range(act.shape[0])); ax.set_xticks(range(act.shape[1]))
    ax.tick_params(labelsize=7); plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    → {path}")

# =====================================================================
# Main analysis loop
# =====================================================================
def analyze_method(wrapper, method, items, args, plots_dir, model_tag):
    num_layers = wrapper.model.config.num_hidden_layers
    num_heads = getattr(wrapper.model.config, "num_key_value_heads",
                        wrapper.model.config.num_attention_heads)

    for phase in ["prefill", "decode"]:
        prefix = f"{method}_{args.task}_{model_tag}_{phase}"
        csv_rows = []
        act_acc  = {v: np.zeros((num_layers, num_heads)) for v in VER_LABELS}
        act_n    = {v: 0 for v in VER_LABELS}
        pca_vecs = defaultdict(list)
        do_decode = (phase == "decode")

        for ti, item in enumerate(items):
            tid = item["task_id"]
            qs = {"original": item["question"],
                  "version1": item.get("version1",""),
                  "version2": item.get("version2",""),
                  "version3": item.get("version3","")}
            print(f"  [{phase}] task {tid}  ({ti+1}/{len(items)})")

            all_keys = {}     # ver → keys or {agent→keys}
            for ver in VER_LABELS:
                q = qs[ver]
                if not q:
                    all_keys[ver] = None; continue
                if method == "baseline":
                    pk, dk = extract_baseline(wrapper, q, args, do_decode)
                    all_keys[ver] = dk if do_decode else pk
                elif method == "text_mas":
                    pka, dk = extract_text_mas(wrapper, q, args, do_decode)
                    all_keys[ver] = dk if do_decode else pka
                elif method == "latent_mas":
                    pk, dk = extract_latent_mas(wrapper, q, args, do_decode)
                    all_keys[ver] = dk if do_decode else pk
                gc.collect(); torch.cuda.empty_cache()

            # ── get "final" keys (judger for text_mas prefill) ──
            def final(ver):
                k = all_keys.get(ver)
                if k is None: return None
                if method == "text_mas" and phase == "prefill" and isinstance(k, dict):
                    return k.get("judger")
                return k

            orig = final("original")
            if orig is None: continue

            # ── pre-compute pair metrics ──
            pm = {}
            for a, b in [("original","version1"),("original","version2"),
                         ("original","version3"),("version1","version2"),
                         ("version1","version3"),("version2","version3")]:
                ka, kb = final(a), final(b)
                if ka and kb:
                    pm[(a,b)] = layer_metrics(ka, kb, num_layers)

            # ── CSV rows ──
            for l in range(num_layers):
                row = {"task_id": tid, "layer": l}
                for v in ["version1","version2","version3"]:
                    m = pm.get(("original", v))
                    row[f"{v}_cos"] = round(m[l][0], 3) if m else float("nan")
                    row[f"{v}_js"]  = round(m[l][1], 3) if m else float("nan")
                for lbl, va, vb in [("ver1to2","version1","version2"),
                                    ("ver1to3","version1","version3"),
                                    ("ver2to3","version2","version3")]:
                    m = pm.get((va, vb))
                    row[f"{lbl}_cos"] = round(m[l][0], 3) if m else float("nan")
                    row[f"{lbl}_js"]  = round(m[l][1], 3) if m else float("nan")
                csv_rows.append(row)

            # ── activation accumulation (running mean) ──
            for ver in VER_LABELS:
                fk = final(ver)
                if fk:
                    a = activation_scores(fk, num_layers, num_heads)
                    n = act_n[ver]
                    act_acc[ver] = (act_acc[ver] * n + a) / (n + 1)
                    act_n[ver] += 1

            # ── PCA/t-SNE vectors (subsampled) ──
            if method == "text_mas" and phase == "prefill":
                for ver in VER_LABELS:
                    k_agents = all_keys.get(ver)
                    if not isinstance(k_agents, dict): continue
                    for ag_role, ag_keys in k_agents.items():
                        alpha = AGENT_ALPHA.get(ag_role, 0.5)
                        col = VER_COLORS[ver]
                        for l in range(num_layers):
                            k = ag_keys.get(l)
                            if k is None or k.size == 0: continue
                            seq = k.shape[1]
                            flat = k.transpose(1,0,2).reshape(seq, -1)
                            if flat.shape[0] > MAX_PCA_TOKENS:
                                idx = np.random.choice(flat.shape[0], MAX_PCA_TOKENS, replace=False)
                                flat = flat[idx]
                            pca_vecs[l].append((flat, col, alpha, f"{ver}_{ag_role}"))
            else:
                for ver in VER_LABELS:
                    fk = final(ver)
                    if fk is None: continue
                    col = VER_COLORS[ver]
                    for l in range(num_layers):
                        k = fk.get(l)
                        if k is None or k.size == 0: continue
                        seq = k.shape[1]
                        flat = k.transpose(1,0,2).reshape(seq, -1)
                        if flat.shape[0] > MAX_PCA_TOKENS:
                            idx = np.random.choice(flat.shape[0], MAX_PCA_TOKENS, replace=False)
                            flat = flat[idx]
                        pca_vecs[l].append((flat, col, 0.6, ver))

            del all_keys; gc.collect(); torch.cuda.empty_cache()

        # ── write CSV ──
        csv_path = os.path.join(plots_dir, f"{prefix}_record.csv")
        if csv_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                w.writeheader(); w.writerows(csv_rows)
            print(f"  CSV → {csv_path}")

        # ── PCA / t-SNE ──
        tpl = os.path.join(plots_dir, f"{prefix}_{{tag}}_{{layers}}.png")
        dim_reduce_plots(pca_vecs, num_layers, tpl, f"{method} {phase}")

        # ── activation heatmaps ──
        for ver in VER_LABELS:
            a = act_acc[ver]
            mx = a.max()
            scaled = a / (mx + 1e-12) if mx > 0 else a
            p = os.path.join(plots_dir, f"{prefix}_activation_{ver}.png")
            plot_heatmap(scaled, p, f"{method} {phase} — {ver}")

        del pca_vecs, csv_rows; gc.collect()


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
    pa.add_argument("--methods", nargs="+",
                    default=["baseline","text_mas","latent_mas"],
                    choices=["baseline","text_mas","latent_mas"])
    pa.add_argument("--prompt", choices=["sequential","hierarchical"], default="sequential")
    pa.add_argument("--max_tasks", type=int, default=3)
    pa.add_argument("--latent_steps", type=int, default=10)
    pa.add_argument("--temperature", type=float, default=0.6)
    pa.add_argument("--top_p", type=float, default=0.95)
    pa.add_argument("--max_new_tokens", type=int, default=4096)
    pa.add_argument("--gen_max_tokens", type=int, default=1024)
    pa.add_argument("--text_mas_context_length", type=int, default=-1)
    pa.add_argument("--think", action="store_true")
    pa.add_argument("--latent_space_realign", action="store_true")
    pa.add_argument("--plots_dir", default="./plots")
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    # ── attrs needed by ModelWrapper but irrelevant for HF-only path ──
    args.use_vllm = False
    args.use_second_HF_model = False
    args.enable_prefix_caching = False
    args.device2 = "cpu"
    args.tensor_parallel_size = 1
    args.gpu_memory_utilization = 0.9
    args.method = "baseline"  # placeholder, overridden per method

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.plots_dir, exist_ok=True)

    # ── load data ──
    items = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    if args.max_tasks > 0:
        items = items[:args.max_tasks]
    print(f"Loaded {len(items)} tasks from {args.data_path}")
    for it in items:
        for v in ["version1","version2","version3"]:
            if not it.get(v):
                print(f"  ⚠ task {it['task_id']} — empty {v}")

    # ── load model (original ModelWrapper, HF path) ──
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    model_tag = args.model_name.split("/")[-1]
    nl = wrapper.model.config.num_hidden_layers
    nh = getattr(wrapper.model.config, "num_key_value_heads",
                 wrapper.model.config.num_attention_heads)
    print(f"Model ready: {model_tag}  layers={nl}  kv_heads={nh}")

    # ── run each method ──
    for method in args.methods:
        print(f"\n{'='*60}\n  Method: {method}\n{'='*60}")
        args.method = method   # prompts.py asserts this
        analyze_method(wrapper, method, items, args, args.plots_dir, model_tag)

    print(f"\nAll done. Outputs in {args.plots_dir}/")


if __name__ == "__main__":
    main()
