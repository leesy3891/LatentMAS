"""
layerwise_hidden_analysis_utils.py (updated)
--------------------------------------------
Changes vs original:
  1. Per-layer independent PCA (each subplot gets its own fit)
  2. Outlier removal: top-5 points by L2 distance from centroid removed before PCA
  3. FID and MMD² vs previous layer computed in current layer's 2D PCA space,
     shown below each subplot
  4. Dot size s=3, max 4 columns per row
  5. Last layer (num_layers-1) labeled "Output Layer" instead of "Layer ##"
  6. PC axes (components + explained variance) saved to .txt (not JSON)
  7. Meta JSON removed; replaced by .txt
"""

import json
import math
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from models import ModelWrapper, _past_length, extract_last_keys_per_layer
from prompts import (
    build_agent_messages_hierarchical_text_mas,
    build_agent_messages_sequential_text_mas,
    build_agent_message_sequential_latent_mas,
    build_agent_message_hierarchical_latent_mas,
)
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout


# ============================================================
# Collector
# ============================================================

class LayerwiseHiddenCollector:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.buffers: Dict[int, list] = {i: [] for i in range(num_layers)}

    def add_all_layers(self, layer_hiddens, case_idx, agent_name, step_idx, step_type):
        for li in range(min(len(layer_hiddens), self.num_layers)):
            vec = layer_hiddens[li].squeeze(0)
            self.buffers[li].append(
                dict(case_idx=case_idx, agent_name=agent_name,
                     step_idx=step_idx, step_type=step_type, vec=vec))

    def total_records(self):
        return sum(len(v) for v in self.buffers.values())


# ============================================================
# Sampling
# ============================================================

def sample_collector(collector, budget_per_layer, budget_last_layer, seed):
    rng = np.random.RandomState(seed)
    sampled = {}
    for li in range(collector.num_layers):
        buf = collector.buffers[li]
        budget = budget_last_layer if li == collector.num_layers - 1 else budget_per_layer
        if len(buf) <= budget:
            sampled[li] = list(buf)
            continue
        by_agent: Dict[str, list] = {}
        for rec in buf:
            by_agent.setdefault(rec["agent_name"], []).append(rec)
        names = sorted(by_agent.keys())
        per_a = budget // len(names)
        rem = budget % len(names)
        result = []
        for i, n in enumerate(names):
            g = by_agent[n]
            q = per_a + (1 if i < rem else 0)
            if len(g) <= q:
                result.extend(g)
            else:
                idx = rng.choice(len(g), size=q, replace=False)
                result.extend(g[j] for j in idx)
        sampled[li] = result
    return sampled


# ============================================================
# Evaluation (mirrors run.py)
# ============================================================

def evaluate_prediction(final_text: str, item: Dict, task: str) -> Dict:
    if task in ["mbppplus", "humanevalplus"]:
        pred = extract_markdown_python_block(final_text)
        gold = item.get("gold", "")
        if pred is None:
            ok, error_msg = False, "No python code block found"
        else:
            ok, error_msg = run_with_timeout(pred + "\n" + gold, timeout=10)
    elif task in ["aime2024", "aime2025"]:
        pred = normalize_answer(extract_gsm8k_answer(final_text))
        gold = str(item.get("gold", "")).strip()
        try:
            ok = int(pred) == int(gold)
            error_msg = None
        except ValueError:
            ok, error_msg = False, f"Parse error: pred={pred}, gold={gold}"
    else:
        pred = normalize_answer(extract_gsm8k_answer(final_text))
        gold = item.get("gold", "")
        ok = (pred == gold) if (pred and gold) else False
        error_msg = None
    return dict(question=item["question"], gold=gold,
                prediction=pred, raw_prediction=final_text,
                correct=ok, error_msg=error_msg)


# ============================================================
# Helpers
# ============================================================

@torch.no_grad()
def _prefill_with_logits(model, input_ids, attention_mask, past_key_values=None):
    """Returns (past_kv, layer_hiddens_cpu, logits_gpu, last_hidden_gpu)."""
    device = model.device
    attn = attention_mask.to(device)
    if past_key_values is not None:
        plen = _past_length(past_key_values)
        if plen > 0:
            pm = torch.ones((attn.shape[0], plen), dtype=attn.dtype, device=device)
            attn = torch.cat([pm, attn], dim=-1)
    outputs = model.model(
        input_ids=input_ids.to(device), attention_mask=attn,
        past_key_values=past_key_values,
        use_cache=True, output_hidden_states=True, return_dict=True)
    layer_h = [h[:, -1, :].detach().cpu().to(torch.float16) for h in outputs.hidden_states]
    last_gpu = outputs.hidden_states[-1][:, -1, :].detach().clone()
    logits = outputs.logits[:, -1, :].detach()
    return outputs.past_key_values, layer_h, logits, last_gpu


def _preselect_steps(max_steps: int, budget: int, seed: int) -> Set[int]:
    """Uniformly select decode step indices for hidden-state collection."""
    if max_steps <= budget:
        return set(range(max_steps))
    indices = np.linspace(0, max_steps - 1, budget, dtype=int)
    return set(indices.tolist())


def _decode_loop(model, past_kv, first_logits, max_tokens, temperature, top_p,
                 collector, case_idx, agent_name, collect_steps,
                 key_collector=None):
    """Full decode loop. Collects hidden states only at steps in collect_steps.
    Returns (generated_ids, past_kv).
    """
    device = model.device
    eos_id = model.tokenizer.eos_token_id

    first_tok = model._sample_from_logits(first_logits, temperature, top_p)
    cur = first_tok.unsqueeze(1)
    gen_ids = [first_tok.item()]

    for ds in range(max_tokens):
        tlen = _past_length(past_kv) + 1
        smask = torch.ones((1, tlen), dtype=torch.long, device=device)

        if ds in collect_steps:
            ntok, past_kv, layer_h = model.decode_step_collect_layerwise(
                cur, smask, past_kv, temperature=temperature, top_p=top_p)
            collector.add_all_layers(layer_h, case_idx, agent_name, ds + 1, "decode")
            if key_collector is not None:
                layer_keys = extract_last_keys_per_layer(past_kv)
                key_collector.add_all_layers(layer_keys, case_idx, agent_name, ds + 1, "decode")
        else:
            ntok, past_kv = model.decode_step_simple(
                cur, smask, past_kv, temperature=temperature, top_p=top_p)

        gen_ids.append(ntok.item())
        cur = ntok.unsqueeze(1)
        if eos_id is not None and ntok.item() == eos_id:
            break

    return gen_ids, past_kv


# ============================================================
# PCA helpers: outlier removal + distance metrics
# ============================================================

def _remove_outliers(vecs_np: np.ndarray, n_remove: int = 5):
    """Remove top n_remove points by L2 distance from centroid.

    Returns (cleaned_vecs, keep_indices).
    """
    n = len(vecs_np)
    if n <= n_remove + 4:
        return vecs_np, list(range(n))

    # Filter out rows containing NaN/inf before computing centroid
    finite_mask = np.isfinite(vecs_np).all(axis=1)
    if not finite_mask.all():
        finite_idx = np.where(finite_mask)[0]
        if len(finite_idx) <= n_remove + 4:
            return vecs_np[finite_idx], finite_idx.tolist()
        vecs_finite = vecs_np[finite_idx]
        centroid = vecs_finite.mean(axis=0)
        dists = np.linalg.norm(vecs_finite - centroid, axis=1)
        outlier_set = set(np.argsort(dists)[-n_remove:].tolist())
        keep_local = [i for i in range(len(finite_idx)) if i not in outlier_set]
        keep = [int(finite_idx[i]) for i in keep_local]
        return vecs_np[keep], keep

    centroid = vecs_np.mean(axis=0)
    dists = np.linalg.norm(vecs_np - centroid, axis=1)
    outlier_set = set(np.argsort(dists)[-n_remove:].tolist())
    keep = [i for i in range(n) if i not in outlier_set]
    return vecs_np[keep], keep


def _compute_fid_2d(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Fréchet distance between two 2-D point sets (Gaussian approximation).

    Both arrays are shape [N, 2].
    """
    from scipy.linalg import sqrtm
    if len(coords_a) < 4 or len(coords_b) < 4:
        return float("nan")
    mu_a = coords_a.mean(axis=0)
    mu_b = coords_b.mean(axis=0)
    cov_a = np.cov(coords_a.T) + 1e-6 * np.eye(2)
    cov_b = np.cov(coords_b.T) + 1e-6 * np.eye(2)
    diff = mu_a - mu_b
    sqrt_prod = sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(sqrt_prod):
        sqrt_prod = sqrt_prod.real
    fid = float(diff @ diff + np.trace(cov_a + cov_b - 2.0 * sqrt_prod))
    return fid


def _compute_mmd2_rbf(coords_a: np.ndarray, coords_b: np.ndarray,
                      max_n: int = 400) -> float:
    """Unbiased MMD² with RBF kernel on 2-D coords."""
    rng = np.random.RandomState(42)
    if len(coords_a) > max_n:
        coords_a = coords_a[rng.choice(len(coords_a), max_n, replace=False)]
    if len(coords_b) > max_n:
        coords_b = coords_b[rng.choice(len(coords_b), max_n, replace=False)]
    all_pts = np.vstack([coords_a, coords_b])
    sq_dists = np.sum((all_pts[:, None] - all_pts[None, :]) ** 2, axis=-1)
    pos = sq_dists[sq_dists > 0]
    sigma2 = float(np.median(pos)) / 2.0 if len(pos) > 0 else 1.0

    def rbf(x, y):
        d = np.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
        return np.exp(-d / (2.0 * sigma2))

    kxx = rbf(coords_a, coords_a)
    kyy = rbf(coords_b, coords_b)
    kxy = rbf(coords_a, coords_b)
    n, m = len(coords_a), len(coords_b)
    val = (
        (kxx.sum() - np.trace(kxx)) / max(n * (n - 1), 1)
        + (kyy.sum() - np.trace(kyy)) / max(m * (m - 1), 1)
        - 2.0 * kxy.mean()
    )
    return float(val)


# ============================================================
# Entropy + KL helpers
# ============================================================

def _gaussian_kl_2d(mu_p: np.ndarray, cov_p: np.ndarray,
                     mu_q: np.ndarray, cov_q: np.ndarray) -> float:
    """KL(P || Q) for two 2-D Gaussians.

    KL = 0.5 * [tr(Σ_q⁻¹ Σ_p) + (μ_q - μ_p)ᵀ Σ_q⁻¹ (μ_q - μ_p) - k + ln|Σ_q| - ln|Σ_p|]
    """
    try:
        cov_q_inv = np.linalg.inv(cov_q)
        _, logdet_p = np.linalg.slogdet(cov_p)
        _, logdet_q = np.linalg.slogdet(cov_q)
        diff = mu_q - mu_p
        kl = 0.5 * (
            np.trace(cov_q_inv @ cov_p)
            + diff @ cov_q_inv @ diff
            - 2
            + logdet_q - logdet_p
        )
        return float(kl)
    except np.linalg.LinAlgError:
        return float("nan")


def compute_entropy_per_layer(
    sampled: Dict[int, list],
    num_layers: int,
    agents_sorted: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per-sample spectral entropy of hidden vectors, grouped by (layer, agent).

    Spectral entropy of a vector h:
        p_i = h_i² / Σ_j h_j²
        H   = -Σ_i p_i log(p_i)

    Returns:
        entropy_mean / entropy_std : dicts  agent_name -> np.array [num_layers]
    """
    entropy_mean = {a: np.full(num_layers, np.nan) for a in agents_sorted}
    entropy_std  = {a: np.full(num_layers, np.nan) for a in agents_sorted}

    for li in range(num_layers):
        recs = sampled.get(li, [])
        if not recs:
            continue
        by_agent: Dict[str, list] = {}
        for rec in recs:
            by_agent.setdefault(rec["agent_name"], []).append(rec)

        for aname in agents_sorted:
            agent_recs = by_agent.get(aname, [])
            if not agent_recs:
                continue
            entropies = []
            for rec in agent_recs:
                h = rec["vec"].float().numpy()   # [D]
                if not np.isfinite(h).all():
                    continue
                sq = h ** 2
                s = sq.sum()
                if not np.isfinite(s) or s < 1e-12:
                    continue
                p = sq / s
                ent = float(-np.sum(p * np.log(p + 1e-12)))
                entropies.append(ent)
            if entropies:
                entropy_mean[aname][li] = float(np.mean(entropies))
                entropy_std[aname][li]  = float(np.std(entropies))

    return entropy_mean, entropy_std


def compute_kl_per_layer(
    pca_coords_dict: Dict[int, Tuple],
    num_layers: int,
    agents_sorted: List[str],
    n_bootstrap: int = 80,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """KL(agent || overall) per layer, via bootstrap on 2-D PCA coords.

    Both the agent distribution and the overall distribution are approximated
    as 2-D Gaussians.  Bootstrap resampling provides the ±1 σ band.

    Returns:
        kl_mean / kl_std : dicts  agent_name -> np.array [num_layers]
    """
    rng = np.random.RandomState(seed)
    kl_mean = {a: np.full(num_layers, np.nan) for a in agents_sorted}
    kl_std  = {a: np.full(num_layers, np.nan) for a in agents_sorted}

    for li in range(num_layers):
        coords_all, recs_all = pca_coords_dict.get(li, (None, []))
        if coords_all is None or len(coords_all) < 4:
            continue

        # Reference (overall) Gaussian – fitted once per layer
        mu_all  = coords_all.mean(axis=0)
        cov_all = np.cov(coords_all.T) + 1e-6 * np.eye(2)

        # Index records by agent
        by_agent_idx: Dict[str, list] = {}
        for j, rec in enumerate(recs_all):
            by_agent_idx.setdefault(rec["agent_name"], []).append(j)

        for aname in agents_sorted:
            idxs = by_agent_idx.get(aname, [])
            if len(idxs) < 4:
                continue
            coords_a = coords_all[idxs]
            n = len(coords_a)
            bs = min(n, 50)   # bootstrap sample size

            kl_vals = []
            for _ in range(n_bootstrap):
                sample = coords_a[rng.choice(n, size=bs, replace=True)]
                mu_s  = sample.mean(axis=0)
                cov_s = np.cov(sample.T) + 1e-6 * np.eye(2)
                kl = _gaussian_kl_2d(mu_s, cov_s, mu_all, cov_all)
                if np.isfinite(kl):
                    kl_vals.append(kl)

            if kl_vals:
                kl_mean[aname][li] = float(np.mean(kl_vals))
                kl_std[aname][li]  = float(np.std(kl_vals))

    return kl_mean, kl_std


def plot_entropy_and_kl(
    entropy_mean: Dict[str, np.ndarray],
    entropy_std:  Dict[str, np.ndarray],
    kl_mean:      Dict[str, np.ndarray],
    kl_std:       Dict[str, np.ndarray],
    agents_sorted: List[str],
    agent_colors:  Dict[str, tuple],
    num_layers: int,
    method: str,
    model_name: str,
    task: str,
    out_dir: str,
) -> str:
    """Two-panel figure: (a) Shannon entropy  (b) KL divergence, per agent.

    Each line = agent mean across samples.  Shaded band = ±1 σ.
    X-axis last tick is labeled 'Output Layer'.

    Returns saved filename.
    """
    os.makedirs(out_dir, exist_ok=True)
    short_model = model_name.split("/")[-1]

    x = np.arange(num_layers)
    # Tick positions: every ~4 layers + last layer
    tick_step = max(1, num_layers // 16)
    xticks = list(range(0, num_layers - 1, tick_step))
    if (num_layers - 1) not in xticks:
        xticks.append(num_layers - 1)

    def _xlabels(ticks):
        return ["Output" if t == num_layers - 1 else str(t) for t in ticks]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    panels = [
        (entropy_mean, entropy_std, "Spectral Shannon Entropy",
         "per-sample H(h)  [nats]"),
        (kl_mean,      kl_std,      "KL Divergence  agent ‖ overall",
         "KL(agent ‖ overall)  [nats]  (bootstrap ±1σ)"),
    ]

    for ax, (mu_dict, std_dict, title, ylabel) in zip(axes, panels):
        any_plotted = False
        for aname in agents_sorted:
            mu  = mu_dict[aname]
            std = std_dict[aname]
            valid = np.isfinite(mu)
            if not np.any(valid):
                continue
            xv   = x[valid]
            muv  = mu[valid]
            stdv = std[valid]
            color = agent_colors[aname]
            ax.plot(xv, muv, color=color, label=aname, linewidth=1.5, zorder=3)
            ax.fill_between(xv, muv - stdv, muv + stdv,
                            color=color, alpha=0.20, zorder=2)
            any_plotted = True

        # Vertical line + label for output layer
        ax.axvline(num_layers - 1, color="gray", linestyle="--",
                   linewidth=0.8, alpha=0.6, zorder=1)
        ax.text(num_layers - 1 + 0.15, ax.get_ylim()[1] if any_plotted else 1,
                "Output\nLayer", fontsize=6.5, color="gray", va="top")

        ax.set_xticks(xticks)
        ax.set_xticklabels(_xlabels(xticks), fontsize=7, rotation=45)
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(
            f"{title}\n{short_model}  |  {task}  |  {method}",
            fontsize=9,
        )
        ax.legend(fontsize=7.5, loc="best", framealpha=0.8)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    fname = f"{task}_{short_model}_{method}_entropy_kl.png"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")
    return fname


# ============================================================
# Collection: latent_mas
# ============================================================

def collect_latent_mas_states(model, agents, items, args, collector,
                              max_decode_analysis_steps=80,
                              key_collector=None):
    """Non-judger: prefill + latent recurrence (all steps collected).
    Judger: prefill + full text decode (sampled hidden collection).

    If ``key_collector`` is provided (a LayerwiseHiddenCollector with
    ``num_layers = num_hidden_layers``), the last-position **key cache** of
    every transformer layer is also collected at the same (case, agent, step)
    points.

    Returns list of evaluation result dicts.
    """
    from tqdm import tqdm
    results = []

    for case_idx, item in enumerate(tqdm(items, desc="latent_mas analysis")):
        past_kv = None
        judger_text = ""

        for agent in agents:
            is_judger = agent.role == "judger"

            if args.prompt == "sequential":
                messages = build_agent_message_sequential_latent_mas(
                    role=agent.role, question=item["question"],
                    context="", method="latent_mas", args=args)
            else:
                messages = build_agent_message_hierarchical_latent_mas(
                    role=agent.role, question=item["question"],
                    context="", method="latent_mas", args=args)

            prompt = model.render_chat(messages, add_generation_prompt=True)
            if getattr(args, "think", False):
                prompt = f"{prompt}<think>"

            enc = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(model.device)
            mask = enc["attention_mask"].to(model.device)

            if not is_judger:
                # Prefill
                past_kv, layer_h, last_gpu = model.forward_collect_layerwise(
                    input_ids=ids, attention_mask=mask, past_key_values=past_kv)
                collector.add_all_layers(layer_h, case_idx, agent.name, 0, "prefill")
                if key_collector is not None:
                    layer_keys = extract_last_keys_per_layer(past_kv)
                    key_collector.add_all_layers(layer_keys, case_idx, agent.name, 0, "prefill")

                # Latent recurrence
                for step in range(args.latent_steps):
                    src = model.HF_model if hasattr(model, "HF_model") else model.model
                    latent_vec = model._apply_latent_realignment(last_gpu, src)
                    latent_emb = latent_vec.unsqueeze(1)
                    plen = _past_length(past_kv)
                    lmask = torch.ones((1, plen + 1), dtype=torch.long, device=model.device)
                    # Position freeze for RoPE (mirrors generate_latent_batch)
                    pos_ids = model._compute_latent_position_ids(step, plen, 1, model.device)
                    past_kv, layer_h, last_gpu = model.forward_collect_layerwise(
                        inputs_embeds=latent_emb, attention_mask=lmask,
                        past_key_values=past_kv, position_ids=pos_ids)
                    collector.add_all_layers(layer_h, case_idx, agent.name, step + 1, "latent")
                    if key_collector is not None:
                        layer_keys = extract_last_keys_per_layer(past_kv)
                        key_collector.add_all_layers(layer_keys, case_idx, agent.name, step + 1, "latent")
            else:
                # Judger: prefill + full decode
                pkv_j, layer_h, logits0, _ = _prefill_with_logits(
                    model, ids, mask, past_key_values=past_kv)
                collector.add_all_layers(layer_h, case_idx, agent.name, 0, "prefill")
                if key_collector is not None:
                    layer_keys = extract_last_keys_per_layer(pkv_j)
                    key_collector.add_all_layers(layer_keys, case_idx, agent.name, 0, "prefill")

                collect_steps = _preselect_steps(
                    args.max_new_tokens, max_decode_analysis_steps, args.seed + case_idx)
                gen_ids, pkv_j = _decode_loop(
                    model, pkv_j, logits0, args.max_new_tokens,
                    args.temperature, args.top_p,
                    collector, case_idx, agent.name, collect_steps,
                    key_collector=key_collector)

                judger_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                del pkv_j

        res = evaluate_prediction(judger_text, item, args.task)
        results.append(res)
        print(f"  #{case_idx + 1}  Pred={res['prediction']}  Gold={res['gold']}  OK={res['correct']}")

        del past_kv
        torch.cuda.empty_cache()

    return results


# ============================================================
# Collection: text_mas
# ============================================================

def collect_text_mas_states(model, agents, items, args, collector,
                            max_decode_analysis_steps=80,
                            key_collector=None):
    """All agents do full decode. Hidden states collected at sampled steps.

    If ``key_collector`` is provided, the per-layer last-position **key cache**
    is also collected at the same (case, agent, step) points.

    Returns list of evaluation result dicts.
    """
    from tqdm import tqdm

    name_map = {
        "Planner": "Math Agent", "Critic": "Science Agent",
        "Refiner": "Code Agent", "Judger": "Task Summrizer",
        "planner": "Math Agent", "critic": "Science Agent",
        "refiner": "Code Agent", "judger": "Task Summrizer",
    }
    results = []

    for case_idx, item in enumerate(tqdm(items, desc="text_mas analysis")):
        context = ""
        judger_text = ""

        for agent in agents:
            is_judger = agent.role == "judger"

            if args.prompt == "hierarchical":
                messages = build_agent_messages_hierarchical_text_mas(
                    role=agent.role, question=item["question"],
                    context=context, method="text_mas", args=args)
            else:
                messages = build_agent_messages_sequential_text_mas(
                    role=agent.role, question=item["question"],
                    context=context, method="text_mas", args=args)

            prompt = model.render_chat(messages, add_generation_prompt=True)
            enc = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(model.device)
            mask = enc["attention_mask"].to(model.device)

            # Prefill
            pkv, layer_h, logits0, _ = _prefill_with_logits(model, ids, mask)
            collector.add_all_layers(layer_h, case_idx, agent.name, 0, "prefill")
            if key_collector is not None:
                layer_keys = extract_last_keys_per_layer(pkv)
                key_collector.add_all_layers(layer_keys, case_idx, agent.name, 0, "prefill")

            # Full decode with sampled hidden collection
            collect_steps = _preselect_steps(
                args.max_new_tokens, max_decode_analysis_steps,
                args.seed + case_idx * 10 + hash(agent.name) % 97)
            gen_ids, pkv = _decode_loop(
                model, pkv, logits0, args.max_new_tokens,
                args.temperature, args.top_p,
                collector, case_idx, agent.name, collect_steps,
                key_collector=key_collector)

            text_out = model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            dname = name_map.get(agent.name, agent.name) if args.prompt == "hierarchical" else agent.name
            formatted = f"[{dname}]:\n{text_out}\n\n"
            if not is_judger:
                context = f"{context}{formatted}"
            else:
                judger_text = text_out

            del pkv
            torch.cuda.empty_cache()

        res = evaluate_prediction(judger_text, item, args.task)
        results.append(res)
        print(f"  #{case_idx + 1}  Pred={res['prediction']}  Gold={res['gold']}  OK={res['correct']}")

    return results


# ============================================================
# PCA + Plotting
# ============================================================

def run_pca_and_plot(sampled, num_layers, method, model_name, task, seed,
                     latent_steps, max_samples, out_dir, pca_chunk_size):
    """Per-layer PCA with outlier removal, FID/MMD² vs previous layer, and txt meta output."""
    os.makedirs(out_dir, exist_ok=True)
    short_model = model_name.split("/")[-1]
    prefix = f"{task}_{short_model}_{method}"

    # --- Agent colour map ---
    all_agents: set = set()
    for recs in sampled.values():
        for r in recs:
            all_agents.add(r["agent_name"])
    agents_sorted = sorted(all_agents)
    cmap = plt.colormaps["tab10"]
    agent_colors = {n: cmap(i) for i, n in enumerate(agents_sorted)}

    # ----------------------------------------------------------------
    # Pass 1: per-layer outlier removal + independent PCA
    # ----------------------------------------------------------------
    pca_objects: Dict[int, Optional[PCA]] = {}
    pca_coords: Dict[int, Tuple] = {}       # li -> (coords [N,2], kept_recs)
    vecs_clean_np: Dict[int, Optional[np.ndarray]] = {}  # li -> float32 [N, D]

    print("Fitting per-layer PCA …")
    for li in range(num_layers):
        recs = sampled.get(li, [])
        if len(recs) < 4:
            pca_objects[li] = None
            pca_coords[li] = (None, recs)
            vecs_clean_np[li] = None
            continue

        vecs = torch.stack([r["vec"].float() for r in recs]).numpy()  # [N, D]

        # Drop records whose vectors contain NaN/inf (fp16 overflow)
        finite_mask = np.isfinite(vecs).all(axis=1)
        if not finite_mask.all():
            finite_idx = np.where(finite_mask)[0]
            vecs = vecs[finite_idx]
            recs = [recs[i] for i in finite_idx]
            if len(recs) < 4:
                pca_objects[li] = None
                pca_coords[li] = (None, recs)
                vecs_clean_np[li] = None
                continue

        # Outlier removal (top-5 by L2 from centroid)
        vecs_c, keep = _remove_outliers(vecs, n_remove=5)
        kept_recs = [recs[i] for i in keep]

        # Independent PCA per layer
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(vecs_c)  # [N, 2]

        pca_objects[li] = pca
        pca_coords[li] = (coords, kept_recs)
        vecs_clean_np[li] = vecs_c

    # ----------------------------------------------------------------
    # Pass 2: FID + MMD² vs previous layer (in current layer's PCA space)
    # ----------------------------------------------------------------
    print("Computing FID / MMD² …")
    metrics: Dict[int, Tuple[float, float]] = {}
    metrics[0] = (float("nan"), float("nan"))

    for li in range(1, num_layers):
        pca_curr = pca_objects.get(li)
        vecs_curr = vecs_clean_np.get(li)
        vecs_prev = vecs_clean_np.get(li - 1)
        if pca_curr is None or vecs_curr is None or vecs_prev is None:
            metrics[li] = (float("nan"), float("nan"))
            continue
        # Project both distributions into current layer's 2D PCA space
        coords_curr = pca_curr.transform(vecs_curr)
        coords_prev = pca_curr.transform(vecs_prev)
        fid = _compute_fid_2d(coords_curr, coords_prev)
        mmd2 = _compute_mmd2_rbf(coords_curr, coords_prev)
        metrics[li] = (fid, mmd2)

    # ----------------------------------------------------------------
    # Save PC axes to .txt
    # ----------------------------------------------------------------
    txt_lines = [
        f"# Layer-wise PCA axes",
        f"# method={method}  model={model_name}  task={task}  seed={seed}",
        f"# latent_steps={latent_steps}  max_samples={max_samples}",
        f"# Per-layer independent PCA (n_components=2)",
        f"# Outlier removal: top-5 by L2 distance from centroid removed before PCA",
        f"# FID/MMD² computed in current-layer PCA space (prev layer projected into current PCA)",
        f"# PC vectors truncated to first 10 dimensions below; full dim saved as float32",
        "",
    ]
    for li in range(num_layers):
        label = "output_layer" if li == num_layers - 1 else f"layer_{li:02d}"
        pca = pca_objects.get(li)
        fid_v, mmd2_v = metrics.get(li, (float("nan"), float("nan")))
        if pca is None:
            txt_lines.append(f"[{label}]  n_samples={len(sampled.get(li, []))}  PCA=N/A")
            txt_lines.append(f"  FID_vs_prev=N/A  MMD2_vs_prev=N/A")
        else:
            evr = pca.explained_variance_ratio_
            pc1 = pca.components_[0]
            pc2 = pca.components_[1]
            pc1_str = "  ".join(f"{x:+.6f}" for x in pc1[:10])
            pc2_str = "  ".join(f"{x:+.6f}" for x in pc2[:10])
            suffix = "  ..." if pc1.shape[0] > 10 else ""
            arr = vecs_clean_np.get(li)
            n_clean = len(arr) if arr is not None else 0
            txt_lines.append(
                f"[{label}]  n_samples={n_clean}"
                f"  evr=[{evr[0]:.4f}, {evr[1]:.4f}]"
            )
            txt_lines.append(f"  PC1: {pc1_str}{suffix}")
            txt_lines.append(f"  PC2: {pc2_str}{suffix}")
            fid_str = f"{fid_v:.4f}" if not math.isnan(fid_v) else "N/A"
            mmd_str = f"{mmd2_v:.6f}" if not math.isnan(mmd2_v) else "N/A"
            txt_lines.append(f"  FID_vs_prev={fid_str}  MMD2_vs_prev={mmd_str}")
        txt_lines.append("")

    txt_path = os.path.join(out_dir, f"{prefix}_layerwise_pca_meta.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))
    print(f"Saved meta: {txt_path}")

    # ----------------------------------------------------------------
    # Plot: max 4 columns, per-layer axis limits, FID/MMD² below subplot
    # ----------------------------------------------------------------
    MAX_COLS = 4
    chunks = [list(range(s, min(s + pca_chunk_size, num_layers)))
              for s in range(0, num_layers, pca_chunk_size)]

    saved_files = []
    for ci, clayers in enumerate(chunks):
        n = len(clayers)
        ncols = min(MAX_COLS, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.2 * ncols, 5.0 * nrows),
            squeeze=False,
        )

        for i, li in enumerate(clayers):
            row, col = divmod(i, ncols)
            ax = axes[row][col]
            coords, recs = pca_coords.get(li, (None, []))
            fid_v, mmd2_v = metrics.get(li, (float("nan"), float("nan")))

            # Title
            if li == num_layers - 1:
                ax.set_title("Output Layer", fontsize=8, fontweight="bold")
            else:
                ax.set_title(f"Layer {li:02d}", fontsize=8)
            ax.tick_params(labelsize=5)

            if coords is None or len(coords) == 0:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
            else:
                # Per-layer axis limits (tight around own PCA coords)
                xpad = max((coords[:, 0].max() - coords[:, 0].min()) * 0.06, 1e-3)
                ypad = max((coords[:, 1].max() - coords[:, 1].min()) * 0.06, 1e-3)
                ax.set_xlim(coords[:, 0].min() - xpad, coords[:, 0].max() + xpad)
                ax.set_ylim(coords[:, 1].min() - ypad, coords[:, 1].max() + ypad)

                for aname in agents_sorted:
                    idxs = [j for j, rec in enumerate(recs) if rec["agent_name"] == aname]
                    if not idxs:
                        continue
                    ax.scatter(
                        coords[idxs, 0], coords[idxs, 1],
                        c=[agent_colors[aname]], s=3, alpha=0.45,
                        label=aname, rasterized=True,
                    )

            # FID / MMD² label below subplot
            if not math.isnan(fid_v):
                xlabel = f"FID={fid_v:.2f}  MMD²={mmd2_v:.4f}"
            else:
                xlabel = "FID=N/A  MMD²=N/A"
            ax.set_xlabel(xlabel, fontsize=5.5, labelpad=2)

        # Hide unused cells
        for i in range(n, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row][col].set_visible(False)

        # Shared legend (top-right of figure)
        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=agent_colors[a], markersize=5, label=a)
            for a in agents_sorted
        ]
        fig.legend(handles=legend_handles, loc="upper right",
                   fontsize=7, framealpha=0.8)

        fig.suptitle(
            f"{method} | {short_model} | {task} | per-layer PCA | colored by agent",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 0.90, 0.97])

        suffix = f"_part{ci + 1:02d}" if len(chunks) > 1 else ""
        fname = f"{prefix}_layerwise_pca{suffix}.png"
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(fname)
        print(f"Saved figure: {fpath}")

    # ----------------------------------------------------------------
    # Pass 3: Shannon entropy + KL divergence plots
    # ----------------------------------------------------------------
    print("Computing per-layer entropy and KL divergence …")
    entropy_mean, entropy_std = compute_entropy_per_layer(sampled, num_layers, agents_sorted)
    kl_mean, kl_std = compute_kl_per_layer(
        pca_coords, num_layers, agents_sorted, n_bootstrap=80, seed=seed
    )
    ek_fname = plot_entropy_and_kl(
        entropy_mean, entropy_std, kl_mean, kl_std,
        agents_sorted, agent_colors, num_layers,
        method, model_name, task, out_dir,
    )
    saved_files.append(ek_fname)

    return saved_files, txt_path
