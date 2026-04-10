"""
layerwise_hidden_analysis_utils.py
----------------------------------
Key design:
- Judger ALWAYS runs full decode (up to max_new_tokens) and its hidden states
  are always collected.
- Hidden states at decode steps are collected only at uniformly pre-selected
  indices to control memory; all other steps use a fast path (no hidden output).
- Evaluation accuracy is computed and returned.
- PCA scatter is colored by agent identity.
"""

import json
import math
import os
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models import ModelWrapper, _past_length
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
                 collector, case_idx, agent_name, collect_steps):
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
        else:
            ntok, past_kv = model.decode_step_simple(
                cur, smask, past_kv, temperature=temperature, top_p=top_p)

        gen_ids.append(ntok.item())
        cur = ntok.unsqueeze(1)
        if eos_id is not None and ntok.item() == eos_id:
            break

    return gen_ids, past_kv


# ============================================================
# Collection: latent_mas
# ============================================================

def collect_latent_mas_states(model, agents, items, args, collector,
                              max_decode_analysis_steps=80):
    """Non-judger: prefill + latent recurrence (all steps collected).
    Judger: prefill + full text decode (sampled hidden collection).
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

                # Latent recurrence
                for step in range(args.latent_steps):
                    src = model.HF_model if hasattr(model, "HF_model") else model.model
                    latent_vec = model._apply_latent_realignment(last_gpu, src)
                    latent_emb = latent_vec.unsqueeze(1)
                    plen = _past_length(past_kv)
                    lmask = torch.ones((1, plen + 1), dtype=torch.long, device=model.device)
                    past_kv, layer_h, last_gpu = model.forward_collect_layerwise(
                        inputs_embeds=latent_emb, attention_mask=lmask, past_key_values=past_kv)
                    collector.add_all_layers(layer_h, case_idx, agent.name, step + 1, "latent")
            else:
                # Judger: prefill + full decode
                pkv_j, layer_h, logits0, _ = _prefill_with_logits(
                    model, ids, mask, past_key_values=past_kv)
                collector.add_all_layers(layer_h, case_idx, agent.name, 0, "prefill")

                collect_steps = _preselect_steps(
                    args.max_new_tokens, max_decode_analysis_steps, args.seed + case_idx)
                gen_ids, pkv_j = _decode_loop(
                    model, pkv_j, logits0, args.max_new_tokens,
                    args.temperature, args.top_p,
                    collector, case_idx, agent.name, collect_steps)

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
                            max_decode_analysis_steps=80):
    """All agents do full decode. Hidden states collected at sampled steps.
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

            # Full decode with sampled hidden collection
            collect_steps = _preselect_steps(
                args.max_new_tokens, max_decode_analysis_steps,
                args.seed + case_idx * 10 + hash(agent.name) % 97)
            gen_ids, pkv = _decode_loop(
                model, pkv, logits0, args.max_new_tokens,
                args.temperature, args.top_p,
                collector, case_idx, agent.name, collect_steps)

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
    os.makedirs(out_dir, exist_ok=True)
    short_model = model_name.split("/")[-1]
    prefix = f"{task}_{short_model}_{method}"

    all_agents = set()
    for recs in sampled.values():
        for r in recs:
            all_agents.add(r["agent_name"])
    agents_sorted = sorted(all_agents)
    cmap = plt.cm.get_cmap("tab10")
    agent_colors = {n: cmap(i) for i, n in enumerate(agents_sorted)}

    pca_results = {}
    gmin, gmax = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
    for li in range(num_layers):
        recs = sampled.get(li, [])
        if len(recs) < 2:
            pca_results[li] = (None, recs)
            continue
        vecs = torch.stack([r["vec"].float() for r in recs]).numpy()
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(vecs)
        pca_results[li] = (coords, recs)
        gmin = np.minimum(gmin, coords.min(axis=0))
        gmax = np.maximum(gmax, coords.max(axis=0))

    margin = (gmax - gmin) * 0.05
    margin = np.where(np.isfinite(margin), margin, 1.0)
    gmin -= margin
    gmax += margin

    chunks = [list(range(s, min(s + pca_chunk_size, num_layers)))
              for s in range(0, num_layers, pca_chunk_size)]

    ncols_g = 4 if num_layers <= 16 else 12 if num_layers <= 48 else 10
    if 16 < num_layers <= 40:
        ncols_g = 10

    saved_files = []
    for ci, clayers in enumerate(chunks):
        n = len(clayers)
        ncols = min(ncols_g, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 4 * nrows), squeeze=False)
        for i, li in enumerate(clayers):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            coords, recs = pca_results.get(li, (None, []))
            ax.set_xlim(gmin[0], gmax[0])
            ax.set_ylim(gmin[1], gmax[1])
            ax.set_title(f"Layer {li:02d}", fontsize=8)
            ax.tick_params(labelsize=6)
            if coords is None:
                continue
            for aname in agents_sorted:
                idxs = [j for j, rec in enumerate(recs) if rec["agent_name"] == aname]
                if not idxs:
                    continue
                ax.scatter(coords[idxs, 0], coords[idxs, 1],
                           c=[agent_colors[aname]], s=8, alpha=0.5,
                           label=aname, rasterized=True)
        for i in range(n, nrows * ncols):
            r, c = divmod(i, ncols)
            axes[r, c].set_visible(False)
        for li in clayers:
            coords, _ = pca_results.get(li, (None, []))
            if coords is not None:
                r, c = divmod(clayers.index(li), ncols)
                h, l = axes[r, c].get_legend_handles_labels()
                if h:
                    fig.legend(h, l, loc="upper right", fontsize=8, markerscale=2)
                break
        fig.suptitle(f"{method} | {short_model} | {task} | colored by agent", fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.95, 0.97])
        suffix = f"_part{ci + 1:02d}" if len(chunks) > 1 else ""
        fname = f"{prefix}_layerwise_pca{suffix}.png"
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(fname)
        print(f"Saved: {fpath}")

    meta = {
        "method": method, "model": model_name, "task": task, "seed": seed,
        "latent_steps": latent_steps, "max_samples": max_samples,
        "num_layers": num_layers,
        "per_layer_sample_counts": {str(k): len(v) for k, v in sampled.items()},
        "agent_color_mapping": {n: [round(x, 4) for x in agent_colors[n][:3]] for n in agents_sorted},
        "sampling_policy": "balanced across agents, uniform random within each agent",
        "figures_split": len(chunks) > 1, "figure_files": saved_files,
        "pca_chunk_size": pca_chunk_size,
    }
    meta_path = os.path.join(out_dir, f"{prefix}_layerwise_pca_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved: {meta_path}")
    return saved_files, meta_path
