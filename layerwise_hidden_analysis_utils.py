"""
layerwise_hidden_analysis_utils.py
----------------------------------
Utilities for collecting, sampling, and visualizing layer-wise hidden-state
drift across multi-agent reasoning steps (latent_mas & text_mas).

Color encoding: agent identity (not temporal phase).
"""

import json
import math
import os
from typing import Dict, List, Tuple

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


# ============================================================
# Collector
# ============================================================

class LayerwiseHiddenCollector:
    """Accumulates per-layer hidden-state vectors with metadata."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.buffers: Dict[int, list] = {i: [] for i in range(num_layers)}

    def add_all_layers(
        self,
        layer_hiddens: List[torch.Tensor],
        case_idx: int,
        agent_name: str,
        step_idx: int,
        step_type: str,
    ):
        for li in range(min(len(layer_hiddens), self.num_layers)):
            vec = layer_hiddens[li].squeeze(0)  # [D]
            self.buffers[li].append(
                dict(case_idx=case_idx, agent_name=agent_name,
                     step_idx=step_idx, step_type=step_type, vec=vec)
            )

    def total_records(self) -> int:
        return sum(len(v) for v in self.buffers.values())


# ============================================================
# Balanced sampling
# ============================================================

def sample_collector(
    collector: LayerwiseHiddenCollector,
    budget_per_layer: int,
    budget_last_layer: int,
    seed: int,
) -> Dict[int, list]:
    """Down-sample each layer buffer to budget, balanced across agents."""
    rng = np.random.RandomState(seed)
    sampled: Dict[int, list] = {}
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
        per_agent_base = budget // len(names)
        remainder = budget % len(names)
        result = []
        for i, name in enumerate(names):
            group = by_agent[name]
            quota = per_agent_base + (1 if i < remainder else 0)
            if len(group) <= quota:
                result.extend(group)
            else:
                idx = rng.choice(len(group), size=quota, replace=False)
                result.extend(group[j] for j in idx)
        sampled[li] = result
    return sampled


# ============================================================
# Internal helpers
# ============================================================

@torch.no_grad()
def _prefill_with_logits(model: ModelWrapper, input_ids, attention_mask, past_key_values=None):
    """Single prefill forward returning hidden states, logits, past_kv, and GPU last hidden.

    Returns (past_kv, layer_hiddens_cpu, logits_gpu [1,V], last_hidden_gpu [1,D])
    """
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
        use_cache=True, output_hidden_states=True, return_dict=True,
    )
    layer_h = [h[:, -1, :].detach().cpu().to(torch.float16) for h in outputs.hidden_states]
    last_gpu = outputs.hidden_states[-1][:, -1, :].detach().clone()
    logits = outputs.logits[:, -1, :].detach()
    return outputs.past_key_values, layer_h, logits, last_gpu


def _sample_token(logits, temperature, top_p):
    """Sample one token from logits [1, V]. Returns [1, 1] ids."""
    if temperature > 0:
        logits = logits / temperature
        sl, si = torch.sort(logits, descending=True, dim=-1)
        cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
        remove = cum - torch.softmax(sl, dim=-1) >= top_p
        sl[remove] = float("-inf")
        probs = torch.softmax(sl, dim=-1)
        sampled = torch.multinomial(probs, 1)
        return si.gather(1, sampled)
    return logits.argmax(dim=-1, keepdim=True)


# ============================================================
# Collection: latent_mas
# ============================================================

def collect_latent_mas_states(
    model: ModelWrapper,
    agents,
    items: List[Dict],
    args,
    collector: LayerwiseHiddenCollector,
    include_judger: bool = False,
    max_decode_steps: int = 80,
):
    """Mirror the latent_mas inference path item-by-item, collecting hidden states.

    Non-judger agents: prefill + latent recurrence with accumulated KV cache.
    Judger (optional): prefill + manual decode on top of the accumulated KV.
    """
    from tqdm import tqdm

    for case_idx, item in enumerate(tqdm(items, desc="latent_mas collection")):
        past_kv = None

        for agent in agents:
            is_judger = agent.role == "judger"
            if is_judger and not include_judger:
                continue

            # Build prompt
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
                # ---- Prefill ----
                past_kv, layer_h, last_gpu = model.forward_collect_layerwise(
                    input_ids=ids, attention_mask=mask, past_key_values=past_kv)
                collector.add_all_layers(layer_h, case_idx, agent.name, 0, "prefill")

                # ---- Latent recurrence ----
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
                # ---- Judger: prefill + decode ----
                pkv_j, layer_h, logits0, _ = _prefill_with_logits(
                    model, ids, mask, past_key_values=past_kv)
                collector.add_all_layers(layer_h, case_idx, agent.name, 0, "prefill")

                cur = _sample_token(logits0, args.temperature, args.top_p)
                eos = model.tokenizer.eos_token_id
                for ds in range(max_decode_steps):
                    tl = _past_length(pkv_j) + 1
                    sm = torch.ones((1, tl), dtype=torch.long, device=model.device)
                    ntok, pkv_j, layer_h = model.decode_step_collect_layerwise(
                        cur, sm, pkv_j, temperature=args.temperature, top_p=args.top_p)
                    collector.add_all_layers(layer_h, case_idx, agent.name, ds + 1, "decode")
                    cur = ntok.unsqueeze(1)
                    if eos is not None and ntok.item() == eos:
                        break
                del pkv_j

        del past_kv
        torch.cuda.empty_cache()


# ============================================================
# Collection: text_mas
# ============================================================

def collect_text_mas_states(
    model: ModelWrapper,
    agents,
    items: List[Dict],
    args,
    collector: LayerwiseHiddenCollector,
    include_judger: bool = True,
    max_decode_steps: int = 80,
):
    """Mirror the text_mas inference path item-by-item with manual decode."""
    from tqdm import tqdm

    name_map = {
        "Planner": "Math Agent", "Critic": "Science Agent",
        "Refiner": "Code Agent", "Judger": "Task Summarizer",
        "planner": "Math Agent", "critic": "Science Agent",
        "refiner": "Code Agent", "judger": "Task Summarizer",
    }

    for case_idx, item in enumerate(tqdm(items, desc="text_mas collection")):
        context = ""

        for agent in agents:
            is_judger = agent.role == "judger"
            if is_judger and not include_judger:
                continue

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

            # ---- Prefill (single forward → hidden states + logits) ----
            pkv, layer_h, logits0, _ = _prefill_with_logits(model, ids, mask)
            collector.add_all_layers(layer_h, case_idx, agent.name, 0, "prefill")

            # ---- Manual decode ----
            cur = _sample_token(logits0, args.temperature, args.top_p)
            eos = model.tokenizer.eos_token_id
            gen_ids: List[int] = [cur.item()]

            for ds in range(max_decode_steps):
                tl = _past_length(pkv) + 1
                sm = torch.ones((1, tl), dtype=torch.long, device=model.device)
                ntok, pkv, layer_h = model.decode_step_collect_layerwise(
                    cur, sm, pkv, temperature=args.temperature, top_p=args.top_p)
                collector.add_all_layers(layer_h, case_idx, agent.name, ds + 1, "decode")
                gen_ids.append(ntok.item())
                cur = ntok.unsqueeze(1)
                if eos is not None and ntok.item() == eos:
                    break

            # Accumulate text context for next agent
            text_out = model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            dname = name_map.get(agent.name, agent.name) if args.prompt == "hierarchical" else agent.name
            formatted = f"[{dname}]:\n{text_out}\n\n"
            if not is_judger:
                context = f"{context}{formatted}"

            del pkv
            torch.cuda.empty_cache()


# ============================================================
# PCA + Plotting
# ============================================================

def run_pca_and_plot(
    sampled: Dict[int, list],
    num_layers: int,
    method: str,
    model_name: str,
    task: str,
    seed: int,
    latent_steps: int,
    max_samples: int,
    out_dir: str,
    pca_chunk_size: int,
    include_judger: bool,
) -> Tuple[List[str], str]:
    """Run 2-D PCA per layer and produce scatter plots colored by agent."""
    os.makedirs(out_dir, exist_ok=True)
    short_model = model_name.split("/")[-1]
    prefix = f"{task}_{short_model}_{method}"

    # Discover agents and assign colors
    all_agents: set = set()
    for recs in sampled.values():
        for r in recs:
            all_agents.add(r["agent_name"])
    agents_sorted = sorted(all_agents)
    cmap = plt.cm.get_cmap("tab10")
    agent_colors = {n: cmap(i) for i, n in enumerate(agents_sorted)}

    # PCA per layer
    pca_results: Dict[int, Tuple] = {}
    gmin = np.array([np.inf, np.inf])
    gmax = np.array([-np.inf, -np.inf])

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

    # Split layers into chunks
    chunks = [list(range(s, min(s + pca_chunk_size, num_layers)))
              for s in range(0, num_layers, pca_chunk_size)]

    if num_layers <= 16:
        ncols_g = 4
    elif num_layers <= 40:
        ncols_g = 10
    elif num_layers <= 48:
        ncols_g = 12
    else:
        ncols_g = 10

    saved_files: List[str] = []
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

        # Hide unused subplot cells
        for i in range(n, nrows * ncols):
            r, c = divmod(i, ncols)
            axes[r, c].set_visible(False)

        # Single legend per figure
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

    # Metadata JSON
    meta = {
        "method": method,
        "model": model_name,
        "task": task,
        "seed": seed,
        "latent_steps": latent_steps,
        "max_samples": max_samples,
        "num_layers": num_layers,
        "per_layer_sample_counts": {str(k): len(v) for k, v in sampled.items()},
        "agent_color_mapping": {
            n: [round(x, 4) for x in agent_colors[n][:3]] for n in agents_sorted
        },
        "sampling_policy": "balanced across agents, uniform random within each agent",
        "figures_split": len(chunks) > 1,
        "figure_files": saved_files,
        "include_judger": include_judger,
        "pca_chunk_size": pca_chunk_size,
    }
    meta_path = os.path.join(out_dir, f"{prefix}_layerwise_pca_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved: {meta_path}")
    return saved_files, meta_path
