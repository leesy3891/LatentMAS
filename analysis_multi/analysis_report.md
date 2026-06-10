# Multi-source representation comparison report

## 1. Input files

- **qkv**: `./plots_multi/baseline_gpqa-gsm8k_orgn-ver1_Qwen3-14B_prefill_qkv_headwise_metrics.csv`
- **head_output**: `./plots_multi/baseline_gpqa-gsm8k_orgn-ver1_Qwen3-14B_prefill_head_output_metrics.csv`
- **residual**: `./plots_multi/baseline_gpqa-gsm8k_orgn-ver1_Qwen3-14B_prefill_residual_stream_metrics.csv`
- **heatmap_long**: `./plots_multi/baseline_gpqa-gsm8k_orgn-ver1_Qwen3-14B_heatmap_long.csv`
- **metadata**: `./plots_multi/baseline_gpqa-gsm8k_orgn-ver1_Qwen3-14B_run_metadata.json`

## 2. Sources & data size

- **gsm8k_orgn**: task=gsm8k, version=original (en), items=50, pooled_tokens=256
- **gsm8k_ver1**: task=gsm8k, version=version1 (zh), items=50, pooled_tokens=256
- **gpqa_orgn**: task=gpqa, version=original (en), items=50, pooled_tokens=256
- **gpqa_ver1**: task=gpqa, version=version1 (zh), items=50, pooled_tokens=256
- qkv: 13440 metric rows
- head_output: 9600 metric rows
- residual: 246 metric rows
- comparison mode: `multi_source_pooled_distribution`

## 3. Pair categories

- gpqa_orgn_vs_gpqa_ver1: **same_semantic_diff_surface**, cross_lang
- gsm8k_orgn_vs_gpqa_orgn: **diff_semantic**, same_lang
- gsm8k_orgn_vs_gpqa_ver1: **diff_semantic**, cross_lang
- gsm8k_orgn_vs_gsm8k_ver1: **same_semantic_diff_surface**, cross_lang
- gsm8k_ver1_vs_gpqa_orgn: **diff_semantic**, cross_lang
- gsm8k_ver1_vs_gpqa_ver1: **diff_semantic**, same_lang

## 4. Headline contrast: cross-domain (diff_semantic) vs paraphrase (same_semantic_diff_surface)


**maxmatch_cosine** (higher=more similar), mean by category × component:
- head_output: diff_semantic=0.603, same_semantic_diff_surface=0.727
- k: diff_semantic=0.774, same_semantic_diff_surface=0.817
- q: diff_semantic=0.755, same_semantic_diff_surface=0.797
- residual: diff_semantic=0.615, same_semantic_diff_surface=0.672
- v: diff_semantic=0.417, same_semantic_diff_surface=0.513

**mmd_rbf** (higher=more different), mean by category × component:
- head_output: diff_semantic=0.059, same_semantic_diff_surface=0.024
- k: diff_semantic=0.053, same_semantic_diff_surface=0.012
- q: diff_semantic=0.057, same_semantic_diff_surface=0.014
- residual: diff_semantic=0.050, same_semantic_diff_surface=0.034
- v: diff_semantic=0.029, same_semantic_diff_surface=0.012

## 5. Residual layer indexing

- layer 0 = embedding output; layer l+1 = transformer block l output.

## 6. Interpretation caveats

> No weighted or composite sensitivity score is computed. Metrics are analyzed separately because cosine, MMD, and L2 have different scales and directions.

> Comparison is distribution-level over pooled, template-stripped question tokens; there is no per-item pairing (required, since cross-domain sources do not correspond).

> MMD is a distribution distance. Its absolute scale depends on the component and the median-heuristic bandwidth, so cross-component absolute comparisons are avoided.

> CKA was removed: it requires paired equal-count samples, which are undefined for unpaired cross-domain token clouds.

> l2_mean is a same-layer comparison metric only; activation scale differs across layers.
