# Representation-sensitivity analysis report

## 1. Input files

- **qkv**: `./plots/baseline_gpqa_Qwen3-14B_prefill_qkv_headwise_metrics.csv`
- **head_output**: `./plots/baseline_gpqa_Qwen3-14B_prefill_head_output_metrics.csv`
- **residual**: `./plots/baseline_gpqa_Qwen3-14B_prefill_residual_stream_metrics.csv`
- **heatmap_long**: `./plots/baseline_gpqa_Qwen3-14B_heatmap_long.csv`
- **metadata**: `./plots/baseline_gpqa_Qwen3-14B_run_metadata.json`

## 2. Data size

- qkv: 672000 rows
- head_output: 480000 rows
- residual: 12300 rows
- model: `Qwen/Qwen3-14B`, task: `gpqa`, layers: 40, q_heads: 40, kv_heads: 8

## 3. Token length difference by pair

- version1_vs_version3: mean |Δtokens| = 13.7
- original_vs_version3: mean |Δtokens| = 12.3
- original_vs_version1: mean |Δtokens| = 10.9
- version1_vs_version2: mean |Δtokens| = 10.8
- version2_vs_version3: mean |Δtokens| = 8.5
- original_vs_version2: mean |Δtokens| = 8.0

## 4. CKA valid rate

- version1_vs_version2: valid rate = 0.02  **(LOW — CKA unreliable)**
- original_vs_version3: valid rate = 0.02  **(LOW — CKA unreliable)**
- version1_vs_version3: valid rate = 0.02  **(LOW — CKA unreliable)**
- original_vs_version1: valid rate = 0.06  **(LOW — CKA unreliable)**
- version2_vs_version3: valid rate = 0.06  **(LOW — CKA unreliable)**
- original_vs_version2: valid rate = 0.10  **(LOW — CKA unreliable)**

## 5. maxmatch_cosine trend by pair (higher = more similar)

- [qkv/k] original_vs_version1: 0.9102
- [qkv/q] original_vs_version1: 0.8873
- [qkv/v] original_vs_version1: 0.8416
- [qkv/k] original_vs_version2: 0.9436
- [qkv/q] original_vs_version2: 0.9298
- [qkv/v] original_vs_version2: 0.9277
- [qkv/k] original_vs_version3: 0.9197
- [qkv/q] original_vs_version3: 0.8987
- [qkv/v] original_vs_version3: 0.8921
- [qkv/k] version1_vs_version2: 0.9102
- [qkv/q] version1_vs_version2: 0.8877
- [qkv/v] version1_vs_version2: 0.8432
- [qkv/k] version1_vs_version3: 0.9036
- [qkv/q] version1_vs_version3: 0.8793
- [qkv/v] version1_vs_version3: 0.8304
- [qkv/k] version2_vs_version3: 0.9301
- [qkv/q] version2_vs_version3: 0.9120
- [qkv/v] version2_vs_version3: 0.9088
- [head_output] original_vs_version1: 0.8926
- [head_output] original_vs_version2: 0.9493
- [head_output] original_vs_version3: 0.9249
- [head_output] version1_vs_version2: 0.8940
- [head_output] version1_vs_version3: 0.8850
- [head_output] version2_vs_version3: 0.9365
- [residual] original_vs_version1: 0.8971
- [residual] original_vs_version2: 0.9589
- [residual] original_vs_version3: 0.9394
- [residual] version1_vs_version2: 0.8979
- [residual] version1_vs_version3: 0.8911
- [residual] version2_vs_version3: 0.9488

## 6. MMD trend by pair (lower = more similar; component-local scale)

- [qkv/k] original_vs_version1: 0.0051
- [qkv/q] original_vs_version1: 0.0067
- [qkv/v] original_vs_version1: 0.0057
- [qkv/k] original_vs_version2: 0.0022
- [qkv/q] original_vs_version2: 0.0028
- [qkv/v] original_vs_version2: 0.0020
- [qkv/k] original_vs_version3: 0.0033
- [qkv/q] original_vs_version3: 0.0041
- [qkv/v] original_vs_version3: 0.0029
- [qkv/k] version1_vs_version2: 0.0049
- [qkv/q] version1_vs_version2: 0.0066
- [qkv/v] version1_vs_version2: 0.0055
- [qkv/k] version1_vs_version3: 0.0055
- [qkv/q] version1_vs_version3: 0.0072
- [qkv/v] version1_vs_version3: 0.0060
- [qkv/k] version2_vs_version3: 0.0026
- [qkv/q] version2_vs_version3: 0.0032
- [qkv/v] version2_vs_version3: 0.0022
- [head_output] original_vs_version1: 0.0269
- [head_output] original_vs_version2: 0.0100
- [head_output] original_vs_version3: 0.0152
- [head_output] version1_vs_version2: 0.0263
- [head_output] version1_vs_version3: 0.0284
- [head_output] version2_vs_version3: 0.0119
- [residual] original_vs_version1: 0.0143
- [residual] original_vs_version2: 0.0025
- [residual] original_vs_version3: 0.0037
- [residual] version1_vs_version2: 0.0140
- [residual] version1_vs_version3: 0.0146
- [residual] version2_vs_version3: 0.0026

## 7. head_output l2_mean caveat

- l2_mean should not be used for direct cross-layer comparison; it is a same-layer pair comparison metric only (activation scale differs across layers).

## 8. Residual stream layer trend

- Residual layer indexing: **layer 0 = embedding output; layer l+1 = transformer block l output.**
- maxmatch_cosine spans 0.831 (layer 0) to 0.938 (layer 39) across layers.

## 9. Sensitive-head rankings

- Rankings are provided **per metric only** (lowest maxmatch / highest MMD / highest L2 / lowest valid-CKA). No composite ranking across metrics is produced.

## 10. Interpretation caveats

> No weighted or composite sensitivity score is computed. Metrics are analyzed separately because cosine, CKA, MMD, and L2 have different scales, directions, and validity conditions.

> CKA is interpreted only when cka_status indicates a valid comparison. Length-mismatched or resampled CKA values should not be treated as primary evidence.

> MMD is a distribution distance. Its absolute scale depends on the component and the median-heuristic bandwidth, so cross-component absolute comparisons are avoided.
