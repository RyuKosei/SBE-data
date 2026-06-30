# Curated 3-Axis v2 Length Diagnostics

Created: 2026-06-23 UTC

## Scope

This diagnostic uses existing curated 3-axis v2 generation and projection outputs. No generation was rerun.

Tables written under:

```text
results/curated3axis_v2_full_eval/tables/
```

Required outputs:

- `length_diagnostics_by_model_axis_alpha.csv`
- `length_error_correlation_with_projection.csv`
- `length_compliant_subset_metrics.csv`
- `too_short_too_long_breakdown.csv`

## Main Findings

Length violation is real but does not appear to strongly contaminate projection metrics in the current curated pilot.

Across both embeddings, correlations between length and projection error are low:

| embedding | model | Pearson chars vs proj error | Spearman chars vs proj error | Pearson violation magnitude vs proj error |
|---|---|---:|---:|---:|
| bge-m3 | qwen35_35b_a3b | 0.008 | -0.008 | 0.028 |
| bge-m3 | qwen35_4b | 0.053 | 0.074 | 0.021 |
| bge-m3 | qwen35_9b | 0.072 | 0.066 | 0.013 |
| text2vec | qwen35_35b_a3b | 0.032 | 0.015 | 0.032 |
| text2vec | qwen35_4b | 0.062 | 0.109 | 0.021 |
| text2vec | qwen35_9b | 0.084 | 0.093 | -0.000 |

The length-compliant subset preserves the embedding ranking:

| embedding | ranking by projection MAE |
|---|---|
| bge-m3 | qwen35_9b, qwen35_35b_a3b, qwen35_4b |
| text2vec | qwen35_9b, qwen35_35b_a3b, qwen35_4b |

## Too Short vs Too Long

| model | axis | violation rate | too short | too long |
|---|---|---:|---:|---:|
| qwen35_35b_a3b | emotion_apology | 0.193 | 0.039 | 0.154 |
| qwen35_35b_a3b | empathy_clinical | 0.139 | 0.014 | 0.125 |
| qwen35_35b_a3b | formality_request | 0.289 | 0.029 | 0.261 |
| qwen35_4b | emotion_apology | 0.354 | 0.325 | 0.029 |
| qwen35_4b | empathy_clinical | 0.182 | 0.164 | 0.018 |
| qwen35_4b | formality_request | 0.246 | 0.100 | 0.146 |
| qwen35_9b | emotion_apology | 0.093 | 0.061 | 0.032 |
| qwen35_9b | empathy_clinical | 0.107 | 0.021 | 0.086 |
| qwen35_9b | formality_request | 0.189 | 0.104 | 0.086 |

Pattern:

- 4B mostly violates by being too short.
- 35B-A3B mostly violates by being too long.
- 9B is closest to acceptable length behavior.
- `formality_request` still creates length-control pressure for 9B and 35B-A3B.
- `emotion_apology` is the worst length axis for 4B.

## Alpha Effects

There is no single global alpha effect across all models and axes.

Examples:

- 35B-A3B on `emotion_apology` has higher violation at alpha 0.75 and 0.9, mostly too long.
- 35B-A3B on `formality_request` has high violation at alpha 0.1, mostly too long.
- 4B on `emotion_apology` is often too short across alpha values, especially alpha 0.1, 0.25, and 0.75.
- 9B shows moderate length variation but no severe alpha-specific collapse.

## Go/No-Go Interpretation

Under the requested rule:

- The length-compliant subset keeps the model ranking unchanged.
- Correlations between length and projection error are low.

Therefore, length does not currently force a prompt revision before the next stage. However, length violation remains high enough that it should be reported and watched in full main generation.

Recommendation:

- Do not modify the prompt solely based on current length diagnostics.
- If modifying the prompt later, make only the proposed minimal relaxation and rerun curated 3-axis v2 before full main generation.

