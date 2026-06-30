# Curated 3-Axis v2 Geometry Diagnostics

Created: 2026-06-23 UTC

## Scope

This diagnostic report uses existing curated 3-axis v2 projection outputs. No generation was rerun.

## Main Geometry Summary

| embedding | model | projection MAE | Spearman | interp dist | off-axis drift | norm off-axis | out-of-range | NN top1 | NN mean rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bge-m3 (cls) | qwen35_9b | 0.121 | 0.889 | 0.278 | 0.265 | 0.494 | 0.001 | 0.287 | 2.950 |
| bge-m3 (cls) | qwen35_35b_a3b | 0.134 | 0.864 | 0.266 | 0.250 | 0.457 | 0.000 | 0.286 | 2.885 |
| bge-m3 (cls) | qwen35_4b | 0.159 | 0.809 | 0.268 | 0.249 | 0.514 | 0.001 | 0.286 | 3.275 |
| text2vec-base-chinese (mean) | qwen35_9b | 0.091 | 0.932 | 0.263 | 0.254 | 0.436 | 0.002 | 0.299 | 2.631 |
| text2vec-base-chinese (mean) | qwen35_35b_a3b | 0.100 | 0.921 | 0.252 | 0.241 | 0.417 | 0.001 | 0.294 | 2.630 |
| text2vec-base-chinese (mean) | qwen35_4b | 0.142 | 0.856 | 0.274 | 0.257 | 0.475 | 0.001 | 0.292 | 3.065 |

## Answers

- Both embeddings keep the same projection ranking: qwen35_9b best, qwen35_35b_a3b second, qwen35_4b weakest.
- 9B's advantage is mainly projection calibration. It has the lowest projection MAE and highest projection Spearman under both embeddings. Its normalized off-axis drift is not consistently lower than 35B-A3B.
- 35B-A3B does not show a higher out-of-range rate; its out-of-range rate is lower than or comparable to 9B. Its off-axis drift is often slightly smaller, but its projection calibration is worse.
- 4B's weakness is primarily ratio calibration and monotonicity/projection ordering rather than uniquely larger off-axis drift. Its nearest-neighbor mean rank is also worst under both embeddings.

## Output Tables

- `results/curated3axis_v2_full_eval/tables/geometry_metrics_by_model_embedding.csv`
- `results/curated3axis_v2_full_eval/tables/geometry_metrics_by_axis_model_embedding.csv`
- `results/curated3axis_v2_full_eval/tables/geometry_metrics_by_alpha_model_embedding.csv`
- `results/curated3axis_v2_full_eval/tables/nearest_neighbor_metrics.csv`
- `results/curated3axis_v2_full_eval/tables/out_of_range_projection_metrics.csv`
