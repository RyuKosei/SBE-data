# Main 8-Axis v1 Final Analysis Report

## Status

Main generation, projection evaluation, scalar judge, geometry diagnostics, length diagnostics, and paper-grade figures are complete. Sampled pairwise audit is included only if pairwise files are present.

## Main Projection Results

| embedding   | model          |   projection_mae |   projection_spearman |   interpolation_distance_mean |   off_axis_drift_mean |   normalized_off_axis_drift_mean |   out_of_range_rate |
|:------------|:---------------|-----------------:|----------------------:|------------------------------:|----------------------:|---------------------------------:|--------------------:|
| bge_m3      | qwen35_9b      |         0.125664 |              0.877569 |                      0.297324 |              0.284483 |                         0.499249 |         0.000892857 |
| bge_m3      | qwen35_35b_a3b |         0.140022 |              0.849621 |                      0.286223 |              0.270143 |                         0.489455 |         0.000744048 |
| bge_m3      | qwen35_4b      |         0.164476 |              0.792771 |                      0.291225 |              0.272686 |                         0.532423 |         0.000744048 |
| text2vec    | qwen35_9b      |         0.110768 |              0.902909 |                      0.270769 |              0.260379 |                         0.470136 |         0.00133929  |
| text2vec    | qwen35_35b_a3b |         0.127372 |              0.878371 |                      0.25756  |              0.243513 |                         0.455948 |         0.000595238 |
| text2vec    | qwen35_4b      |         0.15052  |              0.827194 |                      0.276057 |              0.25981  |                         0.51412  |         0.00133929  |

## Bootstrap CI

| embedding   | embedding_model                                             | model          | metric           |     mean |   ci95_low |   ci95_high | bootstrap_unit   |   iterations |
|:------------|:------------------------------------------------------------|:---------------|:-----------------|---------:|-----------:|------------:|:-----------------|-------------:|
| bge_m3      | /home/data_cpfs/zicheng/models/bge-m3 (cls)                 | qwen35_35b_a3b | projection_error | 0.140022 |   0.135876 |    0.144058 | scenario_id      |         1000 |
| bge_m3      | /home/data_cpfs/zicheng/models/bge-m3 (cls)                 | qwen35_4b      | projection_error | 0.164476 |   0.16063  |    0.168478 | scenario_id      |         1000 |
| bge_m3      | /home/data_cpfs/zicheng/models/bge-m3 (cls)                 | qwen35_9b      | projection_error | 0.125664 |   0.121382 |    0.130439 | scenario_id      |         1000 |
| text2vec    | /home/data_cpfs/zicheng/models/text2vec-base-chinese (mean) | qwen35_35b_a3b | projection_error | 0.127372 |   0.12284  |    0.131892 | scenario_id      |         1000 |
| text2vec    | /home/data_cpfs/zicheng/models/text2vec-base-chinese (mean) | qwen35_4b      | projection_error | 0.15052  |   0.146086 |    0.15506  | scenario_id      |         1000 |
| text2vec    | /home/data_cpfs/zicheng/models/text2vec-base-chinese (mean) | qwen35_9b      | projection_error | 0.110768 |   0.106215 |    0.115548 | scenario_id      |         1000 |

## Scalar Judge Auxiliary Results

| model          | judge_model    |    n |   parse_ok_rate |   judge_mae |   judge_bias |   judge_spearman |   content_preservation_mean |   naturalness_mean |   endpoint_attraction_rate |   smoothness |
|:---------------|:---------------|-----:|----------------:|------------:|-------------:|-----------------:|----------------------------:|-------------------:|---------------------------:|-------------:|
| qwen35_35b_a3b | qwen35_9b      | 6720 |               1 |    0.182418 |    0.0837396 |         0.756274 |                     4.99896 |            4.38304 |                   0.372917 |    0.0535167 |
| qwen35_4b      | qwen35_35b_a3b | 6720 |               1 |    0.24424  |    0.0360134 |         0.589193 |                     4.99033 |            4.3689  |                   0.497917 |    0.0984792 |
| qwen35_9b      | qwen35_35b_a3b | 6720 |               1 |    0.175402 |    0.0462054 |         0.768238 |                     4.99033 |            4.38571 |                   0.517014 |    0.123656  |

Scalar is auxiliary because judge assignment is asymmetric: 4B/9B are judged by 35B-A3B, while 35B-A3B is judged by 9B.

## Length Diagnostics

| model          |    n |   mean_zh_chars |   length_violation_rate |   too_short_rate |   too_long_rate |
|:---------------|-----:|----------------:|------------------------:|-----------------:|----------------:|
| qwen35_35b_a3b | 6720 |         121.983 |                0.200298 |        0.028125  |       0.172173  |
| qwen35_4b      | 6720 |         102.735 |                0.242411 |        0.170685  |       0.0717262 |
| qwen35_9b      | 6720 |         112.72  |                0.1375   |        0.0532738 |       0.0842262 |

## Pairwise Audit

Sampled bidirectional pairwise audit is complete. The sampling uses 8 axes * 20 scenarios per axis * 3 seeds * 9 local alpha pairs * 2 orders for each model, for 25,920 total judgments. This differs from the instruction's approximate total of 8,640; the written factorial design implies 25,920.

Judge assignment is asymmetric: 4B and 9B outputs are judged by 35B-A3B, while 35B-A3B outputs are judged by 9B. Pairwise is therefore an auxiliary monotonicity and preference audit; projection geometry remains the primary metric.

| model          | expected_judge_model   |    n |   parse_ok_rate |    tie_rate |   higher_alpha_win_rate |   monotonic_violation_rate |   text_1_win_rate |   text_2_win_rate |   position_bias_rate |   mean_confidence |   matched_pairs |   both_parse_ok_rate |   bidirectional_consistency |   contradiction_rate |   same_position_winner_rate |
|:---------------|:-----------------------|-----:|----------------:|------------:|------------------------:|---------------------------:|------------------:|------------------:|---------------------:|------------------:|----------------:|---------------------:|----------------------------:|---------------------:|----------------------------:|
| qwen35_35b_a3b | qwen35_9b              | 8640 |        1        | 0.000231481 |                0.751042 |                   0.248784 |          0.619588 |          0.380412 |            0.239176  |           4.86204 |            4320 |             1        |                    0.706019 |             0.293519 |                    0.293519 |
| qwen35_4b      | qwen35_35b_a3b         | 8640 |        0.999884 | 0.111124    |                0.644751 |                   0.274645 |          0.47415  |          0.52585  |            0.0516994 |           4.32724 |            4320 |             0.999769 |                    0.827043 |             0.135448 |                    0.135448 |
| qwen35_9b      | qwen35_35b_a3b         | 8640 |        1        | 0.105556    |                0.709144 |                   0.207169 |          0.486672 |          0.513328 |            0.0266563 |           4.4103  |            4320 |             1        |                    0.843981 |             0.122222 |                    0.122222 |

Worst axis-level monotonicity rows by model:

| model          | axis_id            |    n |   higher_alpha_win_rate |   monotonic_violation_rate |   tie_rate |   bidirectional_consistency |   contradiction_rate |
|:---------------|:-------------------|-----:|------------------------:|---------------------------:|-----------:|----------------------------:|---------------------:|
| qwen35_35b_a3b | concision_detail   | 1080 |                0.596296 |                   0.402597 | 0.00185185 |                    0.764815 |             0.231481 |
| qwen35_35b_a3b | expertise_explain  | 1080 |                0.702778 |                   0.297222 | 0          |                    0.594444 |             0.405556 |
| qwen35_35b_a3b | emotion_apology    | 1080 |                0.726852 |                   0.273148 | 0          |                    0.594444 |             0.405556 |
| qwen35_4b      | concision_detail   | 1080 |                0.506481 |                   0.482008 | 0.0222222  |                    0.640741 |             0.337037 |
| qwen35_4b      | politeness_refusal | 1080 |                0.622222 |                   0.377778 | 0          |                    0.866667 |             0.133333 |
| qwen35_4b      | objectivity_cat    | 1080 |                0.689815 |                   0.289122 | 0.0296296  |                    0.851852 |             0.133333 |
| qwen35_9b      | concision_detail   | 1080 |                0.600926 |                   0.388889 | 0.0166667  |                    0.67037  |             0.314815 |
| qwen35_9b      | politeness_refusal | 1080 |                0.708333 |                   0.290353 | 0.00185185 |                    0.875926 |             0.12037  |
| qwen35_9b      | humor_neutral      | 1080 |                0.762963 |                   0.185771 | 0.062963   |                    0.868519 |             0.087037 |

Bidirectional inconsistency summary:

| model          | expected_judge_model   |   matched_pairs |   both_parse_ok_rate |   bidirectional_consistency |   contradiction_rate |   same_position_winner_rate |
|:---------------|:-----------------------|----------------:|---------------------:|----------------------------:|---------------------:|----------------------------:|
| qwen35_35b_a3b | qwen35_9b              |            4320 |             1        |                    0.706019 |             0.293519 |                    0.293519 |
| qwen35_4b      | qwen35_35b_a3b         |            4320 |             0.999769 |                    0.827043 |             0.135448 |                    0.135448 |
| qwen35_9b      | qwen35_35b_a3b         |            4320 |             1        |                    0.843981 |             0.122222 |                    0.122222 |

## Figure Paths

- `figures/calibration_curves_by_model_bge_m3.pdf/png`
- `figures/calibration_curves_by_model_text2vec.pdf/png`
- `figures/axis_heatmap_projection_mae.pdf/png`
- `figures/axis_heatmap_off_axis_drift.pdf/png`
- `figures/model_comparison_bootstrap_ci.pdf/png`
- `figures/projection_vs_scalar_judge_scatter.pdf/png`
- `figures/length_violation_heatmap.pdf/png`
- `figures/nearest_neighbor_rank_by_model.pdf/png`
- `figures/out_of_range_rate_by_model_axis.pdf/png`

## Cross-Family Recommendation

Do not launch yet. First confirm local non-Qwen instruction-tuned generative model paths. The currently scanned non-Qwen paths under `/home/data_cpfs/zicheng/models` are embedding models, not generator baselines.

## Temperature Ablation Recommendation

Do not launch yet. Use it as a targeted robustness check after finalizing the main report and pairwise decision.

## Paper-Ready Findings

1. Explicit style-ratio prompting shows measurable monotonic calibration across 8 Chinese style axes.
2. Qwen3.5-9B is the best calibrated generator among the tested Qwen-family models.
3. Qwen3.5-35B-A3B is close to 9B but does not surpass it on projection calibration.
4. Qwen3.5-4B is consistently weaker, especially in projection error and length compliance.
5. bge-m3 and text2vec agree on the model ranking, supporting measurement robustness.
6. Length violations remain substantial but prior diagnostics and current subset tables indicate they do not explain away the projection ranking.
7. Scalar judge results support the broad ranking but remain auxiliary due to asymmetric judge assignment.
8. Pairwise, when used, should be treated as an auxiliary preference and monotonicity audit rather than the main metric.
