# Codex Handoff Report: StyleBlend-Bench Main 8-Axis v1

Date: 2026-06-24 09:20 UTC

## Purpose

This report is for the next Codex instance or reviewer. It summarizes the current StyleBlend-Bench state after the 2026-06-24 final-analysis instructions. Do not restart completed generation or judge jobs unless the user explicitly asks for a rerun.

## Current Status

Main 8-axis v1 is complete through final analysis.

- Main generation complete: 8 axes * 40 scenarios * 7 ratios * 3 seeds * 3 Qwen models = 20,160 generations.
- Models: `qwen35_4b`, `qwen35_9b`, `qwen35_35b_a3b`.
- bge-m3 and text2vec projection complete for all generations.
- Scalar judge complete for all generations.
- Sampled bidirectional pairwise audit complete.
- Final tables, figures, figure notes, final report, cross-family plan, temperature-ablation plan, and human-validation sample/protocol are prepared.

Primary final-analysis directory:

`/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/`

Most important final report:

`/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/docs/main8axis_v1_final_analysis_report.md`

Synchronized copy:

`/home/data_cpfs/lizihua/sbe/docs/main8axis_v1_final_analysis_report.md`

## Key Result

Both embedding metrics support the same model ranking:

`qwen35_9b > qwen35_35b_a3b > qwen35_4b`

The 9B advantage is mainly better style-ratio placement along the interpolation axis, not lower off-axis drift. 35B-A3B has competitive drift but weaker projection calibration. 4B is weakest and also has the largest length-compliance issue.

## Projection Metrics

Source table:

`/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/tables/main_projection_metrics_by_embedding.csv`

Headline rows:

| Embedding | Model | Projection MAE | Spearman | Off-axis drift | Normalized drift |
|---|---:|---:|---:|---:|---:|
| bge-m3 | qwen35_9b | 0.126 | 0.878 | 0.284 | 0.499 |
| bge-m3 | qwen35_35b_a3b | 0.140 | 0.850 | 0.270 | 0.489 |
| bge-m3 | qwen35_4b | 0.164 | 0.793 | 0.273 | 0.532 |
| text2vec | qwen35_9b | 0.111 | 0.903 | 0.260 | 0.470 |
| text2vec | qwen35_35b_a3b | 0.127 | 0.878 | 0.244 | 0.456 |
| text2vec | qwen35_4b | 0.151 | 0.827 | 0.260 | 0.514 |

Interpretation:

- 9B has the lowest projection MAE under both embeddings.
- 35B-A3B has slightly lower off-axis drift than 9B, but worse ratio calibration.
- 4B is consistently worse on projection MAE and normalized drift.

## Scalar Judge

Source table:

`/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/tables/main_scalar_metrics.csv`

Summary:

| Model | Judge | Judge MAE | Spearman |
|---|---|---:|---:|
| qwen35_9b | qwen35_35b_a3b | 0.175 | 0.768 |
| qwen35_35b_a3b | qwen35_9b | 0.182 | 0.756 |
| qwen35_4b | qwen35_35b_a3b | 0.244 | 0.589 |

Caveat:

Scalar judge assignment is asymmetric. 4B and 9B are judged by 35B-A3B, while 35B-A3B is judged by 9B. Treat scalar as auxiliary validation, not the main ranking metric.

## Pairwise Audit

Source tables:

- `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/tables/main_pairwise_sampled_metrics.csv`
- `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/tables/main_pairwise_position_bias.csv`
- `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/tables/main_pairwise_inconsistency.csv`
- `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/tables/main_pairwise_by_axis_model.csv`

Audit size:

- 25,920 sampled bidirectional judgments.
- 25,919/25,920 parse_ok.
- The written factorial design implies 25,920 judgments: 8 axes * 20 scenarios * 3 models * 3 seeds * 9 alpha pairs * 2 orders.

Main sampled pairwise metrics:

| Model | Judge | Higher-alpha win rate | Monotonic violation rate | Bidirectional consistency |
|---|---|---:|---:|---:|
| qwen35_9b | qwen35_35b_a3b | 0.709 | 0.207 | 0.844 |
| qwen35_35b_a3b | qwen35_9b | 0.751 | 0.249 | 0.706 |
| qwen35_4b | qwen35_35b_a3b | 0.645 | 0.275 | 0.827 |

Interpretation:

- Pairwise supports that higher-alpha outputs are often preferred as more Style-B-like.
- Position and judge effects are nontrivial, especially for 35B-A3B judged by 9B.
- Pairwise is an auxiliary monotonicity/preference audit. Projection geometry remains primary.

## Length Diagnostics

Source table:

`/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/tables/main_length_metrics.csv`

Summary:

| Model | Mean Chinese chars | Length violation rate | Too short | Too long |
|---|---:|---:|---:|---:|
| qwen35_9b | 112.7 | 0.138 | 0.053 | 0.084 |
| qwen35_35b_a3b | 122.0 | 0.200 | 0.028 | 0.172 |
| qwen35_4b | 102.7 | 0.242 | 0.171 | 0.072 |

Interpretation:

- 9B has the best length compliance.
- 35B-A3B tends to be too long.
- 4B has the largest total violation rate and is often too short.
- Length remains a practical limitation, but current diagnostics do not explain away the projection ranking.

## Figures and Docs

Required figures are saved as both PDF and PNG under:

`/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/figures/`

Figure notes:

`/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/docs/main8axis_figure_notes.md`

Geometry summary:

`/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/docs/main8axis_geometry_summary.md`

## Human Validation Preparation

Sample:

`/home/data_cpfs/lizihua/sbe/data/human_eval/human_validation_sample_v1.jsonl`

Protocol:

- `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/docs/human_validation_protocol.md`
- `/home/data_cpfs/lizihua/sbe/docs/human_validation_protocol.md`

Sample size is 336 rows. It covers 3 models, 8 axes, and 7 alpha ratios, selecting one low-projection-error and one high-projection-error item per `{model, axis, alpha}`.

No human annotation has been run yet.

## Prepared Plans Not Yet Run

Cross-family plan:

- `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/docs/cross_family_experiment_plan.md`
- `/home/data_cpfs/lizihua/sbe/docs/cross_family_experiment_plan.md`

Temperature-ablation plan:

- `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_final_eval/docs/temperature_ablation_plan.md`
- `/home/data_cpfs/lizihua/sbe/docs/temperature_ablation_plan.md`

Do not run these automatically. The next step should be chosen after the user reviews the final report.

## Scripts Relevant to This Stage

- `sbe/scripts/10_main8axis_final_analysis.py`
- `sbe/scripts/11_prepare_main8axis_pairwise_sample.py`
- `sbe/scripts/12_main8axis_pairwise_metrics.py`
- `sbe/scripts/13_prepare_human_validation_sample.py`

Validation already performed:

- `compileall` passed for scripts 10, 12, and 13.
- Required output check returned `missing_count=0`.

## Operational Notes for Next Codex

- Working root: `/home/data_cpfs/lizihua`.
- Read memory first:
  - `/home/data_cpfs/lizihua/memory/INDEX.md`
  - `/home/data_cpfs/lizihua/memory/context/current.md`
  - `/home/data_cpfs/lizihua/memory/logs/2026-06-24-main8axis-final-analysis-complete.md`
- Current SGLang endpoints may not be running. Do not assume live inference is available.
- Do not launch new large generation until the user explicitly chooses a next experiment.
- If continuing analysis, start from the final report and tables, not from raw generation.

## Recommended Next Decisions

1. Review the final report and decide which findings are paper-ready.
2. Start human annotation if external validation is the priority.
3. Run the cross-family subset only after confirming a non-Qwen instruction-tuned local model path.
4. Run temperature ablation only as a targeted robustness check, not as part of the main benchmark.
