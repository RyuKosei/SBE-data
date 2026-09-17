# Completion Report — StyleBlend-Bench MDPI Computers Revision

Generated on 2026-09-15. The unified output root is `revision_computers_final/`. No legacy result file was overwritten.

## Executive status

Cluster-executable generation, embedding, automatic evaluation, geometry, trajectory, ablation, robustness, content, failure, scale, efficiency, statistics, tables, figures, manuscript text, and reproducibility work is complete. Two activities require real human action and are marked **external pending** rather than filled with synthetic data:

1. independent rewriting and two-author approval of 80 shared endpoint pairs;
2. collection of 960 genuine ratings for the blinded 320-text human sample.

Consequently, absolute self-versus-shared ranking results, shared-anchor prompt/temperature geometry, ICC/alpha/kappa, human–geometry correlation, and automatic-checker validation against humans must not yet be reported as completed empirical results.

## Completion checklist

| Requirement | Status | Actual count / evidence | Output |
|---|---|---:|---|
| Locate and audit legacy data/code | Complete | 320 scenarios; 8 axes; 20,160 old generations; 40,320 old projection rows; zero completed human ratings | `audit/data_audit.md`, `audit/audit_summary.json` |
| Correct endpoint aggregation and metric definitions | Complete | Primary means use five internal ratios; raw projection precedes clipping; decomposition residual checked below 1e-6 | `metrics/metric_definitions.md`, `scripts/geometry_metrics.py` |
| Reproduce old-table sanity targets | Complete | All six encoder/model checks within absolute tolerance 0.002 | `audit/data_audit.md` |
| Complete Qwen model registry | Complete | Five BF16 checkpoints: three Qwen3.5 and two Qwen3.6 | `config/model_registry.csv` |
| Run all available Qwen checkpoints | Complete | 5 × 6,720 = 33,600 valid main outputs | `data/qwen_family_outputs.jsonl` |
| Run three encoders | Complete | 100,800 sample-level rows; 4,800 scenario-level rows; 15 model–encoder groups | `metrics/sample_level_metrics.parquet`, `metrics/scenario_level_metrics.parquet` |
| Prepare shared-anchor data | Package complete; approval external pending | 80 scenarios, 160 endpoint text slots, 0 approved pairs | `data/shared_anchor_candidates.xlsx`, `data/shared_anchors_final.csv` |
| Self-anchor analysis | Complete | 14,400 scenario/model/seed/encoder endpoint pairs | `metrics/self_anchor_pairs.parquet`, `tables/table03_self_anchor_diagnostics.csv` |
| Shared-anchor analysis | External pending | Analyzer refuses blank/unapproved endpoints | `scripts/analyze_shared_anchors.py`, `statistics/shared_anchor_status.csv` |
| Linearity and empirical trajectory | Complete | 14,400 seven-point trajectory rows and 14,400 leave-one-seed-out rows | `tables/table05_linearity_diagnostics.csv`, `tables/ablation_endpoint_line_centroid_piecewise.csv` |
| Baselines and ablations | Complete, except shared-anchor branch | Constant, length CV, distance ratio, isotonic CV, scalar judge; endpoint/clipping/encoder/axis/separation/trajectory ablations | `tables/table06_baselines_detailed.csv`, `tables/ablation_*.csv` |
| Prompt/temperature generation | Complete | 9,600 total: 2,400 reused + 7,200 new | `data/robustness_outputs.jsonl` |
| Prompt/temperature fixed-judge analysis | Complete | 3,200 seed-aggregated factorial cells; 444 effect rows | `statistics/robustness_factorial_effects.csv` |
| Prompt/temperature shared-anchor geometry | External pending | Requires approved shared endpoints | `statistics/robustness_shared_anchor_status.csv` |
| Automatic content and quality evaluation | Complete | 33,600 main evaluations; 9,600 robustness evaluations; no missing run IDs | `metrics/llm_evaluations.jsonl`, `metrics/content_quality_sample_level.parquet` |
| Human annotation package | Complete | 320 texts; 960 formal assignments; 24 attention checks | `human_eval/annotation_items.csv`, `human_eval/annotation_form.xlsx`, `human_eval/instructions_zh.md`, `human_eval/randomization_map.csv` |
| Genuine human results | External pending | Blank results template; no simulated ratings | `human_eval/human_results.csv`, `statistics/human_evaluation_results.txt` |
| Qwen scale/version/architecture analysis | Complete, descriptive | Five models; Qwen3.5 and Qwen3.6 distinguished; causal inference prohibited | `statistics/scale_version_architecture.csv` |
| Efficiency and Pareto analysis | Complete | Five generation runs; direct timing comparison limited to three matched TP2/concurrency-32 runs | `metrics/inference_efficiency.csv`, `statistics/efficiency_pareto.csv` |
| Bootstrap, paired tests, hypotheses | Complete | 10,000 scenario-level resamples; 60 paired-test rows; Holm adjustment | `statistics/bootstrap_results.csv`, `statistics/paired_tests.csv`, `statistics/hypothesis_tests.csv` |
| Mixed-effects model | Attempted; singular, fallback complete | Both prespecified MixedLM outcomes were singular; scenario-clustered robust OLS retained | `statistics/mixed_effects_results.txt` |
| Failure taxonomy | Complete | 4,800 scenario/model/encoder classifications, all ten fixed types | `metrics/failure_classifications.parquet`, `cases/failure_cases.csv` |
| Main tables and figures | Complete with explicit pending panels | Ten 300-dpi PNG files and ten vector PDFs | `tables/`, `figures/` |
| English manuscript insertions | Complete with only external-result placeholders | Required section structure present | `manuscript_insertions_en.md` |
| Minimal reproducibility package | Complete | Eleven unit tests pass; included example validates residual and constant baselines | `reproducibility_package/` |

## Actual generation and evaluation counts

| Stage | Planned | Valid final | Missing | Notes |
|---|---:|---:|---:|---|
| Main Qwen generation | 33,600 | 33,600 | 0 | Five models, seven ratios, three seeds |
| Robustness corpus | 9,600 | 9,600 | 0 | Four representative models, four conditions |
| Main fixed LLM evaluation | 33,600 | 33,600 | 0 | Fixed evaluator/temperature/schema |
| Robustness fixed LLM evaluation | 9,600 | 9,600 | 0 | No final parse failures |
| Human formal ratings | 960 | 0 | 960 | External pending |
| Approved shared endpoint texts | 160 | 0 | 160 | External pending; blank cells were not analyzed |

## Failures, repairs, and exclusions

- Qwen3.6-27B initially produced 14 responses that became empty after removal of a thinking block. The invalid raw rows were archived in `logs/invalid_generation_rows_qwen36_27b.jsonl`; the same run keys were regenerated successfully. The final main corpus contains no empty response.
- One Qwen3.6-27B robustness item remained empty over repeated attempts on the first server lifetime. It succeeded after a clean server reload. Every failed attempt remains in `logs/robustness_failures_qwen36_27b.jsonl`.
- The final main corpus has 50 responses with a length finish reason. These are retained and marked as truncations; they were not silently deleted.
- The main automatic evaluator produced schema failures caused by transparent misspellings of `factual_contradiction` or omission of non-decisive fields on unjudgeable items. Raw failures remain logged. The parser accepts only an unambiguous boolean field beginning with `factual_contrad`, preserves its value, and treats missing content fields as unknown only when `unjudgeable=true`. All 33,600 keys were then evaluated successfully.
- No OOM occurred. Qwen3.6-27B has one recorded generated-row retry; other final main rows report zero request retries.
- The specified MixedLM was singular. This is reported explicitly, with a robust scenario-clustered OLS fallback and multicollinearity warning.
- No human score was simulated, copied, or duplicated. The old project contained an unannotated 336-row package, not 160 completed ratings, so none was counted as human evidence.

## Core conclusions currently suitable for the paper

1. Including the two endpoint rows mechanically multiplies five-point ICE by 5/7 and understates internal-ratio error by 28.57%.
2. Qwen3.5-9B has the lowest self-anchor geometric ICE for all three encoders. The complete ICE order is stable across encoders.
3. Larger total parameter count does not guarantee lower ICE in this panel; the evidence is descriptive rather than a scaling law.
4. Endpoint separation is negatively correlated with ICE (Spearman −0.317 to −0.413) and still more strongly with normalized drift (−0.662 to −0.811).
5. ICE and drift have only moderate association (0.338–0.492), and their model orderings differ.
6. The seven-point paths are frequently nonlinear: model-level PC1 variance is 0.349–0.408 and ordered path-length ratio 4.64–5.20.
7. Concision–detail is the hardest tested axis (ICE 0.2366; drift 0.8960; monotonicity 0.1375).
8. Self-anchor endpoint intensity is notably compressed for Qwen3.5-4B, demonstrating why local self coordinates cannot establish absolute cross-model strength.
9. The P2 prompt rewrite increases fixed-judge style MAE by 0.0167 (95% CI 0.0065–0.0274) while shortening text and slightly improving automatic content pass. Temperature reduction has little effect on style MAE but a small positive naturalness effect.
10. Automated content-pass rates are 96.79%–98.17%, but these values require human validation and must be labeled as fixed-judge results.

## Conclusions that must not be placed in the paper

- Style is globally linear in sentence-embedding or human perceptual space.
- These results generalize directly beyond the Qwen family, Chinese tasks, and the eight tested axes.
- Self-anchor metrics prove absolute cross-model control ability.
- Five confounded checkpoints establish a universal parameter scaling law or a causal Dense/MoE advantage.
- The fixed Qwen evaluator is equivalent to human judgment.
- Shared-anchor model rankings or prompt/temperature geometric effects are known before endpoint approval.
- Human correlation, ICC, alpha, kappa, accuracy, precision, recall, F1, or confusion-matrix values exist before real ratings are returned.

## Table and figure provenance

| Asset | Primary source data |
|---|---|
| Table 1: dataset/models/generation | `data/qwen_family_outputs.jsonl`, `config/model_registry.csv` |
| Table 2: all Qwen main metrics | `statistics/bootstrap_results.csv`, `metrics/model_level_summary.csv` |
| Table 3: self/shared anchors | `metrics/self_anchor_pairs.parquet`, `statistics/shared_anchor_status.csv` |
| Table 4: three encoders | `metrics/model_level_summary.csv`, `tables/three_encoder_ranking_consistency.csv` |
| Table 5: linearity | `metrics/fragments/*/*.trajectory.parquet` |
| Table 6: baselines/ablations | `metrics/baseline_scenario_level.parquet`, `tables/ablation_*.csv` |
| Table 7: human agreement | `human_eval/human_results.csv`; currently external-pending status only |
| Table 8: efficiency/Pareto | `metrics/inference_efficiency.csv`, `metrics/model_level_summary.csv`, `metrics/content_quality_sample_level.parquet` |
| Figure 1: method framework | Experiment configurations and methods |
| Figure 2: projection/drift schematic | Metric definitions |
| Figure 3: representative PCA trajectory | BGE-M3 embeddings and main-generation rows |
| Figure 4: five versus seven points | `tables/ablation_5point_7point_raw_clipped.csv` |
| Figure 5: self/shared ranking | Explicit external-pending panel; no fabricated ranking |
| Figure 6: encoder ranking heatmap | `metrics/model_level_summary.csv` |
| Figure 7: scale, ICE, drift | Model registry and bootstrap main metrics |
| Figure 8: prompt/temperature | `statistics/robustness_factorial_effects.csv` |
| Figure 9: failure taxonomy | `metrics/failure_taxonomy_counts.csv` |
| Figure 10: quality-efficiency | `statistics/efficiency_pareto.csv`; comparison-condition caveat shown |

## Required human handoff

1. Fill `data/shared_anchor_candidates.xlsx`, transfer only approved rows into `data/shared_anchors_final.csv`, and set `approval_status=approved` after both authors complete every check. Then run `scripts/analyze_shared_anchors.py` and the shared-anchor robustness branch.
2. Distribute the three blinded worksheets in `human_eval/annotation_form.xlsx`. After genuine ratings are returned, place them in `human_eval/human_results.csv` and run `scripts/analyze_human_eval.py`.
3. Replace only the `[RESULT: ...; status=external_pending]` placeholders in `manuscript_insertions_en.md` after the corresponding files contain real results.
