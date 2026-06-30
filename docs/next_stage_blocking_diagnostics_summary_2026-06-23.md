# StyleBlend-Bench Next-Stage Blocking Diagnostics Summary

Created: 2026-06-23 UTC

## 1. Geometry Diagnostics Summary

Status: complete.

New tables:

- `results/curated3axis_v2_full_eval/tables/geometry_metrics_by_model_embedding.csv`
- `results/curated3axis_v2_full_eval/tables/geometry_metrics_by_axis_model_embedding.csv`
- `results/curated3axis_v2_full_eval/tables/geometry_metrics_by_alpha_model_embedding.csv`
- `results/curated3axis_v2_full_eval/tables/nearest_neighbor_metrics.csv`
- `results/curated3axis_v2_full_eval/tables/out_of_range_projection_metrics.csv`

Report:

- `docs/curated3axis_v2_geometry_diagnostics.md`

Key findings:

- Both embeddings preserve the same projection ranking: `qwen35_9b > qwen35_35b_a3b > qwen35_4b`.
- 9B's advantage is mainly projection calibration: lowest MAE and highest Spearman under both bge-m3 and text2vec.
- 9B does not have the smallest off-axis drift. 35B-A3B often has comparable or lower off-axis drift, but worse projection calibration.
- 35B-A3B does not show a higher out-of-range projection rate. Out-of-range rates are near zero for all models.
- 4B is weaker mainly because of poorer ratio calibration and nearest-neighbor rank, not because it uniquely drifts farther off the A-B axis.

Main geometry table:

| embedding | model | projection MAE | Spearman | norm off-axis drift | out-of-range | NN mean rank |
|---|---|---:|---:|---:|---:|---:|
| bge-m3 | qwen35_9b | 0.121 | 0.889 | 0.494 | 0.001 | 2.950 |
| bge-m3 | qwen35_35b_a3b | 0.134 | 0.864 | 0.457 | 0.000 | 2.885 |
| bge-m3 | qwen35_4b | 0.159 | 0.809 | 0.514 | 0.001 | 3.275 |
| text2vec | qwen35_9b | 0.091 | 0.932 | 0.436 | 0.002 | 2.631 |
| text2vec | qwen35_35b_a3b | 0.100 | 0.921 | 0.417 | 0.001 | 2.630 |
| text2vec | qwen35_4b | 0.142 | 0.856 | 0.475 | 0.001 | 3.065 |

## 2. Length Diagnostics Summary

Status: complete.

New tables:

- `results/curated3axis_v2_full_eval/tables/length_diagnostics_by_model_axis_alpha.csv`
- `results/curated3axis_v2_full_eval/tables/length_error_correlation_with_projection.csv`
- `results/curated3axis_v2_full_eval/tables/length_compliant_subset_metrics.csv`
- `results/curated3axis_v2_full_eval/tables/too_short_too_long_breakdown.csv`

Report:

- `docs/curated3axis_v2_length_diagnostics.md`

Key findings:

- Length violation is real, but current evidence does not show serious contamination of projection results.
- Correlations between output length and projection error are low across both embeddings and all three models.
- The length-compliant subset preserves the same ranking: `qwen35_9b > qwen35_35b_a3b > qwen35_4b`.
- 4B mainly violates by being too short.
- 35B-A3B mainly violates by being too long.
- 9B is closest to acceptable length behavior.

Go/no-go rule result:

- Since length-compliant subset ranking does not reverse and length-error correlations are low, current prompt can be retained for now.
- Length diagnostics should still be reported in the paper and monitored in full main generation.

## 3. 8-Axis Scenario Pool Review

Status: complete.

Generated candidates:

- `data/scenarios/candidate_scenarios_8axis_v1.jsonl`
- 650 rows, 0 failures.

Filtered pool:

- `data/scenarios/styleblend_8axis_scenarios_v1_filtered50.jsonl`
- 50 rows per axis, 400 rows total.

Final pool:

- `data/scenarios/styleblend_8axis_scenarios_v1.jsonl`
- 40 rows per axis, 320 rows total.

Review report:

- `docs/scenario_pool_review_8axis_v1.md`

Review tables:

- `results/scenario_pool_8axis_v1/tables/scenario_domain_distribution.csv`
- `results/scenario_pool_8axis_v1/tables/scenario_validation_summary.csv`
- Also copied to:
  - `tables/scenario_domain_distribution.csv`
  - `tables/scenario_validation_summary.csv`

Domain coverage:

| axis | final scenarios | domain count |
|---|---:|---:|
| concision_detail | 40 | 11 |
| emotion_apology | 40 | 11 |
| empathy_clinical | 40 | 12 |
| expertise_explain | 40 | 10 |
| formality_request | 40 | 11 |
| humor_neutral | 40 | 10 |
| objectivity_cat | 40 | 11 |
| politeness_refusal | 40 | 10 |

Review notes:

- All axes exceed the minimum requirement of 6 domain categories.
- Final validation found no missing required fields.
- Rule-based safety scan found no final-pool fact/safety risks.
- Some `concision_detail` and `expertise_explain` scenarios are flagged as potentially sparse, but still usable; manual spot-check before full main generation is recommended.

## 4. Prompt Modification Recommendation

Recommendation: do not modify the main prompt yet.

Reason:

- Length violation is not strongly correlated with projection error.
- Length-compliant subset preserves the model ranking.
- Changing the prompt would require rerunning curated 3-axis v2 and comparing old vs new prompt.

Keep the proposed relaxed prompt as a fallback only if full main smoke/pilot shows stronger length contamination.

## 5. Full Main Generation Go/No-Go

Current recommendation: GO with caution after one final manual spot-check of the 8-axis scenario pool.

Blocking diagnostics status:

- Complete geometry tables: GO.
- Length diagnostics: GO, no prompt change required now.
- 8-axis scenario pool: GO, with optional manual spot-check for sparse scenarios.
- Embedding scripts: GO, bge-m3 and text2vec have both run stably.
- Generation script: GO, supports resume, failure logging, seed recording, raw prompt/output fields, and complete JSONL fields.

Remaining non-blocking risks:

- Scenario pool was LLM-generated and rule-reviewed, not deeply human-reviewed.
- Some scenario contents may be semantically sparse for `concision_detail` or `expertise_explain`.
- Pairwise remains auxiliary and should not be used as the main metric.

## 6. Concrete Next Step

Before launching the full 20,160-row run, do a short full-main smoke using the new 8-axis scenario file:

- 8 axes
- 1-2 scenarios per axis
- 7 ratios
- 1 seed
- 1 model, preferably `qwen35_9b`

If field completeness, length behavior, and embedding projection all look normal, proceed to full main generation.

