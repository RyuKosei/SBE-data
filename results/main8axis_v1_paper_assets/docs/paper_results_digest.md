# Paper Results Digest

## One-paragraph Result Summary

StyleBlend-Bench main8axis_v1 shows that explicit style-ratio prompting yields measurable monotonic calibration across eight Chinese style axes, but model scale does not translate monotonically into better ratio control. Across bge-m3 and text2vec projection metrics, Qwen3.5-9B is the best calibrated model, Qwen3.5-35B-A3B is close but less accurately placed along the style-ratio axis, and Qwen3.5-4B is weakest. Auxiliary scalar and pairwise audits broadly support the projection-based findings but should not be used as primary metrics because judge assignment is asymmetric and pairwise order effects are visible.

## Paper-ready Findings

### Finding 1: Qwen3.5-9B achieves the best style-ratio calibration.
Evidence: `tables/table_1_main_model_comparison.md`, `figures/fig_3_model_comparison_projection_mae_ci.pdf`.
Numbers: Qwen3.5-9B has mean projection MAE 0.118 and mean Spearman 0.890.
Interpretation: The mid-sized model follows requested style ratios better than both smaller and larger Qwen-family variants.

### Finding 2: The embedding choice does not change the model ranking.
Evidence: `tables/table_2_embedding_robustness.md`, `figures/fig_1_model_calibration_curves_bge_m3.pdf`, `figures/fig_2_model_calibration_curves_text2vec.pdf`.
Numbers: 9B has the lowest projection MAE under both bge-m3 and text2vec.
Interpretation: The core ranking is robust to the two independent embedding spaces used here.

### Finding 3: 9B's advantage is ratio placement rather than lower off-axis drift.
Evidence: `tables/table_4_geometry_decomposition.md`, `figures/fig_6_geometry_decomposition_bar.pdf`.
Numbers: 35B-A3B has lower normalized drift than 9B, but 9B has lower projection MAE.
Interpretation: Staying near the interpolation line is not sufficient; the model must land at the correct location along that line.

### Finding 4: Axis difficulty is heterogeneous.
Evidence: `tables/table_3_axis_level_projection.md`, `tables/table_3_axis_projection_mae_pivot.md`, `figures/fig_4_axis_heatmap_projection_mae.pdf`.
Numbers: Projection MAE varies visibly by axis and model.
Interpretation: Style-ratio controllability should be reported per axis, not only as a global average.

### Finding 5: Length compliance is a real diagnostic but not the main explanation.
Evidence: `tables/table_5_length_diagnostics.md`, `figures/fig_7_length_violation_heatmap.pdf`.
Numbers: 9B has the lowest length violation rate; 4B has the highest.
Interpretation: Length behavior affects benchmark reliability and should be reported, but current length-error correlations do not explain away the projection ranking.

### Finding 6: Scalar judge agrees broadly but remains auxiliary.
Evidence: `tables/table_6_scalar_and_pairwise_auxiliary.md`, `figures/fig_8_projection_vs_scalar_judge_scatter.pdf`.
Numbers: Scalar MAE ranks 9B and 35B-A3B close together and 4B weakest.
Interpretation: Scalar judgments are useful validation but should not replace geometry because the judge model differs across evaluated models.

### Finding 7: Pairwise judgments support monotonicity but reveal noise and bias.
Evidence: `tables/table_6_scalar_and_pairwise_auxiliary.md`, `figures/fig_10_pairwise_auxiliary_audit.pdf`.
Numbers: Higher-alpha win rates are above chance for all models, but bidirectional consistency is imperfect.
Interpretation: Pairwise audit is best framed as an auxiliary sanity check, not a primary benchmark score.

## Claims That Are Safe to Write

- Qwen3.5-9B is best calibrated among the three tested Qwen-family models under the current main8axis_v1 setup.
- bge-m3 and text2vec agree on the main ranking.
- 35B-A3B's lower off-axis drift does not translate into better projection calibration than 9B.
- 4B is weakest and has the largest length-compliance issue.
- Scalar and pairwise audits are auxiliary checks.

## Claims That Should Not Be Written Yet

- Do not claim universal superiority of 9B across model families; no cross-family generation has been run.
- Do not claim human-validated calibration; the human validation sample is prepared but not annotated.
- Do not treat scalar judge or pairwise judge scores as primary metrics.
- Do not claim temperature robustness; the temperature ablation plan exists but has not been executed.

## Missing Evidence / Next Work

- Human annotations for `data/human_eval/human_validation_sample_v1.jsonl`.
- Cross-family subset generation after confirming a non-Qwen instruction-tuned local model path.
- Temperature ablation if reviewers ask whether sampling temperature drives calibration.
- Optional external judge to reduce asymmetric judge caveats in scalar/pairwise validation.
