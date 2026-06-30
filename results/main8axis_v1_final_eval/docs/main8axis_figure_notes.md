# Main 8-Axis Figure Notes

## calibration_curves_by_model_bge_m3

- PDF: `../figures/calibration_curves_by_model_bge_m3.pdf`
- PNG: `../figures/calibration_curves_by_model_bge_m3.png`
- Note: Mean clipped projection ratio by target ratio for bge-m3. The dashed line is ideal calibration.

## calibration_curves_by_model_text2vec

- PDF: `../figures/calibration_curves_by_model_text2vec.pdf`
- PNG: `../figures/calibration_curves_by_model_text2vec.png`
- Note: Mean clipped projection ratio by target ratio for text2vec.

## axis_heatmap_projection_mae

- PDF: `../figures/axis_heatmap_projection_mae.pdf`
- PNG: `../figures/axis_heatmap_projection_mae.png`
- Note: Axis-level projection absolute error under bge-m3. Lower is better.

## axis_heatmap_off_axis_drift

- PDF: `../figures/axis_heatmap_off_axis_drift.pdf`
- PNG: `../figures/axis_heatmap_off_axis_drift.png`
- Note: Axis-level off-axis drift under bge-m3. Lower indicates generations stay closer to the endpoint interpolation line.

## model_comparison_bootstrap_ci

- PDF: `../figures/model_comparison_bootstrap_ci.pdf`
- PNG: `../figures/model_comparison_bootstrap_ci.png`
- Note: Scenario-bootstrap 95% confidence intervals for projection MAE by embedding model.

## projection_vs_scalar_judge_scatter

- PDF: `../figures/projection_vs_scalar_judge_scatter.pdf`
- PNG: `../figures/projection_vs_scalar_judge_scatter.png`
- Note: Scatter of bge-m3 projection ratio against scalar judge ratio. Scalar is auxiliary and judge assignment is asymmetric for 35B-A3B.

## length_violation_heatmap

- PDF: `../figures/length_violation_heatmap.pdf`
- PNG: `../figures/length_violation_heatmap.png`
- Note: Length violation rate by model and axis.

## nearest_neighbor_rank_by_model

- PDF: `../figures/nearest_neighbor_rank_by_model.pdf`
- PNG: `../figures/nearest_neighbor_rank_by_model.png`
- Note: Mean nearest-neighbor rank of each output against its target interpolation point. Lower is better.

## out_of_range_rate_by_model_axis

- PDF: `../figures/out_of_range_rate_by_model_axis.pdf`
- PNG: `../figures/out_of_range_rate_by_model_axis.png`
- Note: Rate of raw projection ratios outside [0, 1], averaged across embeddings.
