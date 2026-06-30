# Figure Captions

## Figure 1. Calibration Curves under bge-m3
Caption: Mean measured projection ratio by target alpha for each model, with 95% normal-approximation standard-error bands over generated samples. The dashed diagonal is ideal calibration.
Key takeaway: All models are broadly monotonic, but Qwen3.5-9B stays closest to the diagonal.
Use in paper section: Results.

## Figure 2. Calibration Curves under text2vec
Caption: Same calibration view as Figure 1 using text2vec embeddings.
Key takeaway: text2vec reproduces the same model ordering as bge-m3, supporting embedding robustness.
Use in paper section: Results / Robustness.

## Figure 3. Projection MAE with Bootstrap Confidence Intervals
Caption: Projection MAE by model and embedding with scenario-bootstrap 95% confidence intervals.
Key takeaway: Qwen3.5-9B has the lowest projection MAE under both embeddings.
Use in paper section: Results.

## Figure 4. Axis Heatmap of Projection MAE
Caption: Axis-level projection MAE averaged across embeddings, with darker cells indicating larger calibration error.
Key takeaway: Axis difficulty varies substantially; detail/concision is a recurring hard axis.
Use in paper section: Results / Axis analysis.

## Figure 5. Axis Heatmap of Normalized Off-axis Drift
Caption: Axis-level normalized off-axis drift averaged across embeddings.
Key takeaway: Drift patterns differ from projection MAE, separating ratio-placement errors from off-axis geometry.
Use in paper section: Diagnostics.

## Figure 6. Geometry Decomposition
Caption: Model-level grouped bars for projection MAE, normalized off-axis drift, and interpolation distance.
Key takeaway: 35B-A3B has lower drift but worse projection calibration than 9B, so 9B's advantage is not simply lower off-axis movement.
Use in paper section: Diagnostics.

## Figure 7. Length Violation Heatmap
Caption: Length violation rate by axis and model.
Key takeaway: 9B has the best length compliance, while 4B is often too short and 35B-A3B is often too long.
Use in paper section: Diagnostics.

## Figure 8. Projection vs Scalar Judge Scatter
Caption: Sampled bge-m3 projection ratios against scalar judge style-B ratios, colored by model.
Key takeaway: Projection and scalar judgments are correlated, but scalar remains auxiliary due to judge asymmetry.
Use in paper section: Auxiliary validation.

## Figure 9. Nearest-neighbor Mean Rank
Caption: Mean nearest-neighbor rank of generated outputs relative to target interpolation points, faceted by embedding through grouped bars.
Key takeaway: Nearest-neighbor behavior agrees with the main projection ranking.
Use in paper section: Diagnostics.

## Figure 10. Pairwise Auxiliary Audit
Caption: Pairwise higher-alpha win rate, monotonic violation rate, and bidirectional consistency by model.
Key takeaway: Pairwise judgments broadly support monotonicity but show noise and order/judge effects.
Use in paper section: Auxiliary validation.
