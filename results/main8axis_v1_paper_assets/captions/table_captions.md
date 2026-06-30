# Table Captions

## Table 1. Main Model Comparison
Caption: Model-level projection and geometry metrics averaged across bge-m3 and text2vec, with length violation rates.
Key takeaway: Qwen3.5-9B has the best style-ratio calibration.
Use in paper section: Results.

## Table 2. Embedding Robustness
Caption: Projection MAE, Spearman correlation, off-axis drift, and normalized drift by embedding and model.
Key takeaway: bge-m3 and text2vec agree on the model ranking.
Use in paper section: Robustness.

## Table 3. Axis-level Projection
Caption: Axis-level projection, drift, nearest-neighbor, and length metrics by model.
Key takeaway: Style axes differ in difficulty, with detail/concision among the harder axes.
Use in paper section: Axis analysis.

## Table 4. Geometry Decomposition
Caption: Model-level decomposition of projection MAE, interpolation distance, off-axis drift, normalized drift, and out-of-range rate.
Key takeaway: 9B's advantage is ratio-placement calibration, not lower off-axis drift.
Use in paper section: Diagnostics.

## Table 5. Length Diagnostics
Caption: Length-compliance metrics and length/projection-error correlations.
Key takeaway: Length violations are material but do not explain away the projection ranking.
Use in paper section: Diagnostics.

## Table 6. Scalar and Pairwise Auxiliary Metrics
Caption: Scalar judge and sampled pairwise metrics with judge model assignment.
Key takeaway: Auxiliary metrics support broad monotonicity but should not be treated as primary due to judge asymmetry and order effects.
Use in paper section: Auxiliary validation.

## Table 7. Nearest-neighbor and Out-of-range Diagnostics
Caption: Nearest-neighbor rank, top-1 accuracy, out-of-range projection rate, and low/high-alpha overshoot diagnostics by embedding.
Key takeaway: Extreme out-of-range failures are rare; most errors are in-range calibration errors.
Use in paper section: Diagnostics.
