# Temperature Ablation Plan

## Status

Do not run this ablation until the main 8-axis final report is complete.

## Proposed Settings

- Temperatures: 0.2, 0.7, 1.0.
- Models: `qwen35_9b`, `qwen35_35b_a3b`.
- Axes: choose 3 axes from `formality_request`, `emotion_apology`, `empathy_clinical`, `expertise_explain`.
- Scenarios: 20 per axis.
- Ratios: 0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0.
- Seeds: 2.

Scale:

`3 axes * 20 scenarios * 7 ratios * 2 seeds * 2 models * 3 temps = 5040 generations`.

## Questions

1. Does low temperature improve projection calibration?
2. Does high temperature increase naturalness but increase off-axis drift?
3. Does temperature change the 9B vs 35B-A3B ranking?
4. Does length compliance improve or degrade at lower temperature?

## Recommended Metrics

- bge-m3 and text2vec projection MAE/Spearman.
- Interpolation distance.
- Off-axis drift and normalized off-axis drift.
- Length violation rate.
- Scalar judge on a subset only, if needed.

## Execution Recommendation

Run only after the final report identifies whether the current main result needs robustness support. The ablation should be framed as a targeted robustness check, not as part of the primary benchmark.
