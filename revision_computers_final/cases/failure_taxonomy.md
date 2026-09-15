# Failure taxonomy results

Threshold configuration: `/home/data_cpfs/lizihua/sbe/revision_computers_final/config/failure_thresholds.yaml` (frozen before new-model ranking).
Content-style entanglement status: **complete**.

| Failure | Count | Proportion |
|---|---:|---:|
| late_under_response | 2818 / 4800 | 58.7% |
| curved_trajectory | 2459 / 4800 | 51.2% |
| early_overshoot | 2391 / 4800 | 49.8% |
| non_monotonic_reversal | 1275 / 4800 | 26.6% |
| near_constant_response | 577 / 4800 | 12.0% |
| endpoint_collapse | 480 / 4800 | 10.0% |
| high_off_axis_drift | 480 / 4800 | 10.0% |
| content_style_entanglement | 364 / 4800 | 7.6% |
| length_dominated_control | 194 / 4800 | 4.0% |
| out_of_range_extrapolation | 135 / 4800 | 2.8% |

## Unexpected negative finding

The most prevalent preregistered flag was `late_under_response` (58.7% of scenario-model-encoder cells). This prevents interpreting a favorable average ICE as evidence that trajectories are generally clean or one-dimensional.

Representative positive and negative cases, including the midpoint text, are stored in `cases/failure_cases.csv`.
