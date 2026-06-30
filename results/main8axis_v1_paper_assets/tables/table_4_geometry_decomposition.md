| Model           |   Projection MAE |   Interpolation distance |   Off-axis drift |   Normalized drift |   Out-of-range rate |
|:----------------|-----------------:|-------------------------:|-----------------:|-------------------:|--------------------:|
| Qwen3.5-9B      |         0.118216 |                 0.284047 |         0.272431 |           0.484692 |         0.00111607  |
| Qwen3.5-35B-A3B |         0.133697 |                 0.271891 |         0.256828 |           0.472701 |         0.000669643 |
| Qwen3.5-4B      |         0.157498 |                 0.283641 |         0.266248 |           0.523271 |         0.00104167  |

Interpretation: Qwen3.5-9B has the lowest projection MAE, so its advantage is primarily better ratio placement. Qwen3.5-35B-A3B has the lowest off-axis and normalized drift, showing that lower drift alone does not imply better calibration. Qwen3.5-4B has the worst projection MAE and normalized drift. Out-of-range projection is rare for all models, so the dominant errors are in-range under/over-placement rather than extreme overshoot.

