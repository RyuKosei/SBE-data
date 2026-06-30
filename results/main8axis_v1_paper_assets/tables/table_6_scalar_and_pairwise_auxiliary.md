| Model           |   Scalar judge MAE |   Scalar judge Spearman |   Higher-alpha win rate |   Pairwise monotonic violation |   Bidirectional consistency |    Tie rate | Judge model                     |
|:----------------|-------------------:|------------------------:|------------------------:|-------------------------------:|----------------------------:|------------:|:--------------------------------|
| Qwen3.5-9B      |           0.175402 |                0.768238 |                0.709144 |                       0.207169 |                    0.843981 | 0.105556    | qwen35_35b_a3b / qwen35_35b_a3b |
| Qwen3.5-35B-A3B |           0.182418 |                0.756274 |                0.751042 |                       0.248784 |                    0.706019 | 0.000231481 | qwen35_9b / qwen35_9b           |
| Qwen3.5-4B      |           0.24424  |                0.589193 |                0.644751 |                       0.274645 |                    0.827043 | 0.111124    | qwen35_35b_a3b / qwen35_35b_a3b |

Note: scalar and pairwise are auxiliary due to judge asymmetry and order effects.

