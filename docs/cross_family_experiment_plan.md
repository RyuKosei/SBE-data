# Cross-Family Experiment Plan

## Status

Do not run this experiment until the main 8-axis final report is complete.

## Available Non-Qwen Model Paths Found

| path                                                 | model_type   |   hidden_size |   num_hidden_layers | notes                                                             |
|:-----------------------------------------------------|:-------------|--------------:|--------------------:|:------------------------------------------------------------------|
| /home/data_cpfs/zicheng/models/bge-large-en-v1.5     | bert         |          1024 |                  24 | embedding model; not suitable as generative cross-family baseline |
| /home/data_cpfs/zicheng/models/bge-m3                | xlm-roberta  |          1024 |                  24 | embedding model; not suitable as generative cross-family baseline |
| /home/data_cpfs/zicheng/models/text2vec-base-chinese | bert         |           768 |                  12 | embedding model; not suitable as generative cross-family baseline |

## Current Assessment

The scanned non-Qwen paths under `/home/data_cpfs/zicheng/models` appear to be embedding models rather than instruction-tuned generative LLMs. They are useful for measurement, not as tested generator baselines.

Before running a cross-family generation experiment, identify at least one instruction-tuned non-Qwen model, such as a Llama-family, Mistral-family, Gemma-family, Yi-family, or DeepSeek-chat/instruct model, with a local path and license suitable for evaluation.

## Recommended Subset

- Axes: `formality_request`, `emotion_apology`, `empathy_clinical`, `expertise_explain`.
- Scenarios: 20 per axis.
- Ratios: 0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0.
- Seeds: 2.
- Scale: `4 axes * 20 scenarios * 7 ratios * 2 seeds * N models = 1120 * N generations`.

## Goals

- Test whether explicit style-ratio calibration exists outside the Qwen family.
- Compare Qwen3.5-9B against similarly sized 7B/8B/9B instruction models.
- Keep bge-m3/text2vec projection as the main metric for comparability.

## Execution Recommendation

Do not launch until:

1. Main 8-axis final report is complete.
2. A non-Qwen instruction-tuned model path is confirmed.
3. A short smoke test verifies no reasoning/meta leakage and acceptable Chinese output quality.
