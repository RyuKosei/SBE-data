# Pilot Experiment Summary for Codex

Date: 2026-06-22

## Scope

This summarizes the current StyleBlend-Bench pilot after adding the updated v2 metric flow.

Pilot data:

- 3 axes: `emotion_apology`, `formality_request`, `empathy_clinical`
- 10 scenarios per axis
- 7 ratios: `0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0`
- 2 seeds
- 3 generator models
- 420 generations per model, 1260 generations total

## Completed Experiments

### E1: Live generation pilot

All three Qwen3.5 models generated complete pilot outputs:

- `qwen35_4b`: 420/420 generations, 0 failures
- `qwen35_9b`: 420/420 generations, 0 failures
- `qwen35_35b_a3b`: 420/420 generations, 0 failures

Non-thinking setup worked. No hard `<think>` leakage remained in the fixed outputs.

### E2: Scalar LLM judge

Judge policy:

- 4B and 9B outputs judged by 35B-A3B
- 35B-A3B outputs judged by 9B

All scalar judge rows parsed:

- 4B judged by 35B: 420/420 parse ok
- 9B judged by 35B: 420/420 parse ok
- 35B-A3B judged by 9B: 420/420 parse ok

### E3: bge-m3 embedding projection

Downloaded multilingual bge-m3:

```bash
/home/data_cpfs/zicheng/models/bge-m3
```

`scripts/02_embed_project.py` now supports local `transformers` loading, so no `sentence-transformers` install is required.

Projection outputs:

- `data/metrics/projection_bge_m3_pilot_4b_v1.jsonl`
- `data/metrics/projection_bge_m3_pilot_9b_v1.jsonl`
- `data/metrics/projection_bge_m3_pilot_35b_v1.jsonl`

### E4: Real pairwise pilot

Pairwise comparisons use local ratio pairs:

- adjacent: `0-.1`, `.1-.25`, `.25-.5`, `.5-.75`, `.75-.9`, `.9-1`
- cross-level: `0-.5`, `.5-1`, `.25-.75`

Outputs:

- 4B judged by 35B: 540 pairwise comparisons
- 9B judged by 35B: 540 pairwise comparisons
- 35B-A3B judged by 9B: 540 pairwise comparisons

All valid after treating invalid winner labels as parse failures. A few earlier judge outputs used invalid `text_3`; analysis code now handles this correctly.

### E5: Pairwise order-bias audit

Ran swapped-order pairwise for 35B-A3B judged by 9B:

- normal order: lower alpha in `text_1`, higher alpha in `text_2`
- swapped order: higher alpha in `text_1`, lower alpha in `text_2`

This is a judge reliability check, not a generator-quality metric.

## Main Pilot Metrics

File:

```bash
results/pilot_bge_m3_pairwise_plus_swap/tables/main_metrics_combined.csv
```

| model | Proj MAE | Proj Spearman | Judge MAE | Judge Spearman | Pairwise violation |
|---|---:|---:|---:|---:|---:|
| qwen35_4b | 0.155 | 0.825 | 0.222 | 0.691 | 0.179 |
| qwen35_9b | 0.117 | 0.905 | 0.126 | 0.884 | 0.133 |
| qwen35_35b_a3b | 0.122 | 0.891 | 0.154 | 0.834 | 0.207 |

Interpretation:

- 9B is best on all three headline metrics.
- 35B-A3B is close to 9B on bge-m3 projection, but weaker on scalar judge and pairwise.
- 4B is consistently weaker.
- This does not support a simple "larger is always better" claim.

## Cross-Metric Agreement

Projection ratio and scalar judge ratio correlate reasonably:

| model | Pearson | Spearman |
|---|---:|---:|
| qwen35_4b | 0.655 | 0.678 |
| qwen35_9b | 0.807 | 0.828 |
| qwen35_35b_a3b | 0.802 | 0.844 |

Interpretation:

- The main embedding metric and scalar judge are aligned enough to continue.
- 4B has weaker cross-metric agreement, likely because outputs are less cleanly calibrated.

## Axis-Level Findings

Mean projection MAE / scalar judge MAE:

| model | axis | projection MAE | judge MAE |
|---|---|---:|---:|
| 4B | emotion_apology | 0.138 | 0.184 |
| 4B | empathy_clinical | 0.165 | 0.334 |
| 4B | formality_request | 0.161 | 0.147 |
| 9B | emotion_apology | 0.098 | 0.119 |
| 9B | empathy_clinical | 0.142 | 0.170 |
| 9B | formality_request | 0.112 | 0.088 |
| 35B-A3B | emotion_apology | 0.117 | 0.098 |
| 35B-A3B | empathy_clinical | 0.130 | 0.247 |
| 35B-A3B | formality_request | 0.117 | 0.117 |

Interpretation:

- `empathy_clinical` is the hardest axis, especially for scalar judge.
- `formality_request` is easiest for 9B and 35B-A3B.
- `emotion_apology` is relatively strong in projection but unstable under pairwise around high-alpha endpoints.

## Monotonicity and Endpoint Behavior

Adjacent monotonic violation by measured curve:

Projection:

- 4B: `emotion 0.200`, `empathy 0.292`, `formality 0.208`
- 9B: `emotion 0.150`, `empathy 0.217`, `formality 0.117`
- 35B-A3B: `emotion 0.200`, `empathy 0.200`, `formality 0.158`

Scalar judge:

- 4B: `emotion 0.142`, `empathy 0.167`, `formality 0.142`
- 9B: `emotion 0.050`, `empathy 0.108`, `formality 0.025`
- 35B-A3B: `emotion 0.142`, `empathy 0.083`, `formality 0.092`

Interpretation:

- 9B has the smoothest monotonic behavior.
- Projection detects more local non-monotonicity than scalar judge.
- The ratio curve is not perfectly smooth; this is scientifically useful for the benchmark.

## Pairwise Judge Reliability

Pairwise normal results:

| model | pairwise violation | tie rate |
|---|---:|---:|
| 4B | 0.179 | 0.095 |
| 9B | 0.133 | 0.083 |
| 35B-A3B | 0.344 normal-only, 0.207 after adding swapped audit |

Swapped-order audit for 35B-A3B judged by 9B:

- normal higher-alpha win rate: 0.656
- swapped higher-alpha win rate: 0.930
- exact consistency: 0.711

By axis:

| axis | normal higher win | swapped higher win | exact consistency |
|---|---:|---:|---:|
| emotion_apology | 0.461 | 0.944 | 0.506 |
| empathy_clinical | 0.794 | 0.906 | 0.878 |
| formality_request | 0.711 | 0.939 | 0.750 |

Interpretation:

- Pairwise judge is position/order sensitive.
- Future pairwise should use bidirectional or randomized order, then aggregate at the pair level.
- Single-direction pairwise should not be treated as a main metric.
- The strongest instability is `emotion_apology`, especially near `0.75-0.9` and `0.9-1.0`.

## Length Compliance

Length violation rates by model/axis:

| model | emotion | empathy | formality |
|---|---:|---:|---:|
| 4B | 0.329 | 0.150 | 0.271 |
| 9B | 0.100 | 0.157 | 0.186 |
| 35B-A3B | 0.257 | 0.229 | 0.357 |

Mean errors by length violation were not dramatically worse. Length noncompliance is still a reporting quality issue, but it does not appear to fully explain calibration errors in the pilot.

## Current Conclusions

1. The updated v2 pipeline is executable end-to-end.
2. bge-m3 projection should be the main pilot metric.
3. 9B currently looks like the best calibrated generator.
4. 35B-A3B is not clearly better than 9B despite being larger/MoE.
5. `empathy_clinical` is the hardest axis and should be retained.
6. Pairwise judging is useful but fragile; future pairwise must be bidirectional/randomized.
7. The current deterministic scenario set is adequate for pipeline testing, but should be curated before main runs.

## Recommended Next Experiments

### High Priority

1. Curate scenarios before main generation.
   - Keep 8 axes.
   - Improve from deterministic variants to hand/LLM-filtered diverse scenarios.
   - Add a scenario validation pass for ambiguity, safety, and content checks.

2. Implement bidirectional pairwise aggregation.
   - For each pair, run both text orders or randomly assign order with balanced design.
   - Treat inconsistent pairs as uncertainty, not hard monotonic failures.

3. Add prompt ablations on a small subset.
   - ratio format: percent vs decimal vs Chinese proportion words
   - A/B order swap
   - few-shot anchors at 0/0.5/1

### Medium Priority

4. Add a second embedding metric.
   - `intfloat/multilingual-e5-large` or `text2vec-base-chinese`
   - Purpose: test whether bge-m3 projection conclusions are robust.

5. Add confidence intervals.
   - Bootstrap by scenario, not by raw row.
   - Report uncertainty for model comparisons.

6. Add content-preservation-focused diagnostics.
   - Current scalar content scores are near ceiling.
   - Need stricter content-check judge or rule-assisted checks.

### Lower Priority

7. Lexical midpoint probe as appendix/motivation.
   - Use Qwen input embedding matrix if making claims about model-internal embedding geometry.
   - Keep separate from main style-ratio benchmark.

8. Conflict prompt probe.
   - Example: numeric ratio says 90% A, natural language says "明显接近B".
   - Tests whether model follows number or descriptive phrase.

9. Endpoint-anchor perturbation.
   - Same style axis with short, long, and example-based style definitions.
   - Tests whether measured style axis is stable to wording.

## Recommended Next Decision

Before main 20,160-row generation, decide one of:

1. Run a curated-scenario pilot first.
2. Run prompt ablation on the current pilot subset.
3. Proceed to main generation but mark scenario quality as a known limitation.

The strongest path is: curate scenarios -> rerun 3-axis pilot -> then launch main generation.
