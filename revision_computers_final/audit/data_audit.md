# StyleBlend-Bench Legacy Data Audit

Generated: 2026-09-15T13:29:46.604886Z
Git commit: `ad10aad52cdd974402c98d109e5108a6755ee096`

## Executive findings

- The scenario source contains 320 unique scenarios across eight axes.
- The three historical generation files contain 6,720 rows each, totaling 20,160 = 320 × 7 × 3 × 3.
- Historical ratios are `0, 0.1, 0.25, 0.5, 0.75, 0.9, 1`, not the revision's sixth-based grid.
- Historical rows carry seed labels 1/2/3, but every generation signature has `pass_seed=false`; the seed was not sent to the inference server.
- Legacy primary projection MAE used clipped coordinates and included the two mechanically zero-error endpoints. Primary drift and interpolation-distance means also included those zero endpoints.
- The corrected five-interior-point clipped MAEs reproduce the supplied sanity targets within 0.002. Raw, unclipped values are retained separately and will be primary in the revision.
- Out-of-range rate in the final legacy script was computed from raw projection coordinates before clipping.
- No completed 160-item human-rating dataset exists under the project. The only human file is a 336-row, unannotated selection package; historical handoff notes explicitly say annotation was not run.

## Source files and coverage

| Model | Path | Records | Unique scenarios | Ratios | Seeds | Duplicate keys | Bad lines | Empty outputs | Seed passed |
|---|---|---:|---:|---|---|---:|---:|---:|---|
| qwen35_4b | `/home/data_cpfs/lizihua/sbe/data/generations/qwen35_4b/main8axis_v1_4b.jsonl` | 6720 | 320 | [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] | [1, 2, 3] | 0 | 0 | 0 | False |
| qwen35_9b | `/home/data_cpfs/lizihua/sbe/data/generations/qwen35_9b/main8axis_v1_9b.jsonl` | 6720 | 320 | [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] | [1, 2, 3] | 0 | 0 | 0 | False |
| qwen35_35b_a3b | `/home/data_cpfs/lizihua/sbe/data/generations/qwen35_35b_a3b/main8axis_v1_35b.jsonl` | 6720 | 320 | [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] | [1, 2, 3] | 0 | 0 | 0 | False |

The generation unique key is `{model, axis_id, scenario_id, alpha, seed, prompt_template}`. Full field lists, hashes, missing-value counts, and coverage diagnostics are in `audit_summary.json`.

## Endpoint-inclusion and clipping sanity check

| Encoder | Model | 7-point clipped MAE | 5-point clipped MAE | 5-point raw MAE | Expected interior MAE | |difference| | Pass |
|---|---|---:|---:|---:|---:|---:|---|
| bge_m3 | qwen35_4b | 0.164476 | 0.230266 | 0.230293 | 0.2296 | 0.000666 | True |
| bge_m3 | qwen35_9b | 0.125664 | 0.175930 | 0.176004 | 0.1764 | 0.000470 | True |
| bge_m3 | qwen35_35b_a3b | 0.140022 | 0.196031 | 0.196091 | 0.1960 | 0.000031 | True |
| text2vec | qwen35_4b | 0.150520 | 0.210727 | 0.210789 | 0.2114 | 0.000673 | True |
| text2vec | qwen35_9b | 0.110768 | 0.155075 | 0.155125 | 0.1554 | 0.000325 | True |
| text2vec | qwen35_35b_a3b | 0.127372 | 0.178320 | 0.178340 | 0.1778 | 0.000520 | True |

The endpoint rows have zero projection error, zero off-axis drift, and zero interpolation distance by construction. Including them multiplies a five-point mean by 5/7 and therefore understates the interior error by 28.57% when all cells are present.

## Human-evaluation audit

- File: `/home/data_cpfs/lizihua/sbe/data/human_eval/human_validation_sample_v1.jsonl`
- Rows: 336
- Rows containing any real rating: 0
- Finding: This is a 336-row unannotated selection package, not 160 completed human ratings.
- Consequence: the revision must generate a fresh blinded 320-text annotation package and wait for at least 960 real rater assignments.

## Model/checkpoint history relevant to reuse

- Historical tested checkpoints: Qwen3.5-4B, Qwen3.5-9B, and Qwen3.5-35B-A3B from `/home/data_cpfs/zicheng/models/`.
- Revision additions found on cluster: official Qwen3.6-27B Dense at `/home/data_cpfs/lgx/model/Qwen3.6-27B-base` and Qwen3.6-35B-A3B at `/home/data_cpfs/zicheng/models/Qwen3.6-35B-A3B`; its Accio duplicate has matching config and index hashes.
- The new main grid and server-side seed requirement differ from the legacy run, so legacy generations are audit evidence rather than revision main-experiment rows.

## Implementation differences that change paper values

1. Excluding endpoints raises projection MAE, drift, and interpolation-distance means relative to legacy seven-point summaries.
2. Raw projection MAE is at least as large as clipped MAE whenever extrapolation occurs; the revision uses raw as primary.
3. Normalizing off-axis and interpolation distances by endpoint separation changes cross-scenario weighting and exposes low-separation instability.
4. The revision retains three generation seeds and actually transmits each seed to the server; the legacy rows only carried three nominal labels that were not sent.
5. The new equally spaced sixth grid is not directly interchangeable with the old irregular interior grid.
6. Scenario-level aggregation/bootstrap replaces any text-level independence assumption.

## Audit limitations

- Model-card parameter counts are rounded release values; no full tensor-by-tensor parameter recount was needed for this audit.
- No real human ratings were available to audit for rater demographics, duration, agreement, or reuse compatibility.
- Inference framework performance will be recorded prospectively because historical generation rows contain latency but not GPU memory, framework version, token throughput, or retry telemetry.
