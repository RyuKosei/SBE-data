# StyleBlend-Bench Computers Revision

This directory contains the non-destructive final revision of
StyleBlend-Bench for the planned MDPI *Computers* submission.

The revision is deliberately separate from the 2026-06 experiment outputs.
Every generated artifact must be traceable to a configuration, source-data
hash, model checkpoint, generation seed, command, environment, and Git commit.

## Current protocol decisions

- Scope: official Qwen3.5 and Qwen3.6 text-capable instruction checkpoints
  available on the cluster; no cross-family model comparison.
- Main ratios: `0, 1/6, 2/6, 3/6, 4/6, 5/6, 1`.
- Replication: the main generation, prompt/temperature robustness experiment,
  and leave-one-seed-out empirical trajectory retain three generation seeds.
  Auxiliary analyses reuse those outputs and add no separate seed sweep.
- Primary metrics use the five interior ratios only. Endpoints define the
  coordinate system and are excluded from primary means.
- Primary projection coordinates are never clipped. Clipped projection is an
  explicitly labeled ablation.
- Statistical resampling uses scenarios as the independent unit.
- Human ratings are never simulated. Until real ratings arrive, human/model
  agreement remains `external_pending`.

## Entry points

- `scripts/audit_data.py`: audit the legacy data and reproduce the endpoint
  inclusion sanity check.
- `scripts/geometry_metrics.py`: endpoint-unbiased geometry and trajectory
  metrics.
- `scripts/run_main_generation.py`: resumable OpenAI-compatible generation.
- `scripts/analyze_human_eval.py`: validate and analyze real annotation files.
- `config/`: immutable experiment and threshold definitions.

The top-level `completion_report.md` is the status authority once generated.
