# StyleBlend-Bench

StyleBlend-Bench evaluates whether black-box LLMs can follow explicit two-endpoint style-ratio prompts, such as `30% style A + 70% style B`, while preserving the core content.

This repository is organized to run in three scales:

- `smoke`: smallest end-to-end check.
- `pilot`: enough data to inspect calibration curves.
- `main`: full experiment design from the handoff docs.

## Quick Start

From `/home/data_cpfs/lizihua/sbe`:

```bash
python scripts/00_make_scenarios.py --config configs/style_axes.yaml --out data/scenarios/smoke_scenarios.jsonl --per-axis 2 --axes emotion_apology
python scripts/01_generate.py --scenario-file data/scenarios/smoke_scenarios.jsonl --model qwen35_4b --split smoke --dry-run
python scripts/05_analyze.py --generation-file data/generations/qwen35_4b/smoke.jsonl --out-dir results
```

Before real model inference, edit:

- `configs/models.yaml`
- `configs/generation.yaml`

## Real Generation

Start a vLLM OpenAI-compatible server, then run:

```bash
python scripts/01_generate.py \
  --scenario-file data/scenarios/smoke_scenarios.jsonl \
  --model qwen35_4b \
  --split smoke \
  --base-url http://127.0.0.1:8001/v1 \
  --api-key EMPTY
```

The generation script supports resume by `run_id` and writes failures to `results/logs/generation_failed.jsonl`.

## Outputs

- Generations: `data/generations/<model>/<split>.jsonl`
- Scalar judgments: `data/judgments/scalar_<judge>_<split>.jsonl`
- Pairwise judgments: `data/judgments/pairwise_<judge>_<split>.jsonl`
- Tables: `results/tables/`
- Figures: `results/figures/`
- Logs: `results/logs/`

