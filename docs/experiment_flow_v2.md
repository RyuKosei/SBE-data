# Experiment Flow v2

This file records the current executable StyleBlend-Bench flow. It is meant to keep the pipeline clear before launching large batch experiments.

## Research Flow

1. Build scenarios.
2. Generate style-ratio outputs for each target model.
3. Compute embedding projection metrics. This is the primary measurement.
4. Run scalar LLM judge. This is an auxiliary measurement.
5. Run pairwise local comparisons on a sampled/local set. This is an auxiliary monotonicity check.
6. Analyze and merge generation, projection, scalar judge, and pairwise results into tables and figures.

## Scripts

- `scripts/00_make_scenarios.py`: creates scenario JSONL files for smoke, pilot, and main splits.
- `scripts/01_generate.py`: calls an OpenAI-compatible model server and writes generation JSONL with resume-by-`run_id`.
- `scripts/02_embed_project.py`: computes projection ratio, interpolation distance, off-axis drift, and nearest-neighbor rank. It supports `sentence-transformers`, local `transformers`, and hash smoke-test backends.
- `scripts/03_judge_scalar.py`: runs JSON scalar style/content/naturalness judging.
- `scripts/04_judge_pairwise.py`: builds local ratio comparisons and runs JSON pairwise judging.
- `scripts/05_analyze.py`: writes summary tables and figures. It accepts multiple files per input group.

## Current Pilot Files

Generations:

```bash
data/generations/qwen35_4b/pilot_4b_v1.jsonl
data/generations/qwen35_9b/pilot_9b_v1.jsonl
data/generations/qwen35_35b_a3b/pilot_35b_v1.jsonl
```

Scalar judgments:

```bash
data/judgments/scalar_qwen35_35b_a3b_pilot_4b_v1.jsonl
data/judgments/scalar_qwen35_35b_a3b_pilot_9b_v1.jsonl
data/judgments/scalar_qwen35_9b_pilot_35b_v1.jsonl
```

Offline projection smoke outputs:

```bash
data/metrics/projection_hash_pilot_4b_v1.jsonl
data/metrics/projection_hash_pilot_9b_v1.jsonl
data/metrics/projection_hash_pilot_35b_v1.jsonl
```

Formal bge-m3 projection outputs:

```bash
data/metrics/projection_bge_m3_pilot_4b_v1.jsonl
data/metrics/projection_bge_m3_pilot_9b_v1.jsonl
data/metrics/projection_bge_m3_pilot_35b_v1.jsonl
```

The local bge-m3 model path is:

```bash
/home/data_cpfs/zicheng/models/bge-m3
```

The hash projection backend is only for pipeline validation. Formal metrics should use bge-m3 or another real embedding model.

## Verified Pipeline Commands

Projection smoke:

```bash
python scripts/02_embed_project.py \
  --generation-file data/generations/qwen35_4b/pilot_4b_v1.jsonl \
  --backend hash \
  --out data/metrics/projection_hash_pilot_4b_v1.jsonl

python scripts/02_embed_project.py \
  --generation-file data/generations/qwen35_9b/pilot_9b_v1.jsonl \
  --backend hash \
  --out data/metrics/projection_hash_pilot_9b_v1.jsonl

python scripts/02_embed_project.py \
  --generation-file data/generations/qwen35_35b_a3b/pilot_35b_v1.jsonl \
  --backend hash \
  --out data/metrics/projection_hash_pilot_35b_v1.jsonl
```

Pairwise dry-run schema check:

```bash
python scripts/04_judge_pairwise.py \
  --generation-file data/generations/qwen35_4b/pilot_4b_v1.jsonl \
  --judge-model qwen35_35b_a3b \
  --dry-run \
  --out data/judgments/pairwise_qwen35_35b_a3b_pilot_4b_v1_dry.jsonl
```

Combined analysis:

```bash
python scripts/05_analyze.py \
  --generation-file \
    data/generations/qwen35_4b/pilot_4b_v1.jsonl \
    data/generations/qwen35_9b/pilot_9b_v1.jsonl \
    data/generations/qwen35_35b_a3b/pilot_35b_v1.jsonl \
  --scalar-judge-file \
    data/judgments/scalar_qwen35_35b_a3b_pilot_4b_v1.jsonl \
    data/judgments/scalar_qwen35_35b_a3b_pilot_9b_v1.jsonl \
    data/judgments/scalar_qwen35_9b_pilot_35b_v1.jsonl \
  --projection-file \
    data/metrics/projection_hash_pilot_4b_v1.jsonl \
    data/metrics/projection_hash_pilot_9b_v1.jsonl \
    data/metrics/projection_hash_pilot_35b_v1.jsonl \
  --out-dir results/pilot_hash_flow
```

Formal bge-m3 projection:

```bash
python scripts/02_embed_project.py \
  --generation-file data/generations/qwen35_4b/pilot_4b_v1.jsonl \
  --backend transformers \
  --embedding-model /home/data_cpfs/zicheng/models/bge-m3 \
  --batch-size 16 \
  --out data/metrics/projection_bge_m3_pilot_4b_v1.jsonl
```

Formal bge-m3 combined analysis:

```bash
python scripts/05_analyze.py \
  --generation-file \
    data/generations/qwen35_4b/pilot_4b_v1.jsonl \
    data/generations/qwen35_9b/pilot_9b_v1.jsonl \
    data/generations/qwen35_35b_a3b/pilot_35b_v1.jsonl \
  --scalar-judge-file \
    data/judgments/scalar_qwen35_35b_a3b_pilot_4b_v1.jsonl \
    data/judgments/scalar_qwen35_35b_a3b_pilot_9b_v1.jsonl \
    data/judgments/scalar_qwen35_9b_pilot_35b_v1.jsonl \
  --projection-file \
    data/metrics/projection_bge_m3_pilot_4b_v1.jsonl \
    data/metrics/projection_bge_m3_pilot_9b_v1.jsonl \
    data/metrics/projection_bge_m3_pilot_35b_v1.jsonl \
  --out-dir results/pilot_bge_m3_flow
```

Combined analysis with pairwise dry-run:

```bash
python scripts/05_analyze.py \
  --generation-file \
    data/generations/qwen35_4b/pilot_4b_v1.jsonl \
    data/generations/qwen35_9b/pilot_9b_v1.jsonl \
    data/generations/qwen35_35b_a3b/pilot_35b_v1.jsonl \
  --scalar-judge-file \
    data/judgments/scalar_qwen35_35b_a3b_pilot_4b_v1.jsonl \
    data/judgments/scalar_qwen35_35b_a3b_pilot_9b_v1.jsonl \
    data/judgments/scalar_qwen35_9b_pilot_35b_v1.jsonl \
  --projection-file \
    data/metrics/projection_hash_pilot_4b_v1.jsonl \
    data/metrics/projection_hash_pilot_9b_v1.jsonl \
    data/metrics/projection_hash_pilot_35b_v1.jsonl \
  --pairwise-judge-file \
    data/judgments/pairwise_qwen35_35b_a3b_pilot_4b_v1_dry.jsonl \
    data/judgments/pairwise_qwen35_35b_a3b_pilot_9b_v1_dry.jsonl \
    data/judgments/pairwise_qwen35_9b_pilot_35b_v1_dry.jsonl \
  --out-dir results/pilot_hash_flow_with_pairwise_dry
```

## Output Tables

- `generation_summary.csv`: row counts, scenario counts, seed counts, output length, thinking marker rate.
- `projection_metrics.csv`: model-level embedding projection metrics.
- `projection_axis_metrics.csv`: axis-level embedding projection metrics.
- `main_metrics.csv`: model-level scalar judge metrics.
- `axis_metrics.csv`: axis-level scalar judge metrics.
- `pairwise_metrics.csv`: pairwise monotonic violation and tie rates.
- `main_metrics_combined.csv`: merged model-level table.

## Remaining Before Batch Experiments

1. Run real pairwise judge on the pilot subset if the projection/scalar signals look plausible.
2. Improve or curate scenario quality before main generation; current scenarios are deterministic seed variants.
3. Decide final judge policy for 35B-A3B outputs, since 35B should not judge itself in the main result table.
4. Only after the pilot analysis is satisfactory, launch full main generation.
