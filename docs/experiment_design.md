# Experiment Design

## Question

Can black-box LLMs map explicit style-ratio prompts into outputs whose measured style ratio tracks the target ratio?

## Task

For each core semantic scenario, define two style endpoints A and B. The model receives the scenario, endpoint descriptions, and a target ratio. It must generate a Chinese short text that preserves the scenario content and blends the two styles.

## Core Hypotheses

- H1: Larger or stronger instruction-tuned models will have lower calibration error, but improvement may vary by axis.
- H2: Some axes will show endpoint attraction: middle ratios collapse toward a salient endpoint.
- H3: Embedding projection, scalar judge, and pairwise judge will agree on broad trends but diverge on axes with strong lexical or length confounds.
- H4: Ratio prompt wording and A/B order will measurably affect outputs, revealing prompt fragility.

## Main Metrics

- Calibration MAE between target ratio and measured ratio.
- Spearman correlation across target ratios.
- Smoothness penalty using second differences across ratio curves.
- Monotonic violation rate.
- Endpoint attraction for middle ratios.
- Content preservation and naturalness.
- Prompt invariance error on ablation subsets.

## Experiment Stages

### Stage 0: Environment and Model Paths

Confirm local paths for Qwen3.5-4B, Qwen3.5-9B, and Qwen3.5-35B-A3B. Update `configs/models.yaml`.

### Stage 1: Scenario Construction

Use `scripts/00_make_scenarios.py` to create deterministic seed scenarios. Later, replace or extend them with LLM-assisted candidates and human spot checks.

### Stage 2: Smoke Test

Run one axis, two scenarios, three ratios, one seed, one model. Verify:

- No visible reasoning.
- Output is plain正文.
- JSONL fields are complete.
- Resume logic skips existing `run_id`.
- Scalar judge JSON parsing works.

### Stage 3: Pilot

Run three axes and ten scenarios per axis. Produce first calibration curves and axis-level diagnostics.

### Stage 4: Main

Run all 8 axes, 40 scenarios per axis, 7 ratios, 3 seeds, 3 models.

### Stage 5: Evaluation

Compute embedding projection, scalar judge, pairwise subset, and aggregate analysis.

### Stage 6: Ablations

Run only on a smaller subset:

- Ratio expression variants.
- A/B order swap.
- Few-shot endpoint anchors.
- Length control.
- Temperature.
- Judge robustness.

## Non-Goals For The First Pass

- No Qwen fine-tuning.
- No reward model training.
- No full pairwise comparison.
- No quantized models in the main comparison unless quantization is controlled.

