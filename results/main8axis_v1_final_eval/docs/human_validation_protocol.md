# Human Validation Protocol v1

## Purpose

Validate whether embedding projection scores align with human perception of explicit Style B ratio, while separately checking content preservation and naturalness.

## Sample

- File: `data/human_eval/human_validation_sample_v1.jsonl`
- Size: 336 outputs.
- Coverage: 3 models, 8 axes, 7 alpha ratios.
- Selection: for each `{model, axis, alpha}`, one low-error and one high-error sample by mean bge-m3/text2vec projection error.

## Annotation Fields

Annotate each row independently:

- `style_b_ratio_0_to_100`: perceived percentage of Style B in the output. Use 0 for pure Style A and 100 for pure Style B.
- `content_preservation`: `yes`, `partial`, or `no`.
- `naturalness_1_to_5`: 1 means unnatural or broken; 5 means fluent and natural.
- `notes`: short optional explanation for ambiguous cases.

Optional pairwise follow-up:

- If two outputs are shown for the same scenario, choose which one is closer to the requested Style B ratio.

## Annotator Instructions

Read the core content, Style A, Style B, target alpha, and model output. Judge style ratio based on the output text only. Do not reward length by itself unless length is part of the style definition. Penalize content drift in `content_preservation`, not in the style ratio field.

## Analysis Plan

1. Correlate human `style_b_ratio_0_to_100 / 100` with bge-m3 and text2vec `r_proj_clipped`.
2. Compare high-error and low-error subsets to estimate whether projection errors are perceptually meaningful.
3. Recompute model ranking on human-ratio absolute error for this sample.
4. Report content preservation and naturalness separately from style calibration.
