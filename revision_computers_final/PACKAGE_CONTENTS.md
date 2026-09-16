# Repository Delivery Package

This directory is the repository-ready delivery of the 2026-09-15
StyleBlend-Bench MDPI *Computers* revision experiment.

## Included

- consolidated main-generation and robustness JSONL outputs;
- embeddings from BGE-M3, text2vec-base-chinese, and
  multilingual-E5-large-instruct for all five tested Qwen checkpoints;
- final sample-, scenario-, model-, evaluator-, trajectory-, baseline-,
  ablation-, robustness-, content-, and failure-analysis metrics;
- analysis tables, statistical outputs, logs, and run manifests;
- the blinded human-evaluation package and blank real-rating template;
- ten 300-dpi PNG figures and the corresponding ten vector PDFs;
- scripts, frozen configs, tests, audit material, manuscript insertions, and
  the minimal reproducibility package.

The repository delivery payload totals about 547 MiB across 179 files. Of
those, 171 result artifacts are newly added by the packaging commit. No
individual delivery file reaches GitHub's 100 MiB hard limit.

## Deliberately excluded

- `models/`: local model and encoder checkpoints (about 3.2 GiB), which are
  not experiment results and must not be redistributed in this repository;
- `data/fragments/`: duplicate per-model generation fragments whose contents
  are already consolidated in `data/qwen_family_outputs.jsonl` and
  `data/robustness_outputs.jsonl`;
- Python bytecode and cache directories.

## Validation status

- main generation: 33,600 / 33,600 valid outputs;
- robustness generation: 9,600 / 9,600 valid outputs;
- fixed automatic evaluation: 43,200 / 43,200 completed records;
- final analysis manifest: `success` for all 16 commands;
- main test suite: 11 passed;
- minimal reproducibility package: 11 passed;
- figures: 10 readable PNG files and 10 valid single-page PDF files.

Two activities remain explicitly `external_pending`: independent rewriting
and two-author approval of 160 shared-endpoint texts, and collection of 960
genuine human ratings. Their prepared templates and analyzers are included;
no missing human result was fabricated.
