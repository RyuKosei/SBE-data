# Reproduction runbook

Run from `/home/data_cpfs/lizihua/sbe`. API credentials are read from
`SBE_IAI_API_KEY`; no credential belongs in a command line or file.

## Audit and indices

```bash
python revision_computers_major_r1/scripts/audit_inputs.py --config revision_computers_major_r1/config/experiment.yaml --output-root revision_computers_major_r1
python revision_computers_major_r1/scripts/build_evaluation_index.py --config revision_computers_major_r1/config/experiment.yaml --output-root revision_computers_major_r1
python revision_computers_major_r1/scripts/build_stability_retest_index.py --output-root revision_computers_major_r1
```

## External generation and embeddings

```bash
python revision_computers_major_r1/scripts/run_gpt_external_main.py --config revision_computers_major_r1/config/experiment.yaml --pricing revision_computers_major_r1/config/gpt_pricing.yaml --output-root revision_computers_major_r1 --concurrency 16 --budget-usd 1
python revision_computers_major_r1/scripts/embed_dataset.py --encoder-registry revision_computers_major_r1/config/encoder_registry.csv --encoder-id bge_m3 --input revision_computers_major_r1/data/external_main_outputs.jsonl --output-root revision_computers_major_r1 --dataset-id external_main --device cuda:0
python revision_computers_major_r1/scripts/embed_dataset.py --encoder-registry revision_computers_major_r1/config/encoder_registry.csv --encoder-id text2vec_base_chinese --input revision_computers_major_r1/data/external_main_outputs.jsonl --output-root revision_computers_major_r1 --dataset-id external_main --device cuda:1
python revision_computers_major_r1/scripts/embed_dataset.py --encoder-registry revision_computers_major_r1/config/encoder_registry.csv --encoder-id multilingual_e5_large_instruct --input revision_computers_major_r1/data/external_main_outputs.jsonl --output-root revision_computers_major_r1 --dataset-id external_main --device cuda:2
```

## Automatic judges

The local Qwen judge server used the frozen local checkpoint, two L20Z GPUs,
tensor parallelism 2, BF16, and service concurrency 32. Its client command was:

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m sglang.launch_server --model-path /home/data_cpfs/zicheng/models/Qwen3.6-35B-A3B --served-model-name qwen36_35b_a3b --host 127.0.0.1 --port 18105 --tensor-parallel-size 2 --context-length 4096 --mem-fraction-static 0.88 --max-running-requests 32 --cuda-graph-max-bs 32 --dtype bfloat16
python revision_computers_major_r1/scripts/run_qwen_judge.py --output-root revision_computers_major_r1 --base-url http://127.0.0.1:18105/v1 --served-model-name qwen36_35b_a3b --concurrency 32
```

The GPT judge was resume-safe and ended at concurrency 64 after the recorded
concurrency adjustments:

```bash
python revision_computers_major_r1/scripts/run_gpt_judge.py --config revision_computers_major_r1/config/experiment.yaml --pricing revision_computers_major_r1/config/gpt_pricing.yaml --output-root revision_computers_major_r1 --model gpt-6-sol --budget-usd 239 --concurrency 64
```

The same runners used `data/stability_retest_index.jsonl` with separate output,
failure-log, and progress paths for the 240-item repeat evaluation.

## Geometry, analysis, and final audit

```bash
python revision_computers_major_r1/scripts/compute_anchor_metrics.py --config revision_computers_major_r1/config/experiment.yaml --output-root revision_computers_major_r1 --encoder-id bge_m3
python revision_computers_major_r1/scripts/compute_anchor_metrics.py --config revision_computers_major_r1/config/experiment.yaml --output-root revision_computers_major_r1 --encoder-id text2vec_base_chinese
python revision_computers_major_r1/scripts/compute_anchor_metrics.py --config revision_computers_major_r1/config/experiment.yaml --output-root revision_computers_major_r1 --encoder-id multilingual_e5_large_instruct
python revision_computers_major_r1/scripts/analyze_geometry.py --output-root revision_computers_major_r1
python revision_computers_major_r1/scripts/analyze_judges.py --output-root revision_computers_major_r1
python revision_computers_major_r1/scripts/analyze_robustness_full.py --output-root revision_computers_major_r1
python revision_computers_major_r1/scripts/analyze_stability_retest.py --output-root revision_computers_major_r1
python revision_computers_major_r1/scripts/select_failure_cases.py --output-root revision_computers_major_r1 --scenario-file data/scenarios/styleblend_8axis_scenarios_v1.jsonl
python revision_computers_major_r1/scripts/audit_final_coverage.py --output-root revision_computers_major_r1
pytest -q revision_computers_major_r1/tests
```

The full 96,000-record robustness matrix, per-model resume state, checkpoint
paths, sampling configuration, and served model names are preserved in
`data/`, `config/`, and `manifests/`. The client pattern was
`run_qwen_robustness.py --config ... --output-root ... --model-id MODEL
--base-url LOCAL_SGLANG_URL --served-model-name MODEL --concurrency 32`.
The transient localhost port assigned to each already-stopped generation
service was not persisted in the progress manifests; this is a run-command
metadata limitation and is not reconstructed after the fact. The consolidated
valid-output matrix is `data/robustness_outputs.jsonl`, while all 96,000
planned identities and six terminal empty outputs are in
`data/robustness_condition_records.jsonl`.
