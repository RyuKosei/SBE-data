# Cluster Runbook

## 1. Confirm Models

```bash
ls /path/to/models
```

Update `configs/models.yaml` with real paths and tensor parallel sizes.

## 2. Start SGLang

The slime-related image already includes SGLang. Prefer it for the first smoke/pilot pass.

4B:

```bash
sglang serve \
  --model-path /home/data_cpfs/zicheng/models/Qwen3.5-4B \
  --served-model-name qwen35_4b \
  --host 0.0.0.0 \
  --port 8001 \
  --context-length 8192 \
  --dtype bfloat16 \
  --trust-remote-code
```

9B:

```bash
sglang serve \
  --model-path /home/data_cpfs/zicheng/models/Qwen3.5-9B \
  --served-model-name qwen35_9b \
  --host 0.0.0.0 \
  --port 8002 \
  --context-length 8192 \
  --dtype bfloat16 \
  --trust-remote-code
```

35B-A3B:

```bash
sglang serve \
  --model-path /home/data_cpfs/zicheng/models/Qwen3.5-35B-A3B \
  --served-model-name qwen35_35b_a3b \
  --host 0.0.0.0 \
  --port 8003 \
  --context-length 8192 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --trust-remote-code
```

## Optional: Start vLLM

Example:

```bash
vllm serve /path/to/Qwen3.5-4B \
  --served-model-name qwen35_4b \
  --port 8001 \
  --max-model-len 8192
```

For 35B-A3B:

```bash
vllm serve /path/to/Qwen3.5-35B-A3B \
  --served-model-name qwen35_35b_a3b \
  --port 8003 \
  --tensor-parallel-size 2 \
  --max-model-len 8192
```

## 3. Build Scenarios

```bash
python scripts/00_make_scenarios.py --out data/scenarios/smoke_scenarios.jsonl --per-axis 2 --axes emotion_apology
python scripts/00_make_scenarios.py --out data/scenarios/pilot_scenarios.jsonl --per-axis 10 --axes emotion_apology formality_request empathy_clinical
python scripts/00_make_scenarios.py --out data/scenarios/main_scenarios.jsonl --per-axis 40
```

## 4. Generate

```bash
python scripts/01_generate.py --scenario-file data/scenarios/smoke_scenarios.jsonl --model qwen35_4b --split smoke
```

Use `--dry-run` first to validate prompts without calling a model.

## 5. Judge

```bash
python scripts/03_judge_scalar.py \
  --generation-file data/generations/qwen35_4b/smoke.jsonl \
  --judge-model qwen35_35b_a3b \
  --split smoke
```

## 6. Analyze

```bash
python scripts/05_analyze.py \
  --generation-file data/generations/qwen35_4b/smoke.jsonl \
  --scalar-judge-file data/judgments/scalar_qwen35_35b_a3b_smoke.jsonl \
  --out-dir results
```
