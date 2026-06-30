# Live Smoke Commands

Run these on the 8-GPU machine. The current image already has SGLang, so use SGLang first. vLLM is optional.

## 1. Start Qwen3.5-4B SGLang Server

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

If `sglang serve` is not on `PATH`, use:

```bash
python -m sglang.launch_server \
  --model-path /home/data_cpfs/zicheng/models/Qwen3.5-4B \
  --served-model-name qwen35_4b \
  --host 0.0.0.0 \
  --port 8001 \
  --context-length 8192 \
  --dtype bfloat16 \
  --trust-remote-code
```

## Optional: Start With vLLM Instead

```bash
vllm serve /home/data_cpfs/zicheng/models/Qwen3.5-4B \
  --served-model-name qwen35_4b \
  --port 8001 \
  --max-model-len 8192
```

If vLLM requires explicit trust for this model family, add:

```bash
--trust-remote-code
```

## 2. Run Live Smoke Generation

From `/home/data_cpfs/lizihua/sbe`:

```bash
python scripts/01_generate.py \
  --scenario-file data/scenarios/smoke_scenarios.jsonl \
  --model qwen35_4b \
  --model-name qwen35_4b \
  --split smoke_live \
  --base-url http://127.0.0.1:8001/v1 \
  --api-key EMPTY \
  --ratios 0.0 0.5 1.0 \
  --seeds 1
```

## 3. Quick Output Check

```bash
sed -n '1,3p' data/generations/qwen35_4b/smoke_live.jsonl
```

Check:

- No `<think>` blocks.
- Output is plain Chinese正文.
- The core meaning is preserved.
- Length is roughly in range.

## 4. Judge Smoke

If `qwen35_35b_a3b` judge server is not ready, use `qwen35_9b` temporarily for smoke only.

Start 9B judge with SGLang:

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

```bash
python scripts/03_judge_scalar.py \
  --generation-file data/generations/qwen35_4b/smoke_live.jsonl \
  --judge-model qwen35_9b \
  --model-name qwen35_9b \
  --split smoke_live \
  --base-url http://127.0.0.1:8002/v1 \
  --api-key EMPTY
```

## 5. Analyze Smoke

```bash
python scripts/05_analyze.py \
  --generation-file data/generations/qwen35_4b/smoke_live.jsonl \
  --scalar-judge-file data/judgments/scalar_qwen35_9b_smoke_live.jsonl \
  --out-dir results
```
