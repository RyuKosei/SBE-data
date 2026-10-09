from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_write_json, read_jsonl, utc_now


def summarize(path: Path, service: str, stage: str, latency_field: str = "latency_seconds") -> dict:
    rows = read_jsonl(path) if path.exists() else []
    latency = np.asarray([float(row[latency_field]) for row in rows if row.get(latency_field) is not None], dtype=float)
    successful = sum(row.get("status") in {"ok", "truncated"} if "status" in row else True for row in rows)
    return {
        "service": service,
        "stage": stage,
        "recorded_records": len(rows),
        "successful_records": successful,
        "non_success_records": len(rows) - successful,
        "requested_models": "|".join(sorted({str(row.get("requested_model")) for row in rows if row.get("requested_model")})),
        "response_models": "|".join(sorted({str(row.get("response_model")) for row in rows if row.get("response_model")})),
        "latency_p50_seconds": float(np.quantile(latency, 0.5)) if len(latency) else np.nan,
        "latency_p95_seconds": float(np.quantile(latency, 0.95)) if len(latency) else np.nan,
        "input_or_prompt_tokens": sum(int(row.get("input_tokens") or row.get("prompt_tokens") or 0) for row in rows),
        "output_or_completion_tokens": sum(int(row.get("output_tokens") or row.get("completion_tokens") or 0) for row in rows),
        "deployment_hardware_known": False,
        "comparable_to_local_gpu_benchmark": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        summarize(args.output_root / "data/gpt_fixed_references.jsonl", "Alibaba internal GPT relay", "fixed-reference generation"),
        summarize(args.output_root / "metrics/gpt_judge_evaluations.jsonl", "Alibaba internal GPT relay", "independent judge"),
        summarize(args.output_root / "metrics/gpt_stability_retest.jsonl", "Alibaba internal GPT relay", "judge stability retest"),
        summarize(args.output_root / "data/external_main_outputs.jsonl", "Alibaba internal GPT relay", "cross-family generation"),
    ]
    table = pd.DataFrame(rows)
    table.to_csv(args.output_root / "tables/table_api_service_observations.csv", index=False)
    (args.output_root / "tables/table_api_service_observations.md").write_text(
        table.to_markdown(index=False) + "\n\nAPI latency is a service observation. Provider hardware, batching, and queue state are not controlled, so these rows are excluded from the matched local-GPU efficiency figure.\n",
        encoding="utf-8",
    )
    status = {"generated_at": utc_now(), "rows": len(table), "mixed_with_local_efficiency_figure": False}
    atomic_write_json(args.output_root / "manifests/api_service_observations.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
