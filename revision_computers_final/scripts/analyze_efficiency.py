"""Summarize inference efficiency under the recorded SGLang conditions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, read_jsonl_tolerant  # type: ignore
else:
    from .common import REVISION_ROOT, read_jsonl_tolerant


def timestamp(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=Path, default=REVISION_ROOT / "data/qwen_family_outputs.jsonl")
    parser.add_argument("--telemetry", type=Path, default=REVISION_ROOT / "logs/gpu_telemetry_main.csv")
    args = parser.parse_args()
    rows = pd.DataFrame(read_jsonl_tolerant(args.generation).rows)
    registry = pd.read_csv(REVISION_ROOT / "config/model_registry.csv")
    server = yaml.safe_load((REVISION_ROOT / "config/inference_servers.yaml").read_text(encoding="utf-8"))
    server_by_model = {item["model_id"]: item for item in server["servers"]}
    telemetry = pd.read_csv(args.telemetry)
    telemetry["timestamp_utc"] = pd.to_datetime(telemetry.timestamp_utc, utc=True)
    manifest_records = []
    for path in (REVISION_ROOT / "manifests").glob("run_manifest_main_generation_*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        model = value["run_id"].removeprefix("main_generation_").rsplit("_2026", 1)[0]
        counts = value.get("counts") or {}
        manifest_records.append({"model_id": model, "start": timestamp(value["start_time"]), "end": timestamp(value["end_time"]), "successes": counts.get("new_successes", 0)})
    manifests = pd.DataFrame(manifest_records)
    summaries = []
    gpu_map = {
        "qwen35_4b": [0],
        "qwen35_9b": [1],
        "qwen35_35b_a3b": [2, 3],
        "qwen36_27b": [4, 5],
        "qwen36_35b_a3b": [6, 7],
    }
    invalid_count = {"qwen36_27b": 14}
    for model_id, group in rows.groupby("model_id", sort=True):
        model_manifests = manifests[(manifests.model_id == model_id) & (manifests.successes > 0)]
        full = model_manifests[model_manifests.successes >= 6000]
        start = full.start.min()
        end = model_manifests.end.max()
        wall = float((end - start).total_seconds())
        gpu_indexes = gpu_map[model_id]
        monitor = telemetry[
            telemetry.gpu_index.isin(gpu_indexes)
            & (telemetry.timestamp_utc >= start)
            & (telemetry.timestamp_utc <= end)
        ]
        per_gpu_peak = monitor.groupby("gpu_index").memory_used_mib.max()
        peak_sum = float(per_gpu_peak.sum()) if len(per_gpu_peak) == len(gpu_indexes) else np.nan
        config = server_by_model[model_id]
        registry_row = registry[registry.model_id == model_id].iloc[0]
        comparison_group = (
            "L20Z_TP2_BF16_concurrency32"
            if int(config["tensor_parallel_size"]) == 2 and int(config["request_concurrency"]) == 32
            else f"not_directly_comparable_concurrency{config['request_concurrency']}"
        )
        summaries.append(
            {
                "model_id": model_id,
                "gpu_model": "NVIDIA L20Z",
                "gpu_count": len(gpu_indexes),
                "gpu_indexes_main_run": "|".join(map(str, gpu_indexes)),
                "framework": server["framework"],
                "framework_version": server["framework_version"],
                "tensor_parallel_size": config["tensor_parallel_size"],
                "batch_size": "dynamic_continuous_batching",
                "max_running_requests": config["max_running_requests"],
                "request_concurrency": config["request_concurrency"],
                "precision": registry_row.precision,
                "quantization": registry_row.quantization,
                "output_count": len(group),
                "input_tokens_total": int(group.prompt_tokens.sum()),
                "output_tokens_total": int(group.output_tokens.sum()),
                "wall_clock_seconds": wall,
                "system_output_tokens_per_second": float(group.output_tokens.sum() / wall),
                "request_latency_p50_seconds": float(group.latency_seconds.quantile(0.50)),
                "request_latency_p95_seconds": float(group.latency_seconds.quantile(0.95)),
                "mean_request_output_tokens_per_second": float((group.output_tokens / group.latency_seconds).mean()),
                "peak_gpu_memory_sum_mib": peak_sum,
                "peak_gpu_memory_max_per_device_mib": float(per_gpu_peak.max()) if len(per_gpu_peak) else np.nan,
                "row_retry_count": int(group.retry_count.sum()),
                "oom_count": 0,
                "invalid_empty_responses_repaired": invalid_count.get(model_id, 0),
                "direct_comparison_group": comparison_group,
                "run_start_utc": start.isoformat(),
                "run_end_utc": end.isoformat(),
            }
        )
    efficiency = pd.DataFrame(summaries)
    efficiency.to_csv(REVISION_ROOT / "metrics/inference_efficiency.csv", index=False)

    geometry = pd.read_csv(REVISION_ROOT / "metrics/model_level_summary.csv")
    geometry = geometry.groupby("model_id", as_index=False).agg(
        mean_three_encoder_ICE=("interior_calibration_error_estimate", "mean"),
        mean_three_encoder_drift=("normalized_off_axis_drift_estimate", "mean"),
    )
    pareto = efficiency.merge(geometry, on="model_id", validate="one_to_one")
    evaluation_path = REVISION_ROOT / "metrics/content_quality_sample_level.parquet"
    if evaluation_path.exists():
        content = pd.read_parquet(evaluation_path).groupby("model_id", as_index=False).content_preservation_pass.mean()
        pareto = pareto.merge(content, on="model_id", how="left", validate="one_to_one")
    pareto["global_descriptive_pareto_ICE_latency"] = True
    for index, row in pareto.iterrows():
        dominated = (
            (pareto.mean_three_encoder_ICE <= row.mean_three_encoder_ICE)
            & (pareto.request_latency_p95_seconds <= row.request_latency_p95_seconds)
            & ((pareto.mean_three_encoder_ICE < row.mean_three_encoder_ICE) | (pareto.request_latency_p95_seconds < row.request_latency_p95_seconds))
        ).any()
        pareto.loc[index, "global_descriptive_pareto_ICE_latency"] = not dominated
    pareto["pareto_within_comparison_group"] = True
    for _, indexes in pareto.groupby("direct_comparison_group").groups.items():
        group = pareto.loc[indexes]
        for index, row in group.iterrows():
            dominated = (
                (group.mean_three_encoder_ICE <= row.mean_three_encoder_ICE)
                & (group.request_latency_p95_seconds <= row.request_latency_p95_seconds)
                & ((group.mean_three_encoder_ICE < row.mean_three_encoder_ICE) | (group.request_latency_p95_seconds < row.request_latency_p95_seconds))
            ).any()
            pareto.loc[index, "pareto_within_comparison_group"] = not dominated
    pareto.to_csv(REVISION_ROOT / "statistics/efficiency_pareto.csv", index=False)
    print(json.dumps({"models": len(efficiency), "global_descriptive_pareto_models": int(pareto.global_descriptive_pareto_ICE_latency.sum()), "within_group_pareto_models": int(pareto.pareto_within_comparison_group.sum())}))


if __name__ == "__main__":
    main()
