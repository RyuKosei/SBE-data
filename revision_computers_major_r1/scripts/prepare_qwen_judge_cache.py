from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from common import atomic_write_json, atomic_write_jsonl, read_jsonl, utc_now
from qwen_judge import EVALUATOR_ID, cache_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    legacy = Path(config["legacy_root"])
    index = read_jsonl(args.output_root / "data/evaluation_index.jsonl")
    old_rows = read_jsonl(legacy / "metrics/llm_evaluations.jsonl") + read_jsonl(legacy / "metrics/robustness_llm_evaluations.jsonl")
    old = {row["run_id"]: row for row in old_rows}
    output = args.output_root / "metrics/qwen_judge_evaluations.jsonl"
    existing = read_jsonl(output) if output.exists() else []
    merged = {
        row["evaluation_item_id"]: row
        for row in existing
        if row.get("cache_key_version") == "item_identity_v2"
    }
    reused = 0
    for item in index:
        source_run_id = item["original_source_run_id"]
        if source_run_id not in old:
            continue
        prior = old[source_run_id]
        identity = cache_identity(item)
        if prior["evaluator_id"] != EVALUATOR_ID:
            raise ValueError(f"evaluator mismatch for {source_run_id}")
        if prior["prompt_sha256"] != identity["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch for {source_run_id}")
        record = {
            **prior,
            "evaluation_item_id": item["evaluation_item_id"],
            "source_run_id": source_run_id,
            **identity,
            "cache_reuse_status": "legacy_validated",
            "cache_reuse_provenance": "legacy evaluator ID and prompt hash matched; output text/schema hashes reconstructed from locked source data",
        }
        if item["evaluation_item_id"] not in merged:
            merged[item["evaluation_item_id"]] = record
            reused += 1
    rows = sorted(merged.values(), key=lambda row: row["evaluation_cache_key"])
    atomic_write_jsonl(output, rows)
    status = {"generated_at": utc_now(), "index_rows": len(index), "legacy_evaluation_rows": len(old_rows), "newly_reused": reused, "cache_rows": len(rows), "expected_reusable_when_complete": 40800}
    atomic_write_json(args.output_root / "manifests/qwen_judge_cache_status.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
