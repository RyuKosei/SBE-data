from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from common import atomic_write_json, sha256_file, utc_now


def rows(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def identity_summary(path: Path, field: str) -> dict[str, Any]:
    seen: set[str] = set()
    duplicates = 0
    total = 0
    missing_identity = 0
    for row in rows(path):
        total += 1
        identity = row.get(field)
        if not identity:
            missing_identity += 1
            continue
        if identity in seen:
            duplicates += 1
        seen.add(identity)
    return {
        "path": str(path),
        "rows": total,
        "unique_identities": len(seen),
        "duplicate_identities": duplicates,
        "missing_identity": missing_identity,
        "identities": seen,
    }


def public(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "identities"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    evaluation_status_path = root / "manifests/evaluation_index_status.json"
    evaluation_status = json.loads(evaluation_status_path.read_text(encoding="utf-8")) if evaluation_status_path.exists() else {}

    index = identity_summary(root / "data/evaluation_index.jsonl", "evaluation_item_id")
    qwen = identity_summary(root / "metrics/qwen_judge_evaluations.jsonl", "evaluation_item_id")
    gpt = identity_summary(root / "metrics/gpt_judge_evaluations.jsonl", "evaluation_item_id")
    qwen_fail = identity_summary(root / "logs/qwen_judge_failures.jsonl", "evaluation_item_id")
    gpt_fail = identity_summary(root / "logs/gpt_judge_failures.jsonl", "evaluation_item_id")

    source_counts: Counter[str] = Counter()
    for row in rows(root / "data/evaluation_index.jsonl"):
        source_counts[str(row.get("source_type"))] += 1

    external_status: Counter[str] = Counter()
    external_response_models: Counter[str] = Counter()
    external_requested_models: Counter[str] = Counter()
    external_timestamps: list[str] = []
    external_attempted = 0
    external_usable = 0
    for row in rows(root / "data/external_main_outputs.jsonl"):
        external_attempted += 1
        status = str(row.get("status"))
        external_status[status] += 1
        external_response_models[str(row.get("response_model"))] += 1
        external_requested_models[str(row.get("requested_model"))] += 1
        if row.get("timestamp"):
            external_timestamps.append(str(row["timestamp"]))
        external_usable += int(status in {"ok", "truncated"} and bool(row.get("generated_text")))

    gpt_response_id_count = 0
    gpt_response_models: Counter[str] = Counter()
    gpt_retry_counts: Counter[str] = Counter()
    for row in rows(root / "metrics/gpt_judge_evaluations.jsonl"):
        gpt_response_id_count += int(bool(row.get("response_id_sha256")))
        gpt_response_models[str(row.get("response_model"))] += 1
        gpt_retry_counts[str(row.get("retry_count"))] += 1

    qwen_covered = qwen["identities"] | qwen_fail["identities"]
    gpt_covered = gpt["identities"] | gpt_fail["identities"]
    index_ids = index["identities"]
    critical = [
        root / "data/evaluation_index.jsonl",
        root / "metrics/qwen_judge_evaluations.jsonl",
        root / "metrics/gpt_judge_evaluations.jsonl",
        root / "data/gpt_fixed_references.jsonl",
        root / "data/external_main_outputs.jsonl",
        root / "data/deepseek_main_outputs.jsonl",
    ]
    audit = {
        "generated_at": utc_now(),
        "evaluation_index": public(index),
        "evaluation_index_source_counts": dict(sorted(source_counts.items())),
        "qwen_judge": {
            **public(qwen),
            "terminal_failures": public(qwen_fail),
            "covered_index_items": len(qwen_covered & index_ids),
            "missing_index_items": len(index_ids - qwen_covered),
            "orphan_items": len(qwen_covered - index_ids),
        },
        "gpt_judge": {
            **public(gpt),
            "terminal_failures": public(gpt_fail),
            "covered_index_items": len(gpt_covered & index_ids),
            "missing_index_items": len(index_ids - gpt_covered),
            "orphan_items": len(gpt_covered - index_ids),
            "hashed_response_ids_present": gpt_response_id_count,
            "hashed_response_ids_missing": gpt["rows"] - gpt_response_id_count,
            "response_models": dict(sorted(gpt_response_models.items())),
            "recorded_outer_retry_counts": dict(sorted(gpt_retry_counts.items())),
        },
        "external_generator": {
            "model": "gpt-6-luna",
            "attempted_request_identities": external_attempted,
            "usable_outputs": external_usable,
            "expected_outputs": 6720,
            "status_counts": dict(sorted(external_status.items())),
            "requested_models": dict(sorted(external_requested_models.items())),
            "response_models": dict(sorted(external_response_models.items())),
            "first_response_timestamp": min(external_timestamps) if external_timestamps else None,
            "last_response_timestamp": max(external_timestamps) if external_timestamps else None,
        },
        "all_external_request_identities_attempted": external_attempted == 6720,
        "complete_six_model_evaluation_index": index["rows"] == 112320 and external_usable == 6720,
        "all_planned_generation_identities_accounted_for": bool(evaluation_status.get("all_planned_identities_accounted_for")),
        "planned_generation_failure_count": evaluation_status.get("explicitly_failed_and_excluded"),
        "all_available_index_items_terminally_accounted_for": qwen_covered >= index_ids and gpt_covered >= index_ids,
        "critical_file_hashes": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in critical
            if path.exists()
        ],
    }
    output = root / "manifests/final_coverage_audit.json"
    atomic_write_json(output, audit)
    print(json.dumps({
        "index_rows": index["rows"],
        "qwen_missing": audit["qwen_judge"]["missing_index_items"],
        "gpt_missing": audit["gpt_judge"]["missing_index_items"],
        "external_usable": external_usable,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
