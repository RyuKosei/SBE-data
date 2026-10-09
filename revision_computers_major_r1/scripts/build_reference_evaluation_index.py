from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_text, stable_id, utc_now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    references = read_jsonl(args.output_root / "data/gpt_fixed_references.jsonl")
    rows = []
    for reference in references:
        target = 0.0 if reference["endpoint"] == "A" else 1.0
        rows.append({
            "evaluation_item_id": stable_id("major_r1_reference_eval", reference["reference_id"]),
            "source_type": "gpt_fixed_reference",
            "source_identity": reference["reference_id"],
            "run_id": reference["reference_id"],
            "original_source_run_id": reference["reference_id"],
            "scenario_id": reference["scenario_id"],
            "style_axis": reference["style_axis"],
            "model_id": "gpt_fixed_reference",
            "target_ratio": target,
            "seed": None,
            "replicate_id": None,
            "prompt_template_id": "gpt_fixed_reference_endpoints_v1",
            "temperature": None,
            "robustness_condition_id": None,
            "input_text": reference["input_text"],
            "content_checks": reference["content_checks"],
            "style_A": reference["style_A"],
            "style_B": reference["style_B"],
            "generated_text": reference["reference_text"],
            "generated_text_sha256": sha256_text(reference["reference_text"]),
        })
    rows.sort(key=lambda row: row["evaluation_item_id"])
    output = args.output_root / "data/reference_evaluation_index.jsonl"
    atomic_write_jsonl(output, rows)
    status = {
        "generated_at": utc_now(),
        "rows": len(rows),
        "expected_rows": 640,
        "complete": len(rows) == 640,
        "source_status": "auto_generated",
        "human_verified": False,
    }
    atomic_write_json(args.output_root / "manifests/reference_evaluation_index_status.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
