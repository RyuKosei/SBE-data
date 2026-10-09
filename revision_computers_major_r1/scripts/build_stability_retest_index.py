from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from common import atomic_write_json, atomic_write_jsonl, read_jsonl, stable_id, utc_now


SELECTION_SEED = 20260918


def score(identity: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}::{identity}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.output_root / "data/evaluation_index.jsonl")
    main_rows = [row for row in rows if row["source_type"] in {"main_qwen", "main_external"} and 0 < float(row["target_ratio"]) < 1]
    frame = pd.DataFrame(main_rows)
    selected = []
    for key, group in frame.groupby(["style_axis", "model_id", "target_ratio"], sort=True):
        records = group.to_dict("records")
        chosen = min(records, key=lambda row: score(row["evaluation_item_id"]))
        original_item = chosen["evaluation_item_id"]
        selected.append({
            **chosen,
            "evaluation_item_id": stable_id("major_r1_stability_retest", original_item, SELECTION_SEED),
            "source_type": "stability_retest",
            "source_identity": original_item,
            "original_evaluation_item_id": original_item,
            "stability_selection_seed": SELECTION_SEED,
        })
    selected.sort(key=lambda row: row["evaluation_item_id"])
    output = args.output_root / "data/stability_retest_index.jsonl"
    atomic_write_jsonl(output, selected)
    expected_models = int(frame.model_id.nunique())
    status = {
        "generated_at": utc_now(),
        "selection_seed": SELECTION_SEED,
        "rows": len(selected),
        "expected_rows": 8 * expected_models * 5,
        "full_six_model_design_expected_rows": 240,
        "complete_for_full_six_model_design": len(selected) == 240,
        "models": sorted(frame.model_id.unique().tolist()),
        "one_preattached_output_per_axis_model_internal_alpha": True,
    }
    atomic_write_json(args.output_root / "manifests/stability_retest_index.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
