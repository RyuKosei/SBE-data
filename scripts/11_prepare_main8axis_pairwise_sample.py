from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from utils import ROOT, read_jsonl, write_jsonl


MODEL_FILES = {
    "qwen35_4b": ROOT / "data/generations/qwen35_4b/main8axis_v1_4b.jsonl",
    "qwen35_9b": ROOT / "data/generations/qwen35_9b/main8axis_v1_9b.jsonl",
    "qwen35_35b_a3b": ROOT / "data/generations/qwen35_35b_a3b/main8axis_v1_35b.jsonl",
}


def selected_scenarios(rows: list[dict], per_axis: int) -> dict[str, set[str]]:
    scenarios_by_axis: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        scenarios_by_axis[row["axis_id"]].add(row["scenario_id"])
    selected: dict[str, set[str]] = {}
    for axis_id, scenario_ids in scenarios_by_axis.items():
        selected[axis_id] = set(sorted(scenario_ids)[:per_axis])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-axis", type=int, default=20)
    parser.add_argument("--out-dir", default=ROOT / "data/judgments/pairwise_samples")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for model, path in MODEL_FILES.items():
        rows = read_jsonl(path)
        selected = selected_scenarios(rows, args.per_axis)
        keep = [
            row
            for row in rows
            if row["axis_id"] in selected and row["scenario_id"] in selected[row["axis_id"]]
        ]
        out = out_dir / f"main8axis_v1_pairwise_sample_{model}.jsonl"
        write_jsonl(out, keep)
        manifest.append(
            {
                "model": model,
                "source": str(path),
                "out": str(out),
                "rows": len(keep),
                "axes": len(selected),
                "scenarios_per_axis": {axis: len(ids) for axis, ids in sorted(selected.items())},
                "expected_pairs_per_order": len(selected) * args.per_axis * 3 * 9,
                "expected_pairs_bidirectional": len(selected) * args.per_axis * 3 * 9 * 2,
            }
        )

    manifest_path = out_dir / "main8axis_v1_pairwise_sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote pairwise samples and manifest to {out_dir}")


if __name__ == "__main__":
    main()
