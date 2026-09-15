"""Freeze scenario subsets for shared anchors, robustness, and human rating."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, atomic_write_csv, read_jsonl_tolerant  # type: ignore
else:
    from .common import REVISION_ROOT, atomic_write_csv, read_jsonl_tolerant


SCENARIO_FILE = Path("/home/data_cpfs/lizihua/sbe/data/scenarios/styleblend_8axis_scenarios_v1.jsonl")


def stratified_select(rows: list[dict[str, object]], count: int, seed: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    axes = sorted({str(row["axis_id"]) for row in rows})
    for axis in axes:
        candidates = [row for row in rows if row["axis_id"] == axis]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{axis}:{row['scenario_id']}".encode("utf-8")
            ).digest()
        )
        selected.extend(candidates[:count])
    return selected


def write_selection(path: Path, rows: list[dict[str, object]], seed: int, purpose: str) -> None:
    output = [
        {
            "scenario_id": row["scenario_id"],
            "style_axis": row["axis_id"],
            "selection_seed": seed,
            "purpose": purpose,
            "input_text": row["content"],
            "style_A": row["style_a"],
            "style_B": row["style_b"],
        }
        for row in rows
    ]
    atomic_write_csv(path, list(output[0]), output)


def main() -> None:
    source = read_jsonl_tolerant(SCENARIO_FILE)
    if source.bad_lines or len(source.rows) != 320:
        raise ValueError("expected the audited 320-scenario source")
    write_selection(
        REVISION_ROOT / "config/shared_anchor_scenarios.csv",
        stratified_select(source.rows, 10, 20260915),
        20260915,
        "shared_anchor",
    )
    write_selection(
        REVISION_ROOT / "config/robustness_scenarios.csv",
        stratified_select(source.rows, 5, 20260916),
        20260916,
        "prompt_temperature_robustness",
    )
    write_selection(
        REVISION_ROOT / "config/human_eval_scenarios.csv",
        stratified_select(source.rows, 2, 20260916),
        20260916,
        "human_evaluation",
    )
    print("wrote 80 shared-anchor, 40 robustness, and 16 human-evaluation scenarios")


if __name__ == "__main__":
    main()
