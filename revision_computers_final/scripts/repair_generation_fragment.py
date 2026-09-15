"""Archive invalid generation rows and atomically retain only valid unique rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, atomic_write_jsonl, read_jsonl_tolerant, utc_now  # type: ignore
    from run_main_generation import append_jsonl
else:
    from .common import REVISION_ROOT, atomic_write_jsonl, read_jsonl_tolerant, utc_now
    from .run_main_generation import append_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    args.input = args.input or REVISION_ROOT / "data/fragments/main" / f"{args.model_id}.jsonl"
    args.archive = args.archive or REVISION_ROOT / "logs" / f"invalid_generation_rows_{args.model_id}.jsonl"
    source = read_jsonl_tolerant(args.input)
    if source.bad_lines:
        raise ValueError("malformed rows require manual recovery before repair")
    valid_by_id: dict[str, dict[str, object]] = {}
    invalid: list[dict[str, object]] = []
    duplicates = 0
    for row in source.rows:
        run_id = str(row.get("run_id", ""))
        reason = None
        if not run_id:
            reason = "missing_run_id"
        elif not str(row.get("generated_text", "")).strip():
            reason = "empty_generated_text"
        elif run_id in valid_by_id:
            reason = "duplicate_run_id"
            duplicates += 1
        if reason:
            invalid.append({**row, "archive_reason": reason, "archived_at": utc_now()})
        else:
            valid_by_id[run_id] = row
    if invalid:
        append_jsonl(args.archive, invalid)
        atomic_write_jsonl(args.input, valid_by_id.values())
    print(json.dumps({"input_rows": len(source.rows), "retained": len(valid_by_id), "archived": len(invalid), "duplicates": duplicates, "archive": str(args.archive)}))


if __name__ == "__main__":
    main()
