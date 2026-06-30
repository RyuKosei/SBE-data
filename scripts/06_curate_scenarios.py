from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils import ROOT, read_jsonl, write_jsonl


RISK_PATTERNS = [
    r"自杀",
    r"自残",
    r"杀",
    r"威胁",
    r"诊断",
    r"处方",
    r"政治",
    r"宗教",
    r"股票",
    r"彩票",
]


def zh_chars(text: str) -> int:
    return sum("\u4e00" <= ch <= "\u9fff" for ch in text)


def normalized_content(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def char_ngrams(text: str, n: int = 2) -> set[str]:
    normalized = normalized_content(text)
    if len(normalized) <= n:
        return {normalized} if normalized else set()
    return {normalized[idx : idx + n] for idx in range(len(normalized) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def validate_row(row: dict[str, Any], min_checks: int, max_checks: int) -> tuple[bool, list[str]]:
    reasons = []
    required = ["scenario_id", "axis_id", "content", "style_a", "style_b", "content_checks", "length_range_zh_chars"]
    for key in required:
        if key not in row:
            reasons.append(f"missing:{key}")

    content = str(row.get("content", "")).strip()
    if zh_chars(content) < 12:
        reasons.append("content_too_short")
    if zh_chars(content) > 80:
        reasons.append("content_too_long")
    if re.search(r"第\d+个语境变体", content):
        reasons.append("template_variant_marker")
    if any(re.search(pattern, content) for pattern in RISK_PATTERNS):
        reasons.append("risk_pattern")

    checks = row.get("content_checks", [])
    if not isinstance(checks, list):
        reasons.append("content_checks_not_list")
    else:
        if len(checks) < min_checks:
            reasons.append("too_few_content_checks")
        if len(checks) > max_checks:
            reasons.append("too_many_content_checks")
        for check in checks:
            if not isinstance(check, str) or not check.strip():
                reasons.append("empty_content_check")
                break

    length_range = row.get("length_range_zh_chars")
    if not isinstance(length_range, list) or len(length_range) != 2:
        reasons.append("bad_length_range")
    else:
        try:
            low, high = int(length_range[0]), int(length_range[1])
            if low < 40 or high > 220 or low >= high:
                reasons.append("length_range_out_of_bounds")
        except (TypeError, ValueError):
            reasons.append("bad_length_range_values")

    style_a = str(row.get("style_a", "")).strip()
    style_b = str(row.get("style_b", "")).strip()
    if not style_a or not style_b or style_a == style_b:
        reasons.append("bad_style_endpoints")

    domain_tag = str(row.get("domain_tag", "")).strip()
    if not domain_tag:
        reasons.append("missing:domain_tag")

    risk_note = str(row.get("risk_note", row.get("safety_note", ""))).strip()
    if not risk_note:
        reasons.append("missing:risk_note")

    return not reasons, reasons


def curate(
    rows: list[dict[str, Any]],
    per_axis: int,
    min_checks: int,
    max_checks: int,
    near_duplicate_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted_by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = []
    seen_by_axis: dict[str, set[str]] = defaultdict(set)
    ngrams_by_axis: dict[str, list[set[str]]] = defaultdict(list)

    for row in rows:
        ok, reasons = validate_row(row, min_checks, max_checks)
        axis_id = str(row.get("axis_id", ""))
        content = str(row.get("content", ""))
        norm = normalized_content(content)
        if norm in seen_by_axis[axis_id]:
            ok = False
            reasons.append("duplicate_content")
        grams = char_ngrams(content)
        if ok and any(jaccard(grams, existing) >= near_duplicate_threshold for existing in ngrams_by_axis[axis_id]):
            ok = False
            reasons.append("near_duplicate_content")
        if ok:
            seen_by_axis[axis_id].add(norm)
            ngrams_by_axis[axis_id].append(grams)
            accepted_by_axis[axis_id].append(row)
        else:
            rejected.append({**row, "reject_reasons": sorted(set(reasons))})

    curated = []
    for axis_id in sorted(accepted_by_axis):
        selected = accepted_by_axis[axis_id][:per_axis]
        for idx, row in enumerate(selected, 1):
            normalized = {**row, "scenario_id": f"{axis_id}_{idx:03d}", "curation_rank": idx}
            if "risk_note" not in normalized and "safety_note" in normalized:
                normalized["risk_note"] = normalized["safety_note"]
            curated.append(normalized)
    return curated, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Candidate scenario JSONL.")
    parser.add_argument("--out", required=True, help="Curated scenario JSONL.")
    parser.add_argument("--rejects-out", default=None)
    parser.add_argument("--per-axis", type=int, default=40)
    parser.add_argument("--min-checks", type=int, default=2)
    parser.add_argument("--max-checks", type=int, default=4)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.45)
    args = parser.parse_args()

    rows = read_jsonl(args.candidates)
    curated, rejected = curate(rows, args.per_axis, args.min_checks, args.max_checks, args.near_duplicate_threshold)
    write_jsonl(args.out, curated)
    if args.rejects_out:
        write_jsonl(args.rejects_out, rejected)

    counts: dict[str, int] = defaultdict(int)
    for row in curated:
        counts[row["axis_id"]] += 1
    print(f"Read {len(rows)} candidates; wrote {len(curated)} curated rows to {Path(args.out)}")
    for axis_id in sorted(counts):
        print(f"{axis_id}: {counts[axis_id]}")
    if rejected:
        print(f"Rejected {len(rejected)} rows")


if __name__ == "__main__":
    main()
