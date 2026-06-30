from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from utils import ROOT, read_jsonl, write_jsonl


PREFERRED_DOMAINS = [
    "daily_life",
    "education",
    "science",
    "workplace",
    "public_service",
    "consumer",
    "family",
    "health_non_diagnostic",
    "travel",
    "technology",
    "community",
    "housing",
    "environment",
]

RISK_WORDS = ["诊断", "处方", "自伤", "自杀", "威胁", "政治", "宗教", "投资", "彩票", "违法"]


def zh_chars(text: object) -> int:
    return sum("\u4e00" <= ch <= "\u9fff" for ch in str(text))


def validate(row: dict[str, Any]) -> list[str]:
    reasons = []
    for key in ["scenario_id", "axis_id", "content", "style_a", "style_b", "content_checks", "length_range_zh_chars", "domain_tag", "risk_note"]:
        if key not in row or row.get(key) in (None, ""):
            reasons.append(f"missing:{key}")
    if zh_chars(row.get("content", "")) < 12:
        reasons.append("content_too_short")
    if zh_chars(row.get("content", "")) > 80:
        reasons.append("content_too_long")
    checks = row.get("content_checks")
    if not isinstance(checks, list):
        reasons.append("content_checks_not_list")
    elif not (2 <= len(checks) <= 4):
        reasons.append("content_checks_count")
    if row.get("domain_tag") not in PREFERRED_DOMAINS:
        reasons.append("domain_tag_unpreferred")
    content = str(row.get("content", ""))
    risk_note = str(row.get("risk_note", ""))
    if any(word in content for word in RISK_WORDS):
        reasons.append("content_risk_word")
    if any(word in risk_note for word in ["风险", "争议", "敏感"]) and "低风险" not in risk_note:
        reasons.append("risk_note_review")
    return reasons


def select_diverse(rows: list[dict[str, Any]], per_axis: int) -> list[dict[str, Any]]:
    by_axis_domain: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_axis_domain[str(row.get("axis_id"))][str(row.get("domain_tag", "unknown"))].append(row)

    selected = []
    for axis_id in sorted(by_axis_domain):
        axis_selected = []
        domain_order = sorted(by_axis_domain[axis_id], key=lambda domain: (-len(by_axis_domain[axis_id][domain]), domain))
        while len(axis_selected) < per_axis and any(by_axis_domain[axis_id].values()):
            for domain in domain_order:
                bucket = by_axis_domain[axis_id][domain]
                if bucket and len(axis_selected) < per_axis:
                    axis_selected.append(bucket.pop(0))
        for idx, row in enumerate(axis_selected, 1):
            normalized = dict(row)
            normalized["scenario_id"] = f"{axis_id}_{idx:03d}"
            normalized["review_rank"] = idx
            if "risk_note" not in normalized and "safety_note" in normalized:
                normalized["risk_note"] = normalized["safety_note"]
            selected.append(normalized)
    return selected


def write_tables(candidates: list[dict[str, Any]], filtered: list[dict[str, Any]], final: list[dict[str, Any]], rejects: list[dict[str, Any]], out_tables: Path) -> None:
    out_tables.mkdir(parents=True, exist_ok=True)
    final_df = pd.DataFrame(final)
    final_df.groupby(["axis_id", "domain_tag"]).size().reset_index(name="n").to_csv(out_tables / "scenario_domain_distribution.csv", index=False)

    candidate_counts = Counter(row.get("axis_id") for row in candidates)
    filtered_counts = Counter(row.get("axis_id") for row in filtered)
    final_counts = Counter(row.get("axis_id") for row in final)
    reject_by_axis: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rejects:
        axis_id = str(row.get("axis_id", "unknown"))
        reasons = row.get("reject_reasons", ["unknown"])
        for reason in reasons:
            reject_by_axis[axis_id][str(reason)] += 1

    rows = []
    axes = sorted(set(candidate_counts) | set(filtered_counts) | set(final_counts))
    for axis_id in axes:
        final_axis = [row for row in final if row.get("axis_id") == axis_id]
        domains = {row.get("domain_tag") for row in final_axis}
        validation_reasons = Counter(reason for row in final_axis for reason in validate(row))
        rows.append(
            {
                "axis_id": axis_id,
                "candidate_count": candidate_counts[axis_id],
                "filtered_count": filtered_counts[axis_id],
                "final_count": final_counts[axis_id],
                "domain_count": len(domains),
                "reject_count": sum(reject_by_axis[axis_id].values()),
                "top_reject_reasons": "; ".join(f"{key}:{value}" for key, value in reject_by_axis[axis_id].most_common(5)),
                "final_validation_issues": "; ".join(f"{key}:{value}" for key, value in validation_reasons.most_common()),
            }
        )
    pd.DataFrame(rows).to_csv(out_tables / "scenario_validation_summary.csv", index=False)


def write_report(final: list[dict[str, Any]], candidates: list[dict[str, Any]], filtered: list[dict[str, Any]], rejects: list[dict[str, Any]], out_report: Path, tables_dir: Path) -> None:
    final_by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final:
        final_by_axis[str(row.get("axis_id"))].append(row)
    candidate_counts = Counter(row.get("axis_id") for row in candidates)
    filtered_counts = Counter(row.get("axis_id") for row in filtered)
    reject_by_axis: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rejects:
        for reason in row.get("reject_reasons", ["unknown"]):
            reject_by_axis[str(row.get("axis_id", "unknown"))][str(reason)] += 1

    lines = [
        "# StyleBlend 8-Axis Scenario Pool Review v1",
        "",
        "Created: 2026-06-23 UTC",
        "",
        "## Output Files",
        "",
        "- `data/scenarios/styleblend_8axis_scenarios_v1.jsonl`",
        f"- `{tables_dir / 'scenario_domain_distribution.csv'}`",
        f"- `{tables_dir / 'scenario_validation_summary.csv'}`",
        "",
        "## Summary",
        "",
        "| axis | candidates | filtered | final | final domains | top reject reasons |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for axis_id in sorted(final_by_axis):
        domains = {row.get("domain_tag") for row in final_by_axis[axis_id]}
        reject_text = "; ".join(f"{key}:{value}" for key, value in reject_by_axis[axis_id].most_common(3)) or "none"
        lines.append(f"| {axis_id} | {candidate_counts[axis_id]} | {filtered_counts[axis_id]} | {len(final_by_axis[axis_id])} | {len(domains)} | {reject_text} |")

    lines.extend(["", "## Domain Distribution", ""])
    for axis_id in sorted(final_by_axis):
        counts = Counter(row.get("domain_tag") for row in final_by_axis[axis_id])
        lines.append(f"### {axis_id}")
        lines.append("")
        lines.append(", ".join(f"{domain}: {count}" for domain, count in sorted(counts.items())))
        lines.append("")

    lines.extend(["", "## Random-Like Samples", ""])
    for axis_id in sorted(final_by_axis):
        lines.append(f"### {axis_id}")
        lines.append("")
        for row in final_by_axis[axis_id][:10]:
            checks = "、".join(row.get("content_checks", []))
            lines.append(f"- `{row['scenario_id']}` [{row.get('domain_tag')}]: {row.get('content')} / checks: {checks}")
        lines.append("")

    lines.extend(["", "## Potential A/B Naturalness Issues", ""])
    issues = []
    for row in final:
        content = str(row.get("content", ""))
        if zh_chars(content) < 16 or len(row.get("content_checks", [])) < 2:
            issues.append(row)
    if issues:
        for row in issues[:30]:
            lines.append(f"- `{row['scenario_id']}` may be too sparse for both endpoints: {row.get('content')}")
    else:
        lines.append("- No obvious rule-based A/B naturalness issues were found. Manual review is still recommended before full main generation.")

    lines.extend(["", "## Potential Fact or Safety Risks", ""])
    risk_rows = [row for row in final if validate(row)]
    if risk_rows:
        for row in risk_rows[:30]:
            lines.append(f"- `{row['scenario_id']}` issues={validate(row)}: {row.get('content')} / risk_note={row.get('risk_note')}")
    else:
        lines.append("- No rule-based fact/safety risks were found in the final pool. This does not replace manual review.")

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--filtered", required=True)
    parser.add_argument("--rejects", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=ROOT / "docs/scenario_pool_review_8axis_v1.md")
    parser.add_argument("--tables-dir", default=ROOT / "results/scenario_pool_8axis_v1/tables")
    parser.add_argument("--per-axis", type=int, default=40)
    args = parser.parse_args()

    candidates = read_jsonl(args.candidates)
    filtered = read_jsonl(args.filtered)
    rejects = read_jsonl(args.rejects) if args.rejects else []
    final = select_diverse(filtered, args.per_axis)
    write_jsonl(args.out, final)
    tables_dir = Path(args.tables_dir)
    write_tables(candidates, filtered, final, rejects, tables_dir)
    write_report(final, candidates, filtered, rejects, Path(args.report), tables_dir)
    print(f"Wrote {len(final)} final scenarios to {args.out}")
    print(f"Wrote review report to {args.report}")
    print(f"Wrote review tables under {tables_dir}")


if __name__ == "__main__":
    main()
