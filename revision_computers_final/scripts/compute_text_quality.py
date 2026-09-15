"""Compute deterministic, non-LLM text quality diagnostics for every output."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, read_jsonl_tolerant  # type: ignore
else:
    from .common import REVISION_ROOT, read_jsonl_tolerant


TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*")
FORMAT_PATTERNS = {
    "assistant_prefix": re.compile(r"^\s*(?:assistant|助手)\s*[:：]", re.I),
    "thinking_tag": re.compile(r"</?think>|</?analysis>", re.I),
    "prompt_echo": re.compile(r"目标风格比例|请生成一段中文文本，使其风格", re.I),
    "code_fence": re.compile(r"```"),
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def distinct_n(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return float("nan")
    grams = list(zip(*(tokens[offset:] for offset in range(n))))
    return len(set(grams)) / len(grams)


def repetition_rate(tokens: list[str]) -> float:
    if not tokens:
        return float("nan")
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(tokens)


def longest_repeated_run(text: str) -> int:
    """Return the longest immediately repeated substring run in characters."""

    compact = re.sub(r"\s+", "", text)
    longest = 0
    for width in range(1, min(40, len(compact) // 2) + 1):
        for start in range(0, len(compact) - 2 * width + 1):
            block = compact[start : start + width]
            if block == compact[start + width : start + 2 * width]:
                longest = max(longest, 2 * width)
    return longest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REVISION_ROOT / "data/qwen_family_outputs.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REVISION_ROOT / "metrics/text_quality_metrics.parquet",
    )
    args = parser.parse_args()
    parsed = read_jsonl_tolerant(args.input)
    if parsed.bad_lines:
        raise SystemExit(f"input contains {len(parsed.bad_lines)} bad JSONL lines")

    records: list[dict[str, object]] = []
    for row in parsed.rows:
        text = str(row.get("generated_text", ""))
        tokens = tokenize(text)
        finish_reason = str(row.get("finish_reason", ""))
        format_flags = [name for name, pattern in FORMAT_PATTERNS.items() if pattern.search(text)]
        records.append(
            {
                "run_id": row["run_id"],
                "scenario_id": row["scenario_id"],
                "style_axis": row["style_axis"],
                "model_id": row["model_id"],
                "seed": int(row["seed"]),
                "target_ratio": float(row["target_ratio"]),
                "character_length": len(text),
                "nonspace_character_length": len(re.sub(r"\s+", "", text)),
                "lexical_token_count": len(tokens),
                "server_output_tokens": int(row.get("output_tokens") or 0),
                "repetition_rate": repetition_rate(tokens),
                "distinct_2": distinct_n(tokens, 2),
                "distinct_3": distinct_n(tokens, 3),
                "longest_immediate_repeated_run": longest_repeated_run(text),
                "empty_response": not bool(text.strip()),
                "truncated": finish_reason in {"length", "max_tokens"},
                "format_anomaly": bool(format_flags),
                "format_anomaly_types": "|".join(format_flags),
                "finish_reason": finish_reason,
            }
        )
    frame = pd.DataFrame.from_records(records)
    key = ["run_id"]
    duplicates = int(frame.duplicated(key).sum())
    if duplicates:
        raise SystemExit(f"found {duplicates} duplicate quality metric keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    summary = {
        "rows": len(frame),
        "models": frame["model_id"].nunique(),
        "empty": int(frame["empty_response"].sum()),
        "truncated": int(frame["truncated"].sum()),
        "format_anomalies": int(frame["format_anomaly"].sum()),
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
