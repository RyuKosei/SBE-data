from __future__ import annotations

import argparse
import asyncio
import itertools
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils import (
    ROOT,
    add_common_model_args,
    gather_limited,
    load_existing_ids,
    parse_json_object,
    read_jsonl,
    read_yaml,
    render_pairwise_judge_prompt,
    stable_id,
    write_jsonl,
)


async def call_judge(args: argparse.Namespace, prompt: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    response = await client.chat.completions.create(
        model=args.model_name or args.judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.request_timeout_seconds,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}} if args.disable_thinking else None,
    )
    return response.choices[0].message.content or ""


def build_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_group: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["axis_id"], row["scenario_id"], row["seed"], row["prompt_template"])
        by_group[key].append(row)

    wanted = {(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0), (0.0, 0.5), (0.5, 1.0), (0.25, 0.75)}
    pairs = []
    for group_rows in by_group.values():
        by_alpha = {float(row["alpha"]): row for row in group_rows}
        for left_alpha, right_alpha in sorted(wanted):
            if left_alpha in by_alpha and right_alpha in by_alpha:
                pairs.append((by_alpha[left_alpha], by_alpha[right_alpha]))
    return pairs


async def run_one(args: argparse.Namespace, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    prompt_left, prompt_right = (right, left) if args.swap_order else (left, right)
    pair_id = stable_id(args.judge_model, left["run_id"], right["run_id"], args.swap_order, prefix="pair_")
    prompt = render_pairwise_judge_prompt(prompt_left, prompt_right)
    if args.dry_run:
        if args.swap_order:
            raw = '{"winner": "text_1", "confidence": 3}' if float(right["alpha"]) > float(left["alpha"]) else '{"winner": "tie", "confidence": 1}'
        else:
            raw = '{"winner": "text_2", "confidence": 3}' if float(right["alpha"]) > float(left["alpha"]) else '{"winner": "tie", "confidence": 1}'
    else:
        raw = await call_judge(args, prompt)
    parsed, error = parse_json_object(raw)
    parsed = parsed or {}
    winner = parsed.get("winner")
    valid_winner = winner in {"text_1", "text_2", "tie"}
    return {
        "pair_id": pair_id,
        "left_run_id": left["run_id"],
        "right_run_id": right["run_id"],
        "left_alpha": left["alpha"],
        "right_alpha": right["alpha"],
        "text_1_run_id": prompt_left["run_id"],
        "text_2_run_id": prompt_right["run_id"],
        "swap_order": args.swap_order,
        "judge_model": args.judge_model,
        "model": left["model"],
        "axis_id": left["axis_id"],
        "scenario_id": left["scenario_id"],
        "seed": left["seed"],
        "winner": winner,
        "confidence": parsed.get("confidence"),
        "parse_ok": parsed != {} and valid_winner,
        "raw_judge_output": raw,
        "error": error if valid_winner else f"invalid winner: {winner!r}",
        "latency_seconds": round(time.time() - started, 3),
    }


async def async_main(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.generation_file)
    pairs = build_pairs(rows)
    existing = load_existing_ids(args.out, "pair_id")
    pairs = [(l, r) for l, r in pairs if stable_id(args.judge_model, l["run_id"], r["run_id"], args.swap_order, prefix="pair_") not in existing]
    if args.limit:
        pairs = pairs[: args.limit]
    out_rows = await gather_limited(args.concurrency, [run_one(args, left, right) for left, right in pairs])
    write_jsonl(args.out, out_rows, append=True)
    print(f"pairs_written={len(out_rows)} out={args.out}")


def main() -> None:
    cfg = read_yaml(ROOT / "configs/evaluation.yaml")["pairwise_judge"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-file", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--split", default="custom")
    parser.add_argument("--out", default=None)
    parser.add_argument("--temperature", type=float, default=cfg["temperature"])
    parser.add_argument("--top-p", type=float, default=cfg["top_p"])
    parser.add_argument("--max-tokens", type=int, default=cfg["max_tokens"])
    parser.add_argument("--concurrency", type=int, default=cfg["concurrency"])
    parser.add_argument("--request-timeout-seconds", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true", default=True)
    parser.add_argument("--swap-order", action="store_true", help="Put the higher-alpha item in text_1 to audit pairwise position bias.")
    parser.add_argument("--limit", type=int, default=None)
    add_common_model_args(parser)
    args = parser.parse_args()

    if args.out is None:
        args.out = ROOT / f"data/judgments/pairwise_{args.judge_model}_{args.split}.jsonl"
    args.out = Path(args.out)
    if not args.dry_run and not args.base_url:
        raise SystemExit("Real judging requires --base-url, or use --dry-run.")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
