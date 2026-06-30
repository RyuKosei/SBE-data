from __future__ import annotations

import argparse
import asyncio
import time
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
    render_scalar_judge_prompt,
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


async def run_one(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    judge_id = stable_id(args.judge_model, row["run_id"], prefix="scalar_")
    prompt = render_scalar_judge_prompt(row)
    parsed = None
    raw_output = ""
    error = None
    for attempt in range(args.max_retries + 1):
        try:
            if args.dry_run:
                raw_output = '{"style_b_ratio": %.3f, "content_preservation": 5, "style_mixture_naturalness": 3, "reason": "dry run"}' % float(row["alpha"])
            else:
                raw_output = await call_judge(args, prompt)
            parsed, error = parse_json_object(raw_output)
            if parsed is not None:
                break
        except Exception as exc:
            error = repr(exc)
    parsed = parsed or {}
    return {
        "judge_id": judge_id,
        "run_id": row["run_id"],
        "judge_model": args.judge_model,
        "model": row["model"],
        "axis_id": row["axis_id"],
        "scenario_id": row["scenario_id"],
        "alpha": row["alpha"],
        "seed": row["seed"],
        "style_b_ratio": parsed.get("style_b_ratio"),
        "content_preservation": parsed.get("content_preservation"),
        "style_mixture_naturalness": parsed.get("style_mixture_naturalness"),
        "reason": parsed.get("reason"),
        "parse_ok": parsed != {},
        "raw_judge_output": raw_output,
        "error": error,
        "latency_seconds": round(time.time() - started, 3),
    }


async def async_main(args: argparse.Namespace) -> None:
    generations = read_jsonl(args.generation_file)
    existing = load_existing_ids(args.out, "judge_id")
    todo = [row for row in generations if stable_id(args.judge_model, row["run_id"], prefix="scalar_") not in existing]
    if args.limit:
        todo = todo[: args.limit]
    rows = await gather_limited(args.concurrency, [run_one(args, row) for row in todo])
    write_jsonl(args.out, rows, append=True)
    print(f"generations={len(generations)} existing={len(existing)} wrote={len(rows)} out={args.out}")


def main() -> None:
    cfg = read_yaml(ROOT / "configs/evaluation.yaml")["scalar_judge"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-file", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--split", default="custom")
    parser.add_argument("--out", default=None)
    parser.add_argument("--temperature", type=float, default=cfg["temperature"])
    parser.add_argument("--top-p", type=float, default=cfg["top_p"])
    parser.add_argument("--max-tokens", type=int, default=cfg["max_tokens"])
    parser.add_argument("--max-retries", type=int, default=cfg["max_retries"])
    parser.add_argument("--concurrency", type=int, default=cfg["concurrency"])
    parser.add_argument("--request-timeout-seconds", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    add_common_model_args(parser)
    args = parser.parse_args()

    if args.out is None:
        args.out = ROOT / f"data/judgments/scalar_{args.judge_model}_{args.split}.jsonl"
    args.out = Path(args.out)
    if not args.dry_run and not args.base_url:
        raise SystemExit("Real judging requires --base-url, or use --dry-run.")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
