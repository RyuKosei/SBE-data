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
    read_jsonl,
    read_yaml,
    render_explicit_ratio_prompt,
    stable_id,
    strip_think_blocks,
    write_jsonl,
)


async def call_openai(args: argparse.Namespace, prompt: str, seed: int) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    extra_body: dict[str, Any] = {}
    if args.disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if args.pass_seed:
        extra_body["seed"] = seed

    response = await client.chat.completions.create(
        model=args.model_name or args.model,
        messages=[
            {
                "role": "system",
                "content": "你是一个中文短文本生成器。不要展示推理过程、分析过程或中间步骤。任何情况下都只输出最终中文正文。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        extra_body=extra_body or None,
        timeout=args.request_timeout_seconds,
    )
    return response.choices[0].message.content or ""


def build_jobs(args: argparse.Namespace, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for scenario in scenarios:
        for alpha in args.ratios:
            for seed in args.seeds:
                run_id = stable_id(args.model, scenario["scenario_id"], alpha, seed, args.prompt_template, prefix="run_")
                prompt = render_explicit_ratio_prompt(scenario, alpha)
                jobs.append(
                    {
                        "run_id": run_id,
                        "model": args.model,
                        "axis_id": scenario["axis_id"],
                        "scenario_id": scenario["scenario_id"],
                        "alpha": alpha,
                        "seed": seed,
                        "prompt_template": args.prompt_template,
                        "prompt": prompt,
                        "content": scenario["content"],
                        "style_a": scenario["style_a"],
                        "style_b": scenario["style_b"],
                        "attribute_name": scenario.get("attribute_name"),
                        "content_checks": scenario.get("content_checks", []),
                        "length_range_zh_chars": scenario.get("length_range_zh_chars", [80, 140]),
                    }
                )
    return jobs


async def run_job(args: argparse.Namespace, job: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    started = time.time()
    try:
        if args.dry_run:
            output = f"[DRY_RUN] {job['content']}"
        else:
            output = await call_openai(args, job["prompt"], job["seed"])
        if args.strip_think_blocks:
            output = strip_think_blocks(output)
        row = {
            **job,
            "output": output,
            "generation_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "pass_seed": args.pass_seed,
            },
            "latency_seconds": round(time.time() - started, 3),
        }
        return row, None
    except Exception as exc:
        return None, {**job, "error": repr(exc), "latency_seconds": round(time.time() - started, 3)}


async def async_main(args: argparse.Namespace) -> None:
    scenarios = read_jsonl(args.scenario_file)
    jobs = build_jobs(args, scenarios)
    existing = load_existing_ids(args.out, "run_id")
    todo = [job for job in jobs if job["run_id"] not in existing]

    if args.limit:
        todo = todo[: args.limit]

    results = await gather_limited(args.concurrency, [run_job(args, job) for job in todo])
    rows = [row for row, _failure in results if row is not None]
    failures = [failure for _row, failure in results if failure is not None]

    write_jsonl(args.out, rows, append=True)
    if failures:
        write_jsonl(args.failure_log, failures, append=True)

    print(f"Planned={len(jobs)} existing={len(existing)} wrote={len(rows)} failures={len(failures)} out={args.out}")


def main() -> None:
    generation_cfg = read_yaml(ROOT / "configs/generation.yaml")["generation"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="custom")
    parser.add_argument("--out", default=None)
    parser.add_argument("--failure-log", default=ROOT / "results/logs/generation_failed.jsonl")
    parser.add_argument("--prompt-template", default=generation_cfg["prompt_template"])
    parser.add_argument("--ratios", nargs="*", type=float, default=generation_cfg["ratios"])
    parser.add_argument("--seeds", nargs="*", type=int, default=generation_cfg["seeds"])
    parser.add_argument("--temperature", type=float, default=generation_cfg["temperature"])
    parser.add_argument("--top-p", type=float, default=generation_cfg["top_p"])
    parser.add_argument("--max-tokens", type=int, default=generation_cfg["max_tokens"])
    parser.add_argument("--concurrency", type=int, default=generation_cfg["concurrency"])
    parser.add_argument("--request-timeout-seconds", type=int, default=generation_cfg["request_timeout_seconds"])
    parser.add_argument("--pass-seed", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true", default=True)
    parser.add_argument("--strip-think-blocks", action="store_true", default=generation_cfg["strip_think_blocks"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    add_common_model_args(parser)
    args = parser.parse_args()

    if args.out is None:
        args.out = ROOT / f"data/generations/{args.model}/{args.split}.jsonl"
    args.out = Path(args.out)

    if not args.dry_run and not args.base_url:
        raise SystemExit("Real generation requires --base-url, or use --dry-run.")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
