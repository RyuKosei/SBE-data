from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from common import atomic_write_json, read_jsonl, utc_now
from gpt_common import estimated_cost, load_prices, usage_fields


def usage_from_rows(path: Path) -> dict[str, int]:
    rows = read_jsonl(path) if path.exists() else []
    return {key: sum(int(row.get(key) or 0) for row in rows) for key in usage_fields(None)}


def stage(path: Path, price: dict[str, float]) -> dict[str, Any]:
    rows = read_jsonl(path) if path.exists() else []
    usage = usage_from_rows(path)
    successful = sum(
        row.get("status") in {"ok", "truncated"} if "status" in row else True
        for row in rows
    )
    return {
        "path": str(path),
        "records": len(rows),
        "successful_records": successful,
        "non_success_records": len(rows) - successful,
        "usage": usage,
        "estimated_cost_usd": estimated_cost(usage, price),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pricing", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = config["gpt"]["selected_model"]
    price = load_prices(args.pricing, model)
    stages = {
        "fixed_references": stage(args.output_root / "data/gpt_fixed_references.jsonl", price),
        "main_judge": stage(args.output_root / "metrics/gpt_judge_evaluations.jsonl", price),
        "stability_retest": stage(args.output_root / "metrics/gpt_stability_retest.jsonl", price),
    }
    external_model = config["external_generator"]["requested_model"]
    external_price = load_prices(args.pricing, external_model)
    external_generation = stage(args.output_root / "data/external_main_outputs.jsonl", external_price)
    selection = json.loads((args.output_root / "manifests/gpt_model_selection.json").read_text(encoding="utf-8"))
    pilot_cost = float(selection["pilot_results"]["gpt-6-sol"]["estimated_cost_usd"])
    external_selection = json.loads((args.output_root / "manifests/external_generator_selection.json").read_text(encoding="utf-8"))
    external_smoke_cost = float(external_selection["real_prompt_smoke"]["estimated_cost_usd"])
    # Raw usage for the two inexpensive comparison-model judge pilots was not
    # retained.  Keep that limitation explicit and add a deliberately generous
    # ten-cent allowance to the conservative ceiling calculation.
    unrecorded_candidate_pilot_allowance = 0.10
    formal_cost = sum(value["estimated_cost_usd"] for value in stages.values())
    total = pilot_cost + external_smoke_cost + formal_cost + external_generation["estimated_cost_usd"]
    # Two deliberate concurrency changes interrupted at most 64 and 48
    # in-flight judge calls.  The relay exposes usage only for returned
    # responses, so keep a conservative response-average billing allowance
    # rather than presenting the reconstructed total as an exact invoice.
    main_records = stages["main_judge"]["successful_records"]
    main_average_cost = stages["main_judge"]["estimated_cost_usd"] / main_records if main_records else 0.0
    restart_inflight_cost_allowance = main_average_cost * (64 + 48)
    conservative_total = total + restart_inflight_cost_allowance + unrecorded_candidate_pilot_allowance
    hard_cap = float(config["gpt"]["operational_gpt_hard_cap_usd"])

    deepseek_path = args.output_root / "data/deepseek_main_outputs.jsonl"
    deepseek_rows = read_jsonl(deepseek_path) if deepseek_path.exists() else []
    deepseek_usage = {
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in deepseek_rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in deepseek_rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in deepseek_rows),
    }

    credential = os.environ.get(config["gpt"]["credential_env"], "")
    credential_hits = []
    if credential:
        for path in args.output_root.rglob("*"):
            if not path.is_file() or path.suffix in {".png", ".pdf", ".npz", ".parquet", ".pyc"}:
                continue
            try:
                if credential in path.read_text(encoding="utf-8", errors="ignore"):
                    credential_hits.append(str(path))
            except OSError:
                pass
    report = {
        "generated_at": utc_now(),
        "gpt_model": model,
        "gpt_snapshot_frozen": False,
        "price_per_million_tokens_usd": price,
        "stages": stages,
        "external_generation": {
            "model": external_model,
            "snapshot_frozen": False,
            "price_per_million_tokens_usd": external_price,
            **external_generation,
        },
        "selection_pilot_estimated_cost_usd": pilot_cost,
        "external_generation_smoke_estimated_cost_usd": external_smoke_cost,
        "unrecorded_candidate_pilot_exact_cost_usd": None,
        "conservative_unrecorded_candidate_pilot_allowance_usd": unrecorded_candidate_pilot_allowance,
        "gpt_total_estimated_cost_usd": total,
        "possible_unrecorded_restart_inflight_cost_allowance_usd": restart_inflight_cost_allowance,
        "gpt_conservative_total_with_allowances_usd": conservative_total,
        "gpt_conservative_total_with_restart_allowance_usd": conservative_total,
        "cost_interpretation": "Response-derived estimate, not a relay invoice; the conservative total adds one full in-flight concurrency window for each deliberate judge-process restart and a USD 0.10 allowance for two inexpensive comparison-model pilot batches whose raw usage was not retained.",
        "operational_gpt_hard_cap_usd": hard_cap,
        "estimated_headroom_usd": hard_cap - total,
        "conservative_estimated_headroom_usd": hard_cap - conservative_total,
        "author_stated_ceiling_usd": float(config["gpt"]["budget_usd"]),
        "hard_cap_exceeded": conservative_total > hard_cap,
        "account_balance_available": False,
        "account_balance_note": "The internal relay returned token usage but no wallet/quota balance endpoint or headers in the tested responses.",
        "deepseek_usage": deepseek_usage,
        "deepseek_cost_usd": None,
        "deepseek_cost_note": "No internal DeepSeek price or billable response was available; do not impute a dollar cost.",
        "credential_exact_match_scan_performed": bool(credential),
        "credential_exact_match_hits": credential_hits,
    }
    atomic_write_json(args.output_root / "manifests/api_usage_and_budget.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
