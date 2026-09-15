from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import (
    PROJECT_ROOT,
    REVISION_ROOT,
    atomic_write_json,
    atomic_write_text,
    git_commit,
    read_jsonl_tolerant,
    sha256_file,
    utc_now,
)


SCENARIO_FILE = PROJECT_ROOT / "data/scenarios/styleblend_8axis_scenarios_v1.jsonl"
GENERATION_FILES = {
    "qwen35_4b": PROJECT_ROOT / "data/generations/qwen35_4b/main8axis_v1_4b.jsonl",
    "qwen35_9b": PROJECT_ROOT / "data/generations/qwen35_9b/main8axis_v1_9b.jsonl",
    "qwen35_35b_a3b": PROJECT_ROOT
    / "data/generations/qwen35_35b_a3b/main8axis_v1_35b.jsonl",
}
PROJECTION_FILES = {
    (encoder, model): PROJECT_ROOT / f"data/metrics/projection_{encoder}_main8axis_v1_{suffix}.jsonl"
    for encoder in ("bge_m3", "text2vec")
    for model, suffix in (
        ("qwen35_4b", "4b"),
        ("qwen35_9b", "9b"),
        ("qwen35_35b_a3b", "35b"),
    )
}
SCALAR_FILES = {
    "qwen35_4b": PROJECT_ROOT
    / "data/judgments/scalar_qwen35_35b_a3b_main8axis_v1_4b.jsonl",
    "qwen35_9b": PROJECT_ROOT
    / "data/judgments/scalar_qwen35_35b_a3b_main8axis_v1_9b.jsonl",
    "qwen35_35b_a3b": PROJECT_ROOT
    / "data/judgments/scalar_qwen35_9b_main8axis_v1_35b.jsonl",
}
PAIRWISE_GLOB = "data/judgments/pairwise_*_main8axis_v1_sampled_*.jsonl"
HUMAN_SAMPLE = PROJECT_ROOT / "data/human_eval/human_validation_sample_v1.jsonl"
OLD_RATIOS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
EXPECTED_INTERIOR = {
    ("bge_m3", "qwen35_9b"): 0.1764,
    ("bge_m3", "qwen35_35b_a3b"): 0.1960,
    ("bge_m3", "qwen35_4b"): 0.2296,
    ("text2vec", "qwen35_9b"): 0.1554,
    ("text2vec", "qwen35_35b_a3b"): 0.1778,
    ("text2vec", "qwen35_4b"): 0.2114,
}
EXPECTED_COMBINED = {
    "qwen35_9b": 0.1652,
    "qwen35_35b_a3b": 0.1876,
    "qwen35_4b": 0.2198,
}
HARD_MARKER = re.compile(r"<think>|</think>|Thinking Process|步骤说明", re.IGNORECASE)


def _key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def profile_jsonl(path: Path, unique_fields: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = read_jsonl_tolerant(path)
    fields = sorted({field for row in result.rows for field in row})
    counts = Counter(_key(row, unique_fields) for row in result.rows)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    missing_by_field = {
        field: sum(row.get(field) is None for row in result.rows) for field in fields
    }
    return (
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "records": len(result.rows),
            "blank_lines": result.blank_lines,
            "bad_line_count": len(result.bad_lines),
            "bad_lines": result.bad_lines[:20],
            "fields": fields,
            "unique_key": list(unique_fields),
            "unique_key_count": len(counts),
            "duplicate_key_rows": duplicates,
            "missing_by_field": missing_by_field,
        },
        result.rows,
    )


def generation_audit() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    profiles: list[dict[str, Any]] = []
    all_rows: dict[str, list[dict[str, Any]]] = {}
    key_fields = ("model", "axis_id", "scenario_id", "alpha", "seed", "prompt_template")
    for model, path in GENERATION_FILES.items():
        profile, rows = profile_jsonl(path, key_fields)
        all_rows[model] = rows
        alpha_values = sorted({float(row["alpha"]) for row in rows})
        seeds = sorted({int(row["seed"]) for row in rows})
        axes = sorted({str(row["axis_id"]) for row in rows})
        scenarios = {str(row["scenario_id"]) for row in rows}
        generation_signatures = sorted(
            {
                json.dumps(row.get("generation_params", {}), sort_keys=True)
                for row in rows
            }
        )
        groups: dict[tuple[Any, ...], set[tuple[float, int]]] = defaultdict(set)
        for row in rows:
            groups[(row.get("model"), row.get("scenario_id"))].add(
                (float(row["alpha"]), int(row["seed"]))
            )
        expected_cells = {(alpha, seed) for alpha in OLD_RATIOS for seed in (1, 2, 3)}
        incomplete = sum(cells != expected_cells for cells in groups.values())
        empty_outputs = sum(not str(row.get("output", "")).strip() for row in rows)
        hard_markers = sum(bool(HARD_MARKER.search(str(row.get("output", "")))) for row in rows)
        profile.update(
            {
                "declared_model": model,
                "axes": axes,
                "axis_count": len(axes),
                "scenario_count": len(scenarios),
                "ratios": alpha_values,
                "seeds": seeds,
                "generation_parameter_signatures": generation_signatures,
                "all_requests_passed_seed": all(
                    bool(row.get("generation_params", {}).get("pass_seed")) for row in rows
                ),
                "incomplete_model_scenario_groups": incomplete,
                "empty_output_count": empty_outputs,
                "hard_marker_count": hard_markers,
            }
        )
        profiles.append(profile)
    return profiles, all_rows


def projection_audit() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    profiles: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for (encoder, model), path in PROJECTION_FILES.items():
        profile, rows = profile_jsonl(
            path, ("model", "axis_id", "scenario_id", "alpha", "seed")
        )
        frame = pd.DataFrame(rows)
        frame["encoder"] = encoder
        frame["model"] = model
        for column in (
            "alpha",
            "projection_ratio_raw",
            "projection_ratio_clipped",
            "endpoint_distance",
            "off_axis_drift",
            "interpolation_distance",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["abs_error_raw"] = (
            frame["projection_ratio_raw"] - frame["alpha"]
        ).abs()
        frame["abs_error_clipped"] = (
            frame["projection_ratio_clipped"] - frame["alpha"]
        ).abs()
        frame["is_endpoint"] = frame["alpha"].isin((0.0, 1.0))
        frame["out_of_range_raw"] = (frame["projection_ratio_raw"] < 0.0) | (
            frame["projection_ratio_raw"] > 1.0
        )
        interior = frame.loc[~frame["is_endpoint"]]
        expected = EXPECTED_INTERIOR[(encoder, model)]
        measured = float(interior["abs_error_clipped"].mean())
        profile.update(
            {
                "encoder": encoder,
                "declared_model": model,
                "seven_point_mae_clipped": float(frame["abs_error_clipped"].mean()),
                "seven_point_mae_raw": float(frame["abs_error_raw"].mean()),
                "five_point_mae_clipped": measured,
                "five_point_mae_raw": float(interior["abs_error_raw"].mean()),
                "expected_five_point_mae_clipped": expected,
                "expected_absolute_difference": abs(measured - expected),
                "sanity_within_0_002": abs(measured - expected) <= 0.002,
                "endpoint_abs_error_clipped_mean": float(
                    frame.loc[frame["is_endpoint"], "abs_error_clipped"].mean()
                ),
                "endpoint_off_axis_mean": float(
                    frame.loc[frame["is_endpoint"], "off_axis_drift"].mean()
                ),
                "endpoint_interpolation_distance_mean": float(
                    frame.loc[frame["is_endpoint"], "interpolation_distance"].mean()
                ),
                "raw_out_of_range_rate_all_points": float(frame["out_of_range_raw"].mean()),
                "raw_out_of_range_rate_interior": float(interior["out_of_range_raw"].mean()),
            }
        )
        profiles.append(profile)
        frames.append(frame)
    return profiles, pd.concat(frames, ignore_index=True)


def combined_sanity(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, expected in EXPECTED_COMBINED.items():
        group = frame[(frame["model"] == model) & (~frame["is_endpoint"])]
        measured = float(group["abs_error_clipped"].mean())
        rows.append(
            {
                "model": model,
                "measured_five_point_mae_clipped": measured,
                "expected": expected,
                "absolute_difference": abs(measured - expected),
                "sanity_within_0_002": abs(measured - expected) <= 0.002,
            }
        )
    return rows


def auxiliary_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scalar_profiles: list[dict[str, Any]] = []
    for model, path in SCALAR_FILES.items():
        profile, rows = profile_jsonl(path, ("judge_id",))
        profile.update(
            {
                "declared_model": model,
                "parse_ok_count": sum(bool(row.get("parse_ok")) for row in rows),
                "judge_models": sorted({str(row.get("judge_model")) for row in rows}),
            }
        )
        scalar_profiles.append(profile)

    pairwise_profiles: list[dict[str, Any]] = []
    for path in sorted(PROJECT_ROOT.glob(PAIRWISE_GLOB)):
        profile, rows = profile_jsonl(path, ("pair_id",))
        profile.update(
            {
                "parse_ok_count": sum(bool(row.get("parse_ok")) for row in rows),
                "judge_models": sorted({str(row.get("judge_model")) for row in rows}),
            }
        )
        pairwise_profiles.append(profile)

    human_profile, human_rows = profile_jsonl(HUMAN_SAMPLE, ("sample_id",))
    distributions: dict[str, Any] = {}
    for field in ("model", "axis_id", "alpha", "scenario_id", "selection_reason"):
        distributions[field] = dict(sorted(Counter(str(row.get(field)) for row in human_rows).items()))
    annotation_value_count = 0
    for row in human_rows:
        annotation = row.get("annotation_fields") or {}
        if any(value not in (None, "", "yes/no/partial") for value in annotation.values()):
            annotation_value_count += 1
    human_profile.update(
        {
            "distribution": distributions,
            "rows_with_any_real_annotation_value": annotation_value_count,
            "is_completed_human_rating_dataset": False,
            "finding": "This is a 336-row unannotated selection package, not 160 completed human ratings.",
        }
    )
    return scalar_profiles, pairwise_profiles, human_profile


def script_behavior() -> dict[str, Any]:
    old = (PROJECT_ROOT / "scripts/05_analyze.py").read_text(encoding="utf-8")
    final = (PROJECT_ROOT / "scripts/10_main8axis_final_analysis.py").read_text(
        encoding="utf-8"
    )
    return {
        "old_analysis_uses_clipped_projection_for_mae": 'projection["abs_error"] = (projection["projection_ratio_clipped"]' in old,
        "old_analysis_filters_endpoints_before_primary_means": bool(
            re.search(r"alpha.*isin.*0\.0.*1\.0", old)
        ),
        "final_analysis_uses_clipped_projection_for_primary_mae": 'group["projection_error"].mean()' in final,
        "final_analysis_defines_raw_projection_error_separately": 'projection_error_unclipped' in final,
        "final_analysis_filters_endpoints_before_group_summary": bool(
            re.search(r"proj\s*=\s*proj\[~.*endpoint", final)
        ),
        "final_analysis_out_of_range_uses_raw_projection": '(proj["r_proj"] < 0.0)' in final,
        "conclusion": (
            "Both legacy analysis paths included the mechanically exact endpoints in primary means. "
            "The original path used clipped projections for MAE. The final 2026-06 path retained a raw-error "
            "column and correctly computed out-of-range from raw coordinates, but still reported clipped, "
            "seven-point MAE as primary."
        ),
    }


def markdown_report(audit: dict[str, Any]) -> str:
    generations = audit["legacy_generations"]
    projections = audit["legacy_projection_metrics"]
    human = audit["legacy_human_sample"]
    lines = [
        "# StyleBlend-Bench Legacy Data Audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Git commit: `{audit['git_commit']}`",
        "",
        "## Executive findings",
        "",
        "- The scenario source contains 320 unique scenarios across eight axes.",
        "- The three historical generation files contain 6,720 rows each, totaling 20,160 = 320 × 7 × 3 × 3.",
        "- Historical ratios are `0, 0.1, 0.25, 0.5, 0.75, 0.9, 1`, not the revision's sixth-based grid.",
        "- Historical rows carry seed labels 1/2/3, but every generation signature has `pass_seed=false`; the seed was not sent to the inference server.",
        "- Legacy primary projection MAE used clipped coordinates and included the two mechanically zero-error endpoints. Primary drift and interpolation-distance means also included those zero endpoints.",
        "- The corrected five-interior-point clipped MAEs reproduce the supplied sanity targets within 0.002. Raw, unclipped values are retained separately and will be primary in the revision.",
        "- Out-of-range rate in the final legacy script was computed from raw projection coordinates before clipping.",
        "- No completed 160-item human-rating dataset exists under the project. The only human file is a 336-row, unannotated selection package; historical handoff notes explicitly say annotation was not run.",
        "",
        "## Source files and coverage",
        "",
        "| Model | Path | Records | Unique scenarios | Ratios | Seeds | Duplicate keys | Bad lines | Empty outputs | Seed passed |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for row in generations:
        lines.append(
            f"| {row['declared_model']} | `{row['path']}` | {row['records']} | {row['scenario_count']} | "
            f"{row['ratios']} | {row['seeds']} | {row['duplicate_key_rows']} | {row['bad_line_count']} | "
            f"{row['empty_output_count']} | {row['all_requests_passed_seed']} |"
        )
    lines.extend(
        [
            "",
            "The generation unique key is `{model, axis_id, scenario_id, alpha, seed, prompt_template}`. "
            "Full field lists, hashes, missing-value counts, and coverage diagnostics are in `audit_summary.json`.",
            "",
            "## Endpoint-inclusion and clipping sanity check",
            "",
            "| Encoder | Model | 7-point clipped MAE | 5-point clipped MAE | 5-point raw MAE | Expected interior MAE | |difference| | Pass |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in projections:
        lines.append(
            f"| {row['encoder']} | {row['declared_model']} | {row['seven_point_mae_clipped']:.6f} | "
            f"{row['five_point_mae_clipped']:.6f} | {row['five_point_mae_raw']:.6f} | "
            f"{row['expected_five_point_mae_clipped']:.4f} | {row['expected_absolute_difference']:.6f} | "
            f"{row['sanity_within_0_002']} |"
        )
    lines.extend(
        [
            "",
            "The endpoint rows have zero projection error, zero off-axis drift, and zero interpolation distance by construction. "
            "Including them multiplies a five-point mean by 5/7 and therefore understates the interior error by 28.57% when all cells are present.",
            "",
            "## Human-evaluation audit",
            "",
            f"- File: `{human['path']}`",
            f"- Rows: {human['records']}",
            f"- Rows containing any real rating: {human['rows_with_any_real_annotation_value']}",
            f"- Finding: {human['finding']}",
            "- Consequence: the revision must generate a fresh blinded 320-text annotation package and wait for at least 960 real rater assignments.",
            "",
            "## Model/checkpoint history relevant to reuse",
            "",
            "- Historical tested checkpoints: Qwen3.5-4B, Qwen3.5-9B, and Qwen3.5-35B-A3B from `/home/data_cpfs/zicheng/models/`.",
            "- Revision additions found on cluster: official Qwen3.6-27B Dense at `/home/data_cpfs/lgx/model/Qwen3.6-27B-base` and Qwen3.6-35B-A3B at `/home/accio_data/zicheng/models/Qwen3.6-35B-A3B`.",
            "- The new main grid and server-side seed requirement differ from the legacy run, so legacy generations are audit evidence rather than revision main-experiment rows.",
            "",
            "## Implementation differences that change paper values",
            "",
            "1. Excluding endpoints raises projection MAE, drift, and interpolation-distance means relative to legacy seven-point summaries.",
            "2. Raw projection MAE is at least as large as clipped MAE whenever extrapolation occurs; the revision uses raw as primary.",
            "3. Normalizing off-axis and interpolation distances by endpoint separation changes cross-scenario weighting and exposes low-separation instability.",
            "4. The revision retains three generation seeds and actually transmits each seed to the server; the legacy rows only carried three nominal labels that were not sent.",
            "5. The new equally spaced sixth grid is not directly interchangeable with the old irregular interior grid.",
            "6. Scenario-level aggregation/bootstrap replaces any text-level independence assumption.",
            "",
            "## Audit limitations",
            "",
            "- Model-card parameter counts are rounded release values; no full tensor-by-tensor parameter recount was needed for this audit.",
            "- No real human ratings were available to audit for rater demographics, duration, agreement, or reuse compatibility.",
            "- Inference framework performance will be recorded prospectively because historical generation rows contain latency but not GPU memory, framework version, token throughput, or retry telemetry.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REVISION_ROOT / "audit")
    args = parser.parse_args()

    scenario_profile, scenarios = profile_jsonl(SCENARIO_FILE, ("scenario_id",))
    scenario_profile.update(
        {
            "axis_count": len({row.get("axis_id") for row in scenarios}),
            "axis_distribution": dict(sorted(Counter(row.get("axis_id") for row in scenarios).items())),
        }
    )
    generation_profiles, _generation_rows = generation_audit()
    projection_profiles, projection_frame = projection_audit()
    scalar_profiles, pairwise_profiles, human_profile = auxiliary_audit()
    audit = {
        "generated_at": utc_now(),
        "git_commit": git_commit(),
        "legacy_scenarios": scenario_profile,
        "legacy_generations": generation_profiles,
        "legacy_projection_metrics": projection_profiles,
        "combined_interior_sanity": combined_sanity(projection_frame),
        "legacy_scalar_judgments": scalar_profiles,
        "legacy_pairwise_judgments": pairwise_profiles,
        "legacy_human_sample": human_profile,
        "legacy_script_behavior": script_behavior(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out_dir / "audit_summary.json", audit)
    atomic_write_text(args.out_dir / "data_audit.md", markdown_report(audit))
    failures = [
        row
        for row in projection_profiles + audit["combined_interior_sanity"]
        if not row.get("sanity_within_0_002", True)
    ]
    if failures:
        raise SystemExit(f"Legacy sanity check failed: {failures}")
    print(
        json.dumps(
            {
                "scenario_records": scenario_profile["records"],
                "generation_records": sum(row["records"] for row in generation_profiles),
                "projection_records": sum(row["records"] for row in projection_profiles),
                "human_records": human_profile["records"],
                "sanity": "passed",
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
