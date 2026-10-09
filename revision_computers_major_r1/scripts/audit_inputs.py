from __future__ import annotations

import argparse
import collections
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from common import atomic_write_json, atomic_write_text, read_jsonl, sha256_file, utc_now


def duplicate_count(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> int:
    counts = collections.Counter(tuple(row.get(field) for field in fields) for row in rows)
    return sum(count - 1 for count in counts.values() if count > 1)


def file_fact(path: Path, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
        "rows": len(rows) if rows is not None else None,
    }


def missing_main(rows: list[dict[str, Any]], scenarios: list[dict[str, Any]], config: dict[str, Any]) -> list[tuple[Any, ...]]:
    expected = {
        (model, scenario["scenario_id"], float(alpha), int(seed), config["main"]["prompt_template_id"])
        for model in config["main"]["qwen_models"]
        for scenario in scenarios
        for alpha in config["main"]["target_ratios"]
        for seed in config["main"]["qwen_seeds"]
    }
    actual = {
        (row["model_id"], row["scenario_id"], float(row["target_ratio"]), int(row["seed"]), row["prompt_template_id"])
        for row in rows
    }
    return sorted(expected - actual)


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    project = Path(config["project_root"])
    legacy = Path(config["legacy_root"])
    scenarios_path = Path(config["scenario_file"])
    paths = {
        "scenarios": scenarios_path,
        "main": legacy / "data/qwen_family_outputs.jsonl",
        "robustness": legacy / "data/robustness_outputs.jsonl",
        "main_judge": legacy / "metrics/llm_evaluations.jsonl",
        "robustness_judge": legacy / "metrics/robustness_llm_evaluations.jsonl",
        "main_config": legacy / "config/main_experiment.yaml",
        "robustness_config": legacy / "config/robustness_experiment.yaml",
        "evaluator_config": legacy / "config/evaluator.yaml",
        "model_registry": legacy / "config/model_registry.csv",
        "encoder_registry": legacy / "config/encoder_registry.csv",
    }
    datasets = {name: read_jsonl(path) for name, path in paths.items() if path.suffix == ".jsonl"}
    scenarios = datasets["scenarios"]
    main_rows = datasets["main"]
    robustness = datasets["robustness"]
    main_eval = datasets["main_judge"]
    robust_eval = datasets["robustness_judge"]
    main_key = ("model_id", "scenario_id", "target_ratio", "seed", "prompt_template_id")
    robust_key = main_key + ("temperature",)
    missing = missing_main(main_rows, scenarios, config)
    main_ids = {row["run_id"] for row in main_rows}
    robust_ids = {row["run_id"] for row in robustness}
    reused = sum(1 for row in robustness if row.get("source") == "reused_main" or row["run_id"] in main_ids)
    axes = collections.Counter(row["axis_id"] for row in scenarios)
    checks = [len(row.get("content_checks") or []) for row in scenarios]
    embedding_facts: list[dict[str, Any]] = []
    for path in sorted((legacy / "data/embeddings").glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            count = len(archive["run_ids"])
            shape = list(archive["vectors"].shape)
            duplicate_ids = count - len(set(map(str, archive["run_ids"].tolist())))
        embedding_facts.append({**file_fact(path), "shape": shape, "duplicate_run_ids": duplicate_ids})
    facts: dict[str, Any] = {
        "generated_at": utc_now(),
        "git_commit": git_commit(project),
        "project_root": str(project),
        "legacy_root": str(legacy),
        "output_root": str(args.output_root),
        "files": {name: file_fact(path, datasets.get(name)) for name, path in paths.items()},
        "scenarios": {
            "count": len(scenarios),
            "axes": dict(sorted(axes.items())),
            "duplicate_scenario_ids": len(scenarios) - len({row["scenario_id"] for row in scenarios}),
            "content_check_count_min": min(checks),
            "content_check_count_max": max(checks),
            "content_check_count_distribution": dict(sorted(collections.Counter(checks).items())),
            "missing_content_checks": sum(count == 0 for count in checks),
        },
        "main": {
            "rows": len(main_rows),
            "duplicate_keys": duplicate_count(main_rows, main_key),
            "duplicate_run_ids": len(main_rows) - len(main_ids),
            "missing_combinations": len(missing),
            "missing_preview": [list(key) for key in missing[:20]],
            "models": dict(sorted(collections.Counter(row["model_id"] for row in main_rows).items())),
            "seeds": sorted({row["seed"] for row in main_rows}),
            "ratios": sorted({row["target_ratio"] for row in main_rows}),
            "prompt_templates": sorted({row["prompt_template_id"] for row in main_rows}),
            "temperatures": sorted({row["temperature"] for row in main_rows}),
            "top_p": sorted({row["top_p"] for row in main_rows}),
            "max_new_tokens": sorted({row["max_new_tokens"] for row in main_rows}),
        },
        "robustness": {
            "rows": len(robustness),
            "duplicate_keys": duplicate_count(robustness, robust_key),
            "duplicate_run_ids": len(robustness) - len(robust_ids),
            "reused_main_rows": reused,
            "new_rows": len(robustness) - reused,
            "scenario_count": len({row["scenario_id"] for row in robustness}),
            "models": dict(sorted(collections.Counter(row["model_id"] for row in robustness).items())),
            "conditions": {
                f"{prompt}|temperature={temperature}": count
                for (prompt, temperature), count in sorted(
                    collections.Counter(
                        (row["prompt_template_id"], row["temperature"])
                        for row in robustness
                    ).items()
                )
            },
        },
        "evaluations": {
            "main_rows": len(main_eval),
            "main_duplicate_run_ids": len(main_eval) - len({row["run_id"] for row in main_eval}),
            "main_missing_generation_ids": len(main_ids - {row["run_id"] for row in main_eval}),
            "robustness_rows": len(robust_eval),
            "robustness_duplicate_run_ids": len(robust_eval) - len({row["run_id"] for row in robust_eval}),
            "robustness_missing_generation_ids": len(robust_ids - {row["run_id"] for row in robust_eval}),
        },
        "embeddings": embedding_facts,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "manifests/input_audit.json", facts)
    condition_lines = [
        f"- `{prompt}` at temperature `{temperature}`: {count:,} rows"
        for (prompt, temperature), count in sorted(collections.Counter((row["prompt_template_id"], row["temperature"]) for row in robustness).items())
    ]
    selection_path = args.output_root / "manifests/external_generator_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    formal_outcome = selection.get("formal_outcome", {})
    formal_line = (
        f"- Formal GPT-6-Luna outcome: {formal_outcome.get('attempted_request_identities'):,}/6,720 request identities attempted; "
        f"{formal_outcome.get('usable_outputs'):,} usable outputs; {formal_outcome.get('content_failures')} content failures; "
        f"{formal_outcome.get('transport_failures')} transport failures; estimated cost USD {formal_outcome.get('estimated_cost_usd'):.6f}."
        if formal_outcome
        else "- Formal GPT-6-Luna generation had not yet completed when this audit was written."
    )
    report = f"""# Input audit

Generated: {facts['generated_at']}
Git commit: `{facts['git_commit']}`
Project: `{project}`
Legacy input (read only): `{legacy}`

## Scenario source

- SHA-256: `{facts['files']['scenarios']['sha256']}`
- Rows: {len(scenarios):,}; duplicate IDs: {facts['scenarios']['duplicate_scenario_ids']}
- Axes: {json.dumps(facts['scenarios']['axes'], ensure_ascii=False, sort_keys=True)}
- Atomic checks per scenario: min {min(checks)}, max {max(checks)}, missing {facts['scenarios']['missing_content_checks']}

## Main Qwen experiment

- Rows: {len(main_rows):,}; expected 33,600; duplicate keys: {facts['main']['duplicate_keys']}; missing combinations: {len(missing)}
- Models: {json.dumps(facts['main']['models'], ensure_ascii=False, sort_keys=True)}
- Ratios: {facts['main']['ratios']}; seeds: {facts['main']['seeds']}
- Prompt: {facts['main']['prompt_templates']}; temperature: {facts['main']['temperatures']}; top-p: {facts['main']['top_p']}; max new tokens: {facts['main']['max_new_tokens']}
- Reuse decision: all valid rows are reused; the complete original main matrix is not regenerated.

## Existing robustness experiment

- Rows: {len(robustness):,}; expected 9,600; scenarios: {facts['robustness']['scenario_count']}; duplicate keys: {facts['robustness']['duplicate_keys']}
- Reused main rows: {reused:,}; other-condition rows: {len(robustness)-reused:,}
- Existing models: {json.dumps(facts['robustness']['models'], ensure_ascii=False, sort_keys=True)}
{chr(10).join(condition_lines)}
- Reuse decision: reuse only exact configuration keys and verified prompt/text hashes; expand to 320 scenarios and all five Qwen models in the new output tree.

## Evaluations and embeddings

- Main fixed-Qwen evaluations: {len(main_eval):,}; missing generation IDs: {facts['evaluations']['main_missing_generation_ids']}
- Robustness fixed-Qwen evaluations: {len(robust_eval):,}; missing generation IDs: {facts['evaluations']['robustness_missing_generation_ids']}
- Embedding archives: {len(embedding_facts)} files; every archive records shape and duplicate run-ID count in `manifests/input_audit.json`.

## External API status

- The attempted `deepseek-r1` and `deepseek-v4-pro-0813` routes were blocked before inference by `missing exchange rate from null to CNY`; both returned zero usage tokens.
- The author replaced that planned condition with lightweight `gpt-6-luna`. A separate 20-output real-prompt smoke test passed 20/20 before formal generation; selection and cost evidence are in `manifests/external_generator_selection.json`.
{formal_line}
- The independent second judge is the distinct `gpt-6-sol` alias. Generator and second judge share the GPT family/provider, while the local Qwen judge supplies the cross-family evaluator comparison. Credentials are read only from `SBE_IAI_API_KEY` and are excluded from this audit.
"""
    atomic_write_text(args.output_root / "reports/input_audit.md", report)
    print(json.dumps({"main_rows": len(main_rows), "robustness_rows": len(robustness), "missing_main": len(missing), "embedding_archives": len(embedding_facts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
