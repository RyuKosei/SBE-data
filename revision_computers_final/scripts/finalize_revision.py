"""Run all deterministic final analyses under one auditable manifest."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, atomic_write_json, atomic_write_text, finish_run_manifest, new_run_manifest  # type: ignore
else:
    from .common import REVISION_ROOT, atomic_write_json, atomic_write_text, finish_run_manifest, new_run_manifest


COMMANDS = [
    [sys.executable, "-m", "pytest", "-q", str(REVISION_ROOT / "tests")],
    [sys.executable, str(REVISION_ROOT / "scripts/audit_data.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/consolidate_generations.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/consolidate_robustness.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/aggregate_metrics.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/run_ablations.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/analyze_baselines_detailed.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/run_statistical_analysis.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/analyze_content_quality.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/analyze_self_anchors.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/analyze_shared_anchors.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/classify_failures.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/analyze_robustness.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/analyze_human_eval.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/analyze_efficiency.py")],
    [sys.executable, str(REVISION_ROOT / "scripts/make_tables_and_figures.py")],
]


def main() -> None:
    run_id = f"final_analysis_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    manifest_path = REVISION_ROOT / "manifests" / f"run_manifest_{run_id}.json"
    log_path = REVISION_ROOT / "logs/final_analysis.log"
    configs = sorted((REVISION_ROOT / "config").glob("*"))
    data = [
        REVISION_ROOT / "data/qwen_family_outputs.jsonl",
        REVISION_ROOT / "data/robustness_outputs.jsonl",
        REVISION_ROOT / "metrics/llm_evaluations.jsonl",
        REVISION_ROOT / "metrics/robustness_llm_evaluations.jsonl",
        REVISION_ROOT / "data/shared_anchors_final.csv",
        REVISION_ROOT / "human_eval/human_results.csv",
    ]
    outputs = [
        REVISION_ROOT / "metrics/model_level_summary.csv",
        REVISION_ROOT / "statistics/bootstrap_results.csv",
        REVISION_ROOT / "statistics/paired_tests.csv",
        REVISION_ROOT / "statistics/robustness_factorial_effects.csv",
        REVISION_ROOT / "tables",
        REVISION_ROOT / "figures",
    ]
    manifest = new_run_manifest(
        run_id=run_id,
        command=sys.argv,
        config_paths=configs,
        data_paths=data,
        checkpoint_name=None,
        checkpoint_path=None,
        seeds=[1, 2, 3, 20260915, 20260916, 20260917],
        output_paths=outputs,
    )
    atomic_write_json(manifest_path, manifest)
    log_lines = []
    try:
        for command in COMMANDS:
            completed = subprocess.run(command, cwd=REVISION_ROOT.parent, text=True, capture_output=True)
            log_lines.extend(["$ " + " ".join(command), completed.stdout, completed.stderr])
            if completed.returncode:
                raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")
        atomic_write_text(log_path, "\n".join(log_lines))
    except BaseException as exc:
        atomic_write_text(log_path, "\n".join(log_lines))
        finish_run_manifest(manifest_path, manifest, status="failed", error=repr(exc))
        raise
    finish_run_manifest(manifest_path, manifest, status="success", counts={"commands": len(COMMANDS)})
    print(f"final analysis completed: {manifest_path}")


if __name__ == "__main__":
    main()
