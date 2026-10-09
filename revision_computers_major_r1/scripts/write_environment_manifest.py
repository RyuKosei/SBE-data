from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from common import atomic_write_json, sha256_file, utc_now


PACKAGES = ["openai", "torch", "transformers", "numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "sglang"]


def command(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu_csv = command(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"], args.project_root)
    critical = []
    for relative in [
        "config/experiment.yaml", "config/api_capabilities.yaml", "config/encoder_registry.csv",
        "data/robustness_outputs.jsonl", "data/robustness_condition_records.jsonl",
        "data/gpt_fixed_references.jsonl", "data/external_main_outputs.jsonl", "data/deepseek_main_outputs.jsonl",
        "data/evaluation_index.jsonl", "metrics/qwen_judge_evaluations.jsonl",
        "metrics/gpt_judge_evaluations.jsonl",
    ]:
        path = args.output_root / relative
        if path.exists():
            critical.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "generated_at": utc_now(),
        "project_root": str(args.project_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "git_commit": command(["git", "rev-parse", "HEAD"], args.project_root),
        "git_branch": command(["git", "branch", "--show-current"], args.project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "gpu_inventory": gpu_csv.splitlines() if gpu_csv else [],
        "critical_file_hashes": critical,
        "credential_values_recorded": False,
    }
    atomic_write_json(args.output_root / "manifests/environment_and_hashes.json", manifest)
    print(json.dumps({"git_commit": manifest["git_commit"], "critical_files": len(critical), "gpus": len(manifest["gpu_inventory"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
