from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REVISION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REVISION_ROOT.parent


@dataclass(frozen=True)
class JsonlReadResult:
    rows: list[dict[str, Any]]
    bad_lines: list[dict[str, Any]]
    blank_lines: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def read_jsonl_tolerant(path: str | Path) -> JsonlReadResult:
    rows: list[dict[str, Any]] = []
    bad_lines: list[dict[str, Any]] = []
    blank_lines = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                blank_lines += 1
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                bad_lines.append(
                    {
                        "line_number": line_number,
                        "error": str(exc),
                        "preview": line[:200],
                    }
                )
                continue
            if not isinstance(value, dict):
                bad_lines.append(
                    {
                        "line_number": line_number,
                        "error": "JSON value is not an object",
                        "preview": line[:200],
                    }
                )
                continue
            rows.append(value)
    return JsonlReadResult(rows=rows, bad_lines=bad_lines, blank_lines=blank_lines)


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, payload)


def atomic_write_csv(
    path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def environment_snapshot() -> dict[str, Any]:
    package_names = (
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "torch",
        "transformers",
        "sglang",
        "openai",
        "PyYAML",
    )
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    try:
        gpu_text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        gpus = [line.strip() for line in gpu_text.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        gpus = []
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": bool(
            packages.get("torch") != "not_installed" and _torch_cuda_available()
        ),
        "gpus": gpus,
    }


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def new_run_manifest(
    *,
    run_id: str,
    command: list[str],
    config_paths: Iterable[str | Path],
    data_paths: Iterable[str | Path],
    checkpoint_name: str | None,
    checkpoint_path: str | Path | None,
    seeds: Iterable[int],
    output_paths: Iterable[str | Path],
) -> dict[str, Any]:
    def hashes(paths: Iterable[str | Path]) -> dict[str, str]:
        output: dict[str, str] = {}
        for raw_path in paths:
            path = Path(raw_path)
            output[str(path)] = sha256_file(path) if path.is_file() else "missing"
        return output

    return {
        "run_id": run_id,
        "git_commit": git_commit(),
        "command": command,
        "config_file_hashes": hashes(config_paths),
        "data_file_hashes": hashes(data_paths),
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "environment": environment_snapshot(),
        "seeds": list(seeds),
        "start_time": utc_now(),
        "end_time": None,
        "status": "running",
        "output_paths": [str(path) for path in output_paths],
        "error": None,
    }


def finish_run_manifest(
    path: str | Path,
    manifest: dict[str, Any],
    *,
    status: str,
    error: str | None = None,
    counts: dict[str, int | float] | None = None,
) -> None:
    updated = dict(manifest)
    updated["end_time"] = utc_now()
    updated["status"] = status
    updated["error"] = error
    if counts is not None:
        updated["counts"] = counts
    atomic_write_json(path, updated)
