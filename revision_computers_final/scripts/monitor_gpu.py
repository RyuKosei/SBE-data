"""Append GPU utilization and memory telemetry until interrupted."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "timestamp_utc",
    "gpu_index",
    "gpu_name",
    "memory_used_mib",
    "memory_total_mib",
    "utilization_gpu_percent",
    "temperature_c",
    "power_draw_w",
]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sample() -> list[list[str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [[value.strip() for value in line.split(",")] for line in output.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists() or args.out.stat().st_size == 0
    with args.out.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(FIELDS)
        try:
            while True:
                now = timestamp()
                for row in sample():
                    writer.writerow([now, *row])
                handle.flush()
                time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
