from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import atomic_write_json, utc_now


MATCHED_GROUP = "L20Z_TP2_BF16_concurrency32"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.legacy_root / "metrics/inference_efficiency.csv"
    frame = pd.read_csv(source)
    matched = frame[frame.direct_comparison_group == MATCHED_GROUP].copy()
    if len(matched) != 3:
        raise ValueError(f"expected three matched TP2/concurrency32 models, found {len(matched)}")
    tables = args.output_root / "tables"
    figures = args.output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_id", "gpu_model", "gpu_count", "tensor_parallel_size",
        "request_concurrency", "precision", "quantization", "output_count",
        "input_tokens_total", "output_tokens_total", "wall_clock_seconds",
        "system_output_tokens_per_second", "request_latency_p50_seconds",
        "request_latency_p95_seconds", "peak_gpu_memory_sum_mib",
        "direct_comparison_group",
    ]
    matched[columns].to_csv(tables / "table_efficiency_matched_local.csv", index=False)
    frame[columns].to_csv(tables / "table_efficiency_all_deployments_descriptive.csv", index=False)
    (tables / "table_efficiency_matched_local.md").write_text(
        matched[columns].to_markdown(index=False) + "\n\n"
        "Only these runs share NVIDIA L20Z ×2, TP=2, BF16, no quantization, and request concurrency 32.\n",
        encoding="utf-8",
    )
    descriptive = frame[frame.direct_comparison_group != MATCHED_GROUP]
    (tables / "table_efficiency_unmatched_descriptive.md").write_text(
        descriptive[columns].to_markdown(index=False) + "\n\n"
        "These deployments are descriptive only and are excluded from the matched-hardware comparison figure.\n",
        encoding="utf-8",
    )

    matched = matched.sort_values("system_output_tokens_per_second")
    labels = [value.replace("_", "\n") for value in matched.model_id]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, matched.system_output_tokens_per_second, color="#4C78A8")
    ax.set_ylabel("System output tokens / second")
    ax.set_title("Matched local inference configuration (2× L20Z, TP=2, concurrency=32)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, matched.system_output_tokens_per_second):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "figure_efficiency_matched_local.png", dpi=300)
    fig.savefig(figures / "figure_efficiency_matched_local.pdf")
    plt.close(fig)
    status = {
        "generated_at": utc_now(),
        "source": str(source),
        "matched_group": MATCHED_GROUP,
        "matched_models": matched.model_id.tolist(),
        "excluded_from_comparative_figure": descriptive.model_id.tolist(),
        "api_models_in_local_figure": False,
        "benchmark_rerun": False,
    }
    atomic_write_json(args.output_root / "manifests/efficiency_audit.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
