"""Compute sample and trajectory metrics for one model/encoder pair."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from baselines import constant_half, endpoint_distance_ratio  # type: ignore
    from common import REVISION_ROOT, read_jsonl_tolerant  # type: ignore
    from geometry_metrics import (  # type: ignore
        compute_geometry,
        leave_one_seed_out_diagnostics,
        summarize_interior,
        trajectory_diagnostics,
    )
else:
    from .baselines import constant_half, endpoint_distance_ratio
    from .common import REVISION_ROOT, read_jsonl_tolerant
    from .geometry_metrics import (
        compute_geometry,
        leave_one_seed_out_diagnostics,
        summarize_interior,
        trajectory_diagnostics,
    )


def chinese_character_count(text: str) -> int:
    return sum("\u4e00" <= character <= "\u9fff" for character in text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument("--generation-file", type=Path)
    parser.add_argument("--embedding-file", type=Path)
    parser.add_argument("--sample-out", type=Path)
    parser.add_argument("--trajectory-out", type=Path)
    parser.add_argument("--loso-out", type=Path)
    args = parser.parse_args()
    args.generation_file = args.generation_file or REVISION_ROOT / "data/fragments/main" / f"{args.model_id}.jsonl"
    args.embedding_file = args.embedding_file or REVISION_ROOT / "data/embeddings" / args.encoder_id / f"{args.model_id}.npz"
    metric_root = REVISION_ROOT / "metrics/fragments" / args.encoder_id
    args.sample_out = args.sample_out or metric_root / f"{args.model_id}.sample.parquet"
    args.trajectory_out = args.trajectory_out or metric_root / f"{args.model_id}.trajectory.parquet"
    args.loso_out = args.loso_out or metric_root / f"{args.model_id}.loso.parquet"

    generations = read_jsonl_tolerant(args.generation_file)
    if generations.bad_lines or not generations.rows:
        raise ValueError("generation file is empty or malformed")
    archive = np.load(args.embedding_file)
    vectors = np.asarray(archive["vectors"], dtype=np.float64)
    embedding_ids = [str(value) for value in archive["run_ids"]]
    if len(vectors) != len(generations.rows):
        raise ValueError("generation and embedding row counts differ")
    vector_by_id = dict(zip(embedding_ids, vectors))
    if len(vector_by_id) != len(embedding_ids):
        raise ValueError("embedding archive has duplicate run_ids")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in generations.rows:
        key = (row["scenario_id"], row["style_axis"], row["model_id"], row["seed"], row["prompt_template_id"])
        groups[key].append(row)
    sample_records: list[dict[str, Any]] = []
    trajectory_records: list[dict[str, Any]] = []
    seed_matrices: dict[tuple[Any, ...], dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: float(row["target_ratio"]))
        ratios = np.asarray([float(row["target_ratio"]) for row in rows])
        matrix = np.stack([vector_by_id[row["run_id"]] for row in rows])
        if len(rows) != 7:
            raise ValueError(f"incomplete seven-point trajectory: {key}")
        geometry = compute_geometry(matrix, ratios)
        if not geometry.endpoint_unstable and geometry.decomposition_residual.max() >= 1e-6:
            raise AssertionError(f"projection decomposition residual >= 1e-6: {key}")
        endpoint_a_index = int(np.flatnonzero(np.isclose(ratios, 0.0))[0])
        endpoint_b_index = int(np.flatnonzero(np.isclose(ratios, 1.0))[0])
        distance_baseline = endpoint_distance_ratio(matrix, matrix[endpoint_a_index], matrix[endpoint_b_index])
        lengths = np.asarray([len(str(row["generated_text"]).strip()) for row in rows], dtype=np.float64)
        length_delta = lengths[endpoint_b_index] - lengths[endpoint_a_index]
        length_coordinate = np.full(7, 0.5) if abs(length_delta) <= 1e-12 else (lengths - lengths[endpoint_a_index]) / length_delta
        for index, row in enumerate(rows):
            sample_records.append(
                {
                    "run_id": row["run_id"],
                    "scenario_id": row["scenario_id"],
                    "style_axis": row["style_axis"],
                    "model_id": row["model_id"],
                    "family_version": row["family_version"],
                    "architecture": row["architecture"],
                    "seed": int(row["seed"]),
                    "target_ratio": float(ratios[index]),
                    "is_interior": bool(0 < ratios[index] < 1),
                    "encoder_id": args.encoder_id,
                    "anchor_mode": "self",
                    "projection_raw": geometry.projection_raw[index],
                    "projection_clipped": geometry.projection_clipped[index],
                    "projection_error_raw": geometry.projection_error_raw[index],
                    "projection_error_clipped": geometry.projection_error_clipped[index],
                    "off_axis_drift_normalized": geometry.off_axis_drift_normalized[index],
                    "interpolation_distance_normalized": geometry.interpolation_distance_normalized[index],
                    "out_of_range": bool(geometry.out_of_range[index]),
                    "nearest_neighbor_rank": geometry.nearest_neighbor_rank[index],
                    "nearest_neighbor_rank_normalized": geometry.nearest_neighbor_rank_normalized[index],
                    "decomposition_residual": geometry.decomposition_residual[index],
                    "endpoint_separation": geometry.endpoint_separation,
                    "endpoint_unstable": geometry.endpoint_unstable,
                    "constant_half_estimate": constant_half(7)[index],
                    "endpoint_distance_ratio_estimate": distance_baseline[index],
                    "character_length": int(lengths[index]),
                    "chinese_character_count": chinese_character_count(str(row["generated_text"])),
                    "server_output_tokens": int(row.get("output_tokens", -1)),
                    "length_coordinate": length_coordinate[index],
                    "endpoint_character_length_a": int(lengths[endpoint_a_index]),
                    "endpoint_character_length_b": int(lengths[endpoint_b_index]),
                }
            )
        summary = summarize_interior(geometry)
        trajectory = trajectory_diagnostics(matrix, ratios)
        trajectory_records.append(
            {
                "scenario_id": key[0],
                "style_axis": key[1],
                "model_id": key[2],
                "seed": int(key[3]),
                "prompt_template_id": key[4],
                "encoder_id": args.encoder_id,
                "anchor_mode": "self",
                **summary,
                **{f"trajectory_{name}": value for name, value in trajectory.items() if name not in {"endpoint_unstable", "endpoint_separation"}},
            }
        )
        seed_key = (key[0], key[1], key[2], key[4])
        seed_matrices[seed_key][int(key[3])] = (ratios, matrix)

    loso_records: list[dict[str, Any]] = []
    for key, by_seed in sorted(seed_matrices.items()):
        if len(by_seed) != 3:
            raise ValueError(f"LOSO requires exactly three seeds: {key}")
        reference_ratios = next(iter(by_seed.values()))[0]
        matrices = {seed: value[1] for seed, value in by_seed.items()}
        for record in leave_one_seed_out_diagnostics(matrices, reference_ratios):
            loso_records.append(
                {
                    "scenario_id": key[0],
                    "style_axis": key[1],
                    "model_id": key[2],
                    "prompt_template_id": key[3],
                    "encoder_id": args.encoder_id,
                    **record,
                }
            )

    for path in (args.sample_out, args.trajectory_out, args.loso_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_records).to_parquet(args.sample_out, index=False)
    pd.DataFrame(trajectory_records).to_parquet(args.trajectory_out, index=False)
    pd.DataFrame(loso_records).to_parquet(args.loso_out, index=False)
    print(json.dumps({"sample_rows": len(sample_records), "trajectory_rows": len(trajectory_records), "loso_rows": len(loso_records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
