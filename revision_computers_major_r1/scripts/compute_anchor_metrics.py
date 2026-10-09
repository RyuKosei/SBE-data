"""Compute self, external-reference, and pooled-Qwen geometry diagnostics.

The script never mutates legacy results.  It reuses the frozen legacy Qwen
embeddings and joins them to new embeddings by stable output identity.  The
legacy epsilon convention is retained: collapsed anchors are declared at
L <= 1e-12 and normalized distances divide by L + epsilon.  Exact-L versions
are also saved so the numerical effect of that convention is auditable.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, spearmanr

from common import atomic_write_json, read_jsonl, utc_now


EPSILON = 1.0e-12
INTERIOR = {1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6}


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        id_key = "ids" if "ids" in archive.files else "run_ids"
        ids = [str(value) for value in archive[id_key].tolist()]
        vectors = np.asarray(archive["vectors"], dtype=np.float64)
    if len(ids) != len(vectors) or len(ids) != len(set(ids)):
        raise ValueError(f"invalid embedding archive identity mapping: {path}")
    return dict(zip(ids, vectors))


def fixed_geometry(vector: np.ndarray, alpha: float, anchor_a: np.ndarray, anchor_b: np.ndarray) -> dict[str, Any]:
    delta = anchor_b - anchor_a
    separation = float(np.linalg.norm(delta))
    if separation <= EPSILON:
        return {
            "projection_raw": np.nan,
            "projection_clipped": np.nan,
            "projection_error_raw": np.nan,
            "projection_error_clipped": np.nan,
            "off_axis_drift_normalized": np.nan,
            "off_axis_drift_normalized_exact": np.nan,
            "interpolation_distance_normalized": np.nan,
            "interpolation_distance_normalized_exact": np.nan,
            "out_of_range": None,
            "endpoint_separation": separation,
            "endpoint_unstable": True,
        }
    displacement = vector - anchor_a
    coordinate = float(displacement @ delta / float(delta @ delta))
    clipped = float(np.clip(coordinate, 0.0, 1.0))
    orthogonal = displacement - coordinate * delta
    ideal = anchor_a + float(alpha) * delta
    drift_norm = float(np.linalg.norm(orthogonal))
    interpolation_norm = float(np.linalg.norm(vector - ideal))
    return {
        "projection_raw": coordinate,
        "projection_clipped": clipped,
        "projection_error_raw": abs(coordinate - float(alpha)),
        "projection_error_clipped": abs(clipped - float(alpha)),
        "off_axis_drift_normalized": drift_norm / (separation + EPSILON),
        "off_axis_drift_normalized_exact": drift_norm / separation,
        "interpolation_distance_normalized": interpolation_norm / (separation + EPSILON),
        "interpolation_distance_normalized_exact": interpolation_norm / separation,
        "out_of_range": bool(coordinate < 0.0 or coordinate > 1.0),
        "endpoint_separation": separation,
        "endpoint_unstable": False,
    }


def trajectory_ranks(vectors: list[np.ndarray], ratios: list[float], anchor_a: np.ndarray, anchor_b: np.ndarray) -> list[tuple[float, float]]:
    """Return 1-based midrank and (rank-1)/6 for each point in a 7-point track."""

    if len(vectors) != 7 or len(ratios) != 7:
        return [(np.nan, np.nan)] * len(vectors)
    matrix = np.stack(vectors)
    delta = anchor_b - anchor_a
    if float(np.linalg.norm(delta)) <= EPSILON:
        return [(np.nan, np.nan)] * 7
    output = []
    for index, alpha in enumerate(ratios):
        ideal = anchor_a + float(alpha) * delta
        distances = np.linalg.norm(matrix - ideal, axis=1)
        rank = float(rankdata(distances, method="average")[index])
        output.append((rank, (rank - 1.0) / 6.0))
    return output


def character_count(text: str) -> int:
    return len(str(text).strip())


def chinese_character_count(text: str) -> int:
    return sum("\u4e00" <= char <= "\u9fff" for char in str(text))


def model_rows(config: dict[str, Any], encoder_id: str, output_root: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    legacy = Path(config["legacy_root"])
    all_rows = read_jsonl(legacy / "data/qwen_family_outputs.jsonl")
    vector_by_id: dict[str, np.ndarray] = {}
    for model_id in config["main"]["qwen_models"]:
        vector_by_id.update(load_archive(legacy / "data/embeddings" / encoder_id / f"{model_id}.npz"))
    external_path = output_root / "data/external_main_outputs.jsonl"
    external_embedding = output_root / "embeddings" / encoder_id / "external_main.npz"
    if external_path.exists() and external_embedding.exists():
        external = [row for row in read_jsonl(external_path) if row.get("status") in {"ok", "truncated"} and row.get("generated_text")]
        all_rows.extend(external)
        vector_by_id.update(load_archive(external_embedding))
    return all_rows, vector_by_id


def pooled_anchors(config: dict[str, Any], rows: list[dict[str, Any]], vector_by_id: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    qwen = set(config["main"]["qwen_models"])
    contributors: dict[tuple[str, float], list[np.ndarray]] = collections.defaultdict(list)
    for row in rows:
        alpha = float(row["target_ratio"])
        if row["model_id"] in qwen and alpha in (0.0, 1.0):
            contributors[(row["scenario_id"], alpha)].append(vector_by_id[row["run_id"]])
    output = {}
    for scenario in read_jsonl(config["scenario_file"]):
        scenario_id = scenario["scenario_id"]
        a = contributors[(scenario_id, 0.0)]
        b = contributors[(scenario_id, 1.0)]
        if len(a) != 15 or len(b) != 15:
            raise ValueError(f"pooled anchors require 15 contributors per endpoint: {scenario_id}, {len(a)}, {len(b)}")
        # The contributors are already normalized individual embeddings.  The
        # arithmetic centroids are intentionally not normalized again.
        output[scenario_id] = (np.mean(np.stack(a), axis=0), np.mean(np.stack(b), axis=0))
    return output


def external_anchors(output_root: Path, encoder_id: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    references = read_jsonl(output_root / "data/gpt_fixed_references.jsonl")
    vectors = load_archive(output_root / "embeddings" / encoder_id / "gpt_references.npz")
    grouped: dict[str, dict[str, np.ndarray]] = collections.defaultdict(dict)
    for row in references:
        grouped[row["scenario_id"]][row["endpoint"]] = vectors[row["reference_id"]]
    output = {}
    for scenario_id, endpoints in grouped.items():
        if set(endpoints) != {"A", "B"}:
            raise ValueError(f"incomplete external reference pair: {scenario_id}")
        output[scenario_id] = (endpoints["A"], endpoints["B"])
    return output


def identity_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "scenario_id": row["scenario_id"],
        "style_axis": row["style_axis"],
        "model_id": row["model_id"],
        "replicate": int(row.get("seed") or row.get("replicate_id")),
        "seed": row.get("seed"),
        "replicate_id": row.get("replicate_id"),
        "target_ratio": float(row["target_ratio"]),
        "prompt_template_id": row["prompt_template_id"],
        "character_length": character_count(row["generated_text"]),
        "chinese_character_count": chinese_character_count(row["generated_text"]),
    }


def compute_main(config: dict[str, Any], encoder_id: str, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, vectors = model_rows(config, encoder_id, output_root)
    pooled = pooled_anchors(config, rows, vectors)
    external = external_anchors(output_root, encoder_id)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        replicate = int(row.get("seed") or row.get("replicate_id"))
        groups[(row["scenario_id"], row["style_axis"], row["model_id"], replicate)].append(row)
    samples: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda row: float(row["target_ratio"]))
        complete_trajectory = len(group) == 7
        # Preserve partial API trajectories.  Fixed-reference coordinates and
        # a self axis (when both endpoints exist) remain computable, but the
        # registered nearest-target rank is specifically a seven-candidate
        # quantity and therefore must be missing for an incomplete trajectory.
        group_vectors = [vectors[row["run_id"]] for row in group]
        ratios = [float(row["target_ratio"]) for row in group]
        endpoint_lookup = {float(row["target_ratio"]): vector for row, vector in zip(group, group_vectors)}
        anchors: list[tuple[str, np.ndarray, np.ndarray]] = [
            ("external_fixed_reference", *external[key[0]]),
            ("fixed_pooled_qwen", *pooled[key[0]]),
        ]
        if 0.0 in endpoint_lookup and 1.0 in endpoint_lookup:
            anchors.insert(0, ("self_anchored", endpoint_lookup[0.0], endpoint_lookup[1.0]))
        for protocol, anchor_a, anchor_b in anchors:
            ranks = (
                trajectory_ranks(group_vectors, ratios, anchor_a, anchor_b)
                if complete_trajectory
                else [(np.nan, np.nan)] * len(group)
            )
            protocol_rows = []
            for row, vector, (rank, normalized_rank) in zip(group, group_vectors, ranks):
                geometry = fixed_geometry(vector, float(row["target_ratio"]), anchor_a, anchor_b)
                record = {
                    **identity_fields(row),
                    "encoder_id": encoder_id,
                    "anchor_protocol": protocol,
                    **geometry,
                    "nearest_target_rank": rank,
                    "nearest_target_rank_normalized": normalized_rank,
                }
                samples.append(record)
                protocol_rows.append(record)
            interior = [record for record in protocol_rows if any(math.isclose(record["target_ratio"], value, abs_tol=1e-12) for value in INTERIOR)]
            valid = [record for record in interior if not record["endpoint_unstable"]]
            coordinates = [record["projection_raw"] for record in valid]
            targets = [record["target_ratio"] for record in valid]
            rank_values = [record["nearest_target_rank_normalized"] for record in valid]
            finite_rank_values = [value for value in rank_values if not pd.isna(value)]
            monotonicity = float(spearmanr(targets, coordinates).statistic) if len(valid) >= 2 and len(set(coordinates)) > 1 else np.nan
            trajectories.append({
                "scenario_id": key[0],
                "style_axis": key[1],
                "model_id": key[2],
                "replicate": key[3],
                "encoder_id": encoder_id,
                "anchor_protocol": protocol,
                "expected_interior_points": 5,
                "valid_interior_points": len(valid),
                "complete_seven_point_trajectory": complete_trajectory,
                "endpoint_separation": float(np.linalg.norm(anchor_b - anchor_a)),
                "endpoint_unstable": float(np.linalg.norm(anchor_b - anchor_a)) <= EPSILON,
                "interior_calibration_error": float(np.mean([record["projection_error_raw"] for record in valid])) if valid else np.nan,
                "interior_calibration_error_clipped": float(np.mean([record["projection_error_clipped"] for record in valid])) if valid else np.nan,
                "normalized_off_axis_drift_mean": float(np.mean([record["off_axis_drift_normalized"] for record in valid])) if valid else np.nan,
                "interpolation_distance_mean": float(np.mean([record["interpolation_distance_normalized"] for record in valid])) if valid else np.nan,
                "interpolation_distance_rms": float(np.sqrt(np.mean(np.square([record["interpolation_distance_normalized"] for record in valid])))) if valid else np.nan,
                "out_of_range_rate": float(np.mean([record["out_of_range"] for record in valid])) if valid else np.nan,
                "nearest_target_rank_normalized_mean": float(np.mean(finite_rank_values)) if finite_rank_values else np.nan,
                "projection_spearman_internal": monotonicity,
            })
    return pd.DataFrame(samples), pd.DataFrame(trajectories)


def compute_robustness(config: dict[str, Any], encoder_id: str, output_root: Path) -> pd.DataFrame:
    path = output_root / "data/robustness_outputs.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = read_jsonl(path)
    vectors = load_archive(output_root / "embeddings" / encoder_id / "robustness.npz")
    main_rows, main_vectors = model_rows(config, encoder_id, output_root)
    pooled = pooled_anchors(config, main_rows, main_vectors)
    external = external_anchors(output_root, encoder_id)
    self_endpoints: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    endpoint_groups: dict[tuple[str, str, int], dict[float, np.ndarray]] = collections.defaultdict(dict)
    for row in main_rows:
        if row["model_id"] not in config["main"]["qwen_models"]:
            continue
        alpha = float(row["target_ratio"])
        if alpha in (0.0, 1.0):
            endpoint_groups[(row["scenario_id"], row["model_id"], int(row["seed"]))][alpha] = main_vectors[row["run_id"]]
    for key, endpoints in endpoint_groups.items():
        if set(endpoints) == {0.0, 1.0}:
            self_endpoints[key] = (endpoints[0.0], endpoints[1.0])
    samples = []
    for row in rows:
        vector = vectors[row["run_id"]]
        anchors = [
            ("external_fixed_reference", *external[row["scenario_id"]]),
            ("fixed_pooled_qwen", *pooled[row["scenario_id"]]),
            ("baseline_self_reference", *self_endpoints[(row["scenario_id"], row["model_id"], int(row["seed"]))]),
        ]
        for protocol, anchor_a, anchor_b in anchors:
            samples.append({
                **identity_fields(row),
                "encoder_id": encoder_id,
                "anchor_protocol": protocol,
                "robustness_condition_id": row["robustness_condition_id"],
                "temperature": float(row["temperature"]),
                "robustness_source": row.get("robustness_source"),
                **fixed_geometry(vector, float(row["target_ratio"]), anchor_a, anchor_b),
            })
    return pd.DataFrame(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--encoder-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sample, trajectory = compute_main(config, args.encoder_id, args.output_root)
    robustness = compute_robustness(config, args.encoder_id, args.output_root)
    metric_dir = args.output_root / "metrics" / "geometry" / args.encoder_id
    metric_dir.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(metric_dir / "main_sample.parquet", index=False)
    trajectory.to_parquet(metric_dir / "main_trajectory.parquet", index=False)
    if not robustness.empty:
        robustness.to_parquet(metric_dir / "robustness_sample.parquet", index=False)
    exact_columns = [
        "off_axis_drift_normalized",
        "off_axis_drift_normalized_exact",
        "interpolation_distance_normalized",
        "interpolation_distance_normalized_exact",
    ]
    epsilon_max_difference = {
        "drift": float((sample[exact_columns[0]] - sample[exact_columns[1]]).abs().max()),
        "interpolation_distance": float((sample[exact_columns[2]] - sample[exact_columns[3]]).abs().max()),
    }
    status = {
        "generated_at": utc_now(),
        "encoder_id": args.encoder_id,
        "epsilon": EPSILON,
        "normalization_denominator": "endpoint_separation + epsilon (legacy convention)",
        "exact_denominator_sensitivity_saved": True,
        "epsilon_max_absolute_difference_main": epsilon_max_difference,
        "main_sample_rows": len(sample),
        "main_trajectory_rows": len(trajectory),
        "robustness_sample_rows": len(robustness),
        "main_models": sorted(sample.model_id.unique().tolist()),
        "collapsed_anchor_counts_main": sample.groupby("anchor_protocol").endpoint_unstable.sum().astype(int).to_dict(),
    }
    atomic_write_json(args.output_root / "manifests" / f"geometry_{args.encoder_id}.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
