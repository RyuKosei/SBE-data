"""Compute shared-anchor metrics after the independent endpoints are approved."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, read_jsonl_tolerant  # type: ignore
    from embed_outputs import encode, load_encoder  # type: ignore
else:
    from .common import REVISION_ROOT, read_jsonl_tolerant
    from .embed_outputs import encode, load_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    anchors = pd.read_csv(REVISION_ROOT / "data/shared_anchors_final.csv")
    status_path = REVISION_ROOT / "statistics/shared_anchor_status.csv"
    approval = anchors.approval_status.astype(str).eq("approved")
    texts_present = anchors.reference_text_A.fillna("").str.strip().ne("") & anchors.reference_text_B.fillna("").str.strip().ne("")
    if not (approval & texts_present).all():
        pd.DataFrame(
            [{
                "status": "external_pending",
                "approved_pairs": int((approval & texts_present).sum()),
                "required_pairs": len(anchors),
                "note": "No shared-anchor statistic was computed from blank or unapproved reference text.",
            }]
        ).to_csv(status_path, index=False)
        print(json.dumps({"status": "external_pending", "approved_pairs": int((approval & texts_present).sum())}))
        return
    generation = pd.DataFrame(read_jsonl_tolerant(REVISION_ROOT / "data/qwen_family_outputs.jsonl").rows)
    generation = generation[generation.scenario_id.isin(anchors.scenario_id)].copy()
    encoder_ids = [row["encoder_id"] for row in pd.read_csv(REVISION_ROOT / "config/encoder_registry.csv").to_dict("records")]
    shared_records = []
    for encoder_id in encoder_ids:
        encoder = load_encoder(REVISION_ROOT / "config/encoder_registry.csv", encoder_id)
        reference_texts = anchors.reference_text_A.tolist() + anchors.reference_text_B.tolist()
        reference_vectors = encode(SimpleNamespace(device=args.device, batch_size=None, encoder_id=encoder_id), encoder, reference_texts)
        reference_a = dict(zip(anchors.scenario_id, reference_vectors[: len(anchors)]))
        reference_b = dict(zip(anchors.scenario_id, reference_vectors[len(anchors) :]))
        for model_id, model_frame in generation.groupby("model_id", sort=True):
            archive = np.load(REVISION_ROOT / "data/embeddings" / encoder_id / f"{model_id}.npz")
            vector_by_id = dict(zip(map(str, archive["run_ids"]), np.asarray(archive["vectors"], dtype=float)))
            for row in model_frame.itertuples():
                a = reference_a[row.scenario_id]
                delta = reference_b[row.scenario_id] - a
                separation = float(np.linalg.norm(delta))
                displacement = vector_by_id[row.run_id] - a
                projection = float(displacement @ delta / (delta @ delta))
                orthogonal = displacement - projection * delta
                ideal = a + float(row.target_ratio) * delta
                shared_records.append(
                    {
                        "run_id": row.run_id,
                        "scenario_id": row.scenario_id,
                        "style_axis": row.style_axis,
                        "model_id": model_id,
                        "seed": row.seed,
                        "target_ratio": row.target_ratio,
                        "encoder_id": encoder_id,
                        "anchor_mode": "shared",
                        "projection_raw": projection,
                        "projection_error_raw": abs(projection - row.target_ratio),
                        "off_axis_drift_normalized": np.linalg.norm(orthogonal) / separation,
                        "interpolation_distance_normalized": np.linalg.norm(vector_by_id[row.run_id] - ideal) / separation,
                        "out_of_range": projection < 0 or projection > 1,
                        "endpoint_separation": separation,
                    }
                )
    shared = pd.DataFrame(shared_records)
    shared.to_parquet(REVISION_ROOT / "metrics/shared_anchor_sample_level.parquet", index=False)
    internal = shared[(shared.target_ratio > 0) & (shared.target_ratio < 1)]
    shared_scenario = internal.groupby(["scenario_id", "style_axis", "model_id", "encoder_id"], as_index=False).agg(
        interior_calibration_error=("projection_error_raw", "mean"),
        normalized_off_axis_drift=("off_axis_drift_normalized", "mean"),
        interpolation_distance_mean=("interpolation_distance_normalized", "mean"),
        out_of_range_rate=("out_of_range", "mean"),
        endpoint_separation=("endpoint_separation", "mean"),
    )
    shared_scenario.to_parquet(REVISION_ROOT / "metrics/shared_anchor_scenario_level.parquet", index=False)
    self_scenario = pd.read_parquet(REVISION_ROOT / "metrics/scenario_level_metrics.parquet")
    self_scenario = self_scenario[self_scenario.scenario_id.isin(anchors.scenario_id)]
    comparisons = []
    for encoder_id in encoder_ids:
        self_model = self_scenario[self_scenario.encoder_id == encoder_id].groupby("model_id").interior_calibration_error.mean()
        shared_model = shared_scenario[shared_scenario.encoder_id == encoder_id].groupby("model_id").interior_calibration_error.mean()
        comparisons.append(
            {
                "encoder_id": encoder_id,
                "metric": "interior_calibration_error",
                "spearman_rank_correlation": spearmanr(self_model, shared_model).statistic,
                "kendall_rank_correlation": kendalltau(self_model, shared_model).statistic,
                "pairwise_rank_flips": sum(
                    np.sign(self_model[first] - self_model[second]) != np.sign(shared_model[first] - shared_model[second])
                    for index, first in enumerate(self_model.index) for second in self_model.index[index + 1 :]
                ),
            }
        )
    pd.DataFrame(comparisons).to_csv(REVISION_ROOT / "statistics/self_shared_anchor_rankings.csv", index=False)
    pd.DataFrame([{"status": "complete", "approved_pairs": len(anchors), "required_pairs": len(anchors)}]).to_csv(status_path, index=False)
    print(json.dumps({"status": "complete", "sample_rows": len(shared), "scenario_rows": len(shared_scenario)}))


if __name__ == "__main__":
    main()
