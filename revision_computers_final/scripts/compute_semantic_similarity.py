"""Compute generated-to-source cosine similarity with each frozen encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, read_jsonl_tolerant  # type: ignore
    from embed_outputs import encode, load_encoder  # type: ignore
else:
    from .common import REVISION_ROOT, read_jsonl_tolerant
    from .embed_outputs import encode, load_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument("--encoder-registry", type=Path, default=REVISION_ROOT / "config/encoder_registry.csv")
    parser.add_argument("--generation", type=Path, default=REVISION_ROOT / "data/qwen_family_outputs.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or REVISION_ROOT / "metrics/semantic_similarity" / f"{args.encoder_id}.parquet"
    source = read_jsonl_tolerant(args.generation)
    if source.bad_lines or len(source.rows) != 33600:
        raise ValueError("expected 33,600 valid main outputs")
    scenario_text = {}
    for row in source.rows:
        previous = scenario_text.setdefault(row["scenario_id"], row["input_text"])
        if previous != row["input_text"]:
            raise ValueError(f"inconsistent source text: {row['scenario_id']}")
    scenario_ids = sorted(scenario_text)
    encoder = load_encoder(args.encoder_registry, args.encoder_id)
    source_vectors = encode(
        SimpleNamespace(device=args.device, batch_size=args.batch_size, encoder_id=args.encoder_id),
        encoder,
        [scenario_text[scenario] for scenario in scenario_ids],
    )
    source_by_scenario = dict(zip(scenario_ids, source_vectors))
    records: list[dict[str, object]] = []
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in source.rows:
        by_model.setdefault(str(row["model_id"]), []).append(row)
    for model_id, rows in sorted(by_model.items()):
        archive_path = REVISION_ROOT / "data/embeddings" / args.encoder_id / f"{model_id}.npz"
        archive = np.load(archive_path)
        generated_by_id = dict(zip(map(str, archive["run_ids"]), np.asarray(archive["vectors"], dtype=np.float64)))
        for row in rows:
            generated = generated_by_id[row["run_id"]]
            original = source_by_scenario[row["scenario_id"]]
            denominator = float(np.linalg.norm(generated) * np.linalg.norm(original))
            similarity = float(generated @ original / denominator) if denominator else np.nan
            records.append(
                {
                    "run_id": row["run_id"],
                    "scenario_id": row["scenario_id"],
                    "style_axis": row["style_axis"],
                    "model_id": model_id,
                    "seed": int(row["seed"]),
                    "target_ratio": float(row["target_ratio"]),
                    "encoder_id": args.encoder_id,
                    "source_semantic_cosine_similarity": similarity,
                }
            )
    frame = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(json.dumps({"rows": len(frame), "encoder_id": args.encoder_id, "mean_similarity": float(frame.source_semantic_cosine_similarity.mean())}))


if __name__ == "__main__":
    main()
