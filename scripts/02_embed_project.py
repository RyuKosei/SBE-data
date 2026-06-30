from __future__ import annotations

import argparse
import json
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils import ROOT, read_jsonl, write_jsonl


def projection_ratio(vec: np.ndarray, va: np.ndarray, vb: np.ndarray) -> float:
    direction = vb - va
    denom = float(np.dot(direction, direction))
    if denom <= 1e-12:
        return 0.5
    return float(np.dot(vec - va, direction) / denom)


def hash_embed(texts: list[str], dim: int = 2048) -> np.ndarray:
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text)
        features: list[str] = []
        features.extend(tokens)
        features.extend("".join(tokens[i : i + 2]) for i in range(max(0, len(tokens) - 1)))
        features.extend("".join(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2)))
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vectors[row_idx, bucket] += sign
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def encode_texts(args: argparse.Namespace, texts: list[str]) -> tuple[np.ndarray, str]:
    if args.backend == "hash":
        return hash_embed(texts, dim=args.hash_dim), f"hash-char-ngram-{args.hash_dim}"

    if args.backend == "transformers":
        return transformers_embed(texts, args)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        if args.backend == "auto":
            print("sentence-transformers unavailable; using hash backend for pipeline smoke only.")
            return hash_embed(texts, dim=args.hash_dim), f"hash-char-ngram-{args.hash_dim}"
        raise SystemExit("sentence-transformers is required for --backend sentence-transformers.") from exc

    model = SentenceTransformer(args.embedding_model)
    vectors = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(vectors), args.embedding_model


def transformers_embed(texts: list[str], args: argparse.Namespace) -> tuple[np.ndarray, str]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers and torch are required for --backend transformers.") from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = str(args.embedding_model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(model_path, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    pooling_mode = "mean"
    pooling_config = Path(model_path) / "1_Pooling" / "config.json"
    if pooling_config.exists():
        with pooling_config.open("r", encoding="utf-8") as f:
            pooling = json.load(f)
        if pooling.get("pooling_mode_cls_token"):
            pooling_mode = "cls"

    encoded_batches = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        if pooling_mode == "cls":
            embeddings = hidden[:, 0]
        else:
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        encoded_batches.append(embeddings.cpu().numpy())

    return np.concatenate(encoded_batches, axis=0), f"{model_path} ({pooling_mode})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-file", required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--backend", choices=["auto", "sentence-transformers", "transformers", "hash"], default="auto")
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hash-dim", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N for --backend transformers.")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    args = parser.parse_args()

    rows = read_jsonl(args.generation_file)
    if not rows:
        raise SystemExit(f"No generations found: {args.generation_file}")

    texts = [row["output"] for row in rows]
    vectors, embedding_name = encode_texts(args, texts)

    by_key: dict[tuple, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = (row["model"], row["axis_id"], row["scenario_id"], row["seed"], row["prompt_template"])
        by_key[key].append(idx)

    out_rows = []
    for key, indices in by_key.items():
        endpoint_a = [i for i in indices if float(rows[i]["alpha"]) == 0.0]
        endpoint_b = [i for i in indices if float(rows[i]["alpha"]) == 1.0]
        if not endpoint_a or not endpoint_b:
            continue
        va = vectors[endpoint_a[0]]
        vb = vectors[endpoint_b[0]]
        direction = vb - va
        endpoint_distance = float(np.linalg.norm(direction))
        group_vectors = vectors[indices]
        for i in indices:
            raw = projection_ratio(vectors[i], va, vb)
            alpha = float(rows[i]["alpha"])
            target = (1.0 - alpha) * va + alpha * vb
            interpolation_distance = float(np.linalg.norm(vectors[i] - target))
            off_axis = float(np.linalg.norm((vectors[i] - va) - raw * direction))
            distances = np.linalg.norm(group_vectors - target, axis=1)
            rank = int(1 + np.sum(distances < np.linalg.norm(vectors[i] - target)))
            out_rows.append(
                {
                    "run_id": rows[i]["run_id"],
                    "model": rows[i]["model"],
                    "axis_id": rows[i]["axis_id"],
                    "scenario_id": rows[i]["scenario_id"],
                    "seed": rows[i]["seed"],
                    "alpha": rows[i]["alpha"],
                    "embedding_model": embedding_name,
                    "embedding_backend": args.backend,
                    "endpoint_a_run_id": rows[endpoint_a[0]]["run_id"],
                    "endpoint_b_run_id": rows[endpoint_b[0]]["run_id"],
                    "endpoint_distance": endpoint_distance,
                    "projection_ratio_raw": raw,
                    "projection_ratio_clipped": min(1.0, max(0.0, raw)),
                    "interpolation_distance": interpolation_distance,
                    "off_axis_drift": off_axis,
                    "nearest_neighbor_rank": rank,
                }
            )

    if args.out is None:
        stem = Path(args.generation_file).stem
        args.out = ROOT / f"data/metrics/projection_{stem}.jsonl"
    write_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} projection rows to {args.out}")


if __name__ == "__main__":
    main()
