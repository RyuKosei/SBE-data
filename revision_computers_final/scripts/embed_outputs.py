"""Encode one model fragment with one frozen sentence encoder."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        REVISION_ROOT,
        atomic_write_json,
        finish_run_manifest,
        new_run_manifest,
        read_jsonl_tolerant,
        sha256_file,
    )
else:
    from .common import (
        REVISION_ROOT,
        atomic_write_json,
        finish_run_manifest,
        new_run_manifest,
        read_jsonl_tolerant,
        sha256_file,
    )


def load_encoder(path: Path, encoder_id: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        registry = {row["encoder_id"]: row for row in csv.DictReader(handle)}
    if encoder_id not in registry:
        raise ValueError(f"unknown encoder_id: {encoder_id}")
    return registry[encoder_id]


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def encode(args: argparse.Namespace, encoder: dict[str, str], texts: list[str]) -> np.ndarray:
    checkpoint = encoder["checkpoint_path"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(checkpoint, local_files_only=True)
    model.to(args.device)
    model.eval()
    batch_size = args.batch_size or int(encoder["batch_size"])
    maximum = int(encoder["max_length"])
    prefix = encoder["instruction_prefix"].replace("\\n", "\n")
    output: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[start : start + batch_size]]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=maximum,
            return_tensors="pt",
        )
        tokens = {key: value.to(args.device) for key, value in tokens.items()}
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
        if encoder["pooling"] == "cls":
            pooled = hidden[:, 0]
        elif encoder["pooling"] == "mean":
            pooled = mean_pool(hidden, tokens["attention_mask"])
        else:
            raise ValueError(f"unsupported pooling: {encoder['pooling']}")
        if encoder["normalize"].lower() == "true":
            pooled = F.normalize(pooled, p=2, dim=1)
        output.append(pooled.float().cpu().numpy())
        print(json.dumps({"encoder": args.encoder_id, "encoded": min(start + batch_size, len(texts)), "total": len(texts)}), flush=True)
    return np.concatenate(output, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-registry", type=Path, default=REVISION_ROOT / "config/encoder_registry.csv")
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.input = args.input or REVISION_ROOT / "data/fragments/main" / f"{args.model_id}.jsonl"
    out_dir = REVISION_ROOT / "data/embeddings" / args.encoder_id
    args.out = args.out or out_dir / f"{args.model_id}.npz"
    args.metadata_out = args.metadata_out or out_dir / f"{args.model_id}.metadata.json"
    if args.out.exists() and not args.force:
        raise SystemExit(f"output exists; use --force only after verifying replacement scope: {args.out}")
    source = read_jsonl_tolerant(args.input)
    if source.bad_lines or not source.rows:
        raise ValueError("input generation fragment is empty or malformed")
    encoder = load_encoder(args.encoder_registry, args.encoder_id)
    texts = [str(row["generated_text"]) for row in source.rows]
    run_id = f"embed_{args.encoder_id}_{args.model_id}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    manifest_path = REVISION_ROOT / "manifests" / f"run_manifest_{run_id}.json"
    manifest = new_run_manifest(
        run_id=run_id,
        command=sys.argv,
        config_paths=[args.encoder_registry],
        data_paths=[args.input],
        checkpoint_name=encoder["model_card_name"],
        checkpoint_path=encoder["checkpoint_path"],
        seeds=sorted({int(row["seed"]) for row in source.rows}),
        output_paths=[args.out, args.metadata_out],
    )
    atomic_write_json(manifest_path, manifest)
    try:
        vectors = encode(args, encoder, texts)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp.npz")
        np.savez(
            temporary,
            vectors=vectors,
            run_ids=np.asarray([row["run_id"] for row in source.rows]),
        )
        temporary.replace(args.out)
        metadata: dict[str, Any] = {
            **encoder,
            "input_file": str(args.input),
            "input_sha256": sha256_file(args.input),
            "row_count": len(source.rows),
            "embedding_dimension": int(vectors.shape[1]),
            "dtype": str(vectors.dtype),
            "device": args.device,
            "actual_batch_size": args.batch_size or int(encoder["batch_size"]),
        }
        atomic_write_json(args.metadata_out, metadata)
    except BaseException as exc:
        finish_run_manifest(manifest_path, manifest, status="failed", error=repr(exc))
        raise
    finish_run_manifest(
        manifest_path,
        manifest,
        status="success",
        counts={"rows": len(source.rows), "embedding_dimension": int(vectors.shape[1])},
    )


if __name__ == "__main__":
    main()
