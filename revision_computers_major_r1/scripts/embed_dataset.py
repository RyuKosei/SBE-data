from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from common import atomic_write_json, read_jsonl, sha256_file, utc_now


def load_encoder(path: Path, encoder_id: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        registry = {row["encoder_id"]: row for row in csv.DictReader(handle)}
    if encoder_id not in registry:
        raise ValueError(f"unknown encoder: {encoder_id}")
    return registry[encoder_id]


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def encode_batch(model: AutoModel, tokenizer: AutoTokenizer, texts: list[str], encoder: dict[str, str], device: str) -> np.ndarray:
    prefix = encoder["instruction_prefix"].replace("\\n", "\n")
    tokens = tokenizer([prefix + text for text in texts], padding=True, truncation=True, max_length=int(encoder["max_length"]), return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}
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
    return pooled.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-registry", type=Path, required=True)
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--id-field", default="run_id")
    parser.add_argument("--text-field", default="generated_text")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    ids = [str(row[args.id_field]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("embedding input IDs are not unique")
    texts = [str(row[args.text_field]) for row in rows]
    encoder = load_encoder(args.encoder_registry, args.encoder_id)
    input_hash = sha256_file(args.input)
    out_dir = args.output_root / "embeddings" / args.encoder_id / args.dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(encoder["checkpoint_path"], local_files_only=True)
    model = AutoModel.from_pretrained(encoder["checkpoint_path"], local_files_only=True).to(args.device).eval()
    batch_size = args.batch_size or int(encoder["batch_size"])
    fragments = []
    for chunk_start in range(0, len(rows), args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, len(rows))
        fragment = out_dir / f"chunk_{chunk_start:07d}_{chunk_end:07d}.npz"
        if fragment.exists():
            with np.load(fragment, allow_pickle=False) as archive:
                if list(map(str, archive["ids"].tolist())) != ids[chunk_start:chunk_end]:
                    raise ValueError(f"existing embedding fragment ID mismatch: {fragment}")
            fragments.append(fragment)
            continue
        output = []
        for start in range(chunk_start, chunk_end, batch_size):
            end = min(start + batch_size, chunk_end)
            output.append(encode_batch(model, tokenizer, texts[start:end], encoder, args.device))
            print(json.dumps({"encoder": args.encoder_id, "dataset": args.dataset_id, "encoded": end, "total": len(rows)}), flush=True)
        vectors = np.concatenate(output, axis=0)
        temporary = fragment.with_suffix(".tmp.npz")
        np.savez(temporary, vectors=vectors, ids=np.asarray(ids[chunk_start:chunk_end]))
        temporary.replace(fragment)
        fragments.append(fragment)
    vector_parts = []
    id_parts = []
    for fragment in fragments:
        with np.load(fragment, allow_pickle=False) as archive:
            vector_parts.append(archive["vectors"])
            id_parts.extend(map(str, archive["ids"].tolist()))
    vectors = np.concatenate(vector_parts, axis=0)
    if id_parts != ids:
        raise ValueError("consolidated embedding IDs do not match input order")
    output_path = args.output_root / "embeddings" / args.encoder_id / f"{args.dataset_id}.npz"
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez(temporary, vectors=vectors, ids=np.asarray(ids))
    temporary.replace(output_path)
    metadata = {
        "generated_at": utc_now(),
        "encoder": encoder,
        "input": str(args.input),
        "input_sha256": input_hash,
        "id_field": args.id_field,
        "text_field": args.text_field,
        "rows": len(rows),
        "shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
        "device": args.device,
        "batch_size": batch_size,
        "chunk_size": args.chunk_size,
    }
    atomic_write_json(output_path.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
