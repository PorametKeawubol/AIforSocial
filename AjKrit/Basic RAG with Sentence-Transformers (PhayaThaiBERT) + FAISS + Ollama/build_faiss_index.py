#!/usr/bin/env python3
"""Stage 1: create a reusable FAISS index from 10,000 Thai QA passages."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from rag_common import DEFAULT_DATASET, PROJECT_DIR, as_float32, choose_device, load_json_records, write_jsonl


DEFAULT_OUTPUT_DIR = PROJECT_DIR / "artifacts" / "thai_qa_10000"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS once from Thai QA passages; it is reused by rag_qa.py."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=10_000, help="Number of source rows to index.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Default: automatically select CUDA when available.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing output directory after checking it is an index directory.",
    )
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, replace: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        known_artifacts = {"index.faiss", "documents.jsonl", "manifest.json"}
        existing = {path.name for path in output_dir.iterdir()}
        if not replace:
            raise FileExistsError(
                f"{output_dir} already contains files. Use --replace to recreate this index."
            )
        if not existing.issubset(known_artifacts):
            raise ValueError(
                f"Refusing to remove unexpected files from {output_dir}: {sorted(existing - known_artifacts)}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    records = load_json_records(args.dataset, args.limit)
    prepare_output_dir(args.output_dir, args.replace)

    device = choose_device(args.device)
    print(f"Loading embedding model: {args.embedding_model} ({device})", flush=True)
    embedder = SentenceTransformer(args.embedding_model, device=device)
    print(f"Embedding {len(records):,} passages ...", flush=True)
    embeddings = embedder.encode(
        [record["passage"] for record in records],
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = as_float32(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(args.output_dir / "index.faiss"))
    write_jsonl(args.output_dir / "documents.jsonl", records)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_rows_indexed": len(records),
        "document_field": "input",
        "embedding_model": args.embedding_model,
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_normalized": True,
        "faiss_index": "IndexFlatIP",
        "similarity": "cosine (inner product over L2-normalized embeddings)",
        "device_used": device,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {index.ntotal:,} vectors to {args.output_dir}\n"
        f"  - index.faiss\n  - documents.jsonl\n  - manifest.json"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
