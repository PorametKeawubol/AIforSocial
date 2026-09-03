"""Shared helpers for the two-stage Thai QA RAG workflow.

The index builder is deliberately independent of Ollama.  It writes the FAISS
index, the passages, and a manifest once; the QA runner only reads those
artifacts and never embeds the whole corpus again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_DATASET = WORKSPACE_DIR / "DENSE" / "thai_qa_utf8.json"
DEFAULT_QUESTIONS = WORKSPACE_DIR / "DENSE" / "thai_qa_paraphrase_15.csv"


def choose_device(requested: str | None) -> str:
    """Return the requested device, otherwise prefer CUDA when it is usable."""
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_json_records(dataset_path: Path, limit: int) -> list[dict[str, Any]]:
    """Load and validate Thai QA records, retaining the first ``limit`` rows."""
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    try:
        raw_records = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Supply --dataset with thai_qa_utf8.json."
        ) from error

    if not isinstance(raw_records, list):
        raise ValueError(f"Dataset must be a JSON array, got {type(raw_records).__name__}")
    if len(raw_records) < limit:
        raise ValueError(f"Dataset has {len(raw_records):,} records but --limit is {limit:,}")

    records: list[dict[str, Any]] = []
    for position, row in enumerate(raw_records[:limit]):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {position} is not an object")
        text = row.get("input")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Dataset row {position} has no non-empty 'input' passage")
        original_id = row.get("__index_level_0__", position)
        records.append(
            {
                "faiss_id": position,
                "original_id": original_id,
                "passage": text.strip(),
                "instruction": str(row.get("instruction", "")).strip(),
                "answer": str(row.get("answer", "")).strip(),
                "source": str(row.get("source", "")).strip(),
            }
        )
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Index metadata not found: {path}") from error
    return [json.loads(line) for line in lines if line.strip()]


def read_artifacts(index_dir: Path) -> tuple[faiss.Index, list[dict[str, Any]], dict[str, Any]]:
    """Load and cross-check an index produced by ``build_faiss_index.py``."""
    index_path = index_dir / "index.faiss"
    documents_path = index_dir / "documents.jsonl"
    manifest_path = index_dir / "manifest.json"
    if not index_path.exists() or not documents_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "Missing index artifacts. Run build_faiss_index.py before starting RAG QA."
        )
    index = faiss.read_index(str(index_path))
    documents = read_jsonl(documents_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if index.ntotal != len(documents):
        raise ValueError(
            f"Corrupt artifacts: FAISS holds {index.ntotal:,} vectors but metadata has "
            f"{len(documents):,} rows"
        )
    return index, documents, manifest


def safe_filename(value: str) -> str:
    """Make a predictable portable filename segment from an Ollama model name."""
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def as_float32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)
