"""Benchmark FAISS and ChromaDB on the Thai/multilingual QA corpus.

The benchmark deliberately encodes documents and queries once, then sends the
same normalised vectors to both engines.  This keeps model inference outside
the vector-store comparison and makes the measured latency reproducible.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:  # Optional at import time so data-loader tests can run before installation.
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - depends on the local environment
    faiss = None  # type: ignore[assignment]

try:  # Optional at import time so the CLI can report a useful install message.
    import chromadb  # type: ignore
except ImportError:  # pragma: no cover - depends on the local environment
    chromadb = None  # type: ignore[assignment]


DEFAULT_DATA_PATH = Path("thai_qa_utf8.json")
DEFAULT_QUERY_PATH = Path("thai_qa_paraphrase_15.csv")
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_COLLECTION_NAME = "thai_qa_benchmark"


@dataclass(frozen=True)
class DocumentRecord:
    """A corpus row with stable ID and metadata for both engines."""

    id: str
    text: str
    instruction: str
    answer: str
    source: str
    dataset_index: int

    def metadata(self) -> dict[str, str]:
        """Return Chroma-compatible scalar metadata."""

        return {
            "source": self.source,
            "dataset_index": str(self.dataset_index),
        }


@dataclass(frozen=True)
class QueryRecord:
    """A benchmark query loaded from the supplied CSV file."""

    id: str
    text: str
    answer: str
    dataset_index: int | None


@dataclass(frozen=True)
class SearchResult:
    """A unified result shape; FAISS returns similarity, Chroma returns distance."""

    record: DocumentRecord
    score: float
    value_type: str


def _require_dependency(dependency: Any, name: str) -> None:
    if dependency is None:
        raise RuntimeError(
            f"ยังไม่ได้ติดตั้ง {name} กรุณารัน: python -m pip install -r requirements.txt"
        )


def _clean_text(value: Any, field: str, row_number: int) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError(f"แถวที่ {row_number} ไม่มีค่า {field}")
    return text


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() in {"", "nan", "None"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_documents(path: Path, limit: int = 10_000) -> tuple[list[DocumentRecord], int]:
    """Load a deterministic corpus slice and return (records, source row count).

    ``__index_level_0__`` is not used as the document ID because it is repeated
    in this dataset.  The physical row number is stable for a fixed JSON file
    and is therefore used to create unique IDs.
    """

    if limit < 1:
        raise ValueError("limit ต้องมากกว่า 0")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("JSON corpus ต้องเป็น list ที่ไม่ว่าง")

    source_count = len(payload)
    records: list[DocumentRecord] = []
    for raw_row_number, item in enumerate(payload):
        # Several source rows (notably wiki_qa) have no context in ``input``.
        # They cannot be embedded as retrieval documents, so skip them and
        # continue until we have the requested number of usable records.
        if not isinstance(item, dict):
            raise ValueError(f"แถวที่ {raw_row_number} ใน JSON ไม่ใช่ object")
        if not str(item.get("input", "") or "").strip():
            continue
        row_number = len(records)
        if row_number >= limit:
            break
        dataset_index = _optional_int(item.get("__index_level_0__"))
        records.append(
            DocumentRecord(
                id=f"doc-{row_number:05d}",
                text=_clean_text(item.get("input"), "input", raw_row_number),
                instruction=_clean_text(
                    item.get("instruction", ""), "instruction", raw_row_number
                ),
                answer=_clean_text(item.get("answer", ""), "answer", raw_row_number),
                source=_clean_text(item.get("source", "unknown"), "source", raw_row_number),
                dataset_index=row_number if dataset_index is None else dataset_index,
            )
        )
    if len(records) < limit:
        raise ValueError(
            f"พบ document ที่มี input ใช้งานได้เพียง {len(records):,} รายการ "
            f"จากที่ขอ {limit:,} รายการ"
        )
    validate_documents(records)
    return records, source_count


def load_queries(path: Path, limit: int | None = None) -> list[QueryRecord]:
    """Load query text from the worksheet CSV (BOM-safe UTF-8)."""

    if limit is not None and limit < 1:
        raise ValueError("query limit ต้องมากกว่า 0")
    queries: list[QueryRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "instruction" not in reader.fieldnames:
            raise ValueError("query CSV ต้องมีคอลัมน์ instruction")
        for row_number, row in enumerate(reader):
            if limit is not None and row_number >= limit:
                break
            queries.append(
                QueryRecord(
                    id=f"query-{row_number:03d}",
                    text=_clean_text(row.get("instruction"), "instruction", row_number),
                    answer=str(row.get("answer", "") or "").strip(),
                    dataset_index=_optional_int(row.get("__index_level_0__")),
                )
            )
    if not queries:
        raise ValueError("query CSV ไม่มี query ที่ใช้งานได้")
    return queries


def validate_documents(records: Sequence[DocumentRecord]) -> None:
    """Fail early when IDs or fields would make results ambiguous."""

    if not records:
        raise ValueError("corpus ต้องมีอย่างน้อย 1 รายการ")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("พบ document ID ซ้ำ")
    for record in records:
        if not record.text or not record.instruction or not record.answer or not record.source:
            raise ValueError(f"ข้อมูลไม่ครบใน document {record.id}")


def _as_float32_matrix(values: Any, expected_rows: int | None = None) -> np.ndarray:
    matrix = np.ascontiguousarray(np.asarray(values, dtype="float32"))
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embeddings ต้องเป็นเมทริกซ์ 2 มิติที่ไม่ว่าง")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError("จำนวน embeddings ไม่ตรงกับจำนวน records")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings ต้องมีค่า finite เท่านั้น")
    return matrix


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings ต้องมีค่า finite เท่านั้น")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("พบ embedding ที่มี norm เป็นศูนย์")
    return np.ascontiguousarray(matrix / norms, dtype="float32")


def encode_texts(
    embedder: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 64,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode and normalise text once for use by both vector engines."""

    if not texts:
        raise ValueError("ต้องมีข้อความอย่างน้อย 1 รายการ")
    values = embedder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return _normalise_rows(_as_float32_matrix(values, expected_rows=len(texts)))


def load_embedder(model_name: str) -> Any:
    """Load SentenceTransformer lazily so utility tests need no model download."""

    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as error:  # pragma: no cover - model/network dependent
        raise RuntimeError(
            f"โหลดโมเดล {model_name} ไม่สำเร็จ: ตรวจอินเทอร์เน็ตหรือ Hugging Face cache"
        ) from error


def build_faiss_index(embeddings: np.ndarray) -> Any:
    """Build an exact cosine-equivalent FAISS IndexFlatIP."""

    _require_dependency(faiss, "faiss-cpu")
    vectors = _as_float32_matrix(embeddings)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _query_vector(query_embedding: Any, dimension: int) -> np.ndarray:
    query = np.ascontiguousarray(np.asarray(query_embedding, dtype="float32")).reshape(1, -1)
    if query.shape[1] != dimension:
        raise ValueError(
            f"query dimension {query.shape[1]} ไม่ตรงกับ index dimension {dimension}"
        )
    if not np.isfinite(query).all():
        raise ValueError("query embedding ต้องมีค่า finite เท่านั้น")
    return _normalise_rows(query)


def search_faiss(
    index: Any,
    records: Sequence[DocumentRecord],
    query_embedding: Any,
    k: int = 5,
) -> list[SearchResult]:
    """Search FAISS and map vector positions back to document metadata."""

    if k < 1:
        raise ValueError("k ต้องมากกว่า 0")
    if index.ntotal != len(records):
        raise ValueError("จำนวน vector ใน FAISS ไม่ตรงกับจำนวน records")
    query = _query_vector(query_embedding, index.d)
    scores, positions = index.search(query, min(k, len(records)))
    return [
        SearchResult(records[position], float(score), "similarity")
        for score, position in zip(scores[0], positions[0])
        if position >= 0
    ]


def filter_faiss_results(
    ranked_results: Sequence[SearchResult],
    records: Sequence[DocumentRecord],
    predicate: Any,
    k: int = 5,
) -> list[SearchResult]:
    """Apply metadata filtering in application code after a full FAISS ranking."""

    if k < 1:
        raise ValueError("k ต้องมากกว่า 0")
    if len(ranked_results) < len(records):
        raise ValueError("FAISS metadata filter ต้องรับผลลัพธ์ที่ over-fetch ครบ corpus")
    return [result for result in ranked_results if predicate(result.record)][:k]


def _reset_collection(client: Any, collection_name: str) -> None:
    """Reset only the named generated collection, preserving other Chroma data."""

    try:
        client.get_collection(collection_name)
    except Exception as error:
        if error.__class__.__name__ != "NotFoundError":
            raise
    else:
        client.delete_collection(collection_name)


def create_chroma_collection(
    path: Path,
    records: Sequence[DocumentRecord],
    embeddings: np.ndarray,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 500,
) -> Any:
    """Create a persistent Chroma HNSW collection from the same vectors."""

    _require_dependency(chromadb, "chromadb")
    if batch_size < 1:
        raise ValueError("Chroma batch size ต้องมากกว่า 0")
    validate_documents(records)
    vectors = _as_float32_matrix(embeddings, expected_rows=len(records))
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    _reset_collection(client, collection_name)
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        collection.add(
            ids=[record.id for record in batch],
            documents=[record.text for record in batch],
            metadatas=[record.metadata() for record in batch],
            embeddings=vectors[start : start + len(batch)].tolist(),
        )
    return collection


def _record_from_chroma(
    record_id: str,
    document: str,
    metadata: dict[str, Any],
    records_by_id: dict[str, DocumentRecord] | None,
) -> DocumentRecord:
    if records_by_id and record_id in records_by_id:
        return records_by_id[record_id]
    return DocumentRecord(
        id=record_id,
        text=document,
        instruction="",
        answer="",
        source=str(metadata.get("source", "unknown")),
        dataset_index=int(metadata.get("dataset_index", -1)),
    )


def search_chroma(
    collection: Any,
    query_embedding: Any,
    *,
    k: int = 5,
    where: dict[str, Any] | None = None,
    records_by_id: dict[str, DocumentRecord] | None = None,
) -> list[SearchResult]:
    """Search Chroma using cosine distance and an optional metadata filter."""

    if k < 1:
        raise ValueError("k ต้องมากกว่า 0")
    query = _normalise_rows(
        np.ascontiguousarray(np.asarray(query_embedding, dtype="float32")).reshape(1, -1)
    )
    if where is None:
        available = collection.count()
    else:
        available = len(collection.get(where=where, include=[]).get("ids", []))
    if available == 0:
        return []
    payload = collection.query(
        query_embeddings=[query[0].tolist()],
        n_results=min(k, available),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = (payload.get("ids") or [[]])[0]
    documents = (payload.get("documents") or [[]])[0]
    metadatas = (payload.get("metadatas") or [[]])[0]
    distances = (payload.get("distances") or [[]])[0]
    results: list[SearchResult] = []
    for record_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        results.append(
            SearchResult(
                record=_record_from_chroma(
                    str(record_id),
                    str(document),
                    metadata or {},
                    records_by_id,
                ),
                score=float(distance),
                value_type="distance",
            )
        )
    return results


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    """Summarise samples and convert mean latency into QPS."""

    if not samples_ms:
        raise ValueError("ต้องมี latency sample อย่างน้อย 1 ค่า")
    values = [float(value) for value in samples_ms]
    ordered = sorted(values)
    percentile_index = min(len(ordered) - 1, max(0, int(round(0.95 * len(ordered))) - 1))
    mean_ms = statistics.fmean(values)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p95_ms": ordered[percentile_index],
        "qps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
        "samples": float(len(values)),
    }


def benchmark_latency(
    index: Any,
    collection: Any,
    query_vectors: Sequence[np.ndarray],
    *,
    k: int = 5,
    repeats: int = 10,
    warmup_runs: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Measure retrieval only, after all query embeddings already exist."""

    if repeats < 1 or warmup_runs < 0:
        raise ValueError("repeats ต้องมากกว่า 0 และ warmup_runs ต้องไม่ติดลบ")
    if not query_vectors:
        raise ValueError("ต้องมี query vector อย่างน้อย 1 รายการ")

    collection_count = collection.count()
    for query_vector in query_vectors[:1]:
        for _ in range(warmup_runs):
            index.search(_query_vector(query_vector, index.d), min(k, index.ntotal))
            collection.query(
                query_embeddings=[
                    _normalise_rows(
                        np.ascontiguousarray(np.asarray(query_vector, dtype="float32"))
                        .reshape(1, -1)
                    )[0].tolist()
                ],
                n_results=min(k, collection_count),
                include=["distances"],
            )

    samples: list[dict[str, Any]] = []
    for query_number, query_vector in enumerate(query_vectors):
        faiss_query = _query_vector(query_vector, index.d)
        chroma_query = _normalise_rows(
            np.ascontiguousarray(np.asarray(query_vector, dtype="float32")).reshape(1, -1)
        )[0].tolist()
        for repeat in range(repeats):
            started = time.perf_counter()
            index.search(faiss_query, min(k, index.ntotal))
            samples.append(
                {
                    "engine": "FAISS",
                    "query_number": query_number,
                    "repeat": repeat,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
            started = time.perf_counter()
            collection.query(
                query_embeddings=[chroma_query],
                n_results=min(k, collection_count),
                include=["distances"],
            )
            samples.append(
                {
                    "engine": "ChromaDB",
                    "query_number": query_number,
                    "repeat": repeat,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )

    summaries: dict[str, dict[str, float]] = {}
    for engine in ("FAISS", "ChromaDB"):
        values = [row["latency_ms"] for row in samples if row["engine"] == engine]
        summaries[engine] = latency_summary(values)
    return samples, summaries


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _result_row(query: QueryRecord, engine: str, rank: int, result: SearchResult) -> dict[str, Any]:
    return {
        "query_id": query.id,
        "query": query.text,
        "query_answer": query.answer,
        "rank": rank,
        "engine": engine,
        "document_id": result.record.id,
        "dataset_index": result.record.dataset_index,
        "source": result.record.source,
        "score": result.score,
        "score_type": result.value_type,
        "document_preview": result.record.text[:160].replace("\n", " "),
    }


def _metadata_filter_demo(
    index: Any,
    collection: Any,
    records: Sequence[DocumentRecord],
    query_vector: np.ndarray,
    *,
    source: str,
    k: int,
) -> dict[str, Any]:
    """Run the same source constraint through app-side FAISS and Chroma ``where``."""

    full_ranked = search_faiss(index, records, query_vector, k=len(records))
    faiss_filtered = [result for result in full_ranked if result.record.source == source][:k]
    records_by_id = {record.id: record for record in records}
    chroma_filtered = search_chroma(
        collection,
        query_vector,
        k=k,
        where={"source": source},
        records_by_id=records_by_id,
    )
    return {
        "field": "source",
        "value": source,
        "query": "first worksheet query",
        "faiss_application_filter": [result.record.id for result in faiss_filtered],
        "chroma_where_filter": [result.record.id for result in chroma_filtered],
        "note": "FAISS filters after full ranking; ChromaDB applies where during collection query.",
    }


def run_experiment(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    query_path: Path = DEFAULT_QUERY_PATH,
    limit: int = 10_000,
    query_limit: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    top_k: int = 5,
    benchmark_repeats: int = 10,
    warmup_runs: int = 2,
    chroma_dir: Path = Path("chroma_db"),
    chroma_batch_size: int = 500,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedder: Any | None = None,
    show_progress_bar: bool = True,
    faiss_threads: int | None = 1,
) -> dict[str, Any]:
    """Run the complete worksheet benchmark and return serialisable reports."""

    if top_k < 1 or benchmark_repeats < 1 or batch_size < 1:
        raise ValueError("top_k, benchmark_repeats และ batch_size ต้องมากกว่า 0")
    documents, source_row_count = load_documents(data_path, limit=limit)
    queries = load_queries(query_path, limit=query_limit)
    active_embedder = embedder if embedder is not None else load_embedder(model_name)

    if faiss is not None and faiss_threads is not None:
        if faiss_threads < 1:
            raise ValueError("faiss_threads ต้องมากกว่า 0 หรือใช้ None")
        faiss.omp_set_num_threads(faiss_threads)

    started = time.perf_counter()
    document_embeddings = encode_texts(
        active_embedder,
        [record.text for record in documents],
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    document_encode_seconds = time.perf_counter() - started
    started = time.perf_counter()
    query_embeddings = encode_texts(
        active_embedder,
        [query.text for query in queries],
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    query_encode_seconds = time.perf_counter() - started

    started = time.perf_counter()
    faiss_index = build_faiss_index(document_embeddings)
    faiss_build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    chroma_collection = create_chroma_collection(
        chroma_dir,
        documents,
        document_embeddings,
        collection_name=collection_name,
        batch_size=chroma_batch_size,
    )
    chroma_build_seconds = time.perf_counter() - started

    records_by_id = {record.id: record for record in documents}
    result_rows: list[dict[str, Any]] = []
    for query, vector in zip(queries, query_embeddings, strict=True):
        faiss_results = search_faiss(faiss_index, documents, vector, k=top_k)
        chroma_results = search_chroma(
            chroma_collection,
            vector,
            k=top_k,
            records_by_id=records_by_id,
        )
        result_rows.extend(
            _result_row(query, "FAISS", rank, result)
            for rank, result in enumerate(faiss_results, start=1)
        )
        result_rows.extend(
            _result_row(query, "ChromaDB", rank, result)
            for rank, result in enumerate(chroma_results, start=1)
        )

    latency_samples, latency = benchmark_latency(
        faiss_index,
        chroma_collection,
        list(query_embeddings),
        k=top_k,
        repeats=benchmark_repeats,
        warmup_runs=warmup_runs,
    )
    filter_demo = _metadata_filter_demo(
        faiss_index,
        chroma_collection,
        documents,
        query_embeddings[0],
        source=documents[0].source,
        k=top_k,
    )

    dimension = int(document_embeddings.shape[1])
    vector_bytes = len(documents) * dimension * np.dtype("float32").itemsize
    build_rows = [
        {
            "engine": "FAISS",
            "index_type": "IndexFlatIP (exact)",
            "documents": len(documents),
            "dimension": dimension,
            "build_time_seconds": faiss_build_seconds,
            "vector_memory_mb": vector_bytes / (1024**2),
            "persisted_storage_mb": 0.0,
        },
        {
            "engine": "ChromaDB",
            "index_type": "HNSW cosine (default collection index)",
            "documents": len(documents),
            "dimension": dimension,
            "build_time_seconds": chroma_build_seconds,
            "vector_memory_mb": vector_bytes / (1024**2),
            "persisted_storage_mb": _directory_size_bytes(chroma_dir) / (1024**2),
        },
    ]
    metrics = {
        "dataset": {
            "path": str(data_path),
            "source_rows_available": source_row_count,
            "rows_indexed": len(documents),
            "limit": limit,
            "query_path": str(query_path),
            "queries_used": len(queries),
        },
        "embedding": {
            "model": model_name if embedder is None else type(active_embedder).__name__,
            "dimension": dimension,
            "document_encode_seconds": document_encode_seconds,
            "query_encode_seconds": query_encode_seconds,
            "normalised": True,
        },
        "benchmark": {
            "top_k": top_k,
            "repeats_per_query": benchmark_repeats,
            "warmup_runs": warmup_runs,
            "latency_scope": "retrieval only; query embeddings were prepared before timing",
        },
        "build": build_rows,
        "latency": latency,
        "metadata_filter_demo": filter_demo,
        "score_convention": {
            "FAISS": "inner-product similarity; higher is better",
            "ChromaDB": "cosine distance; lower is better",
        },
    }
    return {
        "metrics": metrics,
        "build_rows": build_rows,
        "latency_samples": latency_samples,
        "result_rows": result_rows,
    }


def write_reports(summary: dict[str, Any], output_dir: Path) -> None:
    """Write CSV/JSON artifacts that can be attached to the worksheet."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary["build_rows"]).to_csv(
        output_dir / "build_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary["latency_samples"]).to_csv(
        output_dir / "latency_samples.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary["result_rows"]).to_csv(
        output_dir / "query_results_topk.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(summary["metrics"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("ต้องเป็นจำนวนเต็ม") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("ต้องมากกว่า 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("ต้องเป็นจำนวนเต็ม") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("ต้องไม่ติดลบ")
    return parsed


def _configure_utf8_stdout() -> None:
    stream = sys.stdout
    if getattr(stream, "encoding", "").lower().replace("-", "") == "utf8":
        return
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass


def _print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    metrics = summary["metrics"]
    print("\n=== FAISS vs ChromaDB: ผลการทดลอง ===")
    print(pd.DataFrame(summary["build_rows"]).to_string(index=False))
    print("\nRetrieval latency (ms/query)")
    print(pd.DataFrame(metrics["latency"]).T.to_string(float_format=lambda value: f"{value:.4f}"))
    demo = metrics["metadata_filter_demo"]
    print(f"\nMetadata filter: {demo['field']}={demo['value']}")
    print(f"FAISS application filter: {', '.join(demo['faiss_application_filter'])}")
    print(f"ChromaDB where filter: {', '.join(demo['chroma_where_filter'])}")
    print(f"\nสร้าง index จาก {metrics['dataset']['rows_indexed']:,} รายการ")
    print(f"บันทึกผลไว้ที่ {output_dir.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    _configure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="เปรียบเทียบ FAISS และ ChromaDB บนชุดข้อมูล QA ภาษาไทย/หลายภาษา"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--limit", type=_positive_int, default=10_000)
    parser.add_argument("--query-limit", type=_positive_int, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--benchmark-repeats", type=_positive_int, default=10)
    parser.add_argument("--warmup-runs", type=_nonnegative_int, default=2)
    parser.add_argument("--chroma-dir", type=Path, default=Path("chroma_db"))
    parser.add_argument("--chroma-batch-size", type=_positive_int, default=500)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--faiss-threads",
        type=_positive_int,
        default=1,
        help="จำนวน OpenMP threads ของ FAISS เพื่อให้ latency เทียบซ้ำได้",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = run_experiment(
            data_path=args.data,
            query_path=args.queries,
            limit=args.limit,
            query_limit=args.query_limit,
            model_name=args.model_name,
            batch_size=args.batch_size,
            top_k=args.top_k,
            benchmark_repeats=args.benchmark_repeats,
            warmup_runs=args.warmup_runs,
            chroma_dir=args.chroma_dir,
            chroma_batch_size=args.chroma_batch_size,
            collection_name=args.collection_name,
            show_progress_bar=not args.no_progress,
            faiss_threads=args.faiss_threads,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"เกิดข้อผิดพลาด: {error}", file=sys.stderr)
        return 2

    write_reports(summary, args.output_dir)
    _print_summary(summary, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
