"""เปรียบเทียบ FAISS และ ChromaDB สำหรับการค้นคืนเอกสารภาษาไทย."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import sys
import time

import chromadb
import faiss
import numpy as np
import pandas as pd
from chromadb.errors import NotFoundError


@dataclass(frozen=True)
class DocumentRecord:
    """เอกสารหนึ่งรายการพร้อม metadata ที่ใช้ในใบงาน."""

    id: str
    document: str
    doc_id: str
    category: str
    doc_code: str

    def metadata(self) -> dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "category": self.category,
            "doc_code": self.doc_code,
        }


@dataclass(frozen=True)
class SearchResult:
    """ผลการค้นหาหนึ่งอันดับจาก vector store."""

    record: DocumentRecord
    score: float
    value_type: str


CHROMA_COLLECTION_NAME = "university_documents"
MODEL_NAME = "BAAI/bge-m3"

SEMANTIC_EXERCISES = {
    "นักศึกษาจะลงทะเบียนเรียนออนไลน์ได้อย่างไร": "REG-1001",
    "ต้องการถอนวิชาที่ลงทะเบียนไว้": "REG-3010",
    "ต้องการชำระค่าเทอม": "FIN-4020",
    "ต้องการขอเอกสารรับรองนักศึกษา": "REG-5090",
    "ต้องการเชื่อมต่อ Wi-Fi มหาวิทยาลัย": "IT-6012",
}
EXACT_CODE_EXERCISES = ("REG-2045", "IT-7788", "FIN-4020", "LIB-7033")
CATEGORY_EXERCISE_QUERY = "ระบบสำหรับนักศึกษา"
CATEGORY_EXERCISE_VALUE = "REG"


BASE_RECORDS = [
    DocumentRecord(
        "doc_001",
        "ขั้นตอนการลงทะเบียนเรียนสำหรับนักศึกษาระดับปริญญาตรีผ่านระบบออนไลน์",
        "DOC-001",
        "REG",
        "REG-1001",
    ),
    DocumentRecord(
        "doc_002",
        "คู่มือการใช้งานระบบยื่นคำร้องออนไลน์สำหรับนักศึกษา รหัสเอกสาร REG-2045",
        "DOC-002",
        "REG",
        "REG-2045",
    ),
    DocumentRecord(
        "doc_003",
        "ระเบียบการขอเพิ่มและถอนรายวิชาภายในระยะเวลาที่มหาวิทยาลัยกำหนด",
        "DOC-003",
        "REG",
        "REG-3010",
    ),
    DocumentRecord(
        "doc_004",
        "ขั้นตอนการชำระค่าธรรมเนียมการศึกษาและตรวจสอบสถานะการชำระเงิน",
        "DOC-004",
        "FIN",
        "FIN-4020",
    ),
    DocumentRecord(
        "doc_005",
        "คู่มือการใช้งานระบบสารสนเทศสำหรับบุคลากร รหัสเอกสาร IT-7788",
        "DOC-005",
        "IT",
        "IT-7788",
    ),
    DocumentRecord(
        "doc_006",
        "แนวทางการขอหนังสือรับรองการเป็นนักศึกษาผ่านระบบออนไลน์",
        "DOC-006",
        "REG",
        "REG-5090",
    ),
    DocumentRecord(
        "doc_007",
        "ขั้นตอนการขอใช้บริการ Wi-Fi สำหรับนักศึกษาและบุคลากร",
        "DOC-007",
        "IT",
        "IT-6012",
    ),
    DocumentRecord(
        "doc_008",
        "ระเบียบการยืมคืนหนังสือและทรัพยากรสารสนเทศของห้องสมุด",
        "DOC-008",
        "LIB",
        "LIB-7033",
    ),
]

EXTENSION_RECORDS = [
    DocumentRecord(
        "doc_009",
        "คู่มือการใช้งานระบบประชุมออนไลน์สำหรับบุคลากร รหัสเอกสาร IT-8801",
        "DOC-009",
        "IT",
        "IT-8801",
    ),
    DocumentRecord(
        "doc_010",
        "ระเบียบการขอทุนสนับสนุนการวิจัย รหัสเอกสาร RES-1201",
        "DOC-010",
        "RES",
        "RES-1201",
    ),
    DocumentRecord(
        "doc_011",
        "คู่มือการใช้ฐานข้อมูลอิเล็กทรอนิกส์ของห้องสมุด รหัสเอกสาร LIB-8102",
        "DOC-011",
        "LIB",
        "LIB-8102",
    ),
    DocumentRecord(
        "doc_012",
        "ขั้นตอนการขอใบเสร็จรับเงินค่าเล่าเรียน รหัสเอกสาร FIN-5501",
        "DOC-012",
        "FIN",
        "FIN-5501",
    ),
    DocumentRecord(
        "doc_013",
        "คู่มือการลงทะเบียนกิจกรรมนักศึกษา รหัสเอกสาร ACT-2201",
        "DOC-013",
        "ACT",
        "ACT-2201",
    ),
]


def build_corpus(include_extensions: bool = False) -> list[DocumentRecord]:
    """คืนเอกสารหลัก 8 รายการ และเพิ่ม Exercise 3 เมื่อร้องขอ."""
    records = list(BASE_RECORDS)
    if include_extensions:
        records.extend(EXTENSION_RECORDS)
    return records


def validate_corpus(records: list[DocumentRecord]) -> None:
    """ตรวจความสมบูรณ์ของข้อมูลก่อนนำไปสร้าง vector store."""
    if not records:
        raise ValueError("corpus must contain at least one record")

    for attribute in ("id", "doc_id", "doc_code"):
        values = [getattr(record, attribute) for record in records]
        if any(not value for value in values):
            raise ValueError(f"missing {attribute}")
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {attribute}")

    for record in records:
        if not record.document:
            raise ValueError("missing document")
        if not record.category:
            raise ValueError("missing category")


def compute_recall_at_k(
    results_by_query: dict[str, list[str]],
    expected_codes: dict[str, str],
    k: int,
) -> float:
    """คืนสัดส่วนคำถามที่เอกสารถูกต้องปรากฏในอันดับ Top-K."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not expected_codes:
        return 0.0

    hits = sum(
        expected_code in results_by_query.get(query, [])[:k]
        for query, expected_code in expected_codes.items()
    )
    return hits / len(expected_codes)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """สร้าง FAISS exact inner-product index จาก embedding แบบ float32."""
    vectors = np.ascontiguousarray(embeddings, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings must contain only finite values")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _as_query_matrix(query_embedding: np.ndarray, dimension: int) -> np.ndarray:
    query = np.ascontiguousarray(query_embedding, dtype="float32").reshape(1, -1)
    if query.shape[1] != dimension:
        raise ValueError(
            f"query embedding dimension {query.shape[1]} does not match index dimension {dimension}"
        )
    if not np.isfinite(query).all():
        raise ValueError("query embedding must contain only finite values")
    return query


def search_faiss(
    index: faiss.IndexFlatIP,
    records: list[DocumentRecord],
    query_embedding: np.ndarray,
    k: int,
) -> list[SearchResult]:
    """ค้นหาเอกสารด้วย FAISS cosine similarity (เมื่อ vector ถูก normalize แล้ว)."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if index.ntotal != len(records):
        raise ValueError("FAISS index and record count differ")

    scores, positions = index.search(
        _as_query_matrix(query_embedding, index.d),
        min(k, len(records)),
    )
    return [
        SearchResult(record=records[position], score=float(score), value_type="similarity")
        for score, position in zip(scores[0], positions[0])
        if position >= 0
    ]


def search_faiss_with_code(
    index: faiss.IndexFlatIP,
    records: list[DocumentRecord],
    query_embedding: np.ndarray,
    doc_code: str,
    k: int,
) -> list[SearchResult]:
    """สาธิตการกรอง doc_code ที่ application ต้องเขียนเพิ่มบน FAISS."""
    if k < 1:
        raise ValueError("k must be at least 1")
    ranked_results = search_faiss(index, records, query_embedding, k=len(records))
    return [result for result in ranked_results if result.record.doc_code == doc_code][:k]


def create_chroma_collection(
    path: Path,
    records: list[DocumentRecord],
    embeddings: np.ndarray,
):
    """สร้าง collection แบบ persistent จาก embedding ชุดเดียวกับ FAISS."""
    validate_corpus(records)
    vectors = np.ascontiguousarray(embeddings, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        raise ValueError("Chroma embeddings and record count differ")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings must contain only finite values")

    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    try:
        client.get_collection(CHROMA_COLLECTION_NAME)
    except NotFoundError:
        pass
    else:
        client.delete_collection(CHROMA_COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[record.id for record in records],
        documents=[record.document for record in records],
        metadatas=[record.metadata() for record in records],
        embeddings=vectors.tolist(),
    )
    return collection


def chroma_payload_to_results(payload: dict[str, object]) -> list[SearchResult]:
    """แปลง response ของ ChromaDB ให้เป็นรูปแบบผลลัพธ์เดียวกับ FAISS."""
    ids = payload.get("ids", [[]])
    documents = payload.get("documents", [[]])
    metadatas = payload.get("metadatas", [[]])
    distances = payload.get("distances", [[]])
    first_ids = ids[0] if ids else []
    first_documents = documents[0] if documents else []
    first_metadatas = metadatas[0] if metadatas else []
    first_distances = distances[0] if distances else []

    return [
        SearchResult(
            record=DocumentRecord(
                id=str(record_id),
                document=str(document),
                doc_id=str(metadata["doc_id"]),
                category=str(metadata["category"]),
                doc_code=str(metadata["doc_code"]),
            ),
            score=float(distance),
            value_type="distance",
        )
        for record_id, document, metadata, distance in zip(
            first_ids,
            first_documents,
            first_metadatas,
            first_distances,
        )
    ]


def search_chroma(
    collection,
    query_embedding: np.ndarray,
    k: int,
    where: dict[str, str] | None = None,
) -> list[SearchResult]:
    """ค้นหา ChromaDB และกรอง metadata ได้ด้วย where เมื่อระบุ."""
    if k < 1:
        raise ValueError("k must be at least 1")

    query = np.ascontiguousarray(query_embedding, dtype="float32").reshape(-1)
    if not np.isfinite(query).all():
        raise ValueError("query embedding must contain only finite values")

    if where is None:
        eligible_count = collection.count()
    else:
        eligible_count = len(collection.get(where=where, include=[])["ids"])
    if eligible_count == 0:
        return []

    payload = collection.query(
        query_embeddings=[query.tolist()],
        n_results=min(k, eligible_count),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return chroma_payload_to_results(payload)


def write_reports(summary: dict[str, object], output_dir: Path) -> None:
    """บันทึกผลการทดลองเป็นไฟล์ CSV และ JSON สำหรับส่งใบงาน."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary["semantic_rows"]).to_csv(
        output_dir / "semantic_top3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summary["exact_rows"]).to_csv(
        output_dir / "exact_code_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(summary["metrics"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _encode_texts(embedder, texts: list[str]) -> np.ndarray:
    """แปลงข้อความด้วยตัวเข้ารหัสเดียวและบังคับรูปแบบ vector ที่ปลอดภัย."""
    embeddings = np.ascontiguousarray(
        embedder.encode(texts, normalize_embeddings=True),
        dtype="float32",
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError("embedder returned an unexpected embedding shape")
    if not np.isfinite(embeddings).all():
        raise ValueError("embedder returned non-finite embeddings")
    return embeddings


def _row_from_result(system: str, query: str, rank: int, result: SearchResult) -> dict[str, object]:
    return {
        "system": system,
        "query": query,
        "rank": rank,
        "doc_code": result.record.doc_code,
        "doc_id": result.record.doc_id,
        "category": result.record.category,
        "value_type": result.value_type,
        "value": result.score,
    }


def _exact_row(doc_code: str, approach: str, results: list[SearchResult]) -> dict[str, object]:
    matched_codes = [result.record.doc_code for result in results]
    return {
        "doc_code": doc_code,
        "approach": approach,
        "matched_codes": ", ".join(matched_codes),
        "matched_count": len(matched_codes),
    }


def _latency_summary(samples_ms: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(samples_ms)),
        "median": float(statistics.median(samples_ms)),
    }


def _benchmark_retrieval(
    index: faiss.IndexFlatIP,
    collection,
    query_embedding: np.ndarray,
    k: int,
    runs: int,
) -> dict[str, dict[str, float]]:
    """วัดเฉพาะการค้นหา หลังจากสร้าง query embedding แล้ว."""
    if runs < 1:
        raise ValueError("benchmark_runs must be at least 1")

    faiss_query = _as_query_matrix(query_embedding, index.d)
    chroma_query = np.ascontiguousarray(query_embedding, dtype="float32").reshape(-1)
    faiss_samples = []
    chroma_samples = []
    faiss_k = min(k, index.ntotal)
    chroma_k = min(k, collection.count())

    for _ in range(runs):
        started_at = time.perf_counter()
        index.search(faiss_query, faiss_k)
        faiss_samples.append((time.perf_counter() - started_at) * 1000)

    for _ in range(runs):
        started_at = time.perf_counter()
        collection.query(
            query_embeddings=[chroma_query.tolist()],
            n_results=chroma_k,
            include=["distances"],
        )
        chroma_samples.append((time.perf_counter() - started_at) * 1000)

    return {
        "FAISS": _latency_summary(faiss_samples),
        "ChromaDB": _latency_summary(chroma_samples),
    }


def _load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as error:  # pragma: no cover - depends on local model cache/network
        raise RuntimeError(
            "ไม่สามารถโหลด embedding model ได้: การรันครั้งแรกต้องเชื่อมต่ออินเทอร์เน็ต "
            "หรือมีโมเดลอยู่ใน Hugging Face cache แล้ว"
        ) from error


def run_experiment(
    *,
    include_extensions: bool = False,
    model_name: str = MODEL_NAME,
    chroma_dir: Path = Path("chroma_db_comparison"),
    benchmark_runs: int = 30,
    embedder=None,
) -> dict[str, object]:
    """รันทุก exercise ของใบงานและคืนผลลัพธ์ที่บันทึกเป็นรายงานได้."""
    records = build_corpus(include_extensions=include_extensions)
    validate_corpus(records)
    active_embedder = embedder if embedder is not None else _load_embedder(model_name)

    document_embeddings = _encode_texts(
        active_embedder,
        [record.document for record in records],
    )
    query_texts = list(SEMANTIC_EXERCISES) + list(EXACT_CODE_EXERCISES) + [
        CATEGORY_EXERCISE_QUERY
    ]
    query_embeddings = _encode_texts(active_embedder, query_texts)
    query_vectors = dict(zip(query_texts, query_embeddings, strict=True))

    index = build_faiss_index(document_embeddings)
    collection = create_chroma_collection(chroma_dir, records, document_embeddings)

    semantic_rows: list[dict[str, object]] = []
    faiss_ranked_codes: dict[str, list[str]] = {}
    chroma_ranked_codes: dict[str, list[str]] = {}
    for query in SEMANTIC_EXERCISES:
        vector = query_vectors[query]
        faiss_results = search_faiss(index, records, vector, k=5)
        chroma_results = search_chroma(collection, vector, k=5)
        faiss_ranked_codes[query] = [result.record.doc_code for result in faiss_results]
        chroma_ranked_codes[query] = [result.record.doc_code for result in chroma_results]
        semantic_rows.extend(
            _row_from_result("FAISS", query, rank, result)
            for rank, result in enumerate(faiss_results[:3], start=1)
        )
        semantic_rows.extend(
            _row_from_result("ChromaDB", query, rank, result)
            for rank, result in enumerate(chroma_results[:3], start=1)
        )

    exact_rows: list[dict[str, object]] = []
    for doc_code in EXACT_CODE_EXERCISES:
        vector = query_vectors[doc_code]
        exact_rows.extend(
            [
                _exact_row(
                    doc_code,
                    "FAISS pure semantic",
                    search_faiss(index, records, vector, k=3),
                ),
                _exact_row(
                    doc_code,
                    "FAISS custom metadata filter",
                    search_faiss_with_code(index, records, vector, doc_code, k=3),
                ),
                _exact_row(
                    doc_code,
                    "ChromaDB metadata filter",
                    search_chroma(
                        collection,
                        vector,
                        k=3,
                        where={"doc_code": doc_code},
                    ),
                ),
            ]
        )

    category_results = search_chroma(
        collection,
        query_vectors[CATEGORY_EXERCISE_QUERY],
        k=3,
        where={"category": CATEGORY_EXERCISE_VALUE},
    )
    category_rows = [
        _row_from_result(
            f"ChromaDB category={CATEGORY_EXERCISE_VALUE}",
            CATEGORY_EXERCISE_QUERY,
            rank,
            result,
        )
        for rank, result in enumerate(category_results, start=1)
    ]

    recall = {
        system: {
            f"Recall@{k}": compute_recall_at_k(ranked_codes, SEMANTIC_EXERCISES, k)
            for k in (1, 3, 5)
        }
        for system, ranked_codes in {
            "FAISS": faiss_ranked_codes,
            "ChromaDB": chroma_ranked_codes,
        }.items()
    }

    metrics = {
        "corpus_size": len(records),
        "recall": recall,
        "latency_ms": _benchmark_retrieval(
            index,
            collection,
            query_vectors[next(iter(SEMANTIC_EXERCISES))],
            k=3,
            runs=benchmark_runs,
        ),
        "category_filter": {
            "query": CATEGORY_EXERCISE_QUERY,
            "category": CATEGORY_EXERCISE_VALUE,
            "returned_codes": [result.record.doc_code for result in category_results],
        },
    }
    return {
        "semantic_rows": semantic_rows,
        "exact_rows": exact_rows,
        "category_rows": category_rows,
        "metrics": metrics,
    }


def _configure_utf8_stdout() -> None:
    """ให้ Windows console แสดงสถานะภาษาไทยได้แม้เริ่มต้นด้วย cp1252."""
    stdout = sys.stdout
    encoding = getattr(stdout, "encoding", None)
    if not encoding or encoding.lower().replace("-", "") == "utf8":
        return
    try:
        stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass


def _positive_int(value: str) -> int:
    """Argparse validator for settings that cannot be zero or negative."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _print_table(title: str, rows: list[dict[str, object]], columns: list[str]) -> None:
    """พิมพ์ตารางย่อโดยไม่ซ่อนรายละเอียดผลลัพธ์ของใบงานไว้ใน CSV เท่านั้น."""
    print(f"\n{title}")
    table = pd.DataFrame(rows)
    if table.empty:
        print("ไม่มีผลลัพธ์")
        return
    available_columns = [column for column in columns if column in table.columns]
    print(table.loc[:, available_columns].to_string(index=False))


def _print_exercise_summary(summary: dict[str, object]) -> None:
    _print_table(
        "ผล Semantic Search Top-3",
        summary["semantic_rows"],
        ["system", "query", "rank", "doc_code", "doc_id", "value_type", "value"],
    )
    _print_table(
        "ผล Exact Code Match",
        summary["exact_rows"],
        ["doc_code", "approach", "matched_codes", "matched_count"],
    )
    _print_table(
        "ผล Category Filter",
        summary["category_rows"],
        ["query", "rank", "doc_code", "doc_id", "category", "value_type", "value"],
    )

    metrics = summary["metrics"]
    recall = metrics.get("recall", {})
    if recall:
        print("\nRecall@K")
        for system, values in recall.items():
            formatted_values = ", ".join(
                f"{name}={value:.2f}" for name, value in values.items()
            )
            print(f"{system}: {formatted_values}")

    latency = metrics.get("latency_ms", {})
    if latency:
        print("\nRetrieval latency (ms)")
        for system, values in latency.items():
            print(f"{system}: mean={values['mean']:.4f}, median={values['median']:.4f}")


def main(argv: list[str] | None = None) -> int:
    """รับ options จาก command line แล้วสร้างรายงานใบงาน."""
    _configure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="เปรียบเทียบ FAISS และ ChromaDB สำหรับ Hybrid Retrieval ภาษาไทย"
    )
    parser.add_argument(
        "--include-extensions",
        action="store_true",
        help="เพิ่มเอกสาร 5 รายการของ Exercise 3",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=f"Sentence Transformer model (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("chroma_db_comparison"),
        help="โฟลเดอร์ Persistent ChromaDB",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="โฟลเดอร์รายงาน CSV และ JSON",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=_positive_int,
        default=30,
        help="จำนวนรอบสำหรับวัด latency (default: 30)",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_experiment(
            include_extensions=args.include_extensions,
            model_name=args.model_name,
            chroma_dir=args.chroma_dir,
            benchmark_runs=args.benchmark_runs,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2

    write_reports(summary, args.output_dir)
    _print_exercise_summary(summary)
    metrics = summary["metrics"]
    print(f"สร้าง corpus จำนวน {metrics['corpus_size']} เอกสารเรียบร้อย")
    print(f"บันทึกรายงานแล้วที่: {args.output_dir.resolve()}")
    print("- semantic_top3.csv")
    print("- exact_code_comparison.csv")
    print("- metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
