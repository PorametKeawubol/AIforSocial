# FAISS and ChromaDB Thai Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python experiment that completes the FAISS and ChromaDB Hybrid RAG worksheet using identical Thai `BAAI/bge-m3` embeddings.

**Architecture:** A single importable module owns the corpus, vector-store adapters, evaluation, report generation, and CLI.  Unit tests use deterministic vectors and import the same module, avoiding a model download.  The command-line path loads `BAAI/bge-m3` once, gives the exact normalized vectors to both FAISS and ChromaDB, and writes reports under the project folder.

**Tech Stack:** Python 3.10+, NumPy, FAISS CPU, ChromaDB, Sentence Transformers, pandas, pytest.

**Spec:** `E:/AIforSocial/AjKrit/DENSE/FAISS/docs/superpowers/specs/2026-08-21-faiss-chromadb-hybrid-rag-design.md`

## Global Constraints

- Keep all source, tests, documents, and generated files under `E:/AIforSocial/AjKrit/DENSE/FAISS/`.
- Use `BAAI/bge-m3` exactly once per full CLI run with `normalize_embeddings=True`.
- Send the same precomputed document and query embeddings to FAISS and ChromaDB; do not configure a second Chroma embedding function.
- Use `faiss.IndexFlatIP` only with `float32` normalized vectors, and label output values as FAISS similarity or ChromaDB cosine distance.
- The base corpus has exactly 8 records; `--include-extensions` adds exactly 5 records.
- A Chroma collection reset may delete only the named collection in `chroma_db_comparison/`; never remove a broad directory or another collection.
- Runtime data goes only to `chroma_db_comparison/` and `results/`; both paths are Git-ignored.
- Implement retrieval only: do not add BM25, LLM generation, a chat UI, or external services.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `hybrid_rag_comparison.py` | Corpus data, types, FAISS/Chroma adapters, metrics, report writers, and CLI entry point. |
| `test_hybrid_rag_comparison.py` | Fast deterministic tests without a Sentence Transformer download. |
| `requirements.txt` | Direct runtime and test dependencies. |
| `.gitignore` | Excludes only generated Chroma and report paths plus Python cache files. |
| `README.md` | Install/run steps, outputs, expected interpretation, and discussion answers. |

### Task 1: Create the project scaffold and test contract

**Files:**

- Create: `E:/AIforSocial/AjKrit/DENSE/FAISS/requirements.txt`
- Create: `E:/AIforSocial/AjKrit/DENSE/FAISS/.gitignore`
- Create: `E:/AIforSocial/AjKrit/DENSE/FAISS/test_hybrid_rag_comparison.py`
- Create: `E:/AIforSocial/AjKrit/DENSE/FAISS/hybrid_rag_comparison.py`

**Interfaces:**

- Consumes: none.
- Produces: an importable `hybrid_rag_comparison` module and a runnable pytest command.

- [x] **Step 1: Add the first failing dataset test**

Create `test_hybrid_rag_comparison.py` with this test and imports:

```python
import hybrid_rag_comparison as app


def test_base_corpus_has_eight_unique_records_with_required_metadata():
    records = app.build_corpus(include_extensions=False)

    assert len(records) == 8
    assert len({record.id for record in records}) == 8
    assert all(record.doc_id and record.category and record.doc_code for record in records)
```

- [x] **Step 2: Run the test to verify the missing-module failure**

Run: `python -m pytest test_hybrid_rag_comparison.py::test_base_corpus_has_eight_unique_records_with_required_metadata -v`

Expected: FAIL during collection because `hybrid_rag_comparison` does not exist.

- [x] **Step 3: Add dependency and generated-file configuration**

Create `requirements.txt` containing exactly:

```text
chromadb>=0.5
faiss-cpu>=1.8
numpy>=1.26
pandas>=2.2
pytest>=8.0
sentence-transformers>=3.0
```

Create `.gitignore` containing exactly:

```gitignore
__pycache__/
.pytest_cache/
*.py[cod]
chroma_db_comparison/
results/
```

- [x] **Step 4: Add the minimum importable module and corpus type**

Create `hybrid_rag_comparison.py` with the domain type and public function:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentRecord:
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


def build_corpus(include_extensions: bool = False) -> list[DocumentRecord]:
    return [
        DocumentRecord("doc_001", "ขั้นตอนการลงทะเบียนเรียนสำหรับนักศึกษาระดับปริญญาตรีผ่านระบบออนไลน์", "DOC-001", "REG", "REG-1001"),
        DocumentRecord("doc_002", "คู่มือการใช้งานระบบยื่นคำร้องออนไลน์สำหรับนักศึกษา รหัสเอกสาร REG-2045", "DOC-002", "REG", "REG-2045"),
        DocumentRecord("doc_003", "ระเบียบการขอเพิ่มและถอนรายวิชาภายในระยะเวลาที่มหาวิทยาลัยกำหนด", "DOC-003", "REG", "REG-3010"),
        DocumentRecord("doc_004", "ขั้นตอนการชำระค่าธรรมเนียมการศึกษาและตรวจสอบสถานะการชำระเงิน", "DOC-004", "FIN", "FIN-4020"),
        DocumentRecord("doc_005", "คู่มือการใช้งานระบบสารสนเทศสำหรับบุคลากร รหัสเอกสาร IT-7788", "DOC-005", "IT", "IT-7788"),
        DocumentRecord("doc_006", "แนวทางการขอหนังสือรับรองการเป็นนักศึกษาผ่านระบบออนไลน์", "DOC-006", "REG", "REG-5090"),
        DocumentRecord("doc_007", "ขั้นตอนการขอใช้บริการ Wi-Fi สำหรับนักศึกษาและบุคลากร", "DOC-007", "IT", "IT-6012"),
        DocumentRecord("doc_008", "ระเบียบการยืมคืนหนังสือและทรัพยากรสารสนเทศของห้องสมุด", "DOC-008", "LIB", "LIB-7033"),
    ]
```

- [x] **Step 5: Run the focused test to verify it passes**

Run: `python -m pytest test_hybrid_rag_comparison.py::test_base_corpus_has_eight_unique_records_with_required_metadata -v`

Expected: PASS.

- [ ] **Step 6: Commit the scaffold**

```bash
git add AjKrit/DENSE/FAISS/requirements.txt AjKrit/DENSE/FAISS/.gitignore AjKrit/DENSE/FAISS/test_hybrid_rag_comparison.py AjKrit/DENSE/FAISS/hybrid_rag_comparison.py
git commit -m "feat: scaffold FAISS Chroma comparison"
```

### Task 2: Complete corpus validation and evaluation utilities

**Files:**

- Modify: `E:/AIforSocial/AjKrit/DENSE/FAISS/hybrid_rag_comparison.py`
- Modify: `E:/AIforSocial/AjKrit/DENSE/FAISS/test_hybrid_rag_comparison.py`

**Interfaces:**

- Consumes: `DocumentRecord` and `build_corpus(include_extensions: bool)` from Task 1.
- Produces: `SearchResult`, `validate_corpus(records)`, `compute_recall_at_k(results_by_query, expected_codes, k)`, `SEMANTIC_EXERCISES`, and `EXACT_CODE_EXERCISES`.

- [x] **Step 1: Add failing tests for extensions, validation, and Recall@K**

Append these tests:

```python
import pytest


def test_extensions_add_five_records_and_keep_ids_unique():
    records = app.build_corpus(include_extensions=True)

    assert len(records) == 13
    assert len({record.id for record in records}) == 13
    assert {"IT-8801", "RES-1201", "LIB-8102", "FIN-5501", "ACT-2201"} <= {
        record.doc_code for record in records
    }


def test_validate_corpus_rejects_duplicate_document_code():
    records = app.build_corpus()
    duplicate = app.DocumentRecord("doc_999", "ข้อความใหม่", "DOC-999", "REG", "REG-1001")

    with pytest.raises(ValueError, match="duplicate doc_code"):
        app.validate_corpus([*records, duplicate])


def test_compute_recall_at_k_counts_a_code_inside_the_requested_window():
    ranked = {"ถามเรื่อง Wi-Fi": ["REG-1001", "IT-6012", "IT-7788"]}
    expected = {"ถามเรื่อง Wi-Fi": "IT-6012"}

    assert app.compute_recall_at_k(ranked, expected, k=1) == 0.0
    assert app.compute_recall_at_k(ranked, expected, k=2) == 1.0
```

- [x] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest test_hybrid_rag_comparison.py -k "extensions or validate_corpus or compute_recall" -v`

Expected: FAIL because the extension data and utility functions do not yet exist.

- [x] **Step 3: Implement the data and utility functions**

Extend `build_corpus` with these five records only when `include_extensions` is true:

```python
EXTENSION_RECORDS = [
    DocumentRecord("doc_009", "คู่มือการใช้งานระบบประชุมออนไลน์สำหรับบุคลากร รหัสเอกสาร IT-8801", "DOC-009", "IT", "IT-8801"),
    DocumentRecord("doc_010", "ระเบียบการขอทุนสนับสนุนการวิจัย รหัสเอกสาร RES-1201", "DOC-010", "RES", "RES-1201"),
    DocumentRecord("doc_011", "คู่มือการใช้ฐานข้อมูลอิเล็กทรอนิกส์ของห้องสมุด รหัสเอกสาร LIB-8102", "DOC-011", "LIB", "LIB-8102"),
    DocumentRecord("doc_012", "ขั้นตอนการขอใบเสร็จรับเงินค่าเล่าเรียน รหัสเอกสาร FIN-5501", "DOC-012", "FIN", "FIN-5501"),
    DocumentRecord("doc_013", "คู่มือการลงทะเบียนกิจกรรมนักศึกษา รหัสเอกสาร ACT-2201", "DOC-013", "ACT", "ACT-2201"),
]


def validate_corpus(records: list[DocumentRecord]) -> None:
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
    results_by_query: dict[str, list[str]], expected_codes: dict[str, str], k: int
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    hits = sum(
        expected_codes[query] in results_by_query.get(query, [])[:k]
        for query in expected_codes
    )
    return hits / len(expected_codes) if expected_codes else 0.0
```

Define `SEMANTIC_EXERCISES` as the five worksheet questions with expected
codes `REG-1001`, `REG-3010`, `FIN-4020`, `REG-5090`, and `IT-6012`; define
`EXACT_CODE_EXERCISES = ("REG-2045", "IT-7788", "FIN-4020", "LIB-7033")`.

- [x] **Step 4: Run the utility tests to verify they pass**

Run: `python -m pytest test_hybrid_rag_comparison.py -k "extensions or validate_corpus or compute_recall" -v`

Expected: 3 PASS.

- [ ] **Step 5: Commit corpus and metric utilities**

```bash
git add AjKrit/DENSE/FAISS/hybrid_rag_comparison.py AjKrit/DENSE/FAISS/test_hybrid_rag_comparison.py
git commit -m "feat: add corpus and retrieval metrics"
```

### Task 3: Implement and test FAISS semantic and application-side metadata search

**Files:**

- Modify: `E:/AIforSocial/AjKrit/DENSE/FAISS/hybrid_rag_comparison.py`
- Modify: `E:/AIforSocial/AjKrit/DENSE/FAISS/test_hybrid_rag_comparison.py`

**Interfaces:**

- Consumes: validated `list[DocumentRecord]` from Task 2 and a normalized `numpy.ndarray` shaped `(n, dimensions)`.
- Produces: `build_faiss_index(embeddings)`, `search_faiss(index, records, query_embedding, k)`, and `search_faiss_with_code(index, records, query_embedding, doc_code, k)` returning `list[SearchResult]`.

- [x] **Step 1: Add failing tests with fixed normalized vectors**

Append these tests:

```python
import numpy as np


def test_faiss_search_aligns_top_result_with_its_metadata():
    records = app.build_corpus()[:3]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype="float32")
    index = app.build_faiss_index(vectors)

    results = app.search_faiss(index, records, np.array([1.0, 0.0], dtype="float32"), k=2)

    assert results[0].record.doc_code == "REG-1001"
    assert results[0].value_type == "similarity"
    assert results[0].score > results[1].score


def test_faiss_application_filter_returns_only_the_requested_code():
    records = app.build_corpus()[:3]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype="float32")
    index = app.build_faiss_index(vectors)

    results = app.search_faiss_with_code(
        index, records, np.array([1.0, 0.0], dtype="float32"), "REG-2045", k=3
    )

    assert [result.record.doc_code for result in results] == ["REG-2045"]


def test_faiss_application_filter_returns_empty_for_unknown_code():
    records = app.build_corpus()[:3]
    index = app.build_faiss_index(np.eye(3, dtype="float32"))

    assert app.search_faiss_with_code(index, records, np.array([1.0, 0.0, 0.0], dtype="float32"), "NONE-0000", k=1) == []
```

- [x] **Step 2: Run the FAISS tests to verify they fail**

Run: `python -m pytest test_hybrid_rag_comparison.py -k faiss -v`

Expected: FAIL because FAISS adapter functions and `SearchResult` are missing.

- [x] **Step 3: Implement the FAISS adapter**

Add this public result type and core implementation, validating dimensions and
converting all vectors to contiguous `float32` before calling FAISS:

```python
@dataclass(frozen=True)
class SearchResult:
    record: DocumentRecord
    score: float
    value_type: str


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    vectors = np.ascontiguousarray(embeddings, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search_faiss(index, records, query_embedding: np.ndarray, k: int) -> list[SearchResult]:
    if k < 1:
        raise ValueError("k must be at least 1")
    if index.ntotal != len(records):
        raise ValueError("FAISS index and record count differ")
    query = np.ascontiguousarray(query_embedding, dtype="float32").reshape(1, -1)
    scores, indices = index.search(query, min(k, len(records)))
    return [
        SearchResult(records[position], float(score), "similarity")
        for score, position in zip(scores[0], indices[0])
        if position >= 0
    ]


def search_faiss_with_code(index, records, query_embedding, doc_code: str, k: int) -> list[SearchResult]:
    ranked = search_faiss(index, records, query_embedding, k=len(records))
    return [result for result in ranked if result.record.doc_code == doc_code][:k]
```

- [x] **Step 4: Run FAISS tests to verify they pass**

Run: `python -m pytest test_hybrid_rag_comparison.py -k faiss -v`

Expected: 3 PASS.

- [ ] **Step 5: Commit the FAISS adapter**

```bash
git add AjKrit/DENSE/FAISS/hybrid_rag_comparison.py AjKrit/DENSE/FAISS/test_hybrid_rag_comparison.py
git commit -m "feat: add FAISS semantic and code retrieval"
```

### Task 4: Implement and test persistent ChromaDB retrieval and metadata filtering

**Files:**

- Modify: `E:/AIforSocial/AjKrit/DENSE/FAISS/hybrid_rag_comparison.py`
- Modify: `E:/AIforSocial/AjKrit/DENSE/FAISS/test_hybrid_rag_comparison.py`

**Interfaces:**

- Consumes: `DocumentRecord`, `validate_corpus`, and precomputed `float32` embeddings from Tasks 2–3.
- Produces: `create_chroma_collection(path, records, embeddings)`, `search_chroma(collection, query_embedding, k, where=None)`, and `chroma_payload_to_results(payload)` returning `list[SearchResult]`.

- [x] **Step 1: Add a failing isolated Chroma metadata-filter test**

Append this test, using pytest's temporary directory so no class database is
altered:

```python
def test_chroma_where_filter_returns_only_requested_metadata(tmp_path):
    records = app.build_corpus()[:3]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype="float32")
    collection = app.create_chroma_collection(tmp_path / "chroma", records, vectors)

    results = app.search_chroma(
        collection,
        np.array([1.0, 0.0], dtype="float32"),
        k=3,
        where={"doc_code": "REG-2045"},
    )

    assert [result.record.doc_code for result in results] == ["REG-2045"]
    assert results[0].value_type == "distance"
```

- [x] **Step 2: Run the Chroma test to verify it fails**

Run: `python -m pytest test_hybrid_rag_comparison.py::test_chroma_where_filter_returns_only_requested_metadata -v`

Expected: FAIL because the Chroma adapter does not exist.

- [x] **Step 3: Implement the persistent Chroma adapter with a scoped reset**

Use a constant `CHROMA_COLLECTION_NAME = "university_documents"`.  The
creation function must delete only that existing collection, then create it
again and insert precomputed embeddings:

```python
def create_chroma_collection(path: Path, records: list[DocumentRecord], embeddings: np.ndarray):
    validate_corpus(records)
    vectors = np.ascontiguousarray(embeddings, dtype="float32")
    if vectors.shape[0] != len(records):
        raise ValueError("Chroma embeddings and record count differ")
    client = chromadb.PersistentClient(path=str(path))
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except ValueError:
        pass
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
```

`search_chroma` must pass `query_embeddings=[query.tolist()]`, constrain
`n_results` to the maximum eligible collection count, request documents and
metadata, and convert result rows into `SearchResult` values whose record is
reconstructed from each returned ID/document/metadata and whose `value_type`
is `"distance"`.

- [x] **Step 4: Run the Chroma test to verify it passes**

Run: `python -m pytest test_hybrid_rag_comparison.py::test_chroma_where_filter_returns_only_requested_metadata -v`

Expected: PASS.

- [ ] **Step 5: Commit the Chroma adapter**

```bash
git add AjKrit/DENSE/FAISS/hybrid_rag_comparison.py AjKrit/DENSE/FAISS/test_hybrid_rag_comparison.py
git commit -m "feat: add persistent Chroma metadata retrieval"
```

### Task 5: Add experiment orchestration, reports, latency, and CLI

**Files:**

- Modify: `E:/AIforSocial/AjKrit/DENSE/FAISS/hybrid_rag_comparison.py`
- Modify: `E:/AIforSocial/AjKrit/DENSE/FAISS/test_hybrid_rag_comparison.py`

**Interfaces:**

- Consumes: both stores, `SEMANTIC_EXERCISES`, `EXACT_CODE_EXERCISES`, and `compute_recall_at_k` from earlier tasks.
- Produces: `run_experiment(...) -> dict[str, object]`, `write_reports(summary, output_dir)`, and `main(argv=None) -> int`.

- [x] **Step 1: Add failing report tests that do not load the model**

Append these tests:

```python
import json


def test_write_reports_creates_required_csv_and_json_files(tmp_path):
    summary = {
        "semantic_rows": [{"system": "FAISS", "query": "ทดสอบ", "rank": 1, "doc_code": "REG-1001", "doc_id": "DOC-001", "value_type": "similarity", "value": 1.0}],
        "exact_rows": [{"doc_code": "REG-2045", "approach": "ChromaDB metadata filter", "matched_codes": "REG-2045", "matched_count": 1}],
        "metrics": {"corpus_size": 8, "recall": {"FAISS": {"Recall@1": 1.0}}, "latency_ms": {}},
    }

    app.write_reports(summary, tmp_path)

    assert (tmp_path / "semantic_top3.csv").exists()
    assert (tmp_path / "exact_code_comparison.csv").exists()
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["corpus_size"] == 8
```

- [x] **Step 2: Run the report test to verify it fails**

Run: `python -m pytest test_hybrid_rag_comparison.py::test_write_reports_creates_required_csv_and_json_files -v`

Expected: FAIL because `write_reports` is missing.

- [x] **Step 3: Implement orchestration and report writing**

Implement `run_experiment` with this behavior:

1. Build and validate the selected corpus.
2. Load `SentenceTransformer(model_name)` only inside this function.
3. Encode the entire corpus once and the five semantic queries once with
   `normalize_embeddings=True`; cast to `float32`.
4. Build both vector stores from the same corpus vectors.
5. For each semantic query, collect FAISS and ChromaDB Top-3 rows into
   `semantic_rows`; save document code, document ID, rank, value type, and
   numeric value.
6. For each exact code, record results for pure FAISS Top-3, the FAISS
   application filter, and ChromaDB `where={"doc_code": code}` in
   `exact_rows`.
7. Compute Recall@1, Recall@3, and Recall@5 for each semantic system.
8. Use `time.perf_counter()` to time only the store calls, after query vectors
   are prepared, for 30 repetitions of the first semantic query.  Store mean
   and median milliseconds for FAISS and ChromaDB.
9. Return a `summary` dictionary with `semantic_rows`, `exact_rows`, and a
   JSON-serializable `metrics` object.

Implement `write_reports` using pandas:

```python
def write_reports(summary: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary["semantic_rows"]).to_csv(
        output_dir / "semantic_top3.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary["exact_rows"]).to_csv(
        output_dir / "exact_code_comparison.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(summary["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
```

Implement an argparse CLI with `--include-extensions`, `--model-name`,
`--chroma-dir`, `--output-dir`, and `--benchmark-runs`; defaults must be
`False`, `BAAI/bge-m3`, `chroma_db_comparison`, `results`, and `30`
respectively.  On model load failure, print the exception and a Thai message
that the first run needs a network connection or a cached model, then return
exit code 2.

- [x] **Step 4: Run focused report tests and all tests**

Run: `python -m pytest test_hybrid_rag_comparison.py -v`

Expected: all tests PASS.

- [x] **Step 5: Execute a real base-corpus smoke test**

Run: `python hybrid_rag_comparison.py --benchmark-runs 3`

Expected: console output says the corpus has 8 documents and `results/`
contains `semantic_top3.csv`, `exact_code_comparison.csv`, and `metrics.json`.

- [x] **Step 6: Execute the extension exercise**

Run: `python hybrid_rag_comparison.py --include-extensions --benchmark-runs 3 --output-dir results/extensions`

Expected: console output says the corpus has 13 documents and the extension
report directory contains the same three files.

- [ ] **Step 7: Commit orchestration and reports**

```bash
git add AjKrit/DENSE/FAISS/hybrid_rag_comparison.py AjKrit/DENSE/FAISS/test_hybrid_rag_comparison.py
git commit -m "feat: run comparison exercises and reports"
```

### Task 6: Document usage, worksheet interpretation, and final verification

**Files:**

- Create: `E:/AIforSocial/AjKrit/DENSE/FAISS/README.md`
- Modify: `E:/AIforSocial/AjKrit/DENSE/FAISS/docs/superpowers/plans/2026-08-21-faiss-chromadb-hybrid-rag-implementation.md`

**Interfaces:**

- Consumes: the final CLI, output file names, and test suite from Tasks 1–5.
- Produces: an end-user guide that can reproduce the assignment without reading the source.

- [x] **Step 1: Create README content**

Write `README.md` with these sections and exact commands:

```markdown
# FAISS และ ChromaDB สำหรับ Hybrid RAG

## ติดตั้ง

```powershell
python -m pip install -r requirements.txt
```

## รันชุดข้อมูลหลัก 8 เอกสาร

```powershell
python hybrid_rag_comparison.py
```

## รัน Exercise 3 พร้อมเอกสารเพิ่ม 5 รายการ

```powershell
python hybrid_rag_comparison.py --include-extensions --output-dir results/extensions
```

## ทดสอบ

```powershell
python -m pytest test_hybrid_rag_comparison.py -v
```
```

Explain the three generated reports, that FAISS values are cosine similarity
and Chroma values are cosine distance, and that the latency report is
retrieval-only.  Answer all three worksheet discussion questions directly:

1. Same vectors should produce broadly comparable semantic rankings, while
   HNSW approximation and score conventions can make ordering/value details
   differ.
2. FAISS indexes vectors only; an application must store and filter metadata.
3. ChromaDB applies an exact `where` constraint before/with vector retrieval,
   preventing an unrelated semantic result from standing in for an identifier.

- [x] **Step 2: Run the final test suite**

Run: `python -m pytest test_hybrid_rag_comparison.py -v`

Expected: all tests PASS with no model download.

- [x] **Step 3: Inspect generated reports after the smoke test**

Run: `Get-Content results/metrics.json`

Expected: JSON includes `corpus_size`, Recall@1/3/5 for both systems, and
mean/median retrieval milliseconds for both systems.

- [ ] **Step 4: Update completed checkboxes and commit documentation**

Mark every completed step in this plan with `[x]`, then run:

```bash
git add AjKrit/DENSE/FAISS/README.md AjKrit/DENSE/FAISS/docs/superpowers/plans/2026-08-21-faiss-chromadb-hybrid-rag-implementation.md
git commit -m "docs: explain FAISS Chroma retrieval comparison"
```

## Plan self-review

- Spec coverage: Tasks 1–2 implement corpus correctness and extensions; Task 3 covers FAISS semantic search and application-side code filtering; Task 4 covers persistent ChromaDB and native metadata filtering; Task 5 covers all exercises, Recall@K, latency, reports, and CLI errors; Task 6 covers reproducible usage and the discussion questions.
- Placeholder scan: no incomplete markers or undefined later-stage interfaces remain.  Every named public function is introduced in the task before another task consumes it.
- Type consistency: both adapters return `list[SearchResult]`; corpus functions use `list[DocumentRecord]`; reports receive JSON-compatible dictionaries from `run_experiment`.
