from __future__ import annotations

import json

import numpy as np
import pytest

import faiss_chromadb_benchmark as app


def test_loader_skips_rows_without_context_and_returns_unique_ids(tmp_path):
    source = [
        {"input": "บริบทหนึ่ง", "instruction": "ถามหนึ่ง", "answer": "ตอบหนึ่ง", "source": "a", "__index_level_0__": 7},
        {"input": None, "instruction": "ไม่มีบริบท", "answer": "ตอบ", "source": "wiki_qa", "__index_level_0__": 8},
        {"input": "บริบทสอง", "instruction": "ถามสอง", "answer": "ตอบสอง", "source": "b", "__index_level_0__": 7},
    ]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

    records, source_count = app.load_documents(path, limit=2)

    assert source_count == 3
    assert len(records) == 2
    assert [record.id for record in records] == ["doc-00000", "doc-00001"]
    assert [record.dataset_index for record in records] == [7, 7]


def test_query_loader_handles_utf8_bom_and_uses_instruction_column(tmp_path):
    path = tmp_path / "queries.csv"
    path.write_text("\ufeffinstruction,answer,__index_level_0__\nคำถาม,คำตอบ,12\n", encoding="utf-8")

    queries = app.load_queries(path)

    assert queries[0].text == "คำถาม"
    assert queries[0].answer == "คำตอบ"
    assert queries[0].dataset_index == 12


def test_encode_texts_returns_float32_unit_vectors():
    class FakeEmbedder:
        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return np.array([[3.0, 4.0] for _ in texts], dtype="float64")

    vectors = app.encode_texts(FakeEmbedder(), ["a", "b"], show_progress_bar=False)

    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])


def test_latency_summary_contains_requested_worksheet_metrics():
    summary = app.latency_summary([1.0, 2.0, 3.0])

    assert summary["mean_ms"] == pytest.approx(2.0)
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 3.0
    assert summary["qps"] == pytest.approx(500.0)


def test_faiss_search_preserves_document_metadata():
    if app.faiss is None:
        pytest.skip("faiss-cpu is not installed")
    records = [
        app.DocumentRecord("d0", "เอกสารหนึ่ง", "ถาม", "ตอบ", "x", 0),
        app.DocumentRecord("d1", "เอกสารสอง", "ถาม", "ตอบ", "x", 1),
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    index = app.build_faiss_index(vectors)

    results = app.search_faiss(index, records, np.array([1.0, 0.0]), k=2)

    assert results[0].record.id == "d0"
    assert results[0].value_type == "similarity"
    assert results[0].score > results[1].score


def test_chroma_where_filter_returns_only_requested_source(tmp_path):
    if app.faiss is None or app.chromadb is None:
        pytest.skip("FAISS และ ChromaDB ต้องติดตั้งก่อน")
    records = [
        app.DocumentRecord("d0", "เอกสารหนึ่ง", "ถาม", "ตอบ", "x", 0),
        app.DocumentRecord("d1", "เอกสารสอง", "ถาม", "ตอบ", "y", 1),
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    collection = app.create_chroma_collection(tmp_path / "chroma", records, vectors, collection_name="test_collection")

    results = app.search_chroma(
        collection,
        np.array([0.0, 1.0]),
        k=5,
        where={"source": "x"},
        records_by_id={record.id: record for record in records},
    )

    assert [result.record.id for result in results] == ["d0"]
    assert results[0].value_type == "distance"
