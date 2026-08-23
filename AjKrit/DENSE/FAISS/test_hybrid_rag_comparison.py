import json
import io

import numpy as np
import pytest

import hybrid_rag_comparison as app


def test_base_corpus_has_eight_unique_records_with_required_metadata():
    """Breaks if the worksheet's required base corpus is missing or malformed."""
    records = app.build_corpus(include_extensions=False)

    assert len(records) == 8
    assert len({record.id for record in records}) == 8
    assert all(record.doc_id and record.category and record.doc_code for record in records)


def test_extensions_add_five_records_and_keep_ids_unique():
    """Breaks if Exercise 3 does not add exactly its five defined documents."""
    records = app.build_corpus(include_extensions=True)

    assert len(records) == 13
    assert len({record.id for record in records}) == 13
    assert {"IT-8801", "RES-1201", "LIB-8102", "FIN-5501", "ACT-2201"} <= {
        record.doc_code for record in records
    }


def test_validate_corpus_rejects_duplicate_document_code():
    """Breaks if a conflicting exact identifier can silently enter the corpus."""
    records = app.build_corpus()
    duplicate = app.DocumentRecord("doc_999", "ข้อความใหม่", "DOC-999", "REG", "REG-1001")

    with pytest.raises(ValueError, match="duplicate doc_code"):
        app.validate_corpus([*records, duplicate])


def test_compute_recall_at_k_counts_a_code_inside_the_requested_window():
    """Breaks if Recall@K ignores a correct result at a valid rank."""
    ranked = {"ถามเรื่อง Wi-Fi": ["REG-1001", "IT-6012", "IT-7788"]}
    expected = {"ถามเรื่อง Wi-Fi": "IT-6012"}

    assert app.compute_recall_at_k(ranked, expected, k=1) == 0.0
    assert app.compute_recall_at_k(ranked, expected, k=2) == 1.0


def test_faiss_search_aligns_top_result_with_its_metadata():
    """Breaks if index positions become detached from their document metadata."""
    records = app.build_corpus()[:3]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype="float32")
    index = app.build_faiss_index(vectors)

    results = app.search_faiss(index, records, np.array([1.0, 0.0], dtype="float32"), k=2)

    assert results[0].record.doc_code == "REG-1001"
    assert results[0].value_type == "similarity"
    assert results[0].score > results[1].score


def test_faiss_application_filter_returns_only_the_requested_code():
    """Breaks if custom FAISS filtering leaks a semantically similar document."""
    records = app.build_corpus()[:3]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype="float32")
    index = app.build_faiss_index(vectors)

    results = app.search_faiss_with_code(
        index,
        records,
        np.array([1.0, 0.0], dtype="float32"),
        "REG-2045",
        k=3,
    )

    assert [result.record.doc_code for result in results] == ["REG-2045"]


def test_faiss_application_filter_returns_empty_for_unknown_code():
    """Breaks if an unknown identifier is replaced with an unrelated semantic hit."""
    records = app.build_corpus()[:3]
    index = app.build_faiss_index(np.eye(3, dtype="float32"))

    assert (
        app.search_faiss_with_code(
            index,
            records,
            np.array([1.0, 0.0, 0.0], dtype="float32"),
            "NONE-0000",
            k=1,
        )
        == []
    )


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_faiss_application_filter_rejects_nonpositive_k(invalid_k):
    """Breaks if an invalid requested result count changes Python slice semantics."""
    records = app.build_corpus()[:3]
    index = app.build_faiss_index(np.eye(3, dtype="float32"))

    with pytest.raises(ValueError, match="k must be at least 1"):
        app.search_faiss_with_code(
            index,
            records,
            np.array([1.0, 0.0, 0.0], dtype="float32"),
            "REG-1001",
            k=invalid_k,
        )


def test_chroma_where_filter_returns_only_requested_metadata(tmp_path):
    """Breaks if ChromaDB returns a semantic hit outside an exact where clause."""
    records = app.build_corpus()[:3]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype="float32")
    collection = app.create_chroma_collection(tmp_path / "chroma", records, vectors)

    results = app.search_chroma(
        collection,
        np.array([1.0, 0.0], dtype="float32"),
        k=3,
        where={"doc_code": "REG-2045"},
    )

    assert [result.record.doc_code for result in results] == ["REG-2045"]
    assert results[0].value_type == "distance"


def test_write_reports_creates_required_csv_and_json_files(tmp_path):
    """Breaks if a completed experiment cannot be submitted as report files."""
    summary = {
        "semantic_rows": [
            {
                "system": "FAISS",
                "query": "ทดสอบ",
                "rank": 1,
                "doc_code": "REG-1001",
                "doc_id": "DOC-001",
                "value_type": "similarity",
                "value": 1.0,
            }
        ],
        "exact_rows": [
            {
                "doc_code": "REG-2045",
                "approach": "ChromaDB metadata filter",
                "matched_codes": "REG-2045",
                "matched_count": 1,
            }
        ],
        "metrics": {
            "corpus_size": 8,
            "recall": {"FAISS": {"Recall@1": 1.0}},
            "latency_ms": {},
        },
    }

    app.write_reports(summary, tmp_path)

    assert (tmp_path / "semantic_top3.csv").exists()
    assert (tmp_path / "exact_code_comparison.csv").exists()
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["corpus_size"] == 8


class ConstantEmbedder:
    """Fast local stand-in that exercises the stores without a model download."""

    def encode(self, texts, normalize_embeddings=True):
        assert normalize_embeddings is True
        return np.tile(np.array([[1.0, 0.0]], dtype="float32"), (len(texts), 1))


def test_run_experiment_completes_semantic_exact_and_category_exercises(tmp_path):
    """Breaks if the CLI workflow omits a worksheet exercise or a required metric."""
    summary = app.run_experiment(
        include_extensions=False,
        chroma_dir=tmp_path / "chroma",
        benchmark_runs=1,
        embedder=ConstantEmbedder(),
    )

    assert summary["metrics"]["corpus_size"] == 8
    assert len(summary["semantic_rows"]) == len(app.SEMANTIC_EXERCISES) * 2 * 3
    assert {
        row["approach"] for row in summary["exact_rows"]
    } == {
        "FAISS pure semantic",
        "FAISS custom metadata filter",
        "ChromaDB metadata filter",
    }
    assert all(row["category"] == "REG" for row in summary["category_rows"])
    assert set(summary["metrics"]["recall"]["FAISS"]) == {
        "Recall@1",
        "Recall@3",
        "Recall@5",
    }


def test_main_writes_reports_using_the_cli_paths(tmp_path, monkeypatch):
    """Breaks if CLI options do not produce submission files in the selected directory."""
    summary = {
        "semantic_rows": [],
        "exact_rows": [],
        "category_rows": [],
        "metrics": {"corpus_size": 8, "recall": {}, "latency_ms": {}},
    }

    def fake_run_experiment(**kwargs):
        assert kwargs["include_extensions"] is True
        assert kwargs["chroma_dir"] == tmp_path / "chroma"
        assert kwargs["benchmark_runs"] == 1
        return summary

    monkeypatch.setattr(app, "run_experiment", fake_run_experiment)

    exit_code = app.main(
        [
            "--include-extensions",
            "--chroma-dir",
            str(tmp_path / "chroma"),
            "--output-dir",
            str(tmp_path / "reports"),
            "--benchmark-runs",
            "1",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "reports" / "metrics.json").exists()


def test_main_reconfigures_cp1252_stdout_before_printing_thai(tmp_path, monkeypatch):
    """Breaks if a Windows cp1252 console crashes after a successful experiment."""
    summary = {
        "semantic_rows": [],
        "exact_rows": [],
        "category_rows": [],
        "metrics": {"corpus_size": 8, "recall": {}, "latency_ms": {}},
    }
    cp1252_buffer = io.BytesIO()
    cp1252_stdout = io.TextIOWrapper(cp1252_buffer, encoding="cp1252")

    monkeypatch.setattr(app, "run_experiment", lambda **kwargs: summary)
    monkeypatch.setattr(app.sys, "stdout", cp1252_stdout)

    exit_code = app.main(["--output-dir", str(tmp_path / "reports")])
    cp1252_stdout.flush()

    assert exit_code == 0
    assert "สร้าง corpus" in cp1252_buffer.getvalue().decode("utf-8")


@pytest.mark.parametrize("invalid_runs", ["0", "-2"])
def test_main_rejects_nonpositive_benchmark_runs_before_starting_work(monkeypatch, invalid_runs):
    """Breaks if invalid CLI input loads a model or resets a Chroma collection first."""

    def must_not_run(**kwargs):
        raise AssertionError("run_experiment must not be called for an invalid benchmark count")

    monkeypatch.setattr(app, "run_experiment", must_not_run)

    with pytest.raises(SystemExit) as error:
        app.main(["--benchmark-runs", invalid_runs])

    assert error.value.code == 2


def test_main_prints_compact_tables_for_all_worksheet_exercises(tmp_path, monkeypatch, capsys):
    """Breaks if users must open CSV files before they can inspect lab results."""
    summary = {
        "semantic_rows": [
            {
                "system": "FAISS",
                "query": "ลงทะเบียนเรียน",
                "rank": 1,
                "doc_code": "REG-1001",
                "doc_id": "DOC-001",
                "category": "REG",
                "value_type": "similarity",
                "value": 0.99,
            }
        ],
        "exact_rows": [
            {
                "doc_code": "REG-2045",
                "approach": "ChromaDB metadata filter",
                "matched_codes": "REG-2045",
                "matched_count": 1,
            }
        ],
        "category_rows": [
            {
                "system": "ChromaDB category=REG",
                "query": "ระบบสำหรับนักศึกษา",
                "rank": 1,
                "doc_code": "REG-1001",
                "doc_id": "DOC-001",
                "category": "REG",
                "value_type": "distance",
                "value": 0.01,
            }
        ],
        "metrics": {"corpus_size": 8, "recall": {}, "latency_ms": {}},
    }
    monkeypatch.setattr(app, "run_experiment", lambda **kwargs: summary)

    assert app.main(["--output-dir", str(tmp_path / "reports")]) == 0

    output = capsys.readouterr().out
    assert "ผล Semantic Search Top-3" in output
    assert "ผล Exact Code Match" in output
    assert "ผล Category Filter" in output
    assert "REG-1001" in output
