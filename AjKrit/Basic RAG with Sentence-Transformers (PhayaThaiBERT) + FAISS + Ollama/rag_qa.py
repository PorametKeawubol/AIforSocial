#!/usr/bin/env python3
"""Stage 2: run the 15 Thai paraphrase questions over a saved FAISS index."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from sentence_transformers import SentenceTransformer

from build_faiss_index import DEFAULT_EMBEDDING_MODEL, DEFAULT_OUTPUT_DIR
from rag_common import DEFAULT_QUESTIONS, PROJECT_DIR, as_float32, choose_device, read_artifacts, safe_filename


DEFAULT_MODELS = ("qwen2.5:1.5b", "llama3.2:3b", "qwen2.5:7b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved FAISS retrieval and Ollama answers on thai_qa_paraphrase_15.csv."
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Ollama models, run one at a time.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=8_000,
        help="Maximum combined Context length sent to Ollama (default: 8000 characters).",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Default: automatically select CUDA when available.")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results" / "thai_qa_15")
    parser.add_argument("--pull-missing", action="store_true", help="Pull a model that is absent locally.")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Verify retrieval and write the report without contacting Ollama.",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            expected = {"instruction", "answer", "__index_level_0__"}
            if not reader.fieldnames or not expected.issubset(reader.fieldnames):
                raise ValueError(f"CSV needs columns {sorted(expected)}; got {reader.fieldnames}")
            rows = list(reader)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Question CSV not found: {path}. Supply --questions with thai_qa_paraphrase_15.csv."
        ) from error
    if not rows:
        raise ValueError("Question CSV has no records")
    for number, row in enumerate(rows, start=1):
        if not row["instruction"].strip():
            raise ValueError(f"Question row {number} has an empty instruction")
    return rows


def model_is_available(model: str) -> bool:
    completed = subprocess.run(
        ["ollama", "show", model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )
    return completed.returncode == 0


def ensure_model(model: str, pull_missing: bool) -> None:
    if model_is_available(model):
        return
    if not pull_missing:
        raise RuntimeError(f"Ollama model '{model}' is not installed. Re-run with --pull-missing.")
    print(f"Pulling Ollama model: {model}", flush=True)
    subprocess.run(["ollama", "pull", model], check=True)


def ask_ollama(host: str, model: str, context: str, question: str, timeout: int) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "ตอบเป็นภาษาไทย โดยอ้างอิงเฉพาะ Context ที่ให้มาเท่านั้น "
                    "ตอบอย่างกระชับ และหาก Context ไม่มีคำตอบ ให้ตอบว่า 'ไม่พบข้อมูลใน Context'"
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
        "stream": False,
        "keep_alive": "0",
        "options": {"temperature": 0},
    }
    started = time.perf_counter()
    response = requests.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    try:
        return str(data["message"]["content"]).strip(), elapsed
    except KeyError as error:
        raise RuntimeError(f"Unexpected Ollama response: {data}") from error


def retrieve_questions(
    questions: list[dict[str, str]], embedder: SentenceTransformer, index: Any, documents: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    query_vectors = as_float32(
        embedder.encode(
            [row["instruction"].strip() for row in questions],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )
    scores, indices = index.search(query_vectors, top_k)
    output: list[dict[str, Any]] = []
    indexed_original_ids = {str(document["original_id"]) for document in documents}
    for position, row in enumerate(questions):
        gold_id = str(row["__index_level_0__"]).strip()
        retrieved = [
            {
                "rank": rank + 1,
                "faiss_id": int(document_id),
                "original_id": documents[int(document_id)]["original_id"],
                "score": float(scores[position][rank]),
                "passage": documents[int(document_id)]["passage"],
            }
            for rank, document_id in enumerate(indices[position])
            if document_id >= 0
        ]
        hit_at_k = any(str(item["original_id"]) == gold_id for item in retrieved)
        output.append(
            {
                "question_number": position + 1,
                "question": row["instruction"].strip(),
                "original_question": row.get("instruction_org", "").strip(),
                "expected_answer": row["answer"].strip(),
                "gold_original_id": gold_id,
                "gold_in_index": gold_id in indexed_original_ids,
                "retrieval_hit_at_k": hit_at_k,
                "retrieved": retrieved,
            }
        )
    return output


def context_from(retrieved: list[dict[str, Any]], max_context_chars: int) -> str:
    """Keep every retrieved passage represented without exceeding the prompt budget.

    The Thai QA passages can be much longer than an Ollama context window.  A
    dynamic equal-share allocation avoids spending the entire prompt on rank 1
    and causing the model server to silently truncate the request.
    """
    pieces: list[str] = []
    remaining = max_context_chars
    for position, item in enumerate(retrieved):
        remaining_documents = len(retrieved) - position
        header = f"[เอกสาร {item['rank']}; id={item['original_id']}]\n"
        passage_budget = max(0, (remaining - len(header)) // remaining_documents)
        passage = str(item["passage"])
        if len(passage) > passage_budget:
            passage = passage[: max(0, passage_budget - 1)].rstrip() + "…"
        piece = header + passage
        pieces.append(piece)
        remaining -= len(piece) + 2  # separator added by join
    return "\n\n".join(pieces)


def retrieval_summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    evaluable = [row for row in rows if row["gold_in_index"]]
    hits = sum(bool(row["retrieval_hit_at_k"]) for row in evaluable)
    return {
        "questions": len(rows),
        "gold_passages_in_index": len(evaluable),
        "gold_passages_outside_index": len(rows) - len(evaluable),
        f"retrieval_hit_at_{top_k}": hits,
        f"retrieval_recall_at_{top_k}": round(hits / len(evaluable), 4) if evaluable else None,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "question_number", "question", "expected_answer", "gold_original_id", "gold_in_index",
        "retrieval_hit_at_k", "answer_contains_expected", "generation_seconds", "answer",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_model(
    model: str, rows: list[dict[str, Any]], args: argparse.Namespace, retrieval: dict[str, Any]
) -> dict[str, Any]:
    print(f"\nRunning Ollama model: {model}", flush=True)
    ensure_model(model, args.pull_missing)
    model_rows: list[dict[str, Any]] = []
    for row in rows:
        answer, elapsed = ask_ollama(
            args.ollama_host,
            model,
            context_from(row["retrieved"], args.max_context_chars),
            row["question"],
            args.timeout,
        )
        result = dict(row)
        result["answer"] = answer
        result["generation_seconds"] = round(elapsed, 4)
        result["answer_contains_expected"] = bool(
            row["expected_answer"] and row["expected_answer"] in answer
        )
        model_rows.append(result)
        print(f"  {row['question_number']:>2}/{len(rows)} {elapsed:6.2f}s | {answer[:100]}", flush=True)

    answer_hits = sum(bool(row["answer_contains_expected"]) for row in model_rows)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "retrieval": retrieval,
        "max_context_chars": args.max_context_chars,
        "generation": {
            "answer_contains_expected": answer_hits,
            "answer_contains_expected_rate": round(answer_hits / len(model_rows), 4),
            "mean_generation_seconds": round(
                sum(float(row["generation_seconds"]) for row in model_rows) / len(model_rows), 4
            ),
        },
        "results": model_rows,
    }
    safe_model = safe_filename(model)
    write_json(args.output_dir / f"{safe_model}.json", report)
    write_csv(args.output_dir / f"{safe_model}.csv", model_rows)
    return report


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.timeout < 1:
        raise ValueError("--timeout must be at least 1")
    if args.max_context_chars < 1:
        raise ValueError("--max-context-chars must be at least 1")

    index, documents, manifest = read_artifacts(args.index_dir)
    if args.top_k > index.ntotal:
        raise ValueError(f"--top-k must be no more than index size ({index.ntotal:,})")
    questions = load_questions(args.questions)
    embedding_model = str(manifest.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    device = choose_device(args.device)
    print(f"Loading embedding model: {embedding_model} ({device})", flush=True)
    embedder = SentenceTransformer(embedding_model, device=device)
    rows = retrieve_questions(questions, embedder, index, documents, args.top_k)
    retrieval = retrieval_summary(rows, args.top_k)
    print("\nRetrieval summary:", json.dumps(retrieval, ensure_ascii=False), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_generation:
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "index_manifest": manifest,
            "retrieval": retrieval,
            "max_context_chars": args.max_context_chars,
            "results": rows,
        }
        write_json(args.output_dir / "retrieval_only.json", report)
        write_csv(args.output_dir / "retrieval_only.csv", rows)
        print(f"Saved retrieval-only reports to {args.output_dir}")
        return 0

    reports = [run_model(model, rows, args, retrieval) for model in args.models]
    comparison = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_manifest": manifest,
        "retrieval": retrieval,
        "max_context_chars": args.max_context_chars,
        "models": [
            {
                "model": report["model"],
                **report["generation"],
            }
            for report in reports
        ],
    }
    write_json(args.output_dir / "model_comparison.json", comparison)
    print(f"\nSaved reports and model_comparison.json to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        requests.RequestException,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
