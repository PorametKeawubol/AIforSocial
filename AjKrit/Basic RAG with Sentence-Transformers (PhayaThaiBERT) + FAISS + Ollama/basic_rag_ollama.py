#!/usr/bin/env python3
"""Basic Thai RAG: Sentence-Transformers + FAISS + Ollama.

The default embedding model follows the Full Code in the supplied lab sheet.
Run one Ollama model at a time, for example:
    python basic_rag_ollama.py --model qwen2.5:1.5b --pull-missing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


DOCUMENTS = [
    "โรงพยาบาลสงขลานครินทร์เป็นโรงพยาบาลศูนย์ในภาคใต้",
    "ยาปฏิชีวนะควรใช้ตามคำสั่งแพทย์เพื่อป้องกันการดื้อยา",
    "การออกกำลังกายช่วยลดความเสี่ยงโรคหัวใจและหลอดเลือด",
    "ประเทศไทยมีการพัฒนา AI สำหรับงานด้านสาธารณสุข",
    
    "โรงพยาบาลสงขลานครินทร์ตั้งอยู่ในอำเภอหาดใหญ่ จังหวัดสงขลา",
    "โรงพยาบาลมหาวิทยาลัยมีบทบาทในการรักษาพยาบาล การเรียนการสอน และงานวิจัยทางการแพทย์",
    "ผู้ป่วยควรแจ้งประวัติการแพ้ยาให้แพทย์หรือเภสัชกรทราบทุกครั้งก่อนรับยา",
    "ระบบ AI ด้านสาธารณสุขช่วยคัดกรองข้อมูลและสนับสนุนการตัดสินใจของบุคลากร แต่ไม่ทดแทนการวินิจฉัยของแพทย์",
]

DEFAULT_QUESTION = "โรงพยาบาลที่ใหญ่ที่สุดในภาคใต้คือที่ไหน?"


def choose_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_retriever(
    documents: list[str], embedding_model_name: str, device: str
) -> tuple[SentenceTransformer, faiss.IndexFlatIP]:
    print(f"Loading embedding model: {embedding_model_name} ({device})", flush=True)
    embedder = SentenceTransformer(embedding_model_name, device=device)
    embeddings = embedder.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embedder, index


def retrieve(
    question: str,
    embedder: SentenceTransformer,
    index: faiss.IndexFlatIP,
    documents: list[str],
    top_k: int,
) -> list[dict[str, object]]:
    query_vector = embedder.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    scores, indices = index.search(np.ascontiguousarray(query_vector, dtype=np.float32), top_k)
    return [
        {"rank": rank + 1, "document": documents[doc_index], "score": float(scores[0][rank])}
        for rank, doc_index in enumerate(indices[0])
    ]


def model_is_available(model: str) -> bool:
    completed = subprocess.run(
        ["ollama", "show", model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )
    return completed.returncode == 0


def pull_model(model: str) -> None:
    print(f"Pulling Ollama model: {model}", flush=True)
    subprocess.run(["ollama", "pull", model], check=True)


def ask_ollama(
    host: str, model: str, context: str, question: str, timeout: int
) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "ตอบเป็นภาษาไทยโดยอ้างอิงเฉพาะ Context ที่ให้มา หาก Context ไม่พอให้บอกว่าไม่พบข้อมูล",
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
        return data["message"]["content"], elapsed
    except KeyError as error:
        raise RuntimeError(f"Unexpected Ollama response: {data}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Ollama model through the Basic RAG lab.")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="One Ollama model to test.")
    parser.add_argument("--pull-missing", action="store_true", help="Pull --model if it is not installed.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-Transformers model (matches the lab Full Code by default).",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Embedding device; default auto-detect.")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--output-dir", default="results", help="Directory for each model's JSON result.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1 or args.top_k > len(DOCUMENTS):
        raise ValueError(f"--top-k must be between 1 and {len(DOCUMENTS)}")

    if not model_is_available(args.model):
        if not args.pull_missing:
            raise RuntimeError(f"Ollama model '{args.model}' is not installed. Re-run with --pull-missing.")
        pull_model(args.model)

    device = choose_device(args.device)
    embedder, index = build_retriever(DOCUMENTS, args.embedding_model, device)
    retrieved = retrieve(args.question, embedder, index, DOCUMENTS, args.top_k)
    context = "\n".join(item["document"] for item in retrieved)

    print("\nRetrieved context:")
    for item in retrieved:
        print(f"  {item['rank']}. score={item['score']:.4f} | {item['document']}")

    print(f"\nAsking {args.model} ...", flush=True)
    answer, elapsed = ask_ollama(args.ollama_host, args.model, context, args.question, args.timeout)
    print(f"\nQ: {args.question}\nA ({args.model}, {elapsed:.2f}s): {answer}")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "embedding_device": device,
        "index": "FAISS IndexFlatIP (cosine similarity via normalized embeddings)",
        "question": args.question,
        "top_k": args.top_k,
        "retrieved": retrieved,
        "generation_seconds": elapsed,
        "answer": answer,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = args.model.replace(":", "_").replace("/", "_") + ".json"
    output_path = output_dir / filename
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, requests.RequestException, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
