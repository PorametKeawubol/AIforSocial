#!/usr/bin/env python3
"""Part 5: small, local ChromaDB + Ollama RAG pipeline for networking facts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from lab_utils import generate_with_metrics


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:1b"


def load_documents() -> list[str]:
    """Use one knowledge-base sentence per retrievable document (six documents)."""
    documents = [line.strip() for line in KNOWLEDGE_FILE.read_text(encoding="utf-8").splitlines()]
    return [document for document in documents if document]


def build_collection() -> Any:
    documents = load_documents()
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="network_knowledge",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(documents=documents, ids=[f"doc_{index}" for index in range(len(documents))])
    return collection


def retrieve(collection: Any, query: str, top_k: int) -> tuple[list[str], list[str], list[float]]:
    results = collection.query(query_texts=[query], n_results=top_k)
    documents = results["documents"][0]
    ids = results["ids"][0]
    distances = [float(distance) for distance in results["distances"][0]]
    return documents, ids, distances


def answer_with_rag(collection: Any, query: str, top_k: int = 3) -> dict[str, Any]:
    started = time.perf_counter()
    documents, ids, distances = retrieve(collection, query, top_k)
    retrieval_seconds = time.perf_counter() - started
    context = "\n".join(f"- {document}" for document in documents)
    prompt = f"""You are a fact-constrained answer writer. Answer in the same language as the question.
Use only facts explicitly stated in CONTEXT. The facts in CONTEXT are true. State the direct answer first, then give at most one short explanation. Do not reverse TCP, UDP, or QUIC. Do not say information is unavailable if a CONTEXT sentence answers or explains the question.

CONTEXT:
{context}

QUESTION: {query}
ANSWER:"""
    metrics = generate_with_metrics(LLM_MODEL, prompt, num_ctx=1024, keep_alive="0s")
    return {
        "question": query,
        "retrieved_ids": ids,
        "retrieved_documents": documents,
        "retrieval_distances": distances,
        "retrieval_seconds": retrieval_seconds,
        "end_to_end_seconds": retrieval_seconds + metrics["wall_seconds"],
        "answer": metrics["answer"],
        "generation_metrics": {key: value for key, value in metrics.items() if key != "answer"},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="HTTP/3 ใช้ Transport Protocol อะไร?")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "rag_result.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.top_k <= len(load_documents()):
        raise ValueError(f"--top-k must be between 1 and {len(load_documents())}")
    result = answer_with_rag(build_collection(), args.question, args.top_k)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Question: {result['question']}\n")
    print("--- Top-K Retrieved Docs ---")
    for index, document in enumerate(result["retrieved_documents"], start=1):
        print(f"{index}. {document}")
    print("\n--- Llama 3.2 1B Answer ---")
    print(result["answer"])
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
