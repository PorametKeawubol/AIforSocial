#!/usr/bin/env python3
"""Activity 6: compare a direct Llama 1B answer against the local RAG answer."""

from __future__ import annotations

import json
from pathlib import Path

from lab_utils import generate_with_metrics
from rag_pipeline import answer_with_rag, build_collection


BASE_DIR = Path(__file__).resolve().parent
QUESTION = "According to the knowledge base, what transport protocol does HTTP/3 use?"


def main() -> int:
    direct = generate_with_metrics("llama3.2:1b", QUESTION, num_ctx=1024, keep_alive="0s")
    rag = answer_with_rag(build_collection(), QUESTION, top_k=3)
    result = {
        "question": QUESTION,
        "direct_llm": direct,
        "rag": rag,
    }
    output = BASE_DIR / "llm_vs_rag_results.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Question: {QUESTION}\n")
    print(f"LLM answer ({direct['wall_seconds']:.2f}s):\n{direct['answer']}\n")
    print(f"RAG answer ({rag['generation_metrics']['wall_seconds']:.2f}s):\n{rag['answer']}")
    print("\nRetrieved evidence:")
    for document in rag["retrieved_documents"]:
        print(f"- {document}")
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
