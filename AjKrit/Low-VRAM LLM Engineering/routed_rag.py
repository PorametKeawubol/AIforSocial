#!/usr/bin/env python3
"""Part 7: route simple prompts to Qwen 0.5B and knowledge prompts to RAG + Llama 1B."""

from __future__ import annotations

import json
import re
from pathlib import Path

from lab_utils import generate_with_metrics
from rag_pipeline import answer_with_rag, build_collection


BASE_DIR = Path(__file__).resolve().parent
ROUTER_MODEL = "qwen2.5:0.5b"


NETWORK_PATTERN = re.compile(
    r"(?:\bhttp/[23]\b|\bquic\b|\btcp\b|\budp\b|\btransport\s+protocol\b)",
    flags=re.IGNORECASE,
)


def route_query(user_query: str) -> tuple[str, str, bool]:
    """Use Qwen as the router, with a narrow domain guardrail for false positives."""
    prompt = f"""Classify the user input into exactly one category: SIMPLE or KNOWLEDGE.
SIMPLE: greeting, small talk, general coding, math, or common knowledge.
KNOWLEDGE: a specific question about HTTP/2, HTTP/3, QUIC, TCP, or UDP that should use the local networking knowledge base.

User Input: {user_query!r}
Category (reply with only SIMPLE or KNOWLEDGE):"""
    router = generate_with_metrics(ROUTER_MODEL, prompt, num_ctx=512, keep_alive="0s")
    decision_text = router["answer"].upper()
    llm_route = "KNOWLEDGE" if decision_text.strip() == "KNOWLEDGE" else "SIMPLE"

    # A 0.5B router can over-classify.  This guardrail preserves the lab's
    # intended networking-only knowledge branch and keeps a casual prompt out
    # of the vector database when it contains no in-domain concept at all.
    has_network_term = bool(NETWORK_PATTERN.search(user_query))
    route = "KNOWLEDGE" if llm_route == "KNOWLEDGE" and has_network_term else "SIMPLE"
    return route, router["answer"], route != llm_route


def process_request(collection: object, query: str) -> dict[str, object]:
    route, raw_router_answer, guardrail_applied = route_query(query)
    if route == "SIMPLE":
        response = generate_with_metrics(ROUTER_MODEL, query, num_ctx=512, keep_alive="0s")
        return {
            "query": query,
            "route": route,
            "raw_router_answer": raw_router_answer,
            "guardrail_applied": guardrail_applied,
            "retrieved_documents": [],
            "answer": response["answer"],
        }

    rag = answer_with_rag(collection, query, top_k=3)
    return {
        "query": query,
        "route": route,
        "raw_router_answer": raw_router_answer,
        "guardrail_applied": guardrail_applied,
        "retrieved_documents": rag["retrieved_documents"],
        "answer": rag["answer"],
    }


def main() -> int:
    collection = build_collection()
    queries = [
        "Hi there! Can you write a quick 3-line poem about coffee?",
        "What transport protocol does HTTP/3 use and why?",
    ]
    results = [process_request(collection, query) for query in queries]
    for result in results:
        print(f"\n[User Query]: {result['query']}")
        print(f"└─► Router Decision: [{result['route']}]")
        if result["route"] == "KNOWLEDGE":
            print("└─► Path: ChromaDB Top-K → Llama 1B")
        else:
            print("└─► Path: Qwen 0.5B")
        print(f"\n[Response]:\n{result['answer']}\n{'=' * 50}")

    output = BASE_DIR / "routed_rag_results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
