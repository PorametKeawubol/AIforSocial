#!/usr/bin/env python3
"""Run the supplied Low-VRAM lab steps without changing their model or prompts.

This runner intentionally follows the sheet's core settings:
* Activity 1: the same HTTP/2-vs-HTTP/3 prompt for the three specified models.
* Activity 2: Llama 3.2 1B with the sheet's RAG prompt.
* Step 9/Activity 4: ChatOllama with num_gpu=99, num_thread=4, and contexts
  512, 1024, and 2048.
* Activity 5/6: all-MiniLM-L6-v2 + in-memory ChromaDB + Llama 3.2 1B,
  RETRIEVAL_K=3, and the supplied knowledge-base text and prompt.

Only two measurement details are added: wall-clock timing and ROCm VRAM samples.
They do not alter the model prompts or answers.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import ChatOllama
from ollama import Client

from lab_utils import VRAMSampler, bytes_to_mib, response_to_dict, vram_used_bytes, wait_for_vram_settle


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "strict_lab_results.json"
OLLAMA = Client(host="http://127.0.0.1:11434")

MICRO_MODELS = (
    ("Qwen2.5", "qwen2.5:0.5b", "0.5B"),
    ("Llama 3.2", "llama3.2:1b", "1B"),
    ("SmolLM2", "smollm2:1.7b", "1.7B"),
)
ACTIVITY_1_PROMPT = "Explain the difference between HTTP/2 and HTTP/3."
VRAM_PROMPT = "Explain Retrieval-Augmented Generation."
APP_PROMPT = "Explain HTTP/3 in simple terms."
RETRIEVAL_K = 3
KNOWLEDGE_FILE = BASE_DIR / "knowledge.txt"


def stop_model(model: str) -> None:
    """Unload after a test, which is the non-interactive equivalent of /bye."""
    subprocess.run(["ollama", "stop", model], check=False, capture_output=True, text=True)


def direct_generate(model: str, prompt: str, *, keep_alive: str = "0s", sample_vram: bool = True) -> dict[str, Any]:
    """Use the sheet's direct ``ollama.generate`` path with no custom options."""
    sampler = VRAMSampler() if sample_vram else None
    if sampler:
        sampler.start()
    started = time.perf_counter()
    try:
        response = response_to_dict(
            OLLAMA.generate(model=model, prompt=prompt, keep_alive=keep_alive)
        )
    finally:
        if sampler:
            sampler.stop()
    wall_seconds = time.perf_counter() - started
    eval_count = int(response.get("eval_count") or 0)
    eval_seconds = int(response.get("eval_duration") or 0) / 1_000_000_000
    return {
        "answer": str(response.get("response", "")).strip(),
        "response_time_seconds": wall_seconds,
        "generated_tokens": eval_count,
        "tokens_per_second": (eval_count / eval_seconds) if eval_seconds else None,
        "peak_vram_mib": bytes_to_mib(sampler.peak_bytes) if sampler else None,
    }


def activity_1() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, model, parameter_count in MICRO_MODELS:
        stop_model(model)
        result = direct_generate(model, ACTIVITY_1_PROMPT)
        rows.append({"model": label, "parameter": parameter_count, **result})
    return rows


def activity_2() -> dict[str, Any]:
    model = "llama3.2:1b"
    stop_model(model)
    before = vram_used_bytes()
    generation = direct_generate(model, VRAM_PROMPT, keep_alive="10m")
    loaded = vram_used_bytes()
    stop_model(model)
    after = wait_for_vram_settle()
    return {
        "model": model,
        "prompt": VRAM_PROMPT,
        "before_run_mib": bytes_to_mib(before),
        "model_loaded_mib": bytes_to_mib(loaded),
        "during_generate_peak_mib": generation["peak_vram_mib"],
        "after_bye_mib": bytes_to_mib(after),
        "answer": generation["answer"],
    }


def chat_ollama_once(num_ctx: int) -> dict[str, Any]:
    """Step 9 parameters exactly, with timing/VRAM observation around invoke()."""
    stop_model("llama3.2:1b")
    llm = ChatOllama(
        model="llama3.2:1b",
        num_gpu=99,
        num_ctx=num_ctx,
        num_thread=4,
    )
    sampler = VRAMSampler()
    sampler.start()
    started = time.perf_counter()
    try:
        response = llm.invoke(APP_PROMPT)
        response_time_seconds = time.perf_counter() - started
    finally:
        sampler.stop()
    stop_model("llama3.2:1b")
    return {
        "num_ctx": num_ctx,
        "response_time_seconds": response_time_seconds,
        "peak_vram_mib": bytes_to_mib(sampler.peak_bytes),
        "answer": str(response.content).strip(),
    }


def build_lab_collection() -> tuple[Any, int]:
    """Use the splitter and embedding model specified in the sheet verbatim."""
    raw_text = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    documents = [document.strip() for document in raw_text.split("\n\n") if document.strip()]
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name=f"network_knowledge_{time.time_ns()}",
        embedding_function=embedding_func,
    )
    collection.add(documents=documents, ids=[f"doc_{index}" for index in range(len(documents))])
    return collection, len(documents)


def ask_rag(collection: Any, document_count: int, query: str) -> dict[str, Any]:
    """The Activity 5 prompt and direct Llama 3.2 1B generation from the sheet."""
    started = time.perf_counter()
    # Invoke ChromaDB with the worksheet's K=3 exactly. Its supplied text
    # becomes one paragraph under the worksheet's blank-line splitter, so
    # ChromaDB correctly returns the one document it actually contains.
    result = collection.query(query_texts=[query], n_results=RETRIEVAL_K)
    retrieved_docs = result["documents"][0]
    effective_k = len(retrieved_docs)
    context = "\n- ".join(retrieved_docs)
    prompt = f"""Answer the question based only on the provided context.
Context:
- {context}
Question: {query}
Answer:"""
    generation = direct_generate("llama3.2:1b", prompt, sample_vram=False)
    return {
        "question": query,
        "requested_retrieval_k": RETRIEVAL_K,
        "effective_retrieval_k": effective_k,
        "retrieved_docs": retrieved_docs,
        "answer": generation["answer"],
        "response_time_seconds": time.perf_counter() - started,
    }


def activity_5_and_6() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    collection, document_count = build_lab_collection()
    activity_5_questions = (
        "HTTP/3 ใช้ Transport Protocol อะไร?",
        "HTTP/2 และ HTTP/3 แตกต่างกันอย่างไร?",
        "QUIC มีความเกี่ยวข้องกับ HTTP/3 อย่างไร?",
    )
    activity_5_results = [ask_rag(collection, document_count, query) for query in activity_5_questions]

    question = "According to the knowledge base, what transport protocol does HTTP/3 use?"
    direct_llm = direct_generate("llama3.2:1b", question, sample_vram=False)
    rag = ask_rag(collection, document_count, question)
    return activity_5_results, {"question": question, "llm": direct_llm, "rag": rag}


def route_query(user_query: str) -> dict[str, Any]:
    """Part 7 router exactly: Qwen decides SIMPLE or KNOWLEDGE with no heuristic override."""
    router_prompt = f"""Classify the user input into exactly one category: 'SIMPLE' or 'KNOWLEDGE'.
- SIMPLE: Greetings, small talk, general coding, math, or common knowledge.
- KNOWLEDGE: Specific questions about networking protocols (HTTP/2, HTTP/3, QUIC, TCP, UDP).
User Input: \"{user_query}\"
Category (Reply with ONLY 'SIMPLE' or 'KNOWLEDGE'):"""
    router = direct_generate("qwen2.5:0.5b", router_prompt, sample_vram=False)
    category = router["answer"].strip().upper()
    return {"decision": "KNOWLEDGE" if "KNOWLEDGE" in category else "SIMPLE", "raw_answer": router["answer"]}


def activity_7() -> list[dict[str, Any]]:
    collection, document_count = build_lab_collection()
    requests = (
        "Hi there! Can you write a quick 3-line poem about coffee?",
        "What transport protocol does HTTP/3 use and why?",
    )
    results: list[dict[str, Any]] = []
    for query in requests:
        route = route_query(query)
        if route["decision"] == "SIMPLE":
            answer = direct_generate("qwen2.5:0.5b", query, sample_vram=False)["answer"]
            retrieved_docs: list[str] = []
        else:
            rag = ask_rag(collection, document_count, query)
            answer = rag["answer"]
            retrieved_docs = rag["retrieved_docs"]
        results.append({"query": query, **route, "retrieved_docs": retrieved_docs, "answer": answer})
    return results


def main() -> int:
    started = time.perf_counter()
    print("[1/6] Activity 1: Micro-LLM comparison", flush=True)
    micro_llm = activity_1()
    print("[2/6] Activity 2: VRAM measurement", flush=True)
    vram = activity_2()
    print("[3/6] Step 9 + Activity 4: ChatOllama context sizes", flush=True)
    context_sizes = [chat_ollama_once(value) for value in (512, 1024, 2048)]
    print("[4/6] Activity 5: Local RAG", flush=True)
    activity_5, activity_6 = activity_5_and_6()
    print("[5/6] Part 7: routed assistant", flush=True)
    activity_7_results = activity_7()
    result = {
        "lab_settings": {
            "activity_1_prompt": ACTIVITY_1_PROMPT,
            "activity_2_prompt": VRAM_PROMPT,
            "step_9_prompt": APP_PROMPT,
            "retrieval_k": RETRIEVAL_K,
            "knowledge_file": KNOWLEDGE_FILE.name,
        },
        "activity_1": micro_llm,
        "activity_2": vram,
        "activity_4": context_sizes,
        "activity_5": activity_5,
        "activity_6": activity_6,
        "activity_7": activity_7_results,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[6/6] Saved actual results: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
