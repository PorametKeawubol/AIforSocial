from pathlib import Path
import time

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


MODEL = "gemma3:12b"
K = 3
QUESTION = "Quantum หรือ ควอนตัม คืออะไรอธิบายมาให้ครอบคลุมที่สุด"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
KNOWLEDGE_PATH = Path(__file__).resolve().parents[2] / "knowledge.txt"


def load_knowledge() -> list[str]:
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def main() -> None:
    embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    knowledge = load_knowledge()

    docs_embeddings = embedder.encode(knowledge)
    docs_embeddings = np.array(docs_embeddings, dtype="float32")
    index = faiss.IndexFlatL2(docs_embeddings.shape[1])
    index.add(docs_embeddings)

    q_embedding = embedder.encode([QUESTION])
    q_embedding = np.array(q_embedding, dtype="float32")
    _, indices = index.search(q_embedding, k=K)

    retrieved_contexts = [knowledge[i] for i in indices[0]]
    context = "\n\n".join(retrieved_contexts)
    prompt = f'จากข้อความนี้:\n"{context}"\n\nตอบคำถาม: {QUESTION}'

    started_at = time.perf_counter()
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=900,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started_at

    print(f"Model: {MODEL}")
    print(f"k: {K}")
    print(f"Question: {QUESTION}")
    print("\nRetrieved context:")
    print(context)
    print(f"\nTime: {elapsed:.2f} seconds")
    print("\nAnswer:")
    print(response.json()["response"])


if __name__ == "__main__":
    main()
