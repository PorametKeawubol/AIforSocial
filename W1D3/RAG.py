import argparse
import time
from pathlib import Path

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODELS = ["qwen3:8b", "gemma3:12b", "llama3.1:8b"]
DEFAULT_QUESTION = "Quantum หรือ ควอนตัม คืออะไร อธิบายมาให้ครอบคลุมที่สุด"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
MAX_CHUNK_CHARS = 900


def find_knowledge_file() -> Path:
    candidates = [
        Path.cwd() / "knowledge.txt",
        Path(__file__).resolve().with_name("knowledge.txt"),
        Path(__file__).resolve().parents[1] / "knowledge.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("ไม่พบไฟล์ knowledge.txt")


def load_documents(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if not blocks:
        blocks = [line.strip() for line in text.splitlines() if line.strip()]

    docs: list[str] = []
    for block in blocks:
        if len(block) <= MAX_CHUNK_CHARS:
            docs.append(block)
            continue

        sentences = [part.strip() for part in block.replace("\n", " ").split(" ") if part.strip()]
        chunk = ""
        for part in sentences:
            next_chunk = f"{chunk} {part}".strip()
            if len(next_chunk) > MAX_CHUNK_CHARS and chunk:
                docs.append(chunk)
                chunk = part
            else:
                chunk = next_chunk
        if chunk:
            docs.append(chunk)

    return docs


def build_index(docs: list[str]) -> tuple[SentenceTransformer, faiss.IndexFlatL2, np.ndarray]:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    docs_embeddings = embedder.encode(docs, normalize_embeddings=True)
    embeddings = np.array(docs_embeddings, dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return embedder, index, embeddings


def retrieve_context(
    question: str,
    docs: list[str],
    embedder: SentenceTransformer,
    index: faiss.IndexFlatL2,
    k: int,
) -> list[str]:
    q_embedding = embedder.encode([question], normalize_embeddings=True)
    query = np.array(q_embedding, dtype="float32")
    _, indices = index.search(query, k=min(k, len(docs)))
    return [docs[i] for i in indices[0]]


def build_prompt(question: str, contexts: list[str]) -> str:
    context_text = "\n\n".join(f"[บริบท {i + 1}]\n{text}" for i, text in enumerate(contexts))
    return f"""ตอบคำถามโดยอ้างอิงจากบริบทต่อไปนี้เท่านั้น ถ้าบริบทไม่พอให้บอกว่ายังไม่พบข้อมูลเพียงพอ

{context_text}

คำถาม: {question}

คำตอบภาษาไทย:"""


def ask_ollama(model: str, prompt: str) -> tuple[str, float]:
    started_at = time.perf_counter()
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 384,
            },
        },
        timeout=900,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started_at
    return response.json().get("response", "").strip(), elapsed


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def write_report(
    output_path: Path,
    question: str,
    k: int,
    knowledge_path: Path,
    contexts: list[str],
    results: list[tuple[str, float, str]],
) -> None:
    lines = [
        "# ผลลัพธ์ Lab RAG ด้วย Ollama + Python",
        "",
        f"- คำถาม: {question}",
        f"- ไฟล์ความรู้: `{knowledge_path}`",
        f"- จำนวนบริบทที่ค้นคืน: k={k}",
        f"- Embedding model: `{EMBEDDING_MODEL}`",
        "",
        "## บริบทที่ระบบค้นคืน",
        "",
    ]

    for i, context in enumerate(contexts, start=1):
        lines.extend([f"### บริบท {i}", "", context, ""])

    lines.extend(
        [
            "## ผลลัพธ์แต่ละโมเดล",
            "",
            "| โมเดล | เวลา (วินาที) | คำตอบ |",
            "| --- | ---: | --- |",
        ]
    )
    for model, elapsed, answer in results:
        lines.append(f"| {model} | {elapsed:.2f} | {markdown_cell(answer)} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple RAG with Ollama and Python")
    parser.add_argument("-q", "--question", default=DEFAULT_QUESTION)
    parser.add_argument("-k", type=int, default=3, help="จำนวนบริบทที่ต้องการค้นคืน")
    parser.add_argument("--knowledge", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("rag_ollama_results.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    knowledge_path = args.knowledge or find_knowledge_file()
    docs = load_documents(knowledge_path)

    print(f"โหลดฐานความรู้: {knowledge_path} ({len(docs)} documents)")
    print(f"โหลด embedding model: {EMBEDDING_MODEL}")
    embedder, index, _ = build_index(docs)

    contexts = retrieve_context(args.question, docs, embedder, index, args.k)
    prompt = build_prompt(args.question, contexts)

    results: list[tuple[str, float, str]] = []
    for model in args.models:
        print(f"\nกำลังถามโมเดล: {model}")
        try:
            answer, elapsed = ask_ollama(model, prompt)
        except requests.RequestException as exc:
            answer = f"ERROR: {exc}"
            elapsed = 0.0
        results.append((model, elapsed, answer))
        print(f"เวลา: {elapsed:.2f} วินาที")
        print(answer)

    write_report(args.output, args.question, args.k, knowledge_path, contexts, results)
    print(f"\nบันทึกผลลัพธ์: {args.output}")


if __name__ == "__main__":
    main()
