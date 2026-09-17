"""Retrieve PDF passages from FAISS and answer with a local Ollama model.

The FAISS index is loaded once per Python process.  It is deliberately not built
here: run build_faiss.py again only after the source PDF changes.
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


PROJECT_DIR = Path(__file__).resolve().parent
FAISS_PATH = Path(os.getenv("FAISS_PATH", PROJECT_DIR / "faiss_index"))
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    if not (FAISS_PATH / "index.faiss").is_file():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_PATH}. Run: python build_faiss.py"
        )
    # The index is created locally by build_faiss.py; do not load untrusted pickle files.
    return FAISS.load_local(
        str(FAISS_PATH), get_embeddings(), allow_dangerous_deserialization=True
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0)


def search_documents(question: str, k: int = DEFAULT_TOP_K) -> list[Document]:
    """Return the top-k most semantically similar chunks."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")
    return get_vectorstore().similarity_search(question, k=k)


def pages_for(documents: Iterable[Document]) -> list[int]:
    return sorted(
        {
            page
            for document in documents
            for page in document.metadata.get("pages", [])
            if isinstance(page, int)
        }
    )


def format_context(documents: Iterable[Document]) -> str:
    parts: list[str] = []
    for number, document in enumerate(documents, start=1):
        pages = document.metadata.get("pages", [])
        parts.append(f"[เอกสารส่วนที่ {number} | หน้า {pages}]\n{document.page_content}")
    return "\n\n".join(parts)


def llm_response(question: str, k: int = DEFAULT_TOP_K) -> str:
    """Answer strictly from retrieved context, with deterministic page citations."""
    documents = search_documents(question, k=k)
    if not documents:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"

    prompt = f"""คุณเป็นผู้ช่วยตอบคำถามจากเอกสารโซลาร์เซลล์
กฎการตอบ:
1. ใช้ข้อมูลจาก Context เท่านั้น
2. ถ้า Context ไม่มีข้อมูลเพียงพอ ให้ตอบว่า "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"
3. ห้ามคาดเดาหรือสร้างข้อมูลขึ้นมาเอง
4. ตอบภาษาไทยแบบสั้น กระชับ และเข้าใจง่าย

Context:
{format_context(documents)}

คำถาม: {question}
คำตอบ:"""
    answer = get_llm().invoke(prompt).content.strip()
    pages = pages_for(documents)
    if pages:
        answer += f"\n\nแหล่งข้อมูล: หน้า {', '.join(map(str, pages))}"
    return answer


def print_retrieved_documents(documents: list[Document]) -> None:
    print("=" * 60)
    print("TOP-K DOCUMENTS")
    print("=" * 60)
    for number, document in enumerate(documents, start=1):
        print(f"\n--- Document {number} ---")
        print(f"Pages: {document.metadata.get('pages', [])}")
        print(document.page_content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a local FAISS RAG index.")
    parser.add_argument("question", nargs="?", help="Question to ask (Thai or English)")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    parser.add_argument(
        "--retrieve-only", action="store_true", help="Print retrieved chunks without calling Ollama"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = args.question or input("Question: ").strip()
    documents = search_documents(question, k=args.k)
    print_retrieved_documents(documents)
    if not args.retrieve_only:
        print("\n" + "=" * 60)
        print("RAG ANSWER")
        print("=" * 60)
        print(llm_response(question, k=args.k))


if __name__ == "__main__":
    main()
