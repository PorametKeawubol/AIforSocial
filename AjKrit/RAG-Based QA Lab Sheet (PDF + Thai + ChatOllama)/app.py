"""Thai PDF RAG chatbot backed by Chroma and Ollama.

Run after installing requirements and starting Ollama:
    ollama pull qwen2.5:7b
    ollama serve
    python app.py
"""

import argparse
import csv
import hashlib
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = Path(os.environ.get("RAG_PDF_PATH", BASE_DIR / "solarcell-basic-knowledge-SolarHub.pdf"))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", BASE_DIR / "chroma_db"))
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "intfloat/multilingual-e5-base")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE")  # e.g. "cuda" or "cpu"
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "120"))
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "4"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# Thai often uses zero-width space (U+200B) as a word boundary. Keep it until
# chunking, so it becomes a useful split point instead of an invisible token.
THAI_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "! ",
    "? ",
    ". ",
    "\u200b",
    " ",
    "",
]


class E5Embeddings(Embeddings):
    """Embedding adapter that applies E5's required query/passage prefixes."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        options = {"device": device} if device else {}
        self._model = SentenceTransformer(model_name, **options)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}"])[0]


def normalize_thai_text(text: str) -> str:
    """Normalize PDF text while retaining Thai zero-width word boundaries."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "").replace("\u200c", "\u200b").replace("\u200d", "\u200b")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages_from_pdf(path: Path) -> list[tuple[int, str]]:
    """Extract each PDF page in reading order with PyMuPDF."""
    if not path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์ PDF: {path}")

    pages: list[tuple[int, str]] = []
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = normalize_thai_text(page.get_text("text", sort=True))
            if text:
                pages.append((page_number, text))
    return pages


def chunk_pages(
    pages: Iterable[tuple[int, str]], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    """Split pages recursively at Thai-friendly semantic boundaries."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size ต้องมากกว่า overlap และ overlap ต้องไม่ติดลบ")

    page_documents: list[Document] = []
    for page_number, text in pages:
        page_documents.append(
            Document(page_content=text, metadata={"source": PDF_PATH.name, "page": page_number})
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=THAI_SEPARATORS,
        keep_separator=False,
    )
    chunks = splitter.split_documents(page_documents)
    return [
        Document(page_content=chunk.page_content, metadata={**chunk.metadata, "chunk_id": index})
        for index, chunk in enumerate(chunks)
    ]


def pdf_fingerprint(path: Path) -> str:
    """Fingerprint input content so an old Chroma index is never silently reused."""
    digest = hashlib.sha256()
    with path.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_or_load_vectorstore() -> Chroma:
    print(f"[EMBEDDING] Loading model: {EMBED_MODEL_NAME}")
    embedding = E5Embeddings(EMBED_MODEL_NAME, EMBED_DEVICE)
    collection_metadata = {
        "pdf_sha256": pdf_fingerprint(PDF_PATH),
        "embedding_model": EMBED_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "ingestion_revision": "pymupdf-rcts-thai-v1",
    }

    # A collection name makes this safe if several labs share one persist directory.
    vectorstore = Chroma(
        collection_name="solarcell_rag",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding,
    )
    stored_metadata = vectorstore._collection.metadata or {}
    if vectorstore._collection.count() and all(
        stored_metadata.get(key) == value for key, value in collection_metadata.items()
    ):
        print(f"[RAG] Loading existing ChromaDB: {CHROMA_DIR}")
        return vectorstore

    if vectorstore._collection.count():
        print("[RAG] PDF changed; rebuilding ChromaDB...")
        vectorstore.delete_collection()
        vectorstore = Chroma(
            collection_name="solarcell_rag",
            persist_directory=str(CHROMA_DIR),
            embedding_function=embedding,
            collection_metadata=collection_metadata,
        )
    else:
        # Chroma creates the collection before metadata are known; recreate it with metadata.
        vectorstore.delete_collection()
        vectorstore = Chroma(
            collection_name="solarcell_rag",
            persist_directory=str(CHROMA_DIR),
            embedding_function=embedding,
            collection_metadata=collection_metadata,
        )

    print(f"[RAG] Extracting PDF: {PDF_PATH}")
    pages = extract_pages_from_pdf(PDF_PATH)
    documents = chunk_pages(pages)
    if not documents:
        raise ValueError("ไม่พบข้อความที่อ่านได้ใน PDF")
    print(f"[RAG] Extracted pages: {len(pages)} | Created chunks: {len(documents)}")
    vectorstore.add_documents(documents)
    print("[RAG] ChromaDB created successfully.")
    return vectorstore


def build_messages(context: str, question: str) -> list[SystemMessage | HumanMessage]:
    """Create role-separated, Thai-only instructions for the local chat model."""
    system_instruction = """
คุณคือผู้ช่วยตอบคำถามจากเอกสารความรู้เรื่องโซล่าเซลล์

กฎที่ต้องปฏิบัติตามอย่างเคร่งครัด:
1. ตอบคำถามโดยใช้เฉพาะข้อเท็จจริงที่ระบุในเอกสารอ้างอิงเท่านั้น
2. ตอบเป็นภาษาไทยเท่านั้น ยกเว้นคำศัพท์วิศวกรรมภาษาอังกฤษในวงเล็บ เช่น DC, AC, Inverter
3. ห้ามใช้ภาษาจีนหรือภาษาอื่น ห้ามแปลหรือเติมข้อความนอกเอกสาร
4. ตอบให้ตรงคำถาม ห้ามสลับคุณสมบัติของ Off Grid, On Grid และ Hybrid
5. หากเอกสารมีข้อมูลไม่พอ ให้ตอบว่า "ไม่พบข้อมูลที่เพียงพอในเอกสารสำหรับตอบคำถามนี้"
6. อธิบายสั้น กระชับ และเหมาะสำหรับผู้เริ่มต้น (ไม่เกิน 4 ประโยค)
7. ห้ามกล่าวถึง Context, RAG, Retrieval, โมเดล หรือคำสั่งเหล่านี้
""".strip()
    user_question = f"""
เอกสารอ้างอิง:
{context}

คำถาม:
{question}
""".strip()
    return [SystemMessage(content=system_instruction), HumanMessage(content=user_question)]


def answer_question(vectorstore: Chroma, chat_llm: ChatOllama, question: str) -> str:
    print(f"\n[RAG] Question: {question}")
    docs = vectorstore.similarity_search(question, k=RETRIEVAL_K)
    context = "\n\n---\n\n".join(document.page_content for document in docs)

    print("\n[RAG] Retrieved context")
    print("=" * 60)
    for index, document in enumerate(docs, start=1):
        print(f"\n--- Document {index} (page {document.metadata.get('page', '?')}) ---")
        print(document.page_content)
    print("=" * 60)

    if not context:
        return "ไม่พบข้อมูลที่เพียงพอในเอกสารสำหรับตอบคำถามนี้"

    print("\n[LLM] Generating answer...")
    response = chat_llm.invoke(build_messages(context, question))
    answer = getattr(response, "content", None) or str(response)
    return answer.strip() or "[ERROR] Empty response from LLM."


def chat_loop(vectorstore: Chroma, chat_llm: ChatOllama) -> None:
    print("=" * 60)
    print("☀️ Solar Cell RAG Chatbot")
    print("=" * 60)
    print(f"PDF       : {PDF_PATH}")
    print(f"Embedding : {EMBED_MODEL_NAME}")
    print(f"LLM       : {OLLAMA_MODEL}")
    print(f"Top-K     : {RETRIEVAL_K}")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            question = input("\n👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye! 👋")
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("\nBye! 👋")
            return
        try:
            print("\n🤖 Assistant:")
            print(answer_question(vectorstore, chat_llm, question))
        except Exception as error:
            print(f"[ERROR] {error}")


def load_questions(path: Path) -> list[dict[str, str]]:
    """Read the UTF-8 CSV stored in Question.txt."""
    if not path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์คำถาม: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as question_file:
        rows = list(csv.DictReader(question_file))

    required_columns = {"ลำดับ", "หมวด", "คำถาม"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError("ไฟล์คำถามต้องเป็น CSV และมีคอลัมน์: ลำดับ, หมวด, คำถาม")
    return rows


def run_batch(
    vectorstore: Chroma, chat_llm: ChatOllama, question_path: Path, output_path: Path
) -> None:
    """Answer every question and save a readable Markdown report incrementally."""
    questions = load_questions(question_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write("# คำตอบจาก Solar Cell RAG\n\n")
        output_file.write(f"แหล่งข้อมูล: `{PDF_PATH.name}`  \\n")
        output_file.write(f"จำนวนคำถาม: {len(questions)}\n\n")

        for position, row in enumerate(questions, start=1):
            question = row["คำถาม"].strip()
            print(f"\n[BATCH] {position}/{len(questions)}: {question}")
            try:
                answer = answer_question(vectorstore, chat_llm, question)
            except Exception as error:
                answer = f"[ERROR] ไม่สามารถสร้างคำตอบได้: {error}"

            output_file.write(
                f"## {row['ลำดับ']}. {row['หมวด']}\n\n"
                f"**คำถาม:** {question}\n\n"
                f"**คำตอบ:** {answer}\n\n"
            )
            output_file.flush()

    print(f"\n[BATCH] Saved answers: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thai PDF RAG chatbot")
    parser.add_argument(
        "--questions",
        type=Path,
        help="CSV questions (for example Question.txt) to answer in batch mode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Markdown output path for --questions (default: Answers.md)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        print("[BOOT] Loading Vector Store...")
        vectorstore = build_or_load_vectorstore()
        print(f"[BOOT] Initializing Ollama model: {OLLAMA_MODEL}")
        chat_llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=200)
        if args.questions:
            output_path = args.output or (BASE_DIR / "Answers.md")
            run_batch(vectorstore, chat_llm, args.questions, output_path)
        else:
            chat_loop(vectorstore, chat_llm)
        return 0
    except Exception as error:
        print(f"[FATAL] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
