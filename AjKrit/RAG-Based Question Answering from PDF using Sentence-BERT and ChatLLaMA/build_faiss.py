"""Create a reusable FAISS index from a PDF document.

Example:
    python build_faiss.py --pdf solarcell-basic-knowledge-SolarHub.pdf
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = PROJECT_DIR / "solarcell-basic-knowledge-SolarHub.pdf"
DEFAULT_INDEX_PATH = PROJECT_DIR / "faiss_index"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def clean_text(text: str) -> str:
    """Normalize Unicode and collapse excess whitespace from extracted PDF text."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def extract_lines_from_pdf(pdf_path: Path) -> tuple[list[dict[str, Any]], int]:
    """Return non-empty lines together with their 1-based source page number."""
    reader = PdfReader(str(pdf_path))
    lines: list[dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for raw_line in page_text.splitlines():
            text = clean_text(raw_line)
            if text:
                lines.append({"text": text, "page": page_number})

    return lines, len(reader.pages)


def chunk_lines(
    lines: list[dict[str, Any]], chunk_size: int = 5, overlap: int = 2
) -> list[Document]:
    """Create overlapping chunks while preserving PDF page and line metadata."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be zero or greater and smaller than chunk_size")

    step = chunk_size - overlap
    documents: list[Document] = []
    for start in range(0, len(lines), step):
        selected = lines[start : start + chunk_size]
        if not selected:
            continue

        documents.append(
            Document(
                page_content="\n".join(item["text"] for item in selected),
                metadata={
                    "source": "",
                    "start_line": start + 1,
                    "end_line": start + len(selected),
                    "pages": sorted({item["page"] for item in selected}),
                },
            )
        )
    return documents


def create_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Load the multilingual Sentence-BERT embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def index_size_bytes(index_path: Path) -> int:
    return sum(path.stat().st_size for path in index_path.glob("*") if path.is_file())


def build_index(
    pdf_path: Path,
    index_path: Path,
    model_name: str,
    chunk_size: int,
    overlap: int,
) -> dict[str, Any]:
    """Extract, chunk, embed, persist the PDF, and return Lab 1 measurements."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    started_at = time.perf_counter()
    lines, page_count = extract_lines_from_pdf(pdf_path)
    if not lines:
        raise ValueError("No extractable text was found in the PDF")

    documents = chunk_lines(lines, chunk_size=chunk_size, overlap=overlap)
    for document in documents:
        document.metadata["source"] = pdf_path.name

    embeddings = create_embeddings(model_name)
    vectorstore = FAISS.from_documents(documents, embeddings)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))

    metrics: dict[str, Any] = {
        "source_pdf": str(pdf_path.resolve()),
        "embedding_model": model_name,
        "pages": page_count,
        "lines": len(lines),
        "chunks": len(documents),
        "chunk_size_lines": chunk_size,
        "chunk_overlap_lines": overlap,
        "embedding_dimension": vectorstore.index.d,
        "index_build_seconds": round(time.perf_counter() - started_at, 3),
        "faiss_index_bytes": index_size_bytes(index_path),
    }
    (index_path / "index_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from a PDF.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH, help="Source PDF path")
    parser.add_argument(
        "--index-dir", type=Path, default=DEFAULT_INDEX_PATH, help="Directory for FAISS files"
    )
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="Sentence-BERT model name")
    parser.add_argument("--chunk-size", type=int, default=5, help="Lines in each chunk")
    parser.add_argument("--overlap", type=int, default=2, help="Overlapping lines between chunks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_index(
        pdf_path=args.pdf,
        index_path=args.index_dir,
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print("FAISS indexing completed\n")
    for label, value in metrics.items():
        print(f"{label}: {value}")


if __name__ == "__main__":
    main()
