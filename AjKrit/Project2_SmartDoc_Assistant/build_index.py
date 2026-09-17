"""Build the persistent SmartDoc hybrid index.

Run after installing dependencies:
    python3 build_index.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config import Settings, settings as default_settings
from ingest import ingest_pdf, layout_noise_statistics
from retriever import HybridRetriever


def build(
    config: Settings,
    chunk_size: int | None = None,
    overlap: int | None = None,
    parent_size: int | None = None,
    batch_size: int = 8,
) -> dict:
    chunk_size = config.chunk_size if chunk_size is None else chunk_size
    overlap = config.chunk_overlap if overlap is None else overlap
    parent_size = config.parent_size if parent_size is None else parent_size
    started = time.perf_counter()
    config.ensure_directories()
    pages, chunks = ingest_pdf(
        config.pdf_path,
        source_title=config.source_title,
        source_url=config.source_url,
        chunk_size=chunk_size,
        overlap=overlap,
        parent_size=parent_size,
    )
    retriever = HybridRetriever.build(
        chunks,
        embedding_model=config.embedding_model,
        rerank_model=config.rerank_model,
        batch_size=batch_size,
    )
    noise_metrics = layout_noise_statistics(pages)
    extracted_characters = sum(len(page.text) for page in pages)
    noise_characters = int(noise_metrics["layout_noise_characters_removed"])
    retained_characters = max(0, extracted_characters - noise_characters)
    metrics = {
        "source_pdf": str(config.pdf_path.resolve()),
        "source_sha256": __import__("hashlib").sha256(config.pdf_path.read_bytes()).hexdigest(),
        "source_title": config.source_title,
        "source_topic": config.source_topic,
        "source_url": config.source_url,
        "pages": len(pages),
        "non_empty_pages": sum(bool(page.text) for page in pages),
        "extracted_characters": extracted_characters,
        "retained_characters_after_layout_cleanup": retained_characters,
        **noise_metrics,
        "chunks": len(chunks),
        "parent_documents": len({chunk.parent_id for chunk in chunks}),
        "chunk_size_characters": chunk_size,
        "chunk_overlap_characters": overlap,
        "parent_size_characters": parent_size,
        "tables_detected": sum(page.table_count for page in pages),
        "pages_with_tables": sum(page.table_count > 0 for page in pages),
        "embedding_model": config.embedding_model,
        "rerank_model": config.rerank_model or None,
        "embedding_dimension": int(retriever.dense_index.d),
        "build_seconds": round(time.perf_counter() - started, 3),
        "layout_noise_description": "detected repeated header/footer lines and isolated page markers only",
    }
    manifest = retriever.save(config.index_dir, source_pdf=config.pdf_path, metrics=metrics)
    (config.artifact_dir / "pages.json").write_text(
        json.dumps(
            [
                {
                    "pdf_page": page.pdf_page,
                    "printed_page": page.printed_page,
                    "characters": len(page.text),
                    "table_count": page.table_count,
                }
                for page in pages
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (config.artifact_dir / "build_metrics.json").write_text(
        json.dumps({**metrics, "manifest": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {**metrics, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SmartDoc FAISS + BM25 index")
    parser.add_argument("--pdf", type=Path, default=default_settings.pdf_path)
    parser.add_argument("--index-dir", type=Path, default=default_settings.index_dir)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--parent-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    config = Settings.from_env()
    config = Settings(**{**config.__dict__, "pdf_path": args.pdf, "index_dir": args.index_dir})
    print(
        json.dumps(
            build(
                config,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                parent_size=args.parent_size,
                batch_size=args.batch_size,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
