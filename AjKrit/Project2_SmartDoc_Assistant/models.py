"""Small serialisable data models shared by ingestion, retrieval and answering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """A citation-preserving child chunk of a parent document.

    ``text`` is the small retrieval unit indexed by FAISS/BM25.  The optional
    parent fields retain the larger coherent passage used when constructing
    the LLM context.  Keeping both levels avoids the usual trade-off between
    precise retrieval and enough surrounding evidence for an answer.
    """

    chunk_id: str
    text: str
    pdf_pages: list[int] = field(default_factory=list)
    printed_pages: list[str] = field(default_factory=list)
    section_id: str = ""
    section_title: str = ""
    parent_id: str = ""
    chunk_index: int = 0
    source_title: str = ""
    source_url: str = ""
    parent_text: str = ""
    parent_pdf_pages: list[int] = field(default_factory=list)
    parent_printed_pages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentChunk":
        return cls(
            chunk_id=str(value["chunk_id"]),
            text=str(value["text"]),
            pdf_pages=[int(x) for x in value.get("pdf_pages", [])],
            printed_pages=[str(x) for x in value.get("printed_pages", [])],
            section_id=str(value.get("section_id", "")),
            section_title=str(value.get("section_title", "")),
            parent_id=str(value.get("parent_id", "")),
            chunk_index=int(value.get("chunk_index", 0)),
            source_title=str(value.get("source_title", "")),
            source_url=str(value.get("source_url", "")),
            parent_text=str(value.get("parent_text", "")),
            parent_pdf_pages=[int(x) for x in value.get("parent_pdf_pages", [])],
            parent_printed_pages=[str(x) for x in value.get("parent_printed_pages", [])],
        )


@dataclass
class RetrievalResult:
    """One fused search result, retaining component scores for auditability."""

    chunk: DocumentChunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    weighted_score: float = 0.0
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["chunk"] = self.chunk.to_dict()
        value["final_score"] = self.final_score
        return value


@dataclass
class AnswerResult:
    """Public answer object used by CLI, API and LINE adapters."""

    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieval: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    grounded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "sources": self.sources,
            "retrieval": self.retrieval,
            "metrics": self.metrics,
            "grounded": self.grounded,
        }
