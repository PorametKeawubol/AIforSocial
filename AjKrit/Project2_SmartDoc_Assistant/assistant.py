"""Application service that connects hybrid retrieval, guardrails and Ollama."""

from __future__ import annotations

import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import Settings, settings as default_settings
from chat_history import is_followup, resolve_followup
from llm import FALLBACK, OllamaClient, citation_sources, format_evidence_text
from models import AnswerResult
from retriever import HybridRetriever, dynamic_top_k

LOGGER = logging.getLogger(__name__)

_GREETING_START_RE = re.compile(r"^(?:สวัสดี|หวัดดี|hello|hi)", flags=re.IGNORECASE)
_QUESTION_MARKERS = (
    "คนหาย",
    "ของหาย",
    "ขโมย",
    "แจ้งความ",
    "ทำอย่างไร",
    "ทำไง",
    "อย่างไร",
    "ยังไง",
    "อะไร",
    "ไหม",
    "หรือไม่",
    "ช่วย",
    "ขอถาม",
    "ที่ไหน",
    "เมื่อไร",
    "เท่าไร",
)


def _is_greeting(question: str) -> bool:
    """Handle casual greetings without wasting retrieval/LLM resources.

    The greeting prefix is intentionally permissive for casual Thai chat such
    as ``หวัดดีไอบ่าว``.  Question/intent markers keep messages like
    ``หวัดดี ช่วยบอกวิธีแจ้งความ`` in the document pipeline.
    """

    normalized = re.sub(r"\s+", " ", question.strip())
    if not normalized or len(normalized) > 80 or not _GREETING_START_RE.match(normalized):
        return False
    return not any(marker in normalized.lower() for marker in _QUESTION_MARKERS)


class SmartDocAssistant:
    """Evidence-first assistant for CLI, HTTP and LINE."""

    def __init__(self, config: Settings = default_settings) -> None:
        self.config = config
        self._retriever: HybridRetriever | None = None
        self._llm = OllamaClient(
            host=config.ollama_host,
            model=config.ollama_model,
            timeout_seconds=config.ollama_timeout_seconds,
            max_answer_tokens=config.max_answer_tokens,
        )

    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever.load(self.config.index_dir)
        return self._retriever

    def _sources_text(self, sources: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for source in sources[:5]:
            pages = ", ".join(str(page) for page in source.get("pdf_pages", [])) or "ไม่ระบุ"
            title = source.get("section_title") or source.get("title") or self.config.source_title
            line = f"[{source['id']}] {title} — PDF หน้า {pages}"
            if source.get("printed_pages"):
                line += f" (หน้าเอกสาร {', '.join(source['printed_pages'])})"
            if source.get("url"):
                line += f" — {source['url']}"
            lines.append(line)
        return "\n".join(lines)

    def _low_signal(self, results: list[Any]) -> bool:
        if not results:
            return True
        top = results[0]
        # Dense cosine and BM25 are deliberately both checked to avoid an accidental answer
        # from a single weak retrieval channel.
        return top.dense_score < 0.18 and top.sparse_score <= 0

    def answer(
        self,
        question: str,
        use_llm: bool = True,
        retrieval_results: list[Any] | None = None,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> AnswerResult:
        """Retrieve, answer from evidence, validate citations, and expose timing metadata."""

        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        if len(question) > 1500:
            raise ValueError("Question is too long")
        started = time.perf_counter()
        if _is_greeting(question):
            return AnswerResult(
                question=question,
                answer=(
                    "สวัสดีครับ ผมช่วยค้นข้อมูลจากคู่มือประชาชนได้ครับ\n"
                    "ลองถามได้ เช่น “คนหายทำอย่างไร” หรือ “ของหายและสงสัยว่าถูกขโมยควรทำอย่างไร”"
                ),
                metrics={
                    "intent": "greeting",
                    "total_seconds": round(time.perf_counter() - started, 4),
                },
                grounded=False,
            )
        history = (chat_history or [])[-4:]
        if chat_history is not None and not history and is_followup(question):
            return AnswerResult(question=question,
                                answer="ต้องการถามต่อเรื่องไหนครับ เช่น คนหาย เอกสารหาย หรือแจ้งความทรัพย์สินหาย",
                                metrics={"intent": "clarify_topic"}, grounded=False)
        search_query = resolve_followup(question, history)
        top_k = dynamic_top_k(search_query, self.config.top_k)
        retrieval_started = time.perf_counter()
        results = retrieval_results
        if results is None:
            results = self.retriever().search(
                search_query,
                top_k=top_k,
                candidate_k=self.config.candidate_k,
            )
        retrieval_seconds = time.perf_counter() - retrieval_started
        sources = citation_sources(results)
        metrics: dict[str, Any] = {
            "search_query": search_query,
            "history_turns": len(history),
            "top_k": top_k,
            "retrieved": len(results),
            "retrieval_seconds": round(retrieval_seconds, 4),
            "retrieval_reused": retrieval_results is not None,
            "embedding_model": self.config.embedding_model,
            "rerank_model": self.config.rerank_model or None,
            "source_title": self.config.source_title,
            "source_topic": self.config.source_topic,
        }

        if self._low_signal(results):
            metrics["fallback_reason"] = "low_retrieval_signal"
            return AnswerResult(
                question=question,
                answer=FALLBACK,
                citations=[],
                sources=[],
                retrieval=[result.to_dict() for result in results],
                metrics={**metrics, "total_seconds": round(time.perf_counter() - started, 4)},
                grounded=False,
            )

        if not use_llm:
            answer = (
                "พบข้อมูลที่เกี่ยวข้องในเอกสาร\n\n"
                f"{format_evidence_text(results[0].chunk.text[:1000])}\n\n"
                "อ้างอิง: [S1]"
            )
            return AnswerResult(
                question=question,
                answer=answer,
                citations=["S1"],
                sources=sources,
                retrieval=[result.to_dict() for result in results],
                metrics={**metrics, "total_seconds": round(time.perf_counter() - started, 4)},
                grounded=True,
            )

        try:
            kwargs: dict[str, Any] = {"max_context_chars": self.config.max_context_chars}
            if history:
                kwargs["chat_history"] = history
            llm_response = self._llm.answer_from_context(search_query, results, **kwargs)
            answer = llm_response.answer
            citations = llm_response.citations
            metrics.update(llm_response.metrics)
            grounded = answer != FALLBACK and bool(citations)
        except Exception as exc:
            LOGGER.exception("Ollama answer failed")
            metrics["llm_error"] = type(exc).__name__
            metrics["llm_error_detail"] = str(exc)[:240]
            answer = (
                "ยังเชื่อมต่อโมเดลตอบคำถามไม่ได้ จึงแสดงหลักฐานที่เกี่ยวข้องแทน:\n\n"
                f"หัวข้อ: {results[0].chunk.section_title or 'ไม่ระบุหัวข้อ'}\n"
                f"{format_evidence_text(results[0].chunk.text[:1200])}"
            )
            citations = ["S1"]
            grounded = True

        displayed_sources = sources
        if grounded and sources:
            cited_ids = set(citations)
            displayed_sources = [source for source in sources if source["id"] in cited_ids]
            if not displayed_sources:
                displayed_sources = sources[:1]
            answer = f"{answer}\n\nที่มา:\n{self._sources_text(displayed_sources)}"
        metrics["total_seconds"] = round(time.perf_counter() - started, 4)
        return AnswerResult(
            question=question,
            answer=answer,
            citations=citations,
            sources=displayed_sources,
            retrieval=[result.to_dict() for result in results],
            metrics=metrics,
            grounded=grounded,
        )


@lru_cache(maxsize=1)
def get_assistant() -> SmartDocAssistant:
    return SmartDocAssistant(default_settings)
