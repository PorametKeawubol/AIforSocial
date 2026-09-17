"""Hybrid Thai retrieval: dense FAISS + sparse BM25 + RRF + optional reranking."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from models import DocumentChunk, RetrievalResult

LOGGER = logging.getLogger(__name__)


def thai_tokens(text: str) -> list[str]:
    """Tokenise Thai for BM25, with a dependency-free fallback for tests."""

    try:
        from pythainlp.tokenize import word_tokenize

        tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    except ImportError:
        import re

        tokens = re.findall(r"[ก-๙]+|[A-Za-z0-9]+", text)
    return [token.strip().lower() for token in tokens if token.strip()]


def dynamic_top_k(question: str, base: int = 5) -> int:
    """Select retrieval breadth from query complexity, bounded for context safety."""

    question = question.strip()
    if not question:
        return max(1, min(base, 3))
    clauses = sum(question.count(mark) for mark in ("?", "？", "และ", "หรือ", "อย่างไร"))
    length = len(question)
    if length <= 35 and clauses <= 1:
        return max(3, min(base, 4))
    if length <= 100 and clauses <= 3:
        return max(4, min(base + 1, 6))
    return max(5, min(base + 2, 8))


def expand_thai_query(question: str) -> str:
    """Add document vocabulary for common Thai intents without changing the user query.

    The expansion is used only for retrieval.  The original question is still
    sent to the answer model, so the model cannot mistake the helper terms for
    user-provided facts.
    """

    compact = question.replace(" ", "")
    additions: list[str] = []
    if any(term in compact for term in ("ของหาย", "ทรัพย์สินหาย", "ถูกขโมย", "ขโมย", "ลักทรัพย์")):
        additions.extend(
            [
                "แจ้งความทรัพย์สินหาย",
                "หลักฐานการเป็นเจ้าของ",
                "ลักทรัพย์",
                "รักษาร่องรอยหลักฐานที่เกิดเหตุ",
            ]
        )
    if any(term in compact for term in ("เอกสารหาย", "บัตรหาย", "ใบขับขี่หาย")):
        additions.extend(["แจ้งความเอกสารสำคัญหาย", "หลักฐานเอกสาร"])
    if any(term in compact for term in ("รถหาย", "รถถูกขโมย", "เรือหาย")):
        additions.extend(["แจ้งความรถหรือเรือหาย", "191", "1192"])
    unique_additions = list(dict.fromkeys(additions))
    return f"{question} {' '.join(unique_additions)}".strip() if unique_additions else question


def reciprocal_rank_fusion(
    dense_ranks: dict[int, int],
    sparse_ranks: dict[int, int],
    dense_weight: float = 0.55,
    sparse_weight: float = 0.45,
    rank_constant: int = 60,
) -> dict[int, float]:
    """Fuse independent rankings using weighted Reciprocal Rank Fusion."""

    scores: dict[int, float] = {}
    for index, rank in dense_ranks.items():
        scores[index] = scores.get(index, 0.0) + dense_weight / (rank_constant + rank)
    for index, rank in sparse_ranks.items():
        scores[index] = scores.get(index, 0.0) + sparse_weight / (rank_constant + rank)
    return scores


def _ranked_indices(scores: np.ndarray, limit: int) -> list[int]:
    limit = max(1, min(limit, len(scores)))
    return np.argsort(-scores, kind="stable")[:limit].tolist()


def _minmax(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


class HybridRetriever:
    """Persistable hybrid retriever with auditable component scores."""

    # Version 3 adds parent-document context fields to every persisted chunk.
    MANIFEST_VERSION = 3

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedding_model: str,
        rerank_model: str = "",
        dense_index: Any | None = None,
        bm25: Any | None = None,
        index_dir: Path | None = None,
    ) -> None:
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.dense_index = dense_index
        self.bm25 = bm25
        self.index_dir = index_dir
        self._encoder: Any | None = None
        self._reranker: Any | None = None

    @classmethod
    def build(
        cls,
        chunks: list[DocumentChunk],
        embedding_model: str,
        rerank_model: str = "",
        batch_size: int = 8,
        show_progress_bar: bool = True,
    ) -> "HybridRetriever":
        """Embed chunks and build exact cosine-equivalent FAISS + BM25 indexes."""

        if not chunks:
            raise ValueError("Cannot build a retriever without chunks")
        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Index building needs faiss-cpu and sentence-transformers; install requirements.txt"
            ) from exc
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Index building needs rank-bm25; install requirements.txt") from exc

        encoder = SentenceTransformer(embedding_model)
        encoder.max_seq_length = min(getattr(encoder, "max_seq_length", 512), 512)
        texts = [chunk.text for chunk in chunks]
        vectors = encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        ).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        tokenised = [thai_tokens(text) for text in texts]
        return cls(
            chunks=chunks,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
            dense_index=index,
            bm25=BM25Okapi(tokenised),
        )

    def _load_encoder(self) -> Any:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Query embedding needs sentence-transformers; install requirements.txt"
                ) from exc
            self._encoder = SentenceTransformer(self.embedding_model)
            self._encoder.max_seq_length = min(getattr(self._encoder, "max_seq_length", 512), 512)
        return self._encoder

    def _load_reranker(self) -> Any | None:
        if not self.rerank_model:
            return None
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._reranker = CrossEncoder(self.rerank_model, max_length=512)
            except Exception as exc:  # optional production enhancement; retrieval remains usable
                LOGGER.warning("Cross-encoder unavailable; using RRF order: %s", exc)
                self.rerank_model = ""
                return None
        return self._reranker

    def save(self, index_dir: Path, source_pdf: Path | None = None, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        """Save native FAISS and JSON metadata; no unsafe pickle deserialisation."""

        if self.dense_index is None or self.bm25 is None:
            raise RuntimeError("Retriever is not built")
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Saving the index needs faiss-cpu") from exc
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.dense_index, str(index_dir / "index.faiss"))
        (index_dir / "chunks.json").write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest: dict[str, Any] = {
            "manifest_version": self.MANIFEST_VERSION,
            "created_at_epoch": time.time(),
            "embedding_model": self.embedding_model,
            "rerank_model": self.rerank_model,
            "chunks": len(self.chunks),
            "dimension": int(self.dense_index.d),
            "source_pdf": str(source_pdf.resolve()) if source_pdf else None,
            "source_sha256": hashlib.sha256(source_pdf.read_bytes()).hexdigest() if source_pdf else None,
            "metrics": metrics or {},
        }
        (index_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    @classmethod
    def load(cls, index_dir: Path) -> "HybridRetriever":
        """Load a generated index and reconstruct BM25 from JSON text."""

        try:
            import faiss  # type: ignore
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Loading the index needs faiss-cpu and rank-bm25") from exc
        required = [index_dir / "index.faiss", index_dir / "chunks.json", index_dir / "manifest.json"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing index files: " + ", ".join(missing))
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        if int(manifest.get("manifest_version", 0)) != cls.MANIFEST_VERSION:
            raise ValueError("Unsupported index manifest version")
        chunks = [
            DocumentChunk.from_dict(value)
            for value in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        ]
        if len(chunks) != int(manifest["chunks"]):
            raise ValueError("Chunk metadata count does not match index manifest")
        index = faiss.read_index(str(index_dir / "index.faiss"))
        if index.ntotal != len(chunks):
            raise ValueError("FAISS vector count does not match chunk metadata")
        bm25 = BM25Okapi([thai_tokens(chunk.text) for chunk in chunks])
        return cls(
            chunks=chunks,
            embedding_model=str(manifest["embedding_model"]),
            rerank_model=str(manifest.get("rerank_model", "")),
            dense_index=index,
            bm25=bm25,
            index_dir=index_dir,
        )

    def search(self, question: str, top_k: int | None = None, candidate_k: int = 24, rerank_k: int = 6) -> list[RetrievalResult]:
        """Return fused results with dense/sparse ranks and optional cross-encoder scores."""

        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        if self.dense_index is None or self.bm25 is None:
            raise RuntimeError("Retriever is not built or loaded")
        requested_k = top_k or dynamic_top_k(question)
        candidate_k = max(candidate_k, requested_k * 4)
        retrieval_question = expand_thai_query(question)
        encoder = self._load_encoder()
        query_vector = encoder.encode(
            [retrieval_question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        dense_scores, dense_ids = self.dense_index.search(
            query_vector, min(candidate_k, len(self.chunks))
        )
        dense_score_map = {
            int(index): float(score)
            for index, score in zip(dense_ids[0].tolist(), dense_scores[0].tolist())
            if int(index) >= 0
        }
        dense_order = [index for index in dense_ids[0].tolist() if int(index) >= 0]
        dense_ranks = {index: rank for rank, index in enumerate(dense_order, start=1)}

        sparse_scores_array = np.asarray(
            self.bm25.get_scores(thai_tokens(retrieval_question)), dtype="float32"
        )
        sparse_order = _ranked_indices(sparse_scores_array, min(candidate_k, len(self.chunks)))
        sparse_ranks = {index: rank for rank, index in enumerate(sparse_order, start=1)}

        fused = reciprocal_rank_fusion(dense_ranks, sparse_ranks)
        sparse_min = float(sparse_scores_array.min()) if len(sparse_scores_array) else 0.0
        sparse_max = float(sparse_scores_array.max()) if len(sparse_scores_array) else 0.0
        dense_values = list(dense_score_map.values()) or [0.0]
        dense_min, dense_max = min(dense_values), max(dense_values)

        candidates = set(dense_ranks) | set(sparse_ranks)
        results: list[RetrievalResult] = []
        for index in candidates:
            dense_score = dense_score_map.get(index, dense_min)
            sparse_score = float(sparse_scores_array[index])
            weighted = 0.60 * _minmax(dense_score, dense_min, dense_max) + 0.40 * _minmax(
                sparse_score, sparse_min, sparse_max
            )
            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    dense_rank=dense_ranks.get(index),
                    sparse_rank=sparse_ranks.get(index),
                    rrf_score=fused.get(index, 0.0),
                    weighted_score=weighted,
                )
            )
        results.sort(key=lambda item: (item.rrf_score, item.weighted_score), reverse=True)

        reranker = self._load_reranker()
        if reranker is not None and results:
            # Keep enough candidates for complex questions requesting more than six hits.
            pool = results[:max(requested_k, rerank_k)]
            pairs = [(question, result.chunk.text) for result in pool]
            try:
                scores = reranker.predict(pairs)
                if len(scores) != len(pool) or not np.isfinite(scores).all():
                    raise ValueError("Reranker returned invalid scores")
                for result, score in zip(pool, scores):
                    result.rerank_score = float(score)
                # Cross-encoder scores and unscored RRF scores are not comparable.
                results = sorted(pool, key=lambda item: item.final_score, reverse=True)
            except Exception as exc:
                LOGGER.warning("Cross-encoder scoring failed; retaining RRF order: %s", exc)
        return results[:requested_k]

    @staticmethod
    def result_summary(results: Iterable[RetrievalResult]) -> list[dict[str, Any]]:
        return [result.to_dict() for result in results]
