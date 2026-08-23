"""Intent classification for the six-intent KFC LINE chatbot lab.

The classifier uses multilingual Sentence-BERT embeddings when the model is
available.  A small lexical matcher is kept as a deterministic fallback so
the webhook can still answer basic conversational intents when the model has
not been downloaded or the embedding service fails.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping

import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_THRESHOLD = 0.70
OTHER_INTENT = "other"

# Keep each class semantically focused.  The examples are also the small
# labelled dataset used by the lab and are encoded once, then reused for all
# incoming LINE messages.
INTENT_EXAMPLES: dict[str, tuple[str, ...]] = {
    "menu": (
        "เมนู",
        "ขอเมนูหน่อย",
        "มีอะไรขายบ้าง",
        "อยากกินไก่",
        "แนะนำเมนู KFC ให้หน่อย",
        "แนะนำอาหารหน่อย",
        "รายการอาหารมีอะไรบ้าง",
        "เมนูอาหารมีอะไรบ้าง",
        "ขอรายการอาหารของ KFC",
        "ค้นหารายการอาหาร",
        "วันนี้มีอะไรอร่อย",
        "อยากดูเมนูไก่ทอด",
    ),
    "promotion": (
        "โปรโมชั่นวันนี้มีอะไรบ้าง",
        "โปร KFC ตอนนี้",
        "มีส่วนลดอะไรบ้าง",
        "วันนี้มีโปรอะไร",
        "ขอดีลไก่ทอดหน่อย",
        "ขอส่วนลด KFC",
        "คูปองมีไหม",
        "คูปอง KFC มีไหม",
        "โปรโมชันล่าสุด",
        "ราคาโปรพิเศษเท่าไหร่",
    ),
    "location": (
        "ร้าน KFC อยู่ที่ไหน",
        "มีสาขา KFC ที่ไหนบ้าง",
        "ขอแผนที่ร้าน KFC",
        "ร้านใกล้ฉันอยู่ตรงไหน",
        "ค้นหาสาขาใกล้ฉัน",
        "KFC สาขาไหนใกล้ที่สุด",
        "ขอพิกัดร้าน KFC",
        "มีร้าน KFC แถวนี้ไหม",
    ),
    "greeting": (
        "สวัสดี",
        "สวัสดีครับ",
        "สวัสดีค่ะ",
        "hello",
        "hi",
        "hi KFC",
        "หวัดดี",
        "ดีจ้า",
        "ขอทักทายหน่อย",
    ),
    "order": (
        "ต้องการสั่งอาหาร",
        "สั่ง KFC ได้อย่างไร",
        "ขอสั่งไก่ทอด",
        "อยากสั่งอาหารเดลิเวอรี",
        "ช่วยสั่งเมนูนี้ให้หน่อย",
        "สั่งกลับบ้านได้ไหม",
        "จะสั่งอาหารออนไลน์",
        "ขอวิธีสั่ง KFC",
    ),
    "thanks": (
        "ขอบคุณ",
        "ขอบคุณมาก",
        "ขอบใจนะ",
        "ขอบคุณสำหรับข้อมูล",
        "โอเค ขอบคุณครับ",
        "ขอบคุณค่ะ",
        "ขอบคุณที่ช่วยเหลือ",
        "ได้ข้อมูลแล้ว ขอบคุณ",
    ),
}

INTENT_LABELS = tuple(INTENT_EXAMPLES)

# Held-out evaluation set: seven natural variations per supported intent.
# These messages are intentionally different from INTENT_EXAMPLES so the Lab
# measures generalisation instead of rewarding exact-example memorisation.
LAB_TEST_CASES: tuple[tuple[str, str], ...] = (
    ("ขอรายการของกินหน่อย", "menu"),
    ("อยากรู้ว่ามีไก่อะไรบ้าง", "menu"),
    ("ช่วยบอกเมนูน่าลอง", "menu"),
    ("มีอาหารอะไรให้เลือก", "menu"),
    ("หิวมาก", "menu"),
    ("วันนี้อยากกิน KFC มีอะไรบ้าง", "menu"),
    ("เปิดดูเมนูให้หน่อย", "menu"),
    ("ช่วงนี้ KFC มีข้อเสนออะไร", "promotion"),
    ("มีราคาพิเศษไหม", "promotion"),
    ("อยากรู้โปรไก่ทอด", "promotion"),
    ("ตอนนี้มีแคมเปญอะไร", "promotion"),
    ("ซื้อชุดไหนคุ้มสุด", "promotion"),
    ("มีโปรสำหรับวันนี้หรือเปล่า", "promotion"),
    ("มีดีลพิเศษช่วงนี้ไหม", "promotion"),
    ("ช่วยบอกสาขาที่อยู่ใกล้บ้านหน่อย", "location"),
    ("ไป KFC สาขาไหนดี", "location"),
    ("แถวบ้านฉันมีร้านไหม", "location"),
    ("อยากได้ที่อยู่ของร้าน", "location"),
    ("ร้านไหนเดินทางสะดวก", "location"),
    ("ขอทางไปร้าน KFC", "location"),
    ("มี KFC ในห้างไหนบ้าง", "location"),
    ("หวัดดีบอต", "greeting"),
    ("เข้ามาทักทายครับ", "greeting"),
    ("สวัสดีจ้า KFC", "greeting"),
    ("ดีจังที่เจอกัน", "greeting"),
    ("ขอเริ่มคุยด้วยนะ", "greeting"),
    ("ฮัลโหล", "greeting"),
    ("มีใครอยู่ไหม", "greeting"),
    ("ช่วยบอกช่องทางสั่งซื้อ", "order"),
    ("อยากให้มาส่งที่บ้าน", "order"),
    ("สั่งผ่านมือถือได้ไหม", "order"),
    ("รับออเดอร์กลับบ้านไหม", "order"),
    ("จะซื้อไก่ต้องทำอย่างไร", "order"),
    ("ขอขั้นตอนการสั่งหน่อย", "order"),
    ("อยากสั่งชุดอาหารตอนนี้", "order"),
    ("รับทราบครับ ขอบใจ", "thanks"),
    ("ขอบคุณมากสำหรับรายละเอียด", "thanks"),
    ("ขอบคุณที่ช่วยตอบ", "thanks"),
    ("ช่วยได้เยอะเลย ขอบคุณนะ", "thanks"),
    ("ขอบพระคุณสำหรับคำตอบ", "thanks"),
    ("ขอบคุณที่อธิบายครับ", "thanks"),
    ("ได้รับคำตอบแล้ว ขอบคุณครับ", "thanks"),
)

# Boundary cases are reported separately: they are deliberately outside the
# six supported intents and reveal false positives instead of hiding them in
# the main accuracy number.
LAB_EDGE_CASES: tuple[tuple[str, str], ...] = (
    ("เปิดร้านกี่โมง", OTHER_INTENT),
    ("อยากสมัครสมาชิก", OTHER_INTENT),
    ("ทำไมไก่แพง", OTHER_INTENT),
    ("โอเค ได้เลย", OTHER_INTENT),
    ("เข้าใจแล้วครับ", OTHER_INTENT),
)

# These words make the fallback useful for a fresh installation without
# changing the Sentence-BERT path used in the lab evaluation.
LEXICAL_HINTS: dict[str, tuple[str, ...]] = {
    "menu": (
        "เมนู",
        "อาหาร",
        "รายการอาหาร",
        "เมนูอาหาร",
        "รายการของกิน",
        "มีอะไรขาย",
        "อยากกิน",
        "แนะนำเมนู",
        "ไก่ทอด",
    ),
    "promotion": (
        "โปรโมชั่น",
        "โปรโมชัน",
        "โปรโมชั่น",
        "โปร",
        "ส่วนลด",
        "ดีล",
        "คูปอง",
    ),
    "location": (
        "ร้านอยู่ไหน",
        "สาขา",
        "แผนที่",
        "ใกล้ฉัน",
        "ใกล้ที่สุด",
        "พิกัด",
        "แถวนี้",
    ),
    "greeting": ("สวัสดี", "หวัดดี", "hello", "hi", "ดีจ้า"),
    "order": (
        "สั่งอาหาร",
        "สั่งไก่",
        "เดลิเวอรี",
        "ออนไลน์",
        "กลับบ้าน",
        "วิธีสั่ง",
    ),
    "thanks": ("ขอบคุณ", "ขอบใจ", "ขอบคุณมาก"),
}


def _clean_text(value: object | None) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split())


def _normalise(value: object | None) -> str:
    text = _clean_text(value).casefold()
    return re.sub(r"[^\w\sก-๙]", " ", text)


def _ngrams(value: str, size: int = 2) -> set[str]:
    compact = re.sub(r"\s+", "", _normalise(value))
    if not compact:
        return set()
    if len(compact) <= size:
        return {compact}
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


@dataclass(frozen=True)
class IntentPrediction:
    """One prediction and the complete similarity vector for the lab."""

    intent: str
    score: float
    scores: dict[str, float]
    backend: str
    threshold: float

    @property
    def accepted(self) -> bool:
        return self.intent != OTHER_INTENT


class IntentClassifier:
    """Classify a message into six KFC intents using cosine similarity."""

    def __init__(
        self,
        examples: Mapping[str, tuple[str, ...] | list[str]] | None = None,
        *,
        model_name: str | None = None,
        threshold: float | None = None,
        semantic_enabled: bool | None = None,
    ) -> None:
        raw_examples = examples or INTENT_EXAMPLES
        self.examples: dict[str, tuple[str, ...]] = {
            intent: tuple(_clean_text(example) for example in texts if _clean_text(example))
            for intent, texts in raw_examples.items()
            if intent != OTHER_INTENT
        }
        if not self.examples:
            raise ValueError("At least one intent with one example is required")
        self.model_name = model_name or os.getenv("BERT_MODEL_NAME", DEFAULT_MODEL_NAME)
        self.threshold = float(
            threshold
            if threshold is not None
            else os.getenv("INTENT_MIN_SCORE", str(DEFAULT_THRESHOLD))
        )
        self.semantic_enabled = (
            semantic_enabled
            if semantic_enabled is not None
            else os.getenv("INTENT_SEMANTIC_ENABLED", "true").casefold()
            in {"1", "true", "yes", "on"}
        )
        self._model: Any | None = None
        self._embeddings: dict[str, np.ndarray] | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def _build_embeddings(self) -> dict[str, np.ndarray]:
        if self._embeddings is not None:
            return self._embeddings
        model = self._load_model()
        self._embeddings = {
            intent: np.asarray(
                model.encode(
                    list(texts),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            for intent, texts in self.examples.items()
        }
        return self._embeddings

    def _semantic_scores(self, text: str) -> dict[str, float]:
        model = self._load_model()
        embeddings = self._build_embeddings()
        query = np.asarray(
            model.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )[0]
        return {
            intent: float(np.max(vectors @ query))
            for intent, vectors in embeddings.items()
        }

    def _lexical_scores(self, text: str) -> dict[str, float]:
        normalized = _normalise(text)
        query_grams = _ngrams(normalized)
        scores: dict[str, float] = {}
        for intent, examples in self.examples.items():
            best = 0.0
            for example in examples:
                candidate = _normalise(example)
                candidate_grams = _ngrams(candidate)
                union = query_grams | candidate_grams
                overlap = (
                    len(query_grams & candidate_grams) / len(union) if union else 0.0
                )
                best = max(
                    best,
                    SequenceMatcher(None, normalized, candidate).ratio() * 0.65
                    + overlap * 0.35,
                )
            for hint in LEXICAL_HINTS.get(intent, ()):
                normalized_hint = _normalise(hint)
                if normalized_hint in normalized:
                    # Prefer a specific phrase such as "สั่งอาหาร" over
                    # the shorter overlapping word "อาหาร".
                    hint_score = 0.70 + min(0.28, len(normalized_hint.replace(" ", "")) / 20)
                    best = max(best, hint_score)
            scores[intent] = best
        return scores

    def detect(self, text: str) -> IntentPrediction:
        """Return the best intent, all six scores, and the backend used."""

        cleaned = _clean_text(text)
        if not cleaned:
            return IntentPrediction(
                intent=OTHER_INTENT,
                score=0.0,
                scores={intent: 0.0 for intent in self.examples},
                backend="empty",
                threshold=self.threshold,
            )

        backend = "lexical"
        try:
            scores = (
                self._semantic_scores(cleaned)
                if self.semantic_enabled
                else self._lexical_scores(cleaned)
            )
            backend = "sentence-bert" if self.semantic_enabled else "lexical"
        except Exception:
            # Intent detection is a convenience layer around the existing QA
            # engine.  A missing model must not take down the LINE webhook.
            scores = self._lexical_scores(cleaned)

        best_intent, best_score = max(scores.items(), key=lambda pair: pair[1])
        if best_score < self.threshold:
            best_intent = OTHER_INTENT
        return IntentPrediction(
            intent=best_intent,
            score=float(best_score),
            scores={intent: float(score) for intent, score in scores.items()},
            backend=backend,
            threshold=self.threshold,
        )


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_THRESHOLD",
    "INTENT_EXAMPLES",
    "INTENT_LABELS",
    "LAB_EDGE_CASES",
    "LAB_TEST_CASES",
    "OTHER_INTENT",
    "IntentClassifier",
    "IntentPrediction",
]
