"""PhayaThaiBERT-backed intent routing with deterministic entity extraction.

``clicknext/phayathaibert`` is the Thai base checkpoint used for semantic intent
routing.  It is used as a fallback for messages that the catalogue-aware rule parser
cannot classify; the rule parser remains responsible for exact constraints such as
price, stock, and brand names.  This keeps a model prediction from accidentally
discarding a hard search filter.

The model is deliberately loaded on the first request, rather than at Flask startup,
so a temporary Hugging Face/cache outage cannot prevent LINE's webhook from starting.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from threading import RLock
from typing import Any, Iterable, Mapping, Protocol

try:  # Package import (``MercularChatbot.bert_nlp``).
    from .nlp import (
        CommandEntities,
        INTENT_SEARCH,
        INTENT_UNKNOWN,
        ParsedCommand,
        ThaiCommandParser,
        normalize_text,
    )
except ImportError:  # pragma: no cover - direct execution from project directory.
    from nlp import (
        CommandEntities,
        INTENT_SEARCH,
        INTENT_UNKNOWN,
        ParsedCommand,
        ThaiCommandParser,
        normalize_text,
    )


LOGGER = logging.getLogger(__name__)

DEFAULT_PHAYATHAIBERT_MODEL = "clicknext/phayathaibert"

# These are intent exemplars, not a hidden catalogue.  PhayaThaiBERT embeds a user
# message and the examples in the same vector space, then chooses the nearest intent.
# Add an explicitly fine-tuned sequence-classifier later if the product gains labelled
# chat traffic; the public parser API remains unchanged.
INTENT_EXAMPLES: Mapping[str, tuple[str, ...]] = {
    "search": (
        "ค้นหาสินค้าให้หน่อย",
        "อยากได้หูฟังสำหรับเล่นเกม",
        "แนะนำสินค้าในงบประมาณนี้",
        "find a product for me",
    ),
    "greeting": ("สวัสดี", "หวัดดี", "hello", "hi there"),
    "help": ("ช่วยบอกวิธีใช้งาน", "ใช้บอตยังไง", "help me", "what can you do"),
    "thanks": ("ขอบคุณมาก", "ขอบใจ", "thank you", "thanks"),
    "contact": (
        "ติดต่อพนักงาน",
        "ขอช่องทางติดต่อร้าน",
        "contact support",
        "talk to staff",
    ),
    "order": ("สั่งซื้อสินค้า", "ซื้ออันนี้", "place an order", "buy this"),
    "refresh": ("สุ่มสินค้าใหม่", "ขอดูอีกชุด", "show more products", "refresh results"),
}


class PhayaThaiBertUnavailable(RuntimeError):
    """Raised when the optional local PhayaThaiBERT runtime/model cannot load."""


@dataclass(frozen=True, slots=True)
class IntentPrediction:
    """A semantic intent and its calibrated relative confidence."""

    intent: str
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", min(1.0, max(0.0, float(self.confidence))))


class IntentClassifier(Protocol):
    """Small interface that makes model inference replaceable in tests/deployment."""

    def predict(self, text: str) -> IntentPrediction: ...


class PhayaThaiBertIntentClassifier:
    """Lazy PhayaThaiBERT semantic classifier based on Thai intent exemplars.

    PhayaThaiBERT is CamemBERT-compatible, so it is loaded through ``AutoModel``.
    This encoder-based classifier does not pretend that the base checkpoint has an
    already-trained MercuMate intent head.  Its dependencies are imported lazily to
    retain a safe rule-only fallback when the optional model extras are unavailable.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_PHAYATHAIBERT_MODEL,
        *,
        local_files_only: bool = False,
        max_length: int = 128,
        temperature: float = 0.12,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.model_name = model_name.strip()
        self.local_files_only = bool(local_files_only)
        self.max_length = int(max_length)
        self.temperature = float(temperature)
        self._lock = RLock()
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._labels: tuple[str, ...] = tuple(INTENT_EXAMPLES)
        self._prototype_vectors: Any | None = None
        self._load_error: PhayaThaiBertUnavailable | None = None

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if self._load_error is not None:
                raise self._load_error
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as error:
                self._load_error = PhayaThaiBertUnavailable(
                    "PhayaThaiBERT requires torch, transformers, and sentencepiece; "
                    "run: pip install -r requirements.txt"
                )
                raise self._load_error from error
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=self.local_files_only
                )
                # Pooling is calculated explicitly in ``_embed``.  Disabling the
                # unused CamemBERT pooler avoids randomly initialised weights.
                model = AutoModel.from_pretrained(
                    self.model_name,
                    local_files_only=self.local_files_only,
                    add_pooling_layer=False,
                )
                model_type = str(getattr(model.config, "model_type", ""))
                if model_type != "camembert":
                    raise ValueError(
                        f"{self.model_name!r} is {model_type or 'not CamemBERT-compatible'}; "
                        "choose a PhayaThaiBERT checkpoint"
                    )
                model.eval()
                self._torch = torch
                self._tokenizer = tokenizer
                self._model = model
                prototype_texts = tuple(
                    phrase for examples in INTENT_EXAMPLES.values() for phrase in examples
                )
                prototype_vectors = self._embed(prototype_texts)
                cursor = 0
                grouped_vectors = []
                for examples in INTENT_EXAMPLES.values():
                    count = len(examples)
                    grouped_vectors.append(prototype_vectors[cursor : cursor + count].mean(dim=0))
                    cursor += count
                self._prototype_vectors = torch.nn.functional.normalize(
                    torch.stack(grouped_vectors), p=2, dim=1
                )
            except PhayaThaiBertUnavailable:
                raise
            except Exception as error:  # Network/cache/model errors must not take down LINE.
                self._model = None
                self._tokenizer = None
                self._torch = None
                self._load_error = PhayaThaiBertUnavailable(
                    f"could not load PhayaThaiBERT model {self.model_name!r}: {error}"
                )
                raise self._load_error from error

    def _embed(self, texts: tuple[str, ...]) -> Any:
        """Mean-pool the final PhayaThaiBERT hidden states, respecting padding."""

        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        encoded = self._tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        with self._torch.inference_mode():
            hidden_states = self._model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self._torch.nn.functional.normalize(pooled, p=2, dim=1)

    def predict(self, text: str) -> IntentPrediction:
        """Classify non-empty user text with PhayaThaiBERT semantic similarity."""

        normalized = normalize_text(text)
        if not normalized:
            return IntentPrediction(INTENT_UNKNOWN, 0.0)
        self._load()
        assert self._torch is not None and self._prototype_vectors is not None
        query_vector = self._embed((normalized,))[0]
        similarities = self._prototype_vectors @ query_vector
        probabilities = self._torch.softmax(similarities / self.temperature, dim=0)
        index = int(self._torch.argmax(probabilities).item())
        return IntentPrediction(self._labels[index], float(probabilities[index].item()))


class PhayaThaiBertCommandParser:
    """Use PhayaThaiBERT intent semantics with deterministic catalogue entities."""

    def __init__(
        self,
        brands: Iterable[str] | None = None,
        categories: Iterable[str] | None = None,
        *,
        model_name: str = DEFAULT_PHAYATHAIBERT_MODEL,
        min_confidence: float = 0.70,
        local_files_only: bool = False,
        classifier: IntentClassifier | None = None,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.min_confidence = float(min_confidence)
        self.rule_parser = ThaiCommandParser(brands=brands, categories=categories)
        self.classifier = classifier or PhayaThaiBertIntentClassifier(
            model_name, local_files_only=local_files_only
        )
        self._unavailable_logged = False

    def parse(self, message: object | None) -> ParsedCommand:
        """Parse entities deterministically, then let PhayaThaiBERT resolve unknowns."""

        rule_result = self.rule_parser.parse(message)
        if not rule_result.normalized_text:
            return rule_result
        try:
            prediction = self.classifier.predict(rule_result.normalized_text)
        except PhayaThaiBertUnavailable as error:
            if not self._unavailable_logged:
                LOGGER.warning("PhayaThaiBERT unavailable; using rule parser: %s", error)
                self._unavailable_logged = True
            return rule_result
        except Exception:  # A custom classifier must never break webhook delivery.
            LOGGER.exception("PhayaThaiBERT intent inference failed; using rule parser")
            return rule_result

        # Rules remain authoritative once they recognize an intent or hard shopping
        # constraints.  PhayaThaiBERT adds semantic coverage for unknown utterances.
        if rule_result.intent != INTENT_UNKNOWN or prediction.confidence < self.min_confidence:
            return rule_result
        if prediction.intent == INTENT_SEARCH:
            entities = CommandEntities(query=rule_result.normalized_text)
        else:
            entities = CommandEntities()
        return ParsedCommand(
            prediction.intent,
            prediction.confidence,
            entities,
            rule_result.raw_text,
            rule_result.normalized_text,
        )


__all__ = [
    "DEFAULT_PHAYATHAIBERT_MODEL",
    "PhayaThaiBertCommandParser",
    "PhayaThaiBertIntentClassifier",
    "PhayaThaiBertUnavailable",
    "IntentClassifier",
    "IntentPrediction",
]
