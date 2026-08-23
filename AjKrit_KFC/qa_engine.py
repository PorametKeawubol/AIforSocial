"""Question answering over the local KFC menu and promotion snapshot."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import load_dotenv

from scraper import DATA_FILE, KfcScraperError, load_snapshot


load_dotenv()
LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_FILE = PROJECT_DIR / "data" / "kfc_embeddings.npz"

HELP_TEXT = (
    "พิมพ์ชื่อเมนูหรือคำถามเกี่ยวกับโปรโมชัน KFC ได้เลย\n"
    "ตัวอย่าง:\n"
    "• เดอะบอกซ์ ซิกเนเจอร์ คืออะไร\n"
    "• มีเมนูไก่ทอดอะไรบ้าง\n"
    "• มีโปรโมชั่นอะไรบ้าง"
)
PROMOTION_ONLY_NOTICE = (
    "บอตนี้แสดงเฉพาะข้อมูลโปรโมชัน KFC\n"
    "ลองถาม “มีโปรโมชั่นอะไรบ้าง” หรือพิมพ์ชื่อ/ราคาของโปรโมชัน"
)
QUESTION_FILLERS = (
    "คืออะไร",
    "มีอะไรบ้าง",
    "อะไรบ้าง",
    "มีอะไร",
    "ขอรายละเอียด",
    "รายละเอียดของ",
    "รายละเอียด",
    "ราคาเท่าไหร่",
    "ราคา",
    "เมนูนี้",
    "หน่อย",
    "ครับ",
    "ค่ะ",
    "คะ",
    "please",
)
PROMOTION_WORDS = ("โปรโมชั่น", "โปรโมชัน", "โปร", "promotion", "promo", "ดีล", "คูปอง")
HELP_WORDS = {"help", "/help", "ช่วยเหลือ", "วิธีใช้", "เริ่มต้น"}
BROAD_LIST_MARKERS = ("อะไรบ้าง", "รายการ", "มีเมนู", "ทั้งหมด")
DETAIL_QUERY_MARKERS = ("เมนูนี้", "ขอรายละเอียด", "รายละเอียดของ")
LIST_SUBJECT_FILLERS = (
    "เมนูนี้",
    "มีเมนู",
    "เมนู",
    "มีรายการ",
    "รายการ",
    "อะไรบ้าง",
    "ทั้งหมด",
    "ขอ",
    "ช่วย",
    "มี",
)
MENU_DISCOVERY_MARKERS = (
    "เมนู",
    "รายการ",
    "รายการอาหาร",
    "รายการของกิน",
    "ของกิน",
    "อาหาร",
    "มีอะไรขาย",
    "มีอะไรให้เลือก",
    "มีอะไรบ้าง",
    "มีเมนู",
    "ขอเมนู",
    "เปิดดูเมนู",
    "ดูเมนู",
    "แนะนำเมนู",
    "อยากดูเมนู",
    "menu",
)
MENU_QUERY_FILLERS = (
    "รายการอาหาร",
    "รายการของกิน",
    "เมนูอาหาร",
    "ของกิน",
    "มีอะไรขาย",
    "มีอะไรให้เลือก",
    "มีอะไรบ้าง",
    "มีเมนู",
    "ขอเมนู",
    "เปิดดูเมนู",
    "ดูเมนู",
    "แนะนำเมนู",
    "อยากดูเมนู",
    "เมนู",
    "อาหาร",
    "รายการ",
    "ขอ",
    "ช่วย",
    "หน่อย",
    "เปิด",
    "ดู",
    "แนะนำ",
    "อยาก",
    "กิน",
    "วันนี้",
    "มี",
    "อะไร",
    "บ้าง",
    "ให้เลือก",
    "kfc",
    "menu",
)
# Some KFC chicken products use product names such as WingZ or Nuggets rather
# than the literal phrase "ไก่ทอด". Expand only this well-defined topic so a
# broad menu question remains useful without letting BERT include unrelated
# menu items such as drinks or salad.
TOPIC_TERMS: dict[str, tuple[str, ...]] = {
    "ไก่ทอด": (
        "ไก่ทอด",
        "ไก่วิงซ์",
        "วิงซ์แซ่บ",
        "wingz",
        "ชิคเก้นป๊อป",
        "chicken pop",
        "นักเก็ตส์",
        "nuggets",
        "ไก่ไม่มีกระดูก",
        "crispy strips",
        "ซิงเกอร์เบอร์เกอร์",
        "zinger burger",
    ),
}


def _clean_text(value: object | None) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split())


def _normalise(value: object | None) -> str:
    text = _clean_text(value).casefold()
    return re.sub(r"[^\w\sก-๙]", " ", text)


# Thai final consonants in the same pronunciation group are often swapped in
# informal typing, for example "บักเกจ" or "บักเกด" for "บักเก็ต".  This
# key is intentionally applied only to a word's final consonant, so an initial
# letter such as the จ in "จุใจ" is left unchanged.
THAI_FINAL_CONSONANT_KEY = {
    **{letter: "ก" for letter in "กขคฆ"},
    **{letter: "ต" for letter in "จฉชซฌฎฏฐฑฒดตถทธศษส"},
    **{letter: "ป" for letter in "บปพฟภ"},
    **{letter: "น" for letter in "ณญนรลฬ"},
    "ม": "ม",
    "ง": "ง",
    "ย": "ย",
    "ว": "ว",
}


def _thai_final_consonant_key(value: object | None) -> str:
    """Normalise only Thai word-final consonants for tolerant menu search."""

    # Tone marks and mai taikhu are commonly omitted in a quick message, for
    # example "บักเกจ" instead of "บักเก็ต".  They are not needed to compare
    # the pronunciation of a product-name fragment.
    text = re.sub(r"[่้๊๋็์]", "", _normalise(value))

    def replace_final(match: re.Match[str]) -> str:
        # Thai leading vowels are written before their initial consonant, e.g.
        # ใจ and ไก่.  That consonant is not a final sound and must not be
        # folded into a final-consonant group.
        if match.start() > 0 and text[match.start() - 1] in "เแโใไ":
            return match.group(1)
        return THAI_FINAL_CONSONANT_KEY.get(match.group(1), match.group(1))

    return re.sub(
        r"([ก-ฮ])(?=\s|$)",
        replace_final,
        text,
    )


THAI_TYPO_CONSONANT_FOLD = str.maketrans({"ศ": "ซ", "ษ": "ซ", "ส": "ซ"})


def _thai_typo_search_key(value: object | None) -> str:
    """Make a compact, typo-tolerant key for Thai menu-name fragments."""

    return re.sub(
        r"\s+", "", _thai_final_consonant_key(value).translate(THAI_TYPO_CONSONANT_FOLD)
    )


def _menu_name_fragments(item: dict[str, Any]) -> tuple[str, ...]:
    """Return product titles and aliases for a high-confidence typo match."""

    aliases = item.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []
    return tuple(
        dict.fromkeys(
            text
            for value in (item.get("name"), *aliases)
            if (text := _clean_text(value))
        )
    )


def _menu_component_fragments(item: dict[str, Any]) -> tuple[str, ...]:
    """Return component labels as a lower-priority search signal."""

    components = item.get("components", [])
    if not isinstance(components, list):
        components = []
    return tuple(
        dict.fromkeys(
            text
            for value in components
            if (text := _clean_text(value))
        )
    )


def _fuzzy_fragment_score(query: str, fragment: str) -> float:
    """Return a partial-string similarity for a plausible Thai menu typo."""

    compact_query = _thai_typo_search_key(query)
    compact_fragment = _thai_typo_search_key(fragment)
    if len(compact_query) < 4 or len(compact_fragment) < 4:
        return 0.0
    if compact_query in compact_fragment:
        return 1.0

    # Compare the short query against similarly sized substrings rather than
    # the entire product title.  Thus "วิงแสบ" can match the phrase
    # "วิงซ์แซ่บ" inside a longer menu name or a component list.
    # Keep the candidate at least as long as the query.  Shorter windows can
    # falsely turn "บักเกจ" into "นักเก็ตส์" merely because their suffixes
    # overlap, while the longer "วิงซ์แซ่บ" still remains a valid match for
    # "วิงแสบ".
    minimum_window = max(4, len(compact_query))
    maximum_window = min(len(compact_fragment), len(compact_query) + 2)
    if minimum_window > maximum_window:
        return SequenceMatcher(None, compact_query, compact_fragment, autojunk=False).ratio()

    best = 0.0
    for window_size in range(minimum_window, maximum_window + 1):
        for start in range(len(compact_fragment) - window_size + 1):
            candidate = compact_fragment[start : start + window_size]
            best = max(
                best,
                SequenceMatcher(None, compact_query, candidate, autojunk=False).ratio(),
            )
    return best


def _fuzzy_menu_fragment_scores(query: str, item: dict[str, Any]) -> tuple[float, float]:
    """Return separate typo scores for titles and secondary components."""

    return (
        max(
            (_fuzzy_fragment_score(query, fragment) for fragment in _menu_name_fragments(item)),
            default=0.0,
        ),
        max(
            (
                _fuzzy_fragment_score(query, fragment)
                for fragment in _menu_component_fragments(item)
            ),
            default=0.0,
        ),
    )


def _query_variants(question: str) -> list[str]:
    cleaned = _normalise(question)
    stripped = cleaned
    for filler in QUESTION_FILLERS:
        stripped = stripped.replace(filler, " ")
    stripped = _clean_text(stripped)
    return list(dict.fromkeys(value for value in (cleaned, stripped) if value))


def _character_ngrams(value: str, size: int = 2) -> set[str]:
    compact = re.sub(r"\s+", "", _normalise(value))
    if len(compact) <= size:
        return {compact} if compact else set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _document_text(item: dict[str, Any]) -> str:
    aliases = item.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []
    components = item.get("components", [])
    if not isinstance(components, list):
        components = []
    choice_text: list[str] = []
    choices = item.get("choices", [])
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            options = choice.get("options", [])
            if not isinstance(options, list):
                options = []
            choice_text.append(
                " ".join(
                    part
                    for part in (
                        _clean_text(choice.get("group")),
                        " ".join(_clean_text(option) for option in options),
                    )
                    if part
                )
            )
    return "\n".join(
        part
        for part in (
            _clean_text(item.get("name")),
            _clean_text(item.get("description")),
            " ".join(_clean_text(component) for component in components),
            " ".join(choice_text),
            _clean_text(item.get("category")),
            " ".join(_clean_text(alias) for alias in aliases),
            "โปรโมชั่น" if item.get("kind") == "promotion" else "เมนู KFC",
        )
        if part
    )


def _fingerprint(items: Iterable[dict[str, Any]]) -> str:
    serialised = json.dumps(
        [
            {
                "id": item.get("id"),
                "kind": item.get("kind"),
                "name": item.get("name"),
                "description": item.get("description"),
                "components": item.get("components", []),
                "choices": item.get("choices", []),
                "category": item.get("category"),
                "aliases": item.get("aliases", []),
            }
            for item in items
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


class KfcQuestionAnswerer:
    """Answer KFC questions with a BERT index when available, otherwise lexical search."""

    def __init__(
        self,
        data_file: Path | str = DATA_FILE,
        index_file: Path | str = DEFAULT_INDEX_FILE,
        model_name: str | None = None,
        semantic_enabled: bool | None = None,
    ) -> None:
        self.data_file = Path(data_file)
        self.index_file = Path(index_file)
        self.model_name = model_name or os.getenv(
            "BERT_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.semantic_enabled = (
            semantic_enabled
            if semantic_enabled is not None
            else os.getenv("SEMANTIC_SEARCH_ENABLED", "true").casefold()
            in {"1", "true", "yes", "on"}
        )
        self.minimum_semantic_score = float(os.getenv("SEMANTIC_MIN_SCORE", "0.26"))
        # A carousel should not show unrelated products merely because every
        # catalog item has a weak lexical resemblance to an unknown phrase.
        self.menu_search_min_score = float(os.getenv("MENU_SEARCH_MIN_SCORE", "0.60"))
        self._items: list[dict[str, Any]] = []
        self._snapshot_metadata: dict[str, Any] = {}
        self._mtime_ns: int | None = None
        self._model: Any | None = None
        self._vectors: np.ndarray | None = None
        self._vector_fingerprint: str | None = None
        self._lock = threading.RLock()

    def _load_items(self) -> list[dict[str, Any]]:
        with self._lock:
            try:
                mtime_ns = self.data_file.stat().st_mtime_ns
            except FileNotFoundError:
                self._items = []
                self._snapshot_metadata = {}
                self._mtime_ns = None
                return self._items
            if self._mtime_ns == mtime_ns:
                return self._items
            snapshot = load_snapshot(self.data_file)
            raw_items = snapshot.get("items", [])
            self._items = [item for item in raw_items if isinstance(item, dict) and item.get("name")]
            self._snapshot_metadata = snapshot
            self._mtime_ns = mtime_ns
            self._vectors = None
            self._vector_fingerprint = None
            return self._items

    def _is_promotion_only_snapshot(self, items: Iterable[dict[str, Any]]) -> bool:
        """Whether this snapshot is deliberately limited to promotion data."""

        if self._snapshot_metadata.get("scope") == "promotions":
            return True
        item_list = list(items)
        return bool(item_list) and all(item.get("kind") == "promotion" for item in item_list)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        # SentenceTransformer uses its local Hugging Face cache after the
        # first download.  The CLI build step intentionally performs that
        # initial download before the bot starts serving LINE webhooks.
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load_cached_vectors(self, expected_fingerprint: str) -> np.ndarray | None:
        if not self.index_file.exists():
            return None
        try:
            with np.load(self.index_file, allow_pickle=False) as payload:
                stored_fingerprint = str(payload["fingerprint"].item())
                stored_model_name = str(payload["model_name"].item())
                vectors = np.asarray(payload["vectors"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.warning("Ignoring invalid BERT cache: %s", exc)
            return None
        if (
            stored_fingerprint != expected_fingerprint
            or stored_model_name != self.model_name
            or len(vectors) != len(self._items)
        ):
            return None
        return vectors

    def build_index(self) -> None:
        """Build/update the on-disk multilingual BERT embedding cache."""

        items = self._load_items()
        if not items:
            raise KfcScraperError(
                "No KFC data exists yet. Run `python scraper.py --refresh` first."
            )
        with self._lock:
            fingerprint = _fingerprint(items)
            model = self._load_model()
            documents = [_document_text(item) for item in items]
            vectors = np.asarray(
                model.encode(documents, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.index_file,
                fingerprint=np.asarray(fingerprint),
                model_name=np.asarray(self.model_name),
                vectors=vectors,
            )
            self._vectors = vectors
            self._vector_fingerprint = fingerprint

    def _semantic_scores(self, question: str, items: list[dict[str, Any]]) -> list[float] | None:
        if not self.semantic_enabled:
            return None
        with self._lock:
            fingerprint = _fingerprint(items)
            if self._vectors is None or self._vector_fingerprint != fingerprint:
                self._vectors = self._load_cached_vectors(fingerprint)
                self._vector_fingerprint = fingerprint if self._vectors is not None else None
            if self._vectors is None:
                # The bot remains responsive when the optional BERT cache has
                # not been built yet; use `--build-index` to enable it.
                return None
            try:
                model = self._load_model()
                query = np.asarray(
                    model.encode([question], normalize_embeddings=True, show_progress_bar=False),
                    dtype=np.float32,
                )[0]
            except Exception as exc:
                LOGGER.warning("BERT query encoding failed; using lexical search: %s", exc)
                return None
            return [float(score) for score in self._vectors @ query]

    @staticmethod
    def _lexical_score(question: str, item: dict[str, Any]) -> float:
        variants = _query_variants(question)
        name = _normalise(item.get("name"))
        aliases = [_normalise(alias) for alias in item.get("aliases", [])]
        document = _normalise(_document_text(item))
        phonetic_name = _thai_final_consonant_key(name)
        phonetic_aliases = [_thai_final_consonant_key(alias) for alias in aliases]
        phonetic_document = _thai_final_consonant_key(document)
        best = 0.0
        for query in variants:
            if not query:
                continue
            name_score = SequenceMatcher(None, query, name).ratio() if name else 0.0
            doc_score = SequenceMatcher(None, query, document).ratio() if document else 0.0
            grams = _character_ngrams(query)
            document_grams = _character_ngrams(document)
            overlap = len(grams & document_grams) / len(grams | document_grams) if grams else 0.0
            score = max(name_score, doc_score * 0.65 + overlap * 0.35)
            if query in name or any(query in alias for alias in aliases):
                score = max(score, 0.98)
            elif query in document:
                score = max(score, 0.76)
            phonetic_query = _thai_final_consonant_key(query)
            compact_phonetic_query = re.sub(r"\s+", "", phonetic_query)
            if len(compact_phonetic_query) >= 4 and (
                phonetic_query in phonetic_name
                or any(phonetic_query in alias for alias in phonetic_aliases)
            ):
                # A phonetic match is intentionally a little weaker than an
                # exact spelling, while still strong enough to open a menu
                # Carousel for a plausible typo.
                score = max(score, 0.94)
            elif len(compact_phonetic_query) >= 4 and phonetic_query in phonetic_document:
                score = max(score, 0.74)
            fuzzy_name_score, fuzzy_component_score = _fuzzy_menu_fragment_scores(
                query, item
            )
            if fuzzy_name_score >= 0.86:
                # Sentence-BERT provides semantic ranking for natural
                # language queries.  This small lexical boost only recovers a
                # likely misspelled product/component phrase before the BERT
                # score is combined in _rank.
                score = max(score, 0.72 + 0.24 * fuzzy_name_score)
            elif fuzzy_component_score >= 0.90:
                # A matching component is useful when there is no matching
                # title, but it should not outrank a product actually named
                # after the user's query.
                score = max(score, 0.52 + 0.24 * fuzzy_component_score)
            best = max(best, score)
        return best

    def _rank(
        self, question: str, items: list[dict[str, Any]]
    ) -> list[tuple[float, dict[str, Any]]]:
        semantic_scores = self._semantic_scores(question, items)
        lexical_scores = [self._lexical_score(question, item) for item in items]
        ranked: list[tuple[float, dict[str, Any]]] = []
        for index, item in enumerate(items):
            lexical = lexical_scores[index]
            if semantic_scores is None:
                score = lexical
            else:
                # Exact Thai/English names remain decisive even when generic
                # semantic wording makes BERT scores close together.
                score = max(semantic_scores[index], lexical)
            ranked.append((score, item))
        return sorted(ranked, key=lambda pair: pair[0], reverse=True)

    @staticmethod
    def _menu_query_subject(question: str) -> str:
        """Remove menu-discovery wording, leaving a possible product name."""

        subject = _normalise(question)
        for filler in MENU_QUERY_FILLERS:
            subject = subject.replace(filler, " ")
        return _clean_text(subject)

    @classmethod
    def _is_menu_discovery_query(cls, question: str) -> bool:
        """Whether the user asked for a menu overview rather than a named item."""

        normalized = _normalise(question)
        return bool(
            normalized
            and not cls._menu_query_subject(question)
            and any(marker in normalized for marker in MENU_DISCOVERY_MARKERS)
        )

    def is_menu_discovery_query(self, question: str) -> bool:
        """Public routing helper for a full, paginated menu overview."""

        return self._is_menu_discovery_query(question)

    def search_items(
        self,
        question: str,
        *,
        kind: str | None = None,
        limit: int = 2,
        minimum_score: float | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Return the closest catalog items for a menu carousel.

        The BERT cache is used through ``_rank`` when available.  A broad menu
        request is intentionally handled as an overview so generic words such
        as ``รายการอาหาร`` do not get mistaken for a product name.  A partial
        or misspelled product name still goes through semantic/lexical ranking.
        """

        question = _clean_text(question)
        result_limit = max(1, int(limit))
        if not question:
            return []
        items = self._load_items()
        if kind is not None:
            candidates = [item for item in items if item.get("kind") == kind]
        else:
            candidates = items
        if not candidates:
            return []

        if kind == "menu" and self._is_menu_discovery_query(question):
            return [(1.0, item) for item in candidates[:result_limit]]

        ranked = self._rank(question, items)
        if kind is not None:
            ranked = [pair for pair in ranked if pair[1].get("kind") == kind]
        threshold = (
            self.menu_search_min_score
            if minimum_score is None
            else float(minimum_score)
        )
        if not ranked or ranked[0][0] < threshold:
            return []
        return ranked[:result_limit]

    def find_item(
        self, item_key: str, *, kind: str | None = None
    ) -> dict[str, Any] | None:
        """Find a catalog item by its postback id, name, or alias."""

        target = _clean_text(item_key)
        if not target:
            return None
        items = self._load_items()
        candidates = (
            [item for item in items if item.get("kind") == kind]
            if kind is not None
            else items
        )
        for item in candidates:
            if str(item.get("id") or "").strip() == target:
                return item

        normalized_target = _normalise(target)
        compact_target = re.sub(r"\s+", "", normalized_target)
        for item in candidates:
            labels = [item.get("name", ""), *item.get("aliases", [])]
            for label in labels:
                normalized_label = _normalise(label)
                if normalized_label == normalized_target:
                    return item
                if compact_target and re.sub(r"\s+", "", normalized_label) == compact_target:
                    return item
        return None

    @staticmethod
    def _is_promotion_question(question: str) -> bool:
        normalized = _normalise(question)
        return any(word in normalized for word in PROMOTION_WORDS)

    @staticmethod
    def _is_general_list_question(question: str) -> bool:
        normalized = _normalise(question)
        return any(marker in normalized for marker in BROAD_LIST_MARKERS)

    @staticmethod
    def _list_subject(question: str) -> str:
        """Return the topic in a broad list question, if one was supplied."""

        subject = _normalise(question)
        for filler in LIST_SUBJECT_FILLERS:
            subject = subject.replace(filler, " ")
        return _clean_text(subject)

    @staticmethod
    def _detail_query_target(question: str) -> str:
        """Extract a menu name placed after a detail-question prompt."""

        normalized = _normalise(question)
        if not any(marker in normalized for marker in DETAIL_QUERY_MARKERS):
            return ""
        variants = _query_variants(question)
        if len(variants) < 2:
            return ""
        target = variants[-1]
        compact_target = re.sub(r"\s+", "", target)
        return target if len(compact_target) >= 4 and target != normalized else ""

    @staticmethod
    def _list_search_terms(subject: str) -> tuple[str, ...]:
        normalized_subject = _normalise(subject)
        if not normalized_subject:
            return ()
        return TOPIC_TERMS.get(normalized_subject, (normalized_subject,))

    @staticmethod
    def _items_for_list_subject(
        items: Iterable[dict[str, Any]], subject: str
    ) -> list[dict[str, Any]]:
        """Find every catalog item that belongs in a broad list response."""

        terms = KfcQuestionAnswerer._list_search_terms(subject)
        if not terms:
            return []
        return [
            item
            for item in items
            if any(term in _normalise(_document_text(item)) for term in terms)
        ]

    @staticmethod
    def _trim_description(value: str, maximum: int = 900) -> str:
        value = _clean_text(value)
        return value if len(value) <= maximum else value[: maximum - 1].rstrip() + "…"

    @staticmethod
    def _item_components(item: dict[str, Any]) -> list[str]:
        values = item.get("components", [])
        if not isinstance(values, list):
            return []
        return list(
            dict.fromkeys(
                component
                for component in (_clean_text(value) for value in values)
                if component
            )
        )

    @staticmethod
    def _item_choices(item: dict[str, Any]) -> list[tuple[str, list[str]]]:
        values = item.get("choices", [])
        if not isinstance(values, list):
            return []
        result: list[tuple[str, list[str]]] = []
        for value in values:
            if not isinstance(value, dict):
                continue
            group = _clean_text(value.get("group"))
            raw_options = value.get("options", [])
            if not isinstance(raw_options, list):
                raw_options = []
            options = list(
                dict.fromkeys(
                    option
                    for option in (_clean_text(option) for option in raw_options)
                    if option
                )
            )
            if group and options:
                result.append((group, options))
        return result

    def _format_item(self, item: dict[str, Any]) -> str:
        icon = "🏷️" if item.get("kind") == "promotion" else "🍗"
        lines = [f"{icon} {item.get('name', '')}"]
        if item.get("price"):
            lines.append(f"💰 ราคา: {item['price']}")
        if item.get("description"):
            lines.append(f"📝 {self._trim_description(item['description'])}")
        components = self._item_components(item)
        if components:
            lines.append("📦 ประกอบด้วย:")
            lines.extend(f"• {component}" for component in components)
        choices = self._item_choices(item)
        if choices:
            # Keep the complete response below LINE's 4,500-character limit
            # while preserving every normal-sized option group in full.
            reserved = [f"🔗 {item['url']}"] if item.get("url") else []
            maximum_length = 4_300

            def can_append(value: str) -> bool:
                return len("\n".join([*lines, value, *reserved])) <= maximum_length

            if can_append("🎛️ เลือกได้:"):
                lines.append("🎛️ เลือกได้:")
            options_truncated = False
            for group, options in choices:
                if not can_append(f"• {group}"):
                    options_truncated = True
                    break
                lines.append(f"• {group}")
                for option in options:
                    if not can_append(f"  - {option}"):
                        options_truncated = True
                        break
                    lines.append(f"  - {option}")
                if options_truncated:
                    break
            if options_truncated and can_append("  - …มีตัวเลือกเพิ่มเติมในหน้าเมนู KFC"):
                lines.append("  - …มีตัวเลือกเพิ่มเติมในหน้าเมนู KFC")
        if item.get("url"):
            lines.append(f"🔗 {item['url']}")
        return "\n".join(lines)

    def _format_promotion_list(self, items: list[dict[str, Any]]) -> str:
        promotions = [item for item in items if item.get("kind") == "promotion"]
        if not promotions:
            return "ยังไม่พบโปรโมชันในข้อมูลล่าสุด ลองรันอัปเดตข้อมูลอีกครั้ง"
        return self._format_compact_list(
            promotions,
            heading=f"🏷️ พบ {len(promotions)} โปรโมชัน KFC",
            footer="พิมพ์ชื่อหรือราคาของโปรโมชันเพื่อดูรายละเอียด",
        )

    def _format_menu_overview(self, items: list[dict[str, Any]]) -> str:
        menu_items = [item for item in items if item.get("kind") == "menu"]
        if not menu_items:
            return "ยังไม่พบรายการเมนูในข้อมูลล่าสุด ลองรันอัปเดตข้อมูลอีกครั้ง"
        return self._format_compact_list(
            menu_items,
            heading=f"🍗 พบ {len(menu_items)} เมนู KFC",
            footer="พิมพ์ชื่อเมนูเพื่อดูรายละเอียดและส่วนประกอบ",
        )

    @staticmethod
    def _format_compact_list(
        items: list[dict[str, Any]], *, heading: str, footer: str
    ) -> str:
        """Format all matching items compactly, keeping a LINE reply safe."""

        # Reserve space for the footer and a transparent "shown / total" note
        # if a future catalog becomes too large for one LINE text message.
        maximum_length = 4_300
        lines = [heading]
        shown = 0
        for index, item in enumerate(items, start=1):
            suffix = f" — {item['price']}" if item.get("price") else ""
            line = f"{index}. {item.get('name', '')}{suffix}"
            if len("\n".join([*lines, line, "", footer])) > maximum_length:
                break
            lines.append(line)
            shown = index
        if shown < len(items):
            lines.extend(("", f"แสดง {shown} จาก {len(items)} รายการ"))
        lines.extend(("", footer))
        return "\n".join(lines)

    @staticmethod
    def _explicit_name_match(
        question: str, items: Iterable[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Prefer a menu explicitly named inside a longer Thai question."""

        normalized_question = _normalise(question)
        compact_question = re.sub(r"\s+", "", normalized_question)
        matches: list[tuple[int, dict[str, Any]]] = []
        for item in items:
            labels = [item.get("name", ""), *item.get("aliases", [])]
            for label in labels:
                normalized_label = _normalise(label)
                compact_label = re.sub(r"\s+", "", normalized_label)
                if len(compact_label) >= 4 and (
                    normalized_label in normalized_question
                    or compact_label in compact_question
                ):
                    matches.append((len(normalized_label), item))
                    break
        return max(matches, key=lambda match: match[0])[1] if matches else None

    def answer(self, question: str) -> str:
        question = _clean_text(question)
        if not question or _normalise(question) in HELP_WORDS:
            return HELP_TEXT
        items = self._load_items()
        if not items:
            return (
                "ยังไม่มีข้อมูลเมนู KFC ในระบบ\n"
                "ผู้ดูแลต้องรัน `python scraper.py --refresh --build-index` ก่อน"
            )
        candidates = items
        promotion_question = self._is_promotion_question(question)
        promotion_only = self._is_promotion_only_snapshot(items)
        if promotion_question:
            candidates = [item for item in items if item.get("kind") == "promotion"] or items
        explicit_item = self._explicit_name_match(question, candidates)
        if explicit_item is not None:
            return self._format_item(explicit_item)

        detail_target = self._detail_query_target(question)
        if detail_target:
            detail_item = self._explicit_name_match(detail_target, candidates)
            if detail_item is not None:
                return self._format_item(detail_item)
            if promotion_only:
                return f"ไม่พบโปรโมชันชื่อ “{detail_target}” ในข้อมูลล่าสุด\n{PROMOTION_ONLY_NOTICE}"
            item_label = "โปรโมชัน" if promotion_question else "เมนู"
            return f"ไม่พบ{item_label}ชื่อ “{detail_target}” ในข้อมูลล่าสุด"

        if promotion_only and not promotion_question:
            return PROMOTION_ONLY_NOTICE

        if self._is_general_list_question(question):
            subject = self._list_subject(question)
            if promotion_question:
                promotion_subjects = {_normalise(word) for word in PROMOTION_WORDS}
                if not subject or subject in promotion_subjects:
                    return self._format_promotion_list(items)
            else:
                menu_items = [item for item in items if item.get("kind") == "menu"]
                if not subject:
                    return self._format_menu_overview(items)
                matches = self._items_for_list_subject(menu_items, subject)
                if matches:
                    return self._format_compact_list(
                        matches,
                        heading=(
                            f"🍗 เมนู KFC ทั้งหมด {len(menu_items)} เมนู\n"
                            f"พบ {len(matches)} เมนูที่เกี่ยวกับ “{subject}”"
                        ),
                        footer="พิมพ์ชื่อเมนูเพื่อดูรายละเอียดและส่วนประกอบ",
                    )
                return f"ยังไม่พบเมนูที่เกี่ยวกับ “{subject}” ในข้อมูลล่าสุด"

        ranked = self._rank(question, candidates)
        if not ranked or ranked[0][0] < 0.13:
            return "ไม่พบข้อมูลที่ค้นหา ลองพิมพ์ชื่อเมนูหรือคำว่า “มีโปรโมชั่นอะไรบ้าง”"

        return self._format_item(ranked[0][1])
