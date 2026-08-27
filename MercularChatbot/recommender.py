"""Constraint-safe, varied Top-5 product recommendations.

The recommender has three explicit stages:

1. de-duplicate and hard-filter every structured/search constraint;
2. rank the surviving products and bound the randomisation pool;
3. return the exact leading products for explicit price/discount ordering, otherwise
   sample without replacement while preferring products not recently shown.

Randomness and time are injectable, making both fairness and TTL behaviour fully
deterministic in tests.  There are no random retry loops; an identical recent set is
avoided with one bounded swap scan when an alternative set exists.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
import math
import random
import re
from threading import RLock
import time
from typing import Callable, Iterable, Mapping, Protocol, Sequence

try:  # Package import (``MercularChatbot.recommender``).
    from .models import Product
    from .nlp import (
        BRAND_ALIASES,
        CATEGORY_ALIASES,
        FEATURE_ALIASES,
        INTENT_REFRESH,
        INTENT_SEARCH,
        SORT_DISCOUNT,
        SORT_NEWEST,
        SORT_POPULAR,
        SORT_PRICE_ASC,
        SORT_PRICE_DESC,
        CommandEntities,
        ParsedCommand,
        category_aliases_for,
        compact_text,
        normalize_text,
    )
except ImportError:  # Direct script/test import from the project directory.
    from models import Product
    from nlp import (
        BRAND_ALIASES,
        CATEGORY_ALIASES,
        FEATURE_ALIASES,
        INTENT_REFRESH,
        INTENT_SEARCH,
        SORT_DISCOUNT,
        SORT_NEWEST,
        SORT_POPULAR,
        SORT_PRICE_ASC,
        SORT_PRICE_DESC,
        CommandEntities,
        ParsedCommand,
        category_aliases_for,
        compact_text,
        normalize_text,
    )


MAX_TOP_K = 5
MAX_CANDIDATE_POOL = 50


class RandomSource(Protocol):
    """Small subset of :mod:`random` accepted for deterministic injection."""

    def sample(self, population: Sequence[Product], k: int) -> list[Product]: ...

    def random(self) -> float: ...


@dataclass(frozen=True, slots=True)
class _HistoryRecord:
    timestamp: float
    product_ids: tuple[str, ...]


def _normal_values(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(value for value in (compact_text(item) for item in values) if value)


@lru_cache(maxsize=256)
def _brand_aliases(brand: str) -> tuple[str, ...]:
    target = compact_text(brand)
    for canonical, aliases in BRAND_ALIASES.items():
        values = (canonical, *aliases)
        compacts = _normal_values(values)
        if target in compacts:
            return compacts
    return (target,) if target else ()


def _equivalent(value: str, aliases: Sequence[str], *, fuzzy: bool = True) -> bool:
    candidate = compact_text(value)
    if not candidate:
        return False
    for alias in aliases:
        target = compact_text(alias)
        if not target:
            continue
        if candidate == target or target in candidate or candidate in target:
            return True
        if fuzzy and min(len(candidate), len(target)) >= 5:
            if SequenceMatcher(None, candidate, target).ratio() >= 0.84:
                return True
    return False


@lru_cache(maxsize=1_024)
def _alias_match_parts(alias: str) -> tuple[str, str, re.Pattern[str] | None]:
    normal_alias = normalize_text(alias)
    if not normal_alias:
        return "", "", None
    if re.search(r"[a-z0-9]", normal_alias):
        escaped = re.escape(normal_alias).replace(r"\ ", r"[\s._/\-]+")
        return (
            normal_alias,
            compact_text(normal_alias),
            re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"),
        )
    return normal_alias, compact_text(normal_alias), None


def _normal_contains_alias(
    normal_value: str,
    compact_value: str,
    alias: str,
) -> bool:
    _, compact_alias, latin_pattern = _alias_match_parts(alias)
    if not compact_alias:
        return False
    if latin_pattern is not None:
        return latin_pattern.search(normal_value) is not None
    return compact_alias in compact_value


def _contains_requested_alias(value: str, aliases: Sequence[str]) -> bool:
    """Return whether a requested alias occurs directionally in ``value``.

    Thai normally has no spaces, so Thai aliases intentionally use compact substring
    matching.  Latin aliases retain token boundaries: the short microphone alias
    ``mic`` must match ``+ MIC`` but never the prefix of ``Micro SD``.  Symmetric
    containment is also deliberately absent; a broad product parent such as
    ``เกมมิ่ง`` cannot satisfy the more specific request ``เมาส์เกมมิ่ง``.
    """

    normal_value = normalize_text(value)
    if not normal_value:
        return False
    compact_value = compact_text(normal_value)
    return any(
        _normal_contains_alias(normal_value, compact_value, alias)
        for alias in aliases
    )


@lru_cache(maxsize=256)
def _category_alias_compacts(category: str) -> tuple[str, ...]:
    # Kept as normalised text rather than separator-free text so English token
    # boundaries remain available to ``_contains_requested_alias``.
    return tuple(
        value
        for value in (normalize_text(item) for item in category_aliases_for(category))
        if value
    )


_MOUSE_TERMS = _normal_values(("เมาส์", "mouse", "mice"))
_MOUSE_PRODUCT_TERMS = ("เมาส์", "mouse", "mice", "trackball", "แทร็กบอล")
_KEYBOARD_TERMS = _normal_values(("คีย์บอร์ด", "keyboard", "แป้นพิมพ์"))
_PHONE_TERMS = _normal_values(
    ("โทรศัพท์", "โทรศัพท์มือถือ", "มือถือ", "สมาร์ทโฟน", "smartphone", "mobile phone")
)
_PHONE_LEAF_PREFIXES = _normal_values(
    ("โทรศัพท์", "มือถือ", "สมาร์ทโฟน", "smartphone", "mobile phone")
)
_COMPUTER_TERMS = _normal_values(
    ("คอมพิวเตอร์", "เครื่องคอมพิวเตอร์", "คอม", "คอมตั้งโต๊ะ", "desktop", "computer", "pc")
)
_COMPUTER_DEVICE_LEAF_PREFIXES = _normal_values(
    (
        "คอมพิวเตอร์ พีซี",
        "คอมพิวเตอร์ ออลอินวัน",
        "คอมพิวเตอร์เกมมิ่ง",
        "มินิ คอมพิวเตอร์",
        "Computer Set",
        "ซูเปอร์คอมพิวเตอร์",
    )
)
_COMPUTER_ACCESSORY_NAME_TERMS = (
    "cable",
    "adapter",
    "stand",
    "case",
    "สายเชื่อมต่อ",
    "สายพ่วง",
    "อะแดปเตอร์",
    "ขาตั้ง",
    "กระเป๋า",
    "เคส",
)
_GENERIC_ACCESSORY_TERMS = frozenset(
    _normal_values(("อุปกรณ์เสริม", *CATEGORY_ALIASES["อุปกรณ์เสริม"]))
)

# Accessory phrases have two jobs.  Request aliases include the short residual left by
# the parser (``mouse pad`` -> category ``เมาส์``, query ``pad``), while product
# aliases stay precise enough not to label an ordinary keyboard containing a switch as
# a bag of replacement switches.
_ACCESSORY_REQUEST_ALIASES: Mapping[str, tuple[str, ...]] = {
    "mouse_pad": (
        "mouse pad",
        "mousepad",
        "desk mat",
        "deskmat",
        "แผ่นรองเมาส์",
        "แผ่นรอง",
    ),
    "wrist_rest": (
        "mouse wrist rest",
        "keyboard wrist rest",
        "wrist rest",
        "palm rest",
        "ที่รองข้อมือเมาส์",
        "ที่รองข้อมือคีย์บอร์ด",
        "ที่รองข้อมือ",
        "รองข้อมือ",
    ),
    "mouse_bungee": ("mouse bungee", "บันจี้เมาส์", "บันจี้"),
    "mouse_grip": ("mouse grip", "grip tape", "กริปเมาส์", "กริป"),
    "mouse_skates": ("mouse skates", "mouse feet", "ฟีตเมาส์", "สเกตเมาส์"),
    "mouse_receiver": (
        "mouse receiver",
        "usb receiver",
        "wireless receiver",
        "ตัวรับสัญญาณเมาส์",
        "ตัวรับสัญญาณ",
        "รีซีฟเวอร์",
    ),
    "keycap": ("keyboard keycap", "keycap set", "key caps", "keycap", "คีย์แคป"),
    "keyboard_switch": (
        "keyboard switch",
        "switch tester",
        "switch set",
        "switch pack",
        "สวิตช์คีย์บอร์ด",
        "ชุดสวิตช์",
    ),
    "keyboard_case": ("keyboard carrying case", "keyboard case", "เคสคีย์บอร์ด"),
}
_ACCESSORY_PRODUCT_ALIASES: Mapping[str, tuple[str, ...]] = {
    **_ACCESSORY_REQUEST_ALIASES,
    "keyboard_switch": _ACCESSORY_REQUEST_ALIASES["keyboard_switch"]
    + ("mechanical switch set",),
}
_CONTEXTUAL_ACCESSORY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "mouse_pad": ("pad",),
    "keyboard_switch": ("switch", "สวิตช์"),
    "keyboard_case": ("case", "เคส"),
}

# Literal subtype groups stay separate from broad category aliases.  A user asking for
# ``earbuds`` has chosen a narrower product type than ``หูฟัง``; likewise a
# ``soundbar`` must not expand to every speaker merely because both share one category.
_SUBTYPE_QUERY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "earbuds": (
        "earbuds",
        "earbud",
        "ear buds",
        "buds",
        "bud",
        "เอียร์บัด",
        "หูฟังอินเอียร์",
        "in-ear",
        "in ear",
        "tws",
        "enco",
    ),
    "soundbar": ("soundbar", "sound bar", "ซาวด์บาร์", "ซาวด์บา", "ซาวบาร์"),
    "printer": (
        "printer",
        "เครื่องพิมพ์",
        "ปริ้นเตอร์",
        "ปรินเตอร์",
        "พรินเตอร์",
    ),
    "toner": (
        "toner",
        "ink cartridge",
        "printer cartridge",
        "ตลับหมึก",
        "หมึกพิมพ์",
        "หมึกปริ้นเตอร์",
    ),
    "scanner": ("scanner", "สแกนเนอร์", "สแกนเนอ", "เครื่องสแกน"),
    "mic_stand": (
        "microphone arm",
        "mic arm",
        "microphone stand",
        "mic stand",
        "boom arm",
        "ขาไมค์",
        "ขาไมโครโฟน",
        "ขาตั้งไมค์",
        "แขนไมค์",
    ),
    "mic_filter": (
        "pop filter",
        "noise shield",
        "windscreen",
        "ป็อปฟิลเตอร์",
        "แผ่นกรองเสียงไมค์",
    ),
    "mic_cable": ("microphone cable", "mic cable", "xlr cable", "สายไมค์"),
    "mic_mount": ("shock mount", "microphone mount", "mic mount", "ช็อคเมาท์"),
    "watch_strap": (
        "watch strap",
        "watch band",
        "apple watch band",
        "สาย apple watch",
        "สายสมาร์ทวอทช์",
        "สายนาฬิกา",
    ),
    "fitness_tracker": (
        "fitness tracker",
        "activity tracker",
        "fitness band",
        "smart band",
        "สายรัดข้อมือสุขภาพ",
        "อุปกรณ์ติดตามสุขภาพ",
    ),
    "gaming_monitor": (
        "gaming monitor",
        "gaming display",
        "gaming",
        "จอเกมมิ่ง",
        "จอเล่นเกม",
        "เกมมิ่ง",
        "odyssey",
        "ultragear",
        "tuf gaming monitor",
        "rog monitor",
        "predator monitor",
    ),
    "ups": ("ups", "uninterruptible power supply", "เครื่องสำรองไฟ", "สำรองไฟ"),
    "flash_drive": (
        "flash drive",
        "usb drive",
        "thumb drive",
        "แฟลชไดรฟ์",
        "แฟลชไดร์ฟ",
    ),
    "usb_hub": ("usb hub", "usb-hub", "ยูเอสบีฮับ", "ฮับยูเอสบี"),
    "conference_camera": (
        "conference camera",
        "conference webcam",
        "meeting camera",
        "กล้องประชุม",
        "เว็บแคมประชุม",
    ),
}
_MIC_ACCESSORY_SUBTYPES = frozenset(
    {"mic_stand", "mic_filter", "mic_cable", "mic_mount"}
)
_MOUSE_ACCESSORY_KINDS = frozenset(
    {
        "mouse_pad",
        "wrist_rest",
        "mouse_bungee",
        "mouse_grip",
        "mouse_skates",
        "mouse_receiver",
    }
)
_KEYBOARD_ACCESSORY_KINDS = frozenset(
    {"wrist_rest", "keycap", "keyboard_switch", "keyboard_case"}
)
_KIND_BASE_CONCEPTS: Mapping[str, tuple[str, ...]] = {
    "mouse_pad": ("เมาส์",),
    "mouse_bungee": ("เมาส์",),
    "mouse_grip": ("เมาส์",),
    "mouse_skates": ("เมาส์",),
    "mouse_receiver": ("เมาส์",),
    "keycap": ("คีย์บอร์ด",),
    "keyboard_switch": ("คีย์บอร์ด",),
    "keyboard_case": ("คีย์บอร์ด",),
    "wrist_rest": ("เมาส์", "คีย์บอร์ด"),
}
_GAMING_TERMS = ("gaming", "เกมมิ่ง", "เล่นเกม")


@lru_cache(maxsize=1_024)
def _accessory_kinds(value: str, *, request: bool) -> frozenset[str]:
    aliases_by_kind = _ACCESSORY_REQUEST_ALIASES if request else _ACCESSORY_PRODUCT_ALIASES
    normal_value = normalize_text(value)
    compact_value = compact_text(normal_value)
    return frozenset(
        kind
        for kind, aliases in aliases_by_kind.items()
        if any(
            _normal_contains_alias(normal_value, compact_value, alias)
            for alias in aliases
        )
    )


@lru_cache(maxsize=512)
def _category_concepts(value: str) -> frozenset[str]:
    """Map a concrete catalogue label to known non-generic product concepts."""

    concepts = {
        canonical
        for canonical, aliases in CATEGORY_ALIASES.items()
        if _contains_requested_alias(value, (canonical, *aliases))
    }
    # A generic accessory bucket is not evidence that an item conflicts with a more
    # specific request; its product name/tags may be the only usable taxonomy data.
    concepts.difference_update({"อุปกรณ์เสริม", "อุปกรณ์คอมพิวเตอร์"})
    return frozenset(concepts)


@lru_cache(maxsize=2_048)
def _subtype_kinds(value: str) -> frozenset[str]:
    normal_value = normalize_text(value)
    compact_value = compact_text(normal_value)
    return frozenset(
        kind
        for kind, aliases in _SUBTYPE_QUERY_ALIASES.items()
        if any(
            _normal_contains_alias(normal_value, compact_value, alias)
            for alias in aliases
        )
    )


def _product_subtype_kinds(product: Product) -> frozenset[str]:
    # The broad category is intentionally absent: ``เครื่องปริ้น / หมึก``
    # must not turn a toner or scanner into a printer subtype.
    metadata = " ".join((product.name, *product.tags, product.description))
    return _subtype_kinds(metadata)


def _literal_subtype_matches(
    product: Product,
    category: str,
    query: str,
    request_text: str,
) -> bool:
    """Enforce literal product identity inside mixed Mercular catalogue leaves."""

    category_concepts = _category_concepts(category)
    requested = _subtype_kinds(" ".join((query, request_text)))

    # A watch strap may be routed through either the smart-watch leaf or the generic
    # accessory intent by different parser/catalogue versions.  Its literal identity
    # is strong enough to resolve the specialised product safely in either case.
    if "watch_strap" in requested:
        return "watch_strap" in _product_subtype_kinds(product)

    if "หูฟัง" in category_concepts and "earbuds" in requested:
        return "earbuds" in _product_subtype_kinds(product)
    if "ลำโพง" in category_concepts and "soundbar" in requested:
        return "soundbar" in _product_subtype_kinds(product)
    if "จอคอม" in category_concepts and "gaming_monitor" in requested:
        return "gaming_monitor" in _product_subtype_kinds(product)

    if "เครื่องพิมพ์" in category_concepts:
        product_subtypes = _product_subtype_kinds(product)
        if "toner" in requested:
            return "toner" in product_subtypes
        if "scanner" in requested:
            return "scanner" in product_subtypes
        # The leaf mixes printers, toner, and scanners.  A generic printer request is
        # still literal and may correctly return no rows in a shallow snapshot.
        return "printer" in product_subtypes

    if "ไมโครโฟน" in category_concepts:
        product_subtypes = _product_subtype_kinds(product)
        requested_accessories = requested & _MIC_ACCESSORY_SUBTYPES
        if requested_accessories:
            return bool(requested_accessories & product_subtypes)
        if product_subtypes & _MIC_ACCESSORY_SUBTYPES:
            return False
        leaf_evidence = " ".join(
            (product.category, product.category_path[-1] if product.category_path else "")
        )
        if _contains_requested_alias(leaf_evidence, ("cable", "สาย")):
            return False
        return _contains_requested_alias(
            " ".join((product.name, *product.tags)),
            ("microphone", "mic", "ไมโครโฟน", "ไมค์"),
        )

    if "สมาร์ทวอทช์" in category_concepts:
        product_subtypes = _product_subtype_kinds(product)
        if "fitness_tracker" in requested:
            return "fitness_tracker" in product_subtypes
        return "watch_strap" not in product_subtypes

    return True


def _product_accessory_kinds(product: Product) -> frozenset[str]:
    # Product identity must come from product-level fields.  A broad category such
    # as ``แผ่นรองเมาส์`` can contain both pads and wrist rests, so including the
    # category path here would make a wrist rest satisfy a specific mouse-pad query.
    metadata = " ".join(
        (product.name, *product.tags, product.description)
    )
    return _accessory_kinds(metadata, request=False)


@lru_cache(maxsize=512)
def _requested_accessory_kinds(category: str, query: str) -> frozenset[str]:
    kinds = set(_accessory_kinds(" ".join((category, query)), request=True))
    requested_category = compact_text(category)
    if any(term in requested_category for term in _MOUSE_TERMS):
        if _contains_requested_alias(query, _CONTEXTUAL_ACCESSORY_ALIASES["mouse_pad"]):
            kinds.add("mouse_pad")
    if any(term in requested_category for term in _KEYBOARD_TERMS):
        for kind in ("keyboard_switch", "keyboard_case"):
            if _contains_requested_alias(query, _CONTEXTUAL_ACCESSORY_ALIASES[kind]):
                kinds.add(kind)
    return frozenset(kinds)


def _accessory_request_matches(product: Product, category: str, query: str) -> bool:
    requested_kinds = _requested_accessory_kinds(category, query)
    if requested_kinds:
        return bool(requested_kinds & _product_accessory_kinds(product))

    requested = compact_text(category)
    if any(term in requested for term in _MOUSE_TERMS):
        return not bool(_product_accessory_kinds(product) & _MOUSE_ACCESSORY_KINDS)
    if any(term in requested for term in _KEYBOARD_TERMS):
        return not bool(_product_accessory_kinds(product) & _KEYBOARD_ACCESSORY_KINDS)
    return True


def _base_product_identity_matches(product: Product, category: str, query: str) -> bool:
    """Reject catalogue-taxonomy mistakes for ordinary base-product requests.

    The live mouse leaf currently also contains drawing tablets.  Treat the leaf as
    candidate discovery, then require product-name evidence for a base mouse.  An
    explicit accessory request keeps its existing specialised matching behaviour.
    Phone accessory leaves can contain the word ``โทรศัพท์`` too, so a phone request
    additionally requires a concrete phone leaf rather than a film/case/stand leaf.
    """

    requested_category = compact_text(category)
    if any(term in requested_category for term in _PHONE_TERMS):
        leaf_fields = tuple(
            compact
            for field in (
                product.category,
                product.category_path[-1] if product.category_path else "",
            )
            if (compact := compact_text(field))
        )
        return any(
            field.startswith(prefix)
            for field in leaf_fields
            for prefix in _PHONE_LEAF_PREFIXES
        )
    if any(term == requested_category for term in _COMPUTER_TERMS):
        # Normalise each leaf once.  The old nested expression repeated the same
        # Unicode/regex work for every prefix and every product in the catalogue.
        leaf_fields = tuple(
            compact
            for field in (
                product.category,
                product.category_path[-1] if product.category_path else "",
            )
            if (compact := compact_text(field))
        )
        is_computer_device = any(
            field.startswith(prefix)
            for field in leaf_fields
            for prefix in _COMPUTER_DEVICE_LEAF_PREFIXES
        )
        if not is_computer_device:
            return False
        product_name = normalize_text(product.name)
        excluded_product_terms = (
            "notebook",
            "laptop",
            "โน้ตบุ๊ก",
            "โน๊ตบุ๊ค",
            *_COMPUTER_ACCESSORY_NAME_TERMS,
        )
        return not _contains_requested_alias(
            product_name,
            excluded_product_terms,
        )
    if not any(term in requested_category for term in _MOUSE_TERMS):
        return True
    if _requested_accessory_kinds(category, query):
        return True
    normal_name = normalize_text(product.name)
    compact_name = compact_text(normal_name)
    return any(
        _normal_contains_alias(normal_name, compact_name, term)
        for term in _MOUSE_PRODUCT_TERMS
    )


def _discount(product: Product) -> float:
    if (
        product.price is None
        or product.original_price is None
        or product.original_price <= 0
        or product.price >= product.original_price
    ):
        return 0.0
    return (product.original_price - product.price) / product.original_price


def _timestamp(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError):
        return 0.0


class ProductRecommender:
    """Recommend at most five unique products while respecting every constraint.

    Args:
        history_ttl_seconds: How long a user/query recommendation remains recent.
        history_size: Maximum result sets retained per user/query key.
        candidate_pool_size: Maximum number of top-ranked products exposed to random
            selection.  It is always at least the requested result count and globally
            capped at :data:`MAX_CANDIDATE_POOL`.
        rng: Object implementing ``sample`` and ``random`` (for example
            ``random.Random(7)``).  A private RNG is used by default.
        clock: Monotonic seconds callable; inject a fake clock for TTL tests.
    """

    def __init__(
        self,
        *,
        history_ttl_seconds: float = 1_800,
        history_size: int = 20,
        candidate_pool_size: int = 15,
        rng: RandomSource | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if history_ttl_seconds <= 0:
            raise ValueError("history_ttl_seconds must be positive")
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        self.history_ttl_seconds = float(history_ttl_seconds)
        self.history_size = int(history_size)
        self.candidate_pool_size = min(MAX_CANDIDATE_POOL, int(candidate_pool_size))
        self._rng: RandomSource = rng or random.Random()
        self._clock = clock or time.monotonic
        self._history: dict[tuple[str, str], deque[_HistoryRecord]] = {}
        self._lock = RLock()

    @staticmethod
    def _entities(command: ParsedCommand | CommandEntities) -> CommandEntities:
        if isinstance(command, CommandEntities):
            return command
        if isinstance(command, ParsedCommand):
            return command.entities
        # A clear error here is preferable to silently dropping shopping constraints.
        raise TypeError("command must be ParsedCommand or CommandEntities")

    @staticmethod
    def _request_text(
        command: ParsedCommand | CommandEntities,
        entities: CommandEntities,
    ) -> str:
        parts = [entities.category or "", entities.query]
        if isinstance(command, ParsedCommand):
            # Raw text retains literal aliases (for example ``earbuds``) even when an
            # older parser version consumes the whole phrase into a broad category.
            parts.append(command.normalized_text or normalize_text(command.raw_text))
        return " ".join(part for part in parts if part)

    @staticmethod
    def _category_matches(product: Product, category: str, query: str = "") -> bool:
        aliases = _category_alias_compacts(category)

        # ``อุปกรณ์เสริม`` is a real leaf in the snapshot and also a suffix of
        # specialised leaves such as ``ไมโครโฟนและอุปกรณ์เสริม``.  The
        # generic request uses exact leaf semantics to avoid swallowing all of those
        # specialised catalogues.
        if compact_text(category) in _GENERIC_ACCESSORY_TERMS:
            if "watch_strap" in _subtype_kinds(query):
                return "watch_strap" in _product_subtype_kinds(product)
            return compact_text(product.category) in _GENERIC_ACCESSORY_TERMS

        # Preserve qualifiers embedded in a catalogue label.  Expanding
        # ``เมาส์เกมมิ่ง`` to the base alias ``mouse`` is useful, but an office mouse
        # must still fail the gaming modifier.
        if _contains_requested_alias(category, _GAMING_TERMS):
            product_metadata = " ".join(
                (
                    product.name,
                    product.category,
                    *product.category_path,
                    *product.tags,
                    product.description,
                )
            )
            if not _contains_requested_alias(product_metadata, _GAMING_TERMS):
                return False

        # Only the product's concrete category and final breadcrumb are leaf evidence.
        # Earlier breadcrumbs in the live snapshot are combined parents such as
        # ``หูฟัง/ลำโพง`` and must never make both sibling categories match.
        leaf_fields = tuple(
            dict.fromkeys(
                field
                for field in (
                    product.category,
                    product.category_path[-1] if product.category_path else "",
                )
                if field
            )
        )
        if any(
            _contains_requested_alias(field, aliases)
            for field in leaf_fields
        ):
            return True

        requested_concepts = set(_category_concepts(category))
        requested_kinds = _requested_accessory_kinds(category, query)
        for kind in requested_kinds:
            requested_concepts.update(_KIND_BASE_CONCEPTS.get(kind, ()))
        product_concepts = set(_category_concepts(product.category))
        if product.category_path:
            product_concepts.update(_category_concepts(product.category_path[-1]))
        if (
            requested_concepts
            and product_concepts
            and product_concepts.isdisjoint(requested_concepts)
        ):
            # A known, different leaf is authoritative.  For example, a speaker that
            # bundles ``+ MIC`` remains a speaker, not a microphone result.
            return False

        # Names/tags/descriptions are a fallback for broad scraper buckets such as
        # ``อุปกรณ์คอมพิวเตอร์``; parent breadcrumbs are intentionally excluded.
        fallback_fields = (*product.tags, product.name, product.description)
        return any(
            _contains_requested_alias(field, aliases)
            for field in fallback_fields
            if field
        )

    @staticmethod
    def _brand_matches(product: Product, brands: Sequence[str]) -> bool:
        # Multiple requested brands mean alternatives (OR), as one product normally has
        # one manufacturer.  An empty product brand cannot satisfy a brand constraint.
        if not product.brand:
            return False
        return any(_equivalent(product.brand, _brand_aliases(brand)) for brand in brands)

    @staticmethod
    @lru_cache(maxsize=512)
    def _query_groups(query: str, category: str = "") -> tuple[tuple[str, ...], ...]:
        """Turn a residual query into AND groups whose members are synonym alternatives."""

        normal = normalize_text(query)
        if not normal:
            return ()
        soft = {
            "ดี",
            "ดีๆ",
            "น่าสนใจ",
            "คุ้ม",
            "คุ้มๆ",
            "สวย",
            "best",
            "good",
            "nice",
            "recommended",
        }
        # Recognise multi-word feature aliases first and remove their spans so their
        # component words do not become duplicate AND constraints.
        groups: list[tuple[str, ...]] = []
        remaining = f" {normal} "
        requested_category = compact_text(category)
        accessory_groups: list[tuple[str, ...]] = []
        for kind, aliases in _ACCESSORY_REQUEST_ALIASES.items():
            contextual = aliases
            if kind == "mouse_pad" and any(
                term in requested_category for term in _MOUSE_TERMS
            ):
                contextual = (*aliases, *_CONTEXTUAL_ACCESSORY_ALIASES[kind])
            elif kind in {"keyboard_switch", "keyboard_case"} and any(
                term in requested_category for term in _KEYBOARD_TERMS
            ):
                contextual = (*aliases, *_CONTEXTUAL_ACCESSORY_ALIASES[kind])
            accessory_groups.append(contextual)
        alias_groups = (
            *tuple(_SUBTYPE_QUERY_ALIASES.values()),
            *accessory_groups,
            *tuple((canonical, *aliases) for canonical, aliases in FEATURE_ALIASES.items()),
        )
        for candidates in alias_groups:
            matched = next(
                (
                    candidate
                    for candidate in candidates
                    if _contains_requested_alias(remaining, (candidate,))
                ),
                None,
            )
            if matched is not None:
                groups.append(tuple(normalize_text(item) for item in candidates))
                remaining = remaining.replace(normalize_text(matched), " ")

        for token in re_tokens(remaining):
            if token in soft or len(compact_text(token)) < 2:
                continue
            groups.append((token,))
        # Preserve order but remove equivalent duplicate groups.
        unique: list[tuple[str, ...]] = []
        keys: set[tuple[str, ...]] = set()
        for group in groups:
            key = tuple(sorted(set(_normal_values(group))))
            if key and key not in keys:
                keys.add(key)
                unique.append(group)
        return tuple(unique)

    @staticmethod
    def _query_group_matches(haystack: str, group: Sequence[str]) -> bool:
        normal_haystack = normalize_text(haystack)
        for alias in group:
            compact_alias = compact_text(alias)
            if not compact_alias:
                continue
            if _contains_requested_alias(normal_haystack, (alias,)):
                return True
            # Model-name/English typo tolerance without relaxing short feature tokens.
            # Expanded feature groups already provide explicit bilingual synonyms.
            # Fuzzy matching each long Thai synonym can confuse a product category
            # (for example ``คีย์บอร์ดเกมมิ่ง``) with ``คีย์บอร์ดกล``.  Keep
            # typo tolerance only for a literal singleton/model token.
            if len(group) == 1 and len(compact_alias) >= 5:
                for word in re_tokens(normal_haystack):
                    compact_word = compact_text(word)
                    if abs(len(compact_word) - len(compact_alias)) <= 2 and SequenceMatcher(
                        None, compact_word, compact_alias
                    ).ratio() >= 0.84:
                        return True
        return False

    @classmethod
    def _query_matches(cls, product: Product, query: str, category: str = "") -> bool:
        groups = cls._query_groups(query, category)
        if not groups:
            return True
        haystack = product.search_text
        return all(cls._query_group_matches(haystack, group) for group in groups)

    @classmethod
    def product_matches(cls, product: Product, command: ParsedCommand | CommandEntities) -> bool:
        """Return whether a product satisfies every populated command constraint."""

        entities = cls._entities(command)
        if entities.category_path:
            candidate_path = product.category_path
            if len(candidate_path) < len(entities.category_path) or any(
                normalize_text(actual) != normalize_text(expected)
                for actual, expected in zip(
                    candidate_path[: len(entities.category_path)],
                    entities.category_path,
                    strict=True,
                )
            ):
                return False
        request_text = cls._request_text(command, entities)
        if not _accessory_request_matches(
            product,
            entities.category or "",
            entities.query,
        ):
            return False
        if not _base_product_identity_matches(
            product,
            entities.category or "",
            entities.query,
        ):
            return False
        if entities.category and not cls._category_matches(
            product,
            entities.category,
            entities.query,
        ):
            return False
        if not _literal_subtype_matches(
            product,
            entities.category or "",
            entities.query,
            request_text,
        ):
            return False
        if entities.brands and not cls._brand_matches(product, entities.brands):
            return False
        if entities.excluded_brands and cls._brand_matches(
            product, entities.excluded_brands
        ):
            return False
        if entities.min_price is not None:
            if product.price is None or (
                product.price < entities.min_price
                if entities.min_price_inclusive
                else product.price <= entities.min_price
            ):
                return False
        if entities.max_price is not None:
            if product.price is None or (
                product.price > entities.max_price
                if entities.max_price_inclusive
                else product.price >= entities.max_price
            ):
                return False
        if entities.in_stock is not None and product.in_stock is not entities.in_stock:
            return False
        if entities.query and not cls._query_matches(
            product,
            entities.query,
            entities.category or "",
        ):
            return False
        return True

    def filter_products(
        self,
        products: Iterable[Product],
        command: ParsedCommand | CommandEntities,
    ) -> list[Product]:
        """De-duplicate by product id and apply all hard filters, preserving input order."""

        result: list[Product] = []
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        for product in products:
            if product.id in seen_ids:
                continue
            seen_ids.add(product.id)
            if self.product_matches(product, command):
                name_key = normalize_text(product.name)
                if name_key and name_key in seen_names:
                    continue
                if name_key:
                    seen_names.add(name_key)
                result.append(product)
        return result

    @classmethod
    def _relevance(cls, product: Product, entities: CommandEntities) -> float:
        score = 0.0
        query_groups = cls._query_groups(entities.query, entities.category or "")
        # Product details can contain thousands of characters.  Only normalise that
        # full text when there is an actual free-text query to score; a bare/category
        # recommendation otherwise paid this cost for all 1,715 products needlessly.
        haystack = normalize_text(product.search_text) if query_groups else ""
        compact_haystack = compact_text(haystack) if haystack else ""
        # ``rank_products`` receives the hard-filtered set, so populated structured
        # constraints are already guaranteed.  Avoid repeating the full taxonomy
        # classifier for every candidate merely to award a constant base score.
        if entities.category:
            score += 5.0
        if entities.category_path:
            score += 5.0
        if entities.brands:
            score += 5.0
        for group in query_groups:
            exact_length = next(
                (
                    len(compact_alias)
                    for alias in group
                    if (compact_alias := compact_text(alias))
                    and _contains_requested_alias(haystack, (alias,))
                ),
                0,
            )
            if exact_length:
                # Synonyms within one group are semantically equivalent; the first
                # exact hit is enough and avoids normalising every remaining alias.
                score += 3.0 + min(1.0, exact_length / 20)
                continue
            score += max(
                (
                    SequenceMatcher(None, compact_alias, compact_haystack).ratio()
                    for alias in group
                    if (compact_alias := compact_text(alias))
                ),
                default=0.0,
            )
        popular_hints = ("ขายดี", "ยอดนิยม", "bestseller", "best seller", "popular", "ฮิต")
        popularity_text = normalize_text(
            " ".join((product.name, product.category, *product.tags))
        )
        if any(normalize_text(hint) in popularity_text for hint in popular_hints):
            score += 0.5
        score += _discount(product) * 0.25
        return score

    def rank_products(
        self,
        products: Iterable[Product],
        command: ParsedCommand | CommandEntities,
    ) -> list[Product]:
        """Rank already-filtered products according to query and sort preference."""

        entities = self._entities(command)
        scored = [(self._relevance(product, entities), product) for product in products]
        stable_name = lambda product: (normalize_text(product.name), product.id)

        if entities.sort == SORT_PRICE_ASC:
            key = lambda pair: (
                pair[1].price is None,
                pair[1].price if pair[1].price is not None else math.inf,
                -pair[0],
                stable_name(pair[1]),
            )
        elif entities.sort == SORT_PRICE_DESC:
            key = lambda pair: (
                pair[1].price is None,
                -(pair[1].price if pair[1].price is not None else -math.inf),
                -pair[0],
                stable_name(pair[1]),
            )
        elif entities.sort == SORT_DISCOUNT:
            key = lambda pair: (-_discount(pair[1]), -pair[0], stable_name(pair[1]))
        elif entities.sort == SORT_NEWEST:
            key = lambda pair: (-_timestamp(pair[1].scraped_at), -pair[0], stable_name(pair[1]))
        elif entities.sort == SORT_POPULAR:
            key = lambda pair: (-pair[0], -_discount(pair[1]), stable_name(pair[1]))
        else:
            key = lambda pair: (-pair[0], stable_name(pair[1]))
        return [product for _, product in sorted(scored, key=key)]

    @staticmethod
    def _signature(entities: CommandEntities) -> str:
        parts = (
            compact_text(entities.category or ""),
            "/".join(compact_text(part) for part in entities.category_path),
            ",".join(sorted(compact_text(brand) for brand in entities.brands)),
            ",".join(
                sorted(compact_text(brand) for brand in entities.excluded_brands)
            ),
            "" if entities.min_price is None else f"{entities.min_price:g}",
            "" if entities.max_price is None else f"{entities.max_price:g}",
            str(entities.min_price_inclusive),
            str(entities.max_price_inclusive),
            "" if entities.in_stock is None else str(entities.in_stock),
            entities.sort or "",
            compact_text(entities.query),
        )
        return "|".join(parts)

    def _prune_history(self, now: float) -> None:
        cutoff = now - self.history_ttl_seconds
        empty_keys: list[tuple[str, str]] = []
        for key, records in self._history.items():
            while records and records[0].timestamp <= cutoff:
                records.popleft()
            if not records:
                empty_keys.append(key)
        for key in empty_keys:
            del self._history[key]

    def _choose(
        self,
        pool: Sequence[Product],
        count: int,
        records: Sequence[_HistoryRecord],
    ) -> list[Product]:
        if count <= 0 or not pool:
            return []
        shown_counts = Counter(
            product_id for record in records for product_id in record.product_ids
        )
        unseen = [product for product in pool if shown_counts[product.id] == 0]

        if len(unseen) >= count:
            chosen = list(self._rng.sample(unseen, count))
        else:
            chosen = list(unseen)
            chosen_ids = {product.id for product in chosen}
            seen_candidates = [product for product in pool if product.id not in chosen_ids]
            # Fair exposure: least-shown items first; RNG only breaks equal-count ties.
            decorated = [
                (shown_counts[product.id], self._rng.random(), product)
                for product in seen_candidates
            ]
            decorated.sort(key=lambda item: (item[0], item[1]))
            chosen.extend(product for _, _, product in decorated[: count - len(chosen)])

        # Sets, rather than display order, define repetition.  Find one bounded swap if
        # this exact set was recently used and any distinct set is available.
        recent_sets = {frozenset(record.product_ids) for record in records}
        chosen_ids = {product.id for product in chosen}
        if frozenset(chosen_ids) in recent_sets:
            alternatives = [product for product in pool if product.id not in chosen_ids]
            selected_order = sorted(
                chosen,
                key=lambda product: (-shown_counts[product.id], product.id),
            )
            alternative_order = sorted(
                alternatives,
                key=lambda product: (shown_counts[product.id], product.id),
            )
            replacement: tuple[Product, Product] | None = None
            for outgoing in selected_order:
                for incoming in alternative_order:
                    candidate_set = frozenset((chosen_ids - {outgoing.id}) | {incoming.id})
                    if candidate_set not in recent_sets:
                        replacement = outgoing, incoming
                        break
                if replacement:
                    break
            if replacement:
                outgoing, incoming = replacement
                chosen[chosen.index(outgoing)] = incoming

        # Randomness changes the set; catalogue relevance still determines the display
        # order for recommendations without an explicit extrema sort.
        rank_position = {product.id: index for index, product in enumerate(pool)}
        chosen.sort(key=lambda product: rank_position[product.id])
        return chosen

    def recommend(
        self,
        products: Iterable[Product],
        command: ParsedCommand | CommandEntities,
        *,
        user_id: str = "anonymous",
        top_k: int = MAX_TOP_K,
    ) -> list[Product]:
        """Return zero to ``top_k`` unique, matching products (never more than five).

        Non-shopping :class:`ParsedCommand` intents return an empty list.  A bare
        :class:`CommandEntities` is treated as a search, which is convenient for batch
        jobs and unit tests.
        """

        try:
            limit = min(MAX_TOP_K, max(0, int(top_k)))
        except (TypeError, ValueError):
            raise ValueError("top_k must be an integer") from None
        if limit == 0:
            return []
        if isinstance(command, ParsedCommand) and command.intent not in {
            INTENT_SEARCH,
            INTENT_REFRESH,
            "product_search",  # compatibility with early notebook versions
        }:
            return []

        entities = self._entities(command)
        filtered = self.filter_products(products, command)
        if not filtered:
            return []
        ranked = self.rank_products(filtered, entities)
        pool_limit = min(len(ranked), max(limit, self.candidate_pool_size))
        pool = ranked[:pool_limit]

        now = float(self._clock())
        key = (str(user_id or "anonymous"), self._signature(entities))
        with self._lock:
            self._prune_history(now)
            records = self._history.get(key, deque())
            count = min(limit, len(pool))
            if entities.sort in {SORT_PRICE_ASC, SORT_PRICE_DESC, SORT_DISCOUNT}:
                # Explicit extrema must mean the actual first N ranked matches.  A
                # random subset of a sorted pool can look ordered while omitting the
                # cheapest, most expensive, or most discounted products.
                chosen = list(pool[:count])
            else:
                chosen = self._choose(pool, count, tuple(records))
            target = self._history.setdefault(key, deque(maxlen=self.history_size))
            target.append(_HistoryRecord(now, tuple(product.id for product in chosen)))
        return chosen

    def clear_history(self, user_id: str | None = None) -> None:
        """Clear all recommendation history, or only records belonging to one user."""

        with self._lock:
            if user_id is None:
                self._history.clear()
                return
            target = str(user_id or "anonymous")
            for key in [key for key in self._history if key[0] == target]:
                del self._history[key]

    def history_snapshot(self) -> Mapping[tuple[str, str], tuple[tuple[str, ...], ...]]:
        """Return an immutable diagnostic view without exposing mutable internals."""

        with self._lock:
            return {
                key: tuple(record.product_ids for record in records)
                for key, records in self._history.items()
            }


def re_tokens(value: str) -> tuple[str, ...]:
    """Tokenise on whitespace/punctuation without requiring a Thai NLP package."""

    return tuple(re.findall(r"[a-z0-9ก-๙][a-z0-9ก-๙._-]*", normalize_text(value)))


def recommend_products(
    products: Iterable[Product],
    command: ParsedCommand | CommandEntities,
    *,
    user_id: str = "anonymous",
    top_k: int = MAX_TOP_K,
    rng: RandomSource | None = None,
) -> list[Product]:
    """Stateless convenience API; use :class:`ProductRecommender` for repeat avoidance."""

    return ProductRecommender(rng=rng).recommend(
        products,
        command,
        user_id=user_id,
        top_k=top_k,
    )


__all__ = [
    "MAX_TOP_K",
    "MAX_CANDIDATE_POOL",
    "ProductRecommender",
    "recommend_products",
]
