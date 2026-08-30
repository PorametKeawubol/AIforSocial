"""Fast, dependency-free Thai/English command parsing for the Mercular bot.

The parser deliberately uses auditable rules instead of a network model.  A command is
normalised once, matched against Thai/English aliases (including common misspellings),
and returned as a :class:`ParsedCommand`.  Brand and category names discovered by the
product repository can be supplied at construction time, so the parser stays useful as
Mercular's catalogue changes.

Public API::

    parser = ThaiCommandParser(brands=["Sony"], categories=["หูฟัง"])
    command = parser.parse("หูฟัง Sony ไร้สาย งบ ๕k พร้อมส่ง")
    assert command.entities.max_price == 5000
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


INTENT_SEARCH = "search"
INTENT_GREETING = "greeting"
INTENT_HELP = "help"
INTENT_THANKS = "thanks"
INTENT_CONTACT = "contact"
INTENT_ORDER = "order"
INTENT_REFRESH = "refresh"
INTENT_PROMOTION = "promotion"
INTENT_UNKNOWN = "unknown"

SUPPORTED_INTENTS = (
    INTENT_SEARCH,
    INTENT_GREETING,
    INTENT_HELP,
    INTENT_THANKS,
    INTENT_CONTACT,
    INTENT_ORDER,
    INTENT_REFRESH,
    INTENT_PROMOTION,
    INTENT_UNKNOWN,
)

SORT_PRICE_ASC = "price_asc"
SORT_PRICE_DESC = "price_desc"
SORT_NEWEST = "newest"
SORT_POPULAR = "popular"
SORT_DISCOUNT = "discount"

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_THAI_NUMBER_WORDS: Mapping[str, float] = {
    "ศูนย์": 0,
    "หนึ่ง": 1,
    "สอง": 2,
    "สาม": 3,
    "สี่": 4,
    "ห้า": 5,
    "หก": 6,
    "เจ็ด": 7,
    "แปด": 8,
    "เก้า": 9,
    "สิบ": 10,
}


def normalize_text(value: object | None) -> str:
    """Return matching-friendly Unicode text while preserving decimal prices.

    Thai digits are converted to ASCII and long character repetitions used in chat
    (for example ``ดีมากกก``) are collapsed.  The function is intentionally public so
    repositories and tests can build signatures using exactly the parser's rules.
    """

    # NFC keeps Thai SARA AM (``ำ``) intact.  NFKC decomposes it into two codepoints,
    # which makes ordinary literals such as ``ขั้นต่ำ`` fail to match.
    text = unicodedata.normalize("NFC", str(value or ""))
    text = text.translate(_THAI_DIGITS).casefold()
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("≥", ">=").replace("≤", "<=")
    # Chat elongation applies to letters, never digits: collapsing ``1000`` would
    # corrupt both prices and model names such as ``WH-1000XM5``.
    text = re.sub(r"([^\W\d_])\1{2,}", r"\1", text, flags=re.UNICODE)
    text = re.sub(r"[^\wก-๙\s.\-฿<>=]", " ", text, flags=re.UNICODE)
    return " ".join(text.split()).strip(" .-")


def compact_text(value: object | None) -> str:
    """Return normalised text without separators, useful for Thai substring matching."""

    return re.sub(r"[^a-z0-9ก-๙]", "", normalize_text(value))


# Canonical values are deliberately human-readable.  The recommender expands the same
# alias table while matching them to arbitrary catalogue category labels and tags.
CATEGORY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "หูฟัง": (
        "หูฟัง",
        "หูฟังบลูทูธ",
        "เฮดโฟน",
        "เฮดเซ็ต",
        "เอียร์บัด",
        "headphone",
        "headphones",
        "headset",
        "earphone",
        "earphones",
        "earbud",
        "earbuds",
    ),
    "คีย์บอร์ด": (
        "คีย์บอร์ด",
        "คีบอร์ด",
        "คีบอด",
        "คียบอด",
        "คีย์บอด",
        "คีย์บอด",
        "แป้นพิมพ์",
        "keyboard",
        "keybord",
        "keybaord",
    ),
    "เมาส์": ("เมาส์", "เมาส", "เม้า", "เม้าส์", "mouse", "mice", "เมา"),
    "ลำโพง": ("ลำโพง", "ลําโพง", "speaker", "speakers", "soundbar", "ซาวด์บาร์"),
    "ไมโครโฟน": ("ไมโครโฟน", "ไมค์", "ไมค", "microphone", "mic"),
    "จอคอม": (
        "จอ",
        "จอคอม",
        "จอมอนิเตอร์",
        "มอนิเตอร์",
        "monitor",
        "computer monitor",
        "gaming monitor",
    ),
    "เก้าอี้เกมมิ่ง": (
        "เก้าอี้เกมมิ่ง",
        "เก้าอี้เกม",
        "เก้าอี้คอม",
        "gaming chair",
        "computer chair",
    ),
    "โต๊ะคอม": ("โต๊ะคอม", "โต๊ะเกมมิ่ง", "gaming desk", "computer desk"),
    "เว็บแคม": ("เว็บแคม", "เว็ปแคม", "เวปแคม", "webcam", "web cam"),
    "โน้ตบุ๊ก": (
        "โน้ตบุ๊ก",
        "โน๊ตบุ๊ค",
        "โน้ตบุก",
        "โน๊ตบุก",
        "แล็ปท็อป",
        "laptop",
        "notebook",
    ),
    "เครื่องพิมพ์": (
        "เครื่องพิมพ์",
        "เครื่องปริ้น",
        "เครื่องปริ้นท์",
        "เครื่องปรินต์",
        "เครื่องปริ๊น",
        "ปริ้นเตอร์",
        "ปรินเตอร์",
        "พรินเตอร์",
        "หมึกพิมพ์",
        "ตลับหมึก",
        "ink cartridge",
        "printer",
    ),
    "สมาร์ทวอทช์": ("สมาร์ทวอทช์", "นาฬิกาอัจฉริยะ", "smartwatch", "smart watch"),
    "เกมคอนโทรลเลอร์": (
        "จอยเกม",
        "จอย",
        "คอนโทรลเลอร์",
        "game controller",
        "controller",
        "gamepad",
    ),
    "พาวเวอร์แบงก์": (
        "พาวเวอร์แบงก์",
        "พาวเวอร์แบงค์",
        "พาวเวอแบง",
        "พาวเวอร์แบง",
        "แบตสำรอง",
        "powerbank",
        "power bank",
    ),
    "โทรศัพท์": (
        "โทรศัพท์",
        "โทรศัพท์มือถือ",
        "โทรสับ",
        "โทรศัพ",
        "มือถือ",
        "มือถึอ",
        "มือถิอ",
        "สมาร์ทโฟน",
        "สมาร์ตโฟน",
        "สมาทโฟน",
        "smartphone",
        "smart phone",
        "mobile phone",
        "phone",
    ),
    "คอมพิวเตอร์": (
        "คอมพิวเตอร์",
        "เครื่องคอมพิวเตอร์",
        "คอม",
        "คอมตั้งโต๊ะ",
        "คอมตังโตะ",
        "คอมประกอบ",
        "พีซี",
        "desktop",
        "desktop computer",
        "computer",
        "pc",
    ),
    "อุปกรณ์คอมพิวเตอร์": (
        "อุปกรณ์คอมพิวเตอร์",
        "อุปกรณ์เสริมคอมพิวเตอร์",
        "computer accessories",
        "computer accessory",
        "pc accessories",
        "pc accessory",
    ),
    "อุปกรณ์เสริม": ("อุปกรณ์เสริม", "accessory", "accessories"),
}

BRAND_ALIASES: Mapping[str, tuple[str, ...]] = {
    "Sony": ("sony", "โซนี่", "โซนี", "sonny"),
    "JBL": ("jbl", "เจบีแอล", "เจบีเอล"),
    "Logitech": ("logitech", "logitec", "logitecch", "โลจิเทค", "โลจิเท็ค"),
    "Razer": ("razer", "razer", "เรเซอร์", "เรเซอ", "razer"),
    "SteelSeries": ("steelseries", "steel series", "สตีลซีรีส์", "สตีลซีรี่"),
    "HyperX": ("hyperx", "hyper x", "ไฮเปอร์เอ็กซ์"),
    "Keychron": ("keychron", "keycron", "คีย์ครอน"),
    "Audio-Technica": ("audio technica", "audio-technica", "ออดิโอเทคนิคก้า"),
    "Sennheiser": ("sennheiser", "senheiser", "เซนไฮเซอร์"),
    "Bose": ("bose", "โบส", "โบสส์"),
    "Marshall": ("marshall", "marshal", "มาร์แชล", "มาร์แชลล์"),
    "Edifier": ("edifier", "เอดิไฟเออร์"),
    "Apple": ("apple", "แอปเปิล", "แอปเปิ้ล"),
    "Samsung": ("samsung", "ซัมซุง", "samsumg"),
    "Xiaomi": ("xiaomi", "เสียวหมี่", "mi"),
    "ASUS": ("asus", "เอซุส"),
    "Acer": ("acer", "เอเซอร์"),
    "Lenovo": ("lenovo", "เลอโนโว"),
    "MSI": ("msi", "เอ็มเอสไอ"),
    "Corsair": ("corsair", "คอร์แซร์"),
    "Cooler Master": ("cooler master", "coolermaster", "คูลเลอร์มาสเตอร์"),
    "Fantech": ("fantech", "แฟนเทค"),
    "Fifine": ("fifine", "ไฟไฟน์", "ไฟน์ไฟน์"),
    "Anker": ("anker", "แองเคอร์", "แอนเคอร์"),
    "Nintendo": ("nintendo", "นินเทนโด"),
    "PlayStation": ("playstation", "play station", "เพลย์สเตชัน", "ps5"),
}

# Feature groups are used by the recommender to preserve residual constraints such as
# "wireless" even when the catalogue describes the feature in the other language.
FEATURE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "gaming": (
        "gaming",
        "เกมมิ่ง",
        "เล่นเกม",
        "เล่น fps",
        "เกม fps",
        "fps",
        "เล่น valorant",
        "valorant",
    ),
    "conference": ("conference", "meeting", "ประชุม", "ไว้ประชุม", "work from home"),
    "wireless": ("wireless", "ไร้สาย", "ไรสาย", "วายเลส", "ไวเลส", "ไม่มีสาย"),
    "bluetooth": (
        "bluetooth",
        "บลูทูธ",
        "บลูทูท",
        "บลูทูด",
        "บลูทุธ",
        "bt",
    ),
    "noise_cancelling": (
        "noise cancelling",
        "noise canceling",
        "noise cancellation",
        "ตัดเสียง",
        "ตัดเสียงรบกวน",
        "anc",
    ),
    "mechanical": ("mechanical", "แมคคานิคอล", "แมคคานิค", "คีย์บอร์ดกล"),
    "rgb": ("rgb", "ไฟ rgb", "ไฟรุ้ง"),
    "ergonomic": ("ergonomic", "ตามหลักสรีรศาสตร์", "เพื่อสุขภาพ"),
    "usb_c": ("usb c", "usb-c", "type c", "type-c", "ไทป์ซี"),
    "layout_75": ("75 percent", "75 เปอร์เซ็นต์", "75"),
    "lightweight": (
        "lightweight",
        "ultralight",
        "superlight",
        "น้ำหนักเบา",
        "เน้นเบา",
        "เบา",
    ),
    "micro_sd": ("micro sd card", "micro sd", "microsd"),
    "white": ("white", "สีขาว", "ขาว"),
    "black": ("black", "สีดำ", "ดำ"),
}


# Narrow product subtypes that live under broader Mercular categories.  Their words
# deliberately remain in the residual query so retrieval can distinguish, for
# example, a fitness tracker from any smart watch and a microphone stand from a mic.
_SUBTYPE_CATEGORY_HINTS: Mapping[str, str] = {
    "gaming monitor": "จอคอม",
    "fitness tracker": "สมาร์ทวอทช์",
    "ups": "อุปกรณ์คอมพิวเตอร์",
    "เครื่องสำรองไฟ": "อุปกรณ์คอมพิวเตอร์",
    "toner": "เครื่องพิมพ์",
    "scanner": "เครื่องพิมพ์",
    "สแกนเนอร์": "เครื่องพิมพ์",
    "สแกนเนอ": "เครื่องพิมพ์",
    "flash drive": "อุปกรณ์เสริม",
    "แฟลชไดรฟ์": "อุปกรณ์เสริม",
    "แฟลชไดร์ฟ": "อุปกรณ์เสริม",
    "usb hub": "อุปกรณ์เสริม",
    "ยูเอสบีฮับ": "อุปกรณ์เสริม",
    "conference camera": "เว็บแคม",
    "กล้องประชุม": "เว็บแคม",
    "ขาไมค์": "ไมโครโฟน",
    "สายไมค์": "ไมโครโฟน",
    "pop filter": "ไมโครโฟน",
    "apple watch strap": "สมาร์ทวอทช์",
    "watch strap": "สมาร์ทวอทช์",
    "สาย apple watch": "สมาร์ทวอทช์",
    "สายนาฬิกา": "สมาร์ทวอทช์",
    "earbuds": "หูฟัง",
    "earbud": "หูฟัง",
    "เอียร์บัด": "หูฟัง",
    "soundbar": "ลำโพง",
    "ซาวด์บาร์": "ลำโพง",
    "ซาวบาร์": "ลำโพง",
}

_SUBTYPE_QUERY_REPLACEMENTS: Mapping[str, str] = {
    # The category already carries the monitor concept; only the gaming qualifier is
    # needed as an additional hard constraint.
    "gaming monitor": "gaming",
}

_WATCH_STRAP_SUBTYPES = frozenset(
    {"apple watch strap", "watch strap", "สาย apple watch", "สายนาฬิกา"}
)


@dataclass(frozen=True, slots=True)
class CommandEntities:
    """Structured constraints extracted from one user command.

    ``brands`` is an OR constraint: a product may belong to any requested brand.  All
    other populated fields are combined with AND semantics by the recommender.
    ``category_path`` is reserved for exact catalogue navigation; ordinary NLP
    commands continue to use the human-facing ``category`` field.
    """

    category: str | None = None
    category_path: tuple[str, ...] = field(default_factory=tuple)
    brands: tuple[str, ...] = field(default_factory=tuple)
    excluded_brands: tuple[str, ...] = field(default_factory=tuple)
    min_price: float | None = None
    max_price: float | None = None
    min_price_inclusive: bool = True
    max_price_inclusive: bool = True
    in_stock: bool | None = None
    sort: str | None = None
    query: str = ""

    @property
    def brand(self) -> str | None:
        """Return the first brand for callers that only support one brand."""

        return self.brands[0] if self.brands else None

    @property
    def availability(self) -> bool | None:
        """Compatibility alias for :attr:`in_stock`."""

        return self.in_stock

    @property
    def sort_preference(self) -> str | None:
        """Compatibility alias for :attr:`sort`."""

        return self.sort

    @property
    def residual_query(self) -> str:
        """Compatibility alias for :attr:`query`."""

        return self.query

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["category_path"] = list(self.category_path)
        result["brands"] = list(self.brands)
        result["excluded_brands"] = list(self.excluded_brands)
        return result


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    """Intent prediction plus the constraints needed to retrieve products."""

    intent: str
    confidence: float
    entities: CommandEntities = field(default_factory=CommandEntities)
    raw_text: str = ""
    normalized_text: str = ""

    def __post_init__(self) -> None:
        if self.intent not in SUPPORTED_INTENTS:
            raise ValueError(f"unsupported intent: {self.intent}")
        object.__setattr__(self, "confidence", min(1.0, max(0.0, float(self.confidence))))

    # Direct delegates make the result ergonomic while retaining the required nested
    # ``entities`` object used by the webhook.
    @property
    def category(self) -> str | None:
        return self.entities.category

    @property
    def brands(self) -> tuple[str, ...]:
        return self.entities.brands

    @property
    def excluded_brands(self) -> tuple[str, ...]:
        return self.entities.excluded_brands

    @property
    def min_price(self) -> float | None:
        return self.entities.min_price

    @property
    def max_price(self) -> float | None:
        return self.entities.max_price

    @property
    def in_stock(self) -> bool | None:
        return self.entities.in_stock

    @property
    def sort(self) -> str | None:
        return self.entities.sort

    @property
    def query(self) -> str:
        return self.entities.query

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "entities": self.entities.to_dict(),
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
        }


@dataclass(frozen=True, slots=True)
class _AliasMatch:
    canonical: str
    alias: str
    start: int
    fuzzy: bool = False


_AMOUNT_ATOM = r"(?:\d+(?:\.\d+)?|ศูนย์|หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ)"
_THAI_DIGIT_WORD = r"(?:หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)"
_THAI_COMPOUND_AMOUNT = (
    rf"(?:{_THAI_DIGIT_WORD}\s*พัน\s*{_THAI_DIGIT_WORD}\s*ร้อย|"
    rf"{_THAI_DIGIT_WORD}?\s*พัน\s*ห้า(?:\s*ร้อย)?|"
    rf"{_THAI_DIGIT_WORD}?\s*หมื่น\s*{_THAI_DIGIT_WORD}\s*พัน)"
)
_AMOUNT = rf"(?:{_THAI_COMPOUND_AMOUNT}|{_AMOUNT_ATOM}\s*(?:k|เค|พัน|หมื่น)?)"
_SPEC_UNIT = r"(?:นิ้ว|inch|inches|dpi|mah|hz|khz|mhz|ghz)"


def _amount_value(value: str) -> float | None:
    normal_value = re.sub(r"\s+", "", normalize_text(value))
    compound = re.fullmatch(
        r"(หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)?พันห้า(?:ร้อย)?",
        normal_value,
    )
    if compound:
        thousands = _THAI_NUMBER_WORDS.get(compound.group(1) or "หนึ่ง", 1)
        return thousands * 1_000 + 500
    compound = re.fullmatch(
        r"(หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)พัน"
        r"(หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)ร้อย",
        normal_value,
    )
    if compound:
        thousands = _THAI_NUMBER_WORDS[compound.group(1)]
        hundreds = _THAI_NUMBER_WORDS[compound.group(2)]
        return thousands * 1_000 + hundreds * 100
    compound = re.fullmatch(
        r"(หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)?หมื่น"
        r"(หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)พัน",
        normal_value,
    )
    if compound:
        ten_thousands = _THAI_NUMBER_WORDS.get(compound.group(1) or "หนึ่ง", 1)
        thousands = _THAI_NUMBER_WORDS[compound.group(2)]
        return ten_thousands * 10_000 + thousands * 1_000
    match = re.fullmatch(
        rf"\s*({_AMOUNT_ATOM})\s*(k|เค|พัน|หมื่น)?\s*",
        normalize_text(value),
    )
    if not match:
        return None
    atom, unit = match.groups()
    try:
        number = float(atom)
    except ValueError:
        number = _THAI_NUMBER_WORDS.get(atom, -1)
    if number < 0:
        return None
    multiplier = {None: 1, "k": 1_000, "เค": 1_000, "พัน": 1_000, "หมื่น": 10_000}[unit]
    return number * multiplier


def _unique_clean(values: Iterable[object]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value or "").split())
        # Spacing and punctuation can be semantically important to downstream word-
        # boundary matching.  For example, both ``smartwatch`` and ``smart watch``
        # must survive so the canonical Thai category can match Mercular's live
        # ``Smart Watch & Fitness Tracker`` label.
        key = normalize_text(cleaned)
        if cleaned and key and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return tuple(result)


@lru_cache(maxsize=256)
def category_aliases_for(category: str) -> tuple[str, ...]:
    """Return aliases for a canonical or catalogue-provided category name."""

    target = compact_text(category)
    for canonical, aliases in CATEGORY_ALIASES.items():
        candidates = (canonical, *aliases)
        compact_candidates = {compact_text(item) for item in candidates}
        if target in compact_candidates or any(
            candidate and candidate in target for candidate in compact_candidates
        ):
            # Keep a catalogue-specific label such as ``เมาส์เกมมิ่ง`` while also
            # expanding its underlying product concept (mouse).  The recommender's
            # directional matcher will not let the broad parent ``เกมมิ่ง`` satisfy it.
            return _unique_clean((category, *candidates))
    return _unique_clean((category,))


class ThaiCommandParser:
    """Parse conversational Thai/English Mercular messages in a few milliseconds.

    Args:
        brands: Current catalogue brands.  Built-in aliases are still available, while
            exact configured spelling is preferred in returned entities.
        categories: Current catalogue category labels.  These are added to the known
            consumer-electronics category aliases.
        fuzzy_threshold: Similarity used for a one-token typo fallback.  Explicit common
            misspellings are handled before fuzzy matching.
    """

    def __init__(
        self,
        brands: Iterable[str] | None = None,
        categories: Iterable[str] | None = None,
        *,
        fuzzy_threshold: float = 0.78,
    ) -> None:
        if not 0.5 <= fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be between 0.5 and 1.0")
        self.fuzzy_threshold = float(fuzzy_threshold)
        self._brand_aliases = self._build_aliases(brands, BRAND_ALIASES)
        self._category_aliases = self._build_aliases(categories, CATEGORY_ALIASES)

    @staticmethod
    def _build_aliases(
        configured: Iterable[str] | None,
        defaults: Mapping[str, Sequence[str]],
    ) -> tuple[tuple[str, str], ...]:
        alias_to_canonical: dict[str, str] = {}
        configured_values = _unique_clean(configured or ())
        # Catalogue spelling wins when it is equivalent to a built-in canonical name.
        for canonical in configured_values:
            alias_to_canonical[normalize_text(canonical)] = canonical
            alias_to_canonical[compact_text(canonical)] = canonical
        for canonical, aliases in defaults.items():
            configured_canonical = next(
                (
                    item
                    for item in configured_values
                    if compact_text(item) == compact_text(canonical)
                ),
                canonical,
            )
            for alias in (canonical, *aliases):
                normal = normalize_text(alias)
                if normal:
                    alias_to_canonical.setdefault(normal, configured_canonical)
        # Longer aliases win over nested terms ("gaming monitor" before "monitor").
        return tuple(
            sorted(
                ((alias, canonical) for alias, canonical in alias_to_canonical.items() if alias),
                key=lambda item: (-len(compact_text(item[0])), item[0]),
            )
        )

    @staticmethod
    def _exact_position(text: str, alias: str) -> int:
        if not alias:
            return -1
        # ASCII aliases need boundaries ("mi" must not match "gaming").  Thai aliases
        # intentionally allow compounds such as "อยากได้หูฟัง".
        if re.fullmatch(r"[a-z0-9 ._-]+", alias):
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            match = re.search(pattern, text)
            return match.start() if match else -1
        return text.find(alias)

    def _find_aliases(
        self,
        text: str,
        alias_table: Sequence[tuple[str, str]],
        *,
        multiple: bool,
    ) -> tuple[_AliasMatch, ...]:
        matches: list[_AliasMatch] = []
        seen: set[str] = set()
        for alias, canonical in alias_table:
            position = self._exact_position(text, alias)
            key = compact_text(canonical)
            if position >= 0 and key not in seen:
                seen.add(key)
                matches.append(_AliasMatch(canonical, alias, position))
                if not multiple:
                    break

        # An exact category/brand phrase is always stronger than a fuzzy match
        # from a dynamically discovered catalogue label (for example,
        # ``soundbar`` must not become ``Sound Card``).
        if matches and not multiple:
            return tuple(matches)

        # A conservative single-token fuzzy pass catches unseen adjacent-key typos.
        tokens = tuple(re.finditer(r"[a-z0-9ก-๙]+", text))
        for token_match in tokens:
            token = token_match.group(0)
            compact_token = compact_text(token)
            if len(compact_token) < 4 or compact_token.isdigit():
                continue
            best: tuple[float, str, str] | None = None
            for alias, canonical in alias_table:
                compact_alias = compact_text(alias)
                key = compact_text(canonical)
                if key in seen or len(compact_alias) < 4 or " " in alias:
                    continue
                if abs(len(compact_token) - len(compact_alias)) > 2:
                    continue
                if compact_token[0] != compact_alias[0]:
                    continue
                ratio = SequenceMatcher(None, compact_token, compact_alias).ratio()
                if ratio >= self.fuzzy_threshold and (best is None or ratio > best[0]):
                    best = (ratio, alias, canonical)
            if best is not None:
                _, alias, canonical = best
                seen.add(compact_text(canonical))
                matches.append(_AliasMatch(canonical, token, token_match.start(), True))
                if not multiple:
                    break

        matches.sort(key=lambda match: (match.start, match.fuzzy, match.canonical.casefold()))
        deduplicated: list[_AliasMatch] = []
        for match in matches:
            candidate = compact_text(match.canonical)
            if any(
                candidate == compact_text(existing.canonical)
                or (
                    match.fuzzy
                    and SequenceMatcher(
                        None, candidate, compact_text(existing.canonical)
                    ).ratio()
                    >= 0.84
                )
                for existing in deduplicated
            ):
                continue
            deduplicated.append(match)
        return tuple(deduplicated if multiple else deduplicated[:1])

    @staticmethod
    def _extract_prices(
        text: str,
    ) -> tuple[float | None, float | None, bool, bool, list[tuple[int, int]]]:
        minimum: float | None = None
        maximum: float | None = None
        minimum_inclusive = True
        maximum_inclusive = True
        spans: list[tuple[int, int]] = []

        between_patterns = (
            (rf"(?:ราคา|งบ)?\s*(?:ระหว่าง|ช่วง|ตั้งแต่|from)\s*({_AMOUNT})\s*(?:บาท)?\s*(?:ถึง|จนถึง|to|-)\s*({_AMOUNT})\s*(?:บาท)?", False),
            (rf"(?:ราคา|งบ)\s*({_AMOUNT})\s*(?:บาท)?\s*(?:ถึง|จนถึง|to|-)\s*({_AMOUNT})\s*(?:บาท)?", False),
            # A whitespace-delimited bare numeric range is a common compact budget
            # after a category/brand (``คีย์บอร์ด 1000-3500``).  The boundaries keep
            # model tokens out.  A contextual check below also rejects spec ranges.
            (rf"(?<!\S)({_AMOUNT})\s*-\s*({_AMOUNT})(?!\S)", True),
            (rf"({_AMOUNT})\s*(?:บาท)?\s*(?:-|ถึง|to)\s*({_AMOUNT})\s*บาท", False),
            (rf"({_AMOUNT_ATOM}\s*(?:k|เค|พัน|หมื่น))\s*-\s*({_AMOUNT_ATOM}\s*(?:k|เค|พัน|หมื่น))", False),
        )
        range_found = False
        for pattern, is_bare_range in between_patterns:
            for match in re.finditer(pattern, text):
                if is_bare_range:
                    before = text[: match.start()].rstrip()
                    after = text[match.end() :].lstrip()
                    spec_before = re.search(rf"(?:^|\s){_SPEC_UNIT}$", before)
                    spec_after = re.match(rf"{_SPEC_UNIT}(?:\s|$)", after)
                    if spec_before or spec_after:
                        continue
                first = _amount_value(match.group(1))
                second = _amount_value(match.group(2))
                if first is not None and second is not None:
                    minimum, maximum = sorted((first, second))
                    spans.append(match.span())
                    range_found = True
                    break
            if range_found:
                break

        max_patterns = (
            (rf"(?:ราคา|งบ)?\s*(?:<=|=<)\s*฿?\s*({_AMOUNT})\s*(?:บาท)?", True),
            (rf"(?:งบ|budget|ภายในงบ|ราคา|ไม่เกินงบ)\s*(?:ไม่เกิน|ประมาณ|ราว|ราวๆ|ที่|ไม่เกีน)?\s*฿?\s*({_AMOUNT})\s*(?:บาท)?\s*(?:หรือต่ำกว่า|ลงมา|หรือน้อยกว่า)?", True),
            (rf"(?:<|ต่ำกว่า|ถูกกว่า|under|below|less than)\s*(?:ราคา|งบ)?\s*฿?\s*({_AMOUNT})\s*(?:บาท)?", False),
            (rf"(?:ไม่เกิน|ไม่เกีน|up to|at most|maximum|max)\s*(?:ราคา|งบ)?\s*฿?\s*({_AMOUNT})\s*(?:บาท)?", True),
            (rf"({_AMOUNT})\s*(?:บาท)?\s*(?:หรือต่ำกว่า|ลงมา|หรือน้อยกว่า|maximum|max)", True),
        )
        min_patterns = (
            (rf"(?:ราคา|งบ)?\s*(?:>=|=>)\s*฿?\s*({_AMOUNT})\s*(?:บาท)?", True),
            (rf"(?:>|มากกว่า|แพงกว่า|over|above|more than)\s*(?:ราคา)?\s*฿?\s*({_AMOUNT})\s*(?:บาท)?", False),
            (rf"(?:อย่างน้อย|ขั้นต่ำ|ตั้งแต่|เริ่มต้น|at least|minimum|min)\s*(?:ราคา)?\s*฿?\s*({_AMOUNT})\s*(?:บาท)?\s*(?:ขึ้นไป)?", True),
            (rf"({_AMOUNT})\s*(?:บาท)?\s*(?:ขึ้นไป|หรือมากกว่า|เป็นต้นไป)", True),
        )

        def overlaps(span: tuple[int, int]) -> bool:
            return any(span[0] < old[1] and old[0] < span[1] for old in spans)

        # Upward postfixes (``5000 บาทขึ้นไป``) must claim their amount before the
        # generic ``ราคา 5000 บาท`` budget rule can interpret it as a maximum.
        for patterns, target in ((min_patterns, "min"), (max_patterns, "max")):
            for pattern, inclusive in patterns:
                for match in re.finditer(pattern, text):
                    if overlaps(match.span()):
                        continue
                    amount = _amount_value(match.group(1))
                    if amount is None:
                        continue
                    if target == "max":
                        maximum = amount
                        maximum_inclusive = inclusive
                    else:
                        minimum = amount
                        minimum_inclusive = inclusive
                    spans.append(match.span())

        # A bare currency amount is conventionally a shopping budget.  Numbers without
        # a price cue are intentionally ignored so product model numbers survive.
        if maximum is None:
            for match in re.finditer(rf"({_AMOUNT})\s*(?:บาท|฿)", text):
                if overlaps(match.span()):
                    continue
                maximum = _amount_value(match.group(1))
                spans.append(match.span())
                break
        return minimum, maximum, minimum_inclusive, maximum_inclusive, spans

    @staticmethod
    def _first_phrase(text: str, phrases: Sequence[str]) -> tuple[str | None, int]:
        found: list[tuple[int, int, str]] = []
        for phrase in phrases:
            normal = normalize_text(phrase)
            if not normal:
                continue
            if normal.isascii():
                # ``hi`` is not an intent inside ``this``/``shipping`` and ``help``
                # must not fire inside ``helpful``.
                match = re.search(
                    rf"(?<![a-z0-9]){re.escape(normal)}(?![a-z0-9])",
                    text,
                )
                position = match.start() if match else -1
            else:
                position = text.find(normal)
            if position >= 0:
                found.append((position, -len(normal), normal))
        if not found:
            return None, -1
        position, _, phrase = min(found)
        return phrase, position

    @staticmethod
    def _remove_spans(text: str, spans: Iterable[tuple[int, int]]) -> str:
        characters = list(text)
        for start, end in spans:
            for index in range(max(0, start), min(len(characters), end)):
                characters[index] = " "
        return "".join(characters)

    @staticmethod
    def _clean_residual(text: str) -> str:
        filler_phrases = (
            "ช่วยหา",
            "ช่วยแนะนำ",
            "ขอแนะนำ",
            "แนะนำ",
            "กำลังหา",
            "อยากได้",
            "อยากซื้อ",
            "ขอดู",
            "ค้นหา",
            "ขอรุ่น",
            "หา",
            "เอา",
            "เอาไว้",
            "เอาเฉพาะ",
            "เฉพาะ",
            "อย่างเดียว",
            "สำหรับ",
            "ไม่เอา",
            "ไม่รับ",
            "ไม่ต้องการ",
            "ยกเว้น",
            "not",
            "except",
            "without",
            "เท่านั้น",
            "สินค้า",
            "รุ่น",
            "หน่อย",
            "ให้หน่อย",
            "ทีครับ",
            "ทีค่ะ",
            "ครับ",
            "ค่ะ",
            "คะ",
            "จ้า",
            "ที",
            "please",
            "show me",
            "find me",
            "find",
            "search for",
            "search",
            "looking for",
            "recommend",
            "i want",
            "product",
            "ราคา",
            "budget",
            "มีสินค้าอะไรแนะนำบ้าง",
            "มีอะไรแนะนำบ้าง",
            "มีอะไรแนะนำ",
            "มีอะไรน่าสนใจ",
            "มีอะไรบ้าง",
            "แนะนำอะไรหน่อย",
            "แนะนำหน่อย",
            "สินค้าอะไรแนะนำบ้าง",
            "ตัวไหนบ้าง",
            "มีไหม",
            "หรือเปล่า",
        )
        result = f" {text} "
        prefix_fillers = {
            normalize_text(phrase)
            for phrase in (
                "ช่วยหา",
                "ช่วยแนะนำ",
                "ขอแนะนำ",
                "แนะนำ",
                "กำลังหา",
                "อยากได้",
                "อยากซื้อ",
                "ขอดู",
                "ค้นหา",
                "ขอรุ่น",
                "หา",
                "เอาไว้",
                "show me",
                "find me",
                "find",
                "search for",
                "search",
                "looking for",
                "recommend",
                "i want",
                "สำหรับ",
            )
        }
        suffix_fillers = {
            normalize_text(phrase)
            for phrase in (
                "หน่อย",
                "ให้หน่อย",
                "ทีครับ",
                "ทีค่ะ",
                "ครับ",
                "ค่ะ",
                "คะ",
                "จ้า",
                "ที",
                "นะ",
                "ด้วย",
                "please",
            )
        }
        left_boundary_fillers = {normalize_text("เอาไว้")}
        for phrase in sorted(filler_phrases, key=lambda item: len(normalize_text(item)), reverse=True):
            normal = normalize_text(phrase)
            stripped = result.strip()
            if normal in prefix_fillers and stripped.startswith(normal):
                offset = result.find(normal)
                result = result[:offset] + " " * len(normal) + result[offset + len(normal) :]
                continue
            if normal in left_boundary_fillers:
                result = re.sub(
                    rf"(?<![a-z0-9ก-๙]){re.escape(normal)}",
                    " ",
                    result,
                )
                continue
            if normal in suffix_fillers and stripped.endswith(normal):
                offset = result.rfind(normal)
                result = result[:offset] + " " * len(normal) + result[offset + len(normal) :]
                continue
            word_class = "a-z0-9" if normal.isascii() else "a-z0-9ก-๙"
            result = re.sub(
                rf"(?<![{word_class}]){re.escape(normal)}(?![{word_class}])",
                " ",
                result,
            )
        # Politeness can be stacked (``หน่อยครับ``).  Strip from the outside in;
        # every pass shortens the text, so this bounded vocabulary cannot loop forever.
        while True:
            stripped = result.strip()
            matched_suffix = next(
                (
                    phrase
                    for phrase in sorted(suffix_fillers, key=len, reverse=True)
                    if stripped.endswith(phrase)
                ),
                None,
            )
            if matched_suffix is None:
                break
            offset = result.rfind(matched_suffix)
            result = (
                result[:offset]
                + " " * len(matched_suffix)
                + result[offset + len(matched_suffix) :]
            )
        result = re.sub(
            r"\b(?:and|or|with|that|the|a|an|for|me|want|need|please)\b",
            " ",
            result,
        )
        result = re.sub(r"(?:^|\s)(?:และ|หรือ|แต่|ที่|แบบ|มี|ของ)(?=\s|$)", " ", result)
        result = re.sub(
            r"(?:^|\s)(?:หา|ขอ|แนะนำ|ค้นหา|เรียง|บ้าง|ไหม|นะ|ด้วย|ตัว)(?=\s|$)",
            " ",
            result,
        )
        result = " ".join(result.split()).strip(" .-")
        return result

    def parse(self, message: object | None) -> ParsedCommand:
        """Parse one message.  Empty and unsupported input safely returns ``unknown``."""

        raw_text = " ".join(str(message or "").replace("\xa0", " ").split())
        text = normalize_text(raw_text)
        if not text:
            return ParsedCommand(INTENT_UNKNOWN, 0.0, raw_text=raw_text, normalized_text=text)

        brand_matches = self._find_aliases(text, self._brand_aliases, multiple=True)
        category_matches = self._find_aliases(text, self._category_aliases, multiple=False)
        subtype_phrase, subtype_position = self._first_phrase(
            text,
            tuple(_SUBTYPE_CATEGORY_HINTS),
        )
        subtype_category = (
            _SUBTYPE_CATEGORY_HINTS[subtype_phrase]
            if subtype_phrase is not None
            else None
        )
        subtype_span = (
            (subtype_position, subtype_position + len(subtype_phrase))
            if subtype_phrase is not None
            else None
        )
        if subtype_phrase in _WATCH_STRAP_SUBTYPES:
            # In a strap query, Apple names the compatible watch ecosystem rather
            # than the strap manufacturer.  Keep it in the residual text and avoid a
            # false hard brand constraint; ordinary Apple searches are unaffected.
            brand_matches = tuple(
                match
                for match in brand_matches
                if compact_text(match.canonical) != "apple"
            )
        (
            minimum,
            maximum,
            minimum_inclusive,
            maximum_inclusive,
            price_spans,
        ) = self._extract_prices(re.sub(r"(?<=\d)o(?=\b)", "0", text))

        all_brand_matches = brand_matches
        excluded_brand_matches = tuple(
            match
            for match in all_brand_matches
            if re.search(
                r"(?:ไม่เอา|ไม่รับ|ไม่ต้องการ|ยกเว้น|not|except|without)\s*$",
                text[max(0, match.start - 24) : match.start],
            )
            is not None
        )
        excluded_keys = {
            (match.start, compact_text(match.canonical)) for match in excluded_brand_matches
        }
        brand_matches = tuple(
            match
            for match in all_brand_matches
            if (match.start, compact_text(match.canonical)) not in excluded_keys
        )

        availability_true = (
            "ไม่เอาของหมด",
            "ไม่เอาสินค้าหมด",
            "ไม่เอาหมดสต็อก",
            "เอาเฉพาะของไม่หมด",
            "ของไม่หมด",
            "not out of stock",
            "exclude sold out",
            "ของพร้อมส่ง",
            "พร้อมส่ง",
            "พร้อมสง",
            "มีของ",
            "มีสต็อก",
            "ของพร้อม",
            "ส่งได้เลย",
            "in stock",
            "available now",
            "ready to ship",
        )
        availability_false = (
            "ของหมด",
            "หมดสต็อก",
            "สินค้าหมด",
            "ไม่พร้อมส่ง",
            "ยังไม่พร้อมส่ง",
            "out of stock",
            "sold out",
            "unavailable",
        )
        false_phrase, false_position = self._first_phrase(text, availability_false)
        true_phrase, true_position = self._first_phrase(text, availability_true)
        in_stock: bool | None = None
        availability_match: tuple[str, int] | None = None
        if false_phrase is not None and (true_phrase is None or false_position <= true_position):
            in_stock = False
            availability_match = (false_phrase, false_position)
        elif true_phrase is not None:
            in_stock = True
            availability_match = (true_phrase, true_position)

        sort_groups: tuple[tuple[str, tuple[str, ...]], ...] = (
            (SORT_PRICE_ASC, ("ตัวถูกๆ", "ถูกๆ", "เรียงถูกสุด", "ถูกสุด", "ถูกที่สุด", "ราคาต่ำสุด", "เรียงราคาถูก", "price low to high", "cheapest")),
            (SORT_PRICE_DESC, ("แพงสุด", "แพงที่สุด", "ราคาสูงสุด", "เรียงราคาแพง", "price high to low", "most expensive")),
            (SORT_NEWEST, ("ใหม่สุด", "ใหม่ล่าสุด", "รุ่นใหม่", "สินค้ามาใหม่", "newest", "latest")),
            (SORT_POPULAR, ("ยอดนิยม", "ขายดี", "ฮิต", "นิยมสุด", "popular", "best seller", "bestseller")),
            (SORT_DISCOUNT, ("ลดเยอะสุด", "ลดแรงสุด", "ส่วนลดสูงสุด", "คุ้มสุด", "ลดราคามากสุด", "biggest discount", "best discount")),
        )
        sort_value: str | None = None
        sort_match: tuple[str, int] | None = None
        found_sorts: list[tuple[int, str, str]] = []
        for value, phrases in sort_groups:
            phrase, position = self._first_phrase(text, phrases)
            if phrase is not None:
                found_sorts.append((position, value, phrase))
        if found_sorts:
            position, sort_value, phrase = min(found_sorts)
            sort_match = (phrase, position)

        brand_label_spans: list[tuple[int, int]] = []
        for match in all_brand_matches:
            before = text[: match.start]
            prefix = re.search(
                r"(?:แบรนด์|(?<![a-z0-9])brand)\s*$",
                before,
            )
            if prefix:
                brand_label_spans.append(prefix.span())
            match_end = match.start + len(match.alias)
            suffix = re.match(
                r"\s*(?:แบรนด์|brand)(?![a-z0-9])",
                text[match_end:],
            )
            if suffix:
                brand_label_spans.append(
                    (match_end + suffix.start(), match_end + suffix.end())
                )

        removal_spans = [*price_spans, *brand_label_spans]
        for match in all_brand_matches:
            removal_spans.append((match.start, match.start + len(match.alias)))
        for match in category_matches:
            match_span = (match.start, match.start + len(match.alias))
            if subtype_span and (
                match_span[0] < subtype_span[1] and subtype_span[0] < match_span[1]
            ):
                # The broad concept remains a hard category, while the narrower words
                # (``earbuds``, ``ขาไมค์``) stay available for subtype filtering.
                continue
            removal_spans.append(match_span)
        if availability_match:
            phrase, position = availability_match
            removal_spans.append((position, position + len(phrase)))
        if sort_match:
            phrase, position = sort_match
            removal_spans.append((position, position + len(phrase)))
        residual = self._clean_residual(self._remove_spans(text, removal_spans))
        replacement = (
            _SUBTYPE_QUERY_REPLACEMENTS.get(subtype_phrase)
            if subtype_phrase is not None
            else None
        )
        if replacement:
            if subtype_phrase in residual:
                residual = residual.replace(subtype_phrase, replacement, 1)
            elif replacement not in residual:
                residual = f"{replacement} {residual}".strip()
            residual = " ".join(residual.split())

        category = (
            category_matches[0].canonical
            if category_matches
            else subtype_category
        )
        brands = tuple(match.canonical for match in brand_matches)
        excluded_brands = tuple(match.canonical for match in excluded_brand_matches)
        entities = CommandEntities(
            category=category,
            brands=brands,
            excluded_brands=excluded_brands,
            min_price=minimum,
            max_price=maximum,
            min_price_inclusive=minimum_inclusive,
            max_price_inclusive=maximum_inclusive,
            in_stock=in_stock,
            sort=sort_value,
            query=residual,
        )

        strong_order_phrases = (
            "อยากสั่งซื้อ",
            "ต้องการสั่ง",
            "สั่งซื้อ",
            "สั่งของ",
            "ออเดอร์",
            "วิธีสั่งซื้อ",
            "สั่งซื้อยังไง",
            "สั่งอย่างไร",
            "ชำระเงิน",
            "สถานะคำสั่งซื้อ",
            "ติดตามออเดอร์",
            "เลขออเดอร์",
            "order status",
            "how to order",
            "place an order",
            "place order",
            "checkout",
            "payment",
        )
        contact_phrases = (
            "ติดต่อ",
            "ติดต่อแอดมิน",
            "ติดต่อร้าน",
            "คุยกับแอดมิน",
            "คุยกับเจ้าหน้าที่",
            "เบอร์โทร",
            "ช่องทางติดต่อ",
            "ฝ่ายบริการลูกค้า",
            "customer service",
            "contact staff",
            "contact us",
        )
        refresh_phrases = (
            "อื่นๆ",
            "ดูอีก",
            "สุ่มใหม่",
            "ขอชุดใหม่",
            "ดูชุดใหม่",
            "ขออันอื่น",
            "ขอดูอันอืน",
            "ดูอย่างอื่น",
            "มีตัวอื่นไหม",
            "เอาใหม่",
            "สุ่มไหม่",
            "refresh",
            "show another",
            "show me another",
            "show more",
            "more options",
        )
        help_phrases = (
            "ช่วยเหลือ",
            "ช้วยเหลือ",
            "ช่วยอะไรได้บ้าง",
            "ทำอะไรได้บ้าง",
            "ใช้งานยังไง",
            "ใช้งานยงไง",
            "วิธีใช้งาน",
            "คำสั่งมีอะไรบ้าง",
            "ขอความช่วยเหลือ",
            "help",
            "how to use",
            "what can you do",
        )
        promotion_phrases = (
            "โปรโมชัน",
            "โปรโมชั่น",
            "โปรโมชั้น",
            "โปรมชัน",
            "โปรโมช่น",
            "โปรล่าสุด",
            "โปรอะไร",
            "มีโปรอะไร",
            "มีคูปองอะไร",
            "มีคูปองลดไหม",
            "คูปองล่าสุด",
            "promotion",
            "promotions",
            "latest deals",
        )
        thanks_phrases = (
            "ขอบคุณ",
            "ขอบคุน",
            "ขอบใจ",
            "ขอบพระคุณ",
            "thanks",
            "thank you",
            "thx",
        )
        greeting_phrases = (
            "สวัสดี",
            "สวัดดี",
            "หวัดดี",
            "ดีจ้า",
            "ฮัลโหล",
            "hello",
            "hi",
            "hey",
            "good morning",
            "good evening",
        )
        search_phrases = (
            "ช่วยหา",
            "กำลังหา",
            "ค้นหา",
            "อยากได้",
            "อยากซื้อ",
            "ขอดู",
            "แนะนำสินค้า",
            "มีอะไรขาย",
            "มีสินค้าอะไรแนะนำบ้าง",
            "มีอะไรแนะนำบ้าง",
            "มีอะไรแนะนำ",
            "มีอะไรน่าสนใจ",
            "แนะนำอะไรหน่อย",
            "แนะนำหน่อย",
            "สินค้าอะไรแนะนำบ้าง",
            "มีรุ่นไหน",
            "หาให้หน่อย",
            "looking for",
            "find me",
            "show me",
            "recommend",
            "i want",
            "i need",
        )
        product_query_hints = (
            "airpods",
            "soundcore",
            "micro sd card",
            "micro sd",
            "sd card",
            "keycap",
            "keyboard switch",
            "wrist rest",
            "คีย์แคป",
            "สวิตช์คีย์บอร์ด",
            "ที่รองข้อมือ",
            "router",
            "ssd",
        )

        has_entities = any(
            (
                category,
                brands,
                excluded_brands,
                minimum is not None,
                maximum is not None,
                in_stock is not None,
                sort_value,
            )
        )
        has_search_phrase = self._first_phrase(text, search_phrases)[0] is not None
        # Keep this exact: a substring match would turn “คำสั่งมีอะไรบ้าง” from
        # the help intent into a product search.
        has_general_discovery = text == normalize_text("มีอะไรบ้าง")
        has_search_prefix = bool(
            re.match(r"^(?:หา|find|search(?:\s+for)?)\s+\S", text)
        )
        has_product_hint = self._first_phrase(text, product_query_hints)[0] is not None
        has_model_token = bool(
            re.search(
                r"\b(?=[a-z0-9-]{3,}\b)(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9-]+\b",
                text,
            )
        )
        fuzzy_used = any(match.fuzzy for match in (*brand_matches, *category_matches))

        # Explicit transactional/contact/refresh commands win; ordinary social words
        # still yield to product signals in messages such as "สวัสดี ขอหูฟัง".
        order_detected = self._first_phrase(text, strong_order_phrases)[0] is not None
        contact_detected = self._first_phrase(text, contact_phrases)[0] is not None
        refresh_phrase = self._first_phrase(text, refresh_phrases)[0]
        refresh_detected = refresh_phrase is not None and (
            refresh_phrase != "refresh"
            or re.fullmatch(r"refresh(?:\s+please)?", text) is not None
        )
        help_detected = self._first_phrase(text, help_phrases)[0] is not None
        promotion_detected = self._first_phrase(text, promotion_phrases)[0] is not None
        if order_detected:
            intent, confidence = INTENT_ORDER, 0.97
        elif contact_detected:
            intent, confidence = INTENT_CONTACT, 0.97
        elif refresh_detected:
            intent, confidence = INTENT_REFRESH, 0.96
        elif promotion_detected:
            intent, confidence = INTENT_PROMOTION, 0.97
        elif (
            has_entities
            or has_search_phrase
            or has_general_discovery
            or has_search_prefix
            or has_product_hint
            or has_model_token
        ):
            intent = INTENT_SEARCH
            evidence = sum(
                (
                    bool(category),
                    bool(brands),
                    bool(excluded_brands),
                    minimum is not None or maximum is not None,
                    in_stock is not None,
                    sort_value is not None,
                    has_search_phrase
                    or has_general_discovery
                    or has_search_prefix
                    or has_product_hint,
                    bool(residual),
                )
            )
            confidence = min(0.99, 0.72 + evidence * 0.045 - (0.05 if fuzzy_used else 0.0))
        elif help_detected:
            intent, confidence = INTENT_HELP, 0.96
        elif self._first_phrase(text, thanks_phrases)[0] is not None:
            intent, confidence = INTENT_THANKS, 0.96
        elif self._first_phrase(text, greeting_phrases)[0] is not None:
            intent, confidence = INTENT_GREETING, 0.95
        else:
            intent, confidence = INTENT_UNKNOWN, 0.20

        # Conversational intents do not carry their utterance as a catalogue query.
        if intent != INTENT_SEARCH:
            entities = CommandEntities()
        return ParsedCommand(intent, confidence, entities, raw_text, text)


_DEFAULT_PARSER = ThaiCommandParser()


def parse_command(
    message: object | None,
    *,
    brands: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
) -> ParsedCommand:
    """Functional parser API; supply catalogue values for one-off dynamic matching."""

    if brands is None and categories is None:
        return _DEFAULT_PARSER.parse(message)
    return ThaiCommandParser(brands=brands, categories=categories).parse(message)


__all__ = [
    "BRAND_ALIASES",
    "CATEGORY_ALIASES",
    "FEATURE_ALIASES",
    "CommandEntities",
    "ParsedCommand",
    "ThaiCommandParser",
    "SUPPORTED_INTENTS",
    "INTENT_SEARCH",
    "INTENT_GREETING",
    "INTENT_HELP",
    "INTENT_THANKS",
    "INTENT_CONTACT",
    "INTENT_ORDER",
    "INTENT_REFRESH",
    "INTENT_PROMOTION",
    "INTENT_UNKNOWN",
    "SORT_PRICE_ASC",
    "SORT_PRICE_DESC",
    "SORT_NEWEST",
    "SORT_POPULAR",
    "SORT_DISCOUNT",
    "normalize_text",
    "compact_text",
    "category_aliases_for",
    "parse_command",
]
