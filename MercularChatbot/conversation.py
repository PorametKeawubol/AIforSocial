"""Deterministic comparison and contextual product-question helpers."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Iterable, Sequence

try:
    from .models import Product, clean_text
    from .nlp import compact_text, normalize_text
except ImportError:  # pragma: no cover
    from models import Product, clean_text
    from nlp import compact_text, normalize_text


_COMPARE_SUFFIX_RE = re.compile(
    r"\s*(?:(?:ต่างกัน|เทียบกัน)(?:ยังไง|อย่างไร|ไง)?|อันไหนดีกว่า(?:กัน)?|compare)?"
    r"\s*(?:หน่อย|ที|ครับ|ค่ะ|คะ)?\s*$"
)
_QUESTION_REFERENCE_RE = re.compile(
    r"(?:ตัวนี้|ตัวนี|รุ่นนี้|รุ่นนี|อันนี้|อันนี|สินค้านี้|สินค้านี)"
)
_QUESTION_CUE_RE = re.compile(
    r"(?:ได้ไหม|ได้มั้ย|หรือเปล่า|รองรับ|ใช้.*ได้|หนัก|น้ำหนัก|กี่กรัม|สเปก|สเปค|spec|"
    r"bluetooth|บลูทู|แบต|ชาร์จ|ประกัน|เชื่อมต่อ|พอร์ต|ขนาด|กี่นิ้ว|รีเฟรชเรต|"
    r"ความละเอียด|hz|กี่ปี|กี่ชั่วโมง|ราคา|เท่าไหร่|มีของ|สต็อก|ของหมด|รีวิว|"
    r"กี่ดาว|สีอะไร)"
)
_CHEAPER_RE = re.compile(r"(?:ขอ|เอา|มี)?\s*(?:ตัว)?ถูกกว่า(?:นี้|นี)")
_ALTERNATIVE_RE = re.compile(
    r"(?:คล้าย|ค้าย|คลาย|ใกล้เคียง|ทดแทน).*(?:ถูกกว่า|ประหยัดกว่า)"
)
_NAME_NOISE = {
    "mouse",
    "เมาส์",
    "keyboard",
    "คีย์บอร์ด",
    "headset",
    "headphone",
    "หูฟัง",
    "wireless",
    "bluetooth",
    "black",
    "white",
    "กับ",
    "the",
}


def comparison_queries(message: object | None) -> tuple[str, str] | None:
    """Return two named sides only for an explicit comparison sentence."""

    text = normalize_text(message)
    if not text or not re.search(r"(?:ต่างกัน|เทียบ|เปรียบเทียบ|อันไหนดีกว่า|compare)", text):
        return None
    text = re.sub(r"^(?:ช่วย)?(?:เปรียบเทียบ|เทียบ|compare)\s*", "", text)
    parts = re.split(r"\s+(?:เทียบกับ|กับ|กะ|vs\.?|versus)\s+", text, maxsplit=1)
    if len(parts) != 2:
        return None
    left = clean_text(parts[0], limit=250)
    right = clean_text(_COMPARE_SUFFIX_RE.sub("", parts[1]), limit=250)
    return (left, right) if left and right else None


def _name_tokens(value: object) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-z0-9ก-๙]+", normalize_text(value))
        if len(compact_text(token)) >= 2 and token not in _NAME_NOISE
    )


def find_named_product(products: Iterable[Product], query: object) -> Product | None:
    """Conservatively match a model phrase; ambiguous weak matches return ``None``."""

    query_text = normalize_text(query)
    query_compact = compact_text(query_text)
    tokens = _name_tokens(query_text)
    if not query_compact or not tokens:
        return None
    scored: list[tuple[float, int, str, Product]] = []
    for product in products:
        name = normalize_text(product.name)
        name_compact = compact_text(name)
        name_tokens = _name_tokens(name)
        if query_compact in name_compact:
            score = 2.0 + len(query_compact) / max(1, len(name_compact))
            scored.append((score, -len(name_compact), product.id, product))
            continue
        matched = 0
        long_matched = 0
        for token in tokens:
            best = max(
                (SequenceMatcher(None, token, candidate).ratio() for candidate in name_tokens),
                default=0.0,
            )
            if best >= 0.88:
                matched += 1
                long_matched += int(len(token) >= 5)
        coverage = matched / len(tokens)
        if coverage >= 0.75 and long_matched:
            scored.append((coverage, -len(name_compact), product.id, product))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    if len(scored) > 1 and scored[0][:2] == scored[1][:2]:
        # Colour variants of one model are acceptable; unrelated ties are not.
        first_core = re.sub(r"(?:black|white|สีดำ|สีขาว)", "", normalize_text(scored[0][3].name))
        second_core = re.sub(r"(?:black|white|สีดำ|สีขาว)", "", normalize_text(scored[1][3].name))
        if compact_text(first_core) != compact_text(second_core):
            return None
    return scored[0][3]


def is_product_question(message: object | None) -> bool:
    text = normalize_text(message)
    return bool(_QUESTION_REFERENCE_RE.search(text) and _QUESTION_CUE_RE.search(text))


def product_question_answer(product: Product, message: object | None) -> str:
    """Answer Bluetooth/weight/spec questions only from stored product evidence."""

    text = normalize_text(message)
    if "ราคา" in text or "เท่าไหร่" in text:
        if product.price is None:
            return (
                f"ยังยืนยันราคาของ {product.name} จากข้อมูลที่มีตอนนี้ไม่ได้ครับ "
                f"ตรวจสอบหน้าสินค้าได้ที่ {product.product_url}"
            )
        original = (
            f" (ราคาปกติ ฿{product.original_price:,.0f})"
            if product.original_price is not None and product.original_price > product.price
            else ""
        )
        return (
            f"ราคาจากข้อมูลที่มีสำหรับ {product.name} คือ ฿{product.price:,.0f}{original} ครับ\n"
            f"ตรวจสอบราคาล่าสุดอีกครั้ง: {product.product_url}"
        )
    if any(term in text for term in ("มีของ", "สต็อก", "ของหมด")):
        stock = (
            "มีสินค้า"
            if product.in_stock is True
            else "สินค้าหมด"
            if product.in_stock is False
            else "ยังยืนยันสต็อกไม่ได้"
        )
        return (
            f"สถานะจากข้อมูลที่มีสำหรับ {product.name}: {stock} ครับ\n"
            f"ตรวจสอบสต็อกล่าสุดอีกครั้ง: {product.product_url}"
        )
    if any(term in text for term in ("รีวิว", "กี่ดาว", "rating")):
        if product.rating is None and product.review_count is None:
            return (
                f"ยังไม่มีคะแนนรีวิวของ {product.name} ในข้อมูลที่มีครับ "
                f"ตรวจสอบหน้าสินค้าได้ที่ {product.product_url}"
            )
        rating = f"{product.rating:g}/5" if product.rating is not None else "ไม่ระบุคะแนน"
        reviews = (
            f" จาก {product.review_count:,} รีวิว"
            if product.review_count is not None
            else ""
        )
        return f"{product.name} ได้คะแนน {rating}{reviews} ครับ\n{product.product_url}"

    general_specs = False
    if "bluetooth" in text or "บลูทู" in text:
        terms = ("bluetooth", "บลูทูธ", "บลูทูท", "บลูทูด", "บลูทุธ")
        label = "Bluetooth"
    elif any(term in text for term in ("หนัก", "น้ำหนัก", "กี่กรัม", "weight")):
        terms = ("น้ำหนัก", "weight", "กรัม", "gram", "kg", "กิโลกรัม")
        label = "น้ำหนัก"
    elif "ประกัน" in text or "warranty" in text:
        terms = ("ประกัน", "warranty")
        label = "การรับประกัน"
    elif any(term in text for term in ("แบต", "ชาร์จ", "battery")):
        terms = ("แบตเตอรี่", "แบต", "battery", "ชาร์จ", "mah", "ชั่วโมง")
        label = "แบตเตอรี่"
    elif any(term in text for term in ("เชื่อมต่อ", "พอร์ต", "connection", "port")):
        terms = (
            "การเชื่อมต่อ",
            "เชื่อมต่อ",
            "connection",
            "พอร์ต",
            "port",
            "usb",
            "hdmi",
        )
        label = "การเชื่อมต่อ"
    elif any(term in text for term in ("ขนาด", "กี่นิ้ว", "dimension", "size")):
        terms = ("ขนาด", "dimension", "size", "นิ้ว", "inch")
        label = "ขนาด"
    elif any(term in text for term in ("รีเฟรชเรต", "ความละเอียด", "hz", "resolution")):
        terms = ("รีเฟรชเรต", "refresh rate", "hz", "ความละเอียด", "resolution")
        label = "หน้าจอ"
    elif any(term in text for term in ("สเปก", "สเปค", "spec")):
        terms = ()
        label = "สเปก"
        general_specs = True
    elif "สีอะไร" in text:
        terms = ("สี", "color", "colour", "black", "white")
        label = "สี"
    else:
        terms = tuple(token for token in _name_tokens(text) if len(token) >= 3)
        label = "สเปกที่ถาม"

    evidence: list[str] = []
    for name, value in product.specifications:
        haystack = normalize_text(f"{name} {value}")
        if general_specs or any(normalize_text(term) in haystack for term in terms):
            evidence.append(f"• {clean_text(name, limit=120)}: {clean_text(value, limit=300)}")
    for item in (*product.highlights, product.overview):
        haystack = normalize_text(item)
        if item and (
            general_specs or any(normalize_text(term) in haystack for term in terms)
        ):
            evidence.append(f"• {clean_text(item, limit=350)}")
    if label == "การรับประกัน" and product.warranty:
        evidence.append(f"• การรับประกัน: {clean_text(product.warranty, limit=350)}")
    for item in product.service_notes:
        haystack = normalize_text(item)
        if item and any(normalize_text(term) in haystack for term in terms):
            evidence.append(f"• {clean_text(item, limit=350)}")
    evidence = list(dict.fromkeys(evidence))[:6]
    if not evidence:
        return (
            f"ยังยืนยันข้อมูล{label}ของ {product.name} จากข้อมูลที่มีตอนนี้ไม่ได้ครับ "
            f"ตรวจสอบหน้าสินค้าก่อนตัดสินใจได้ที่ {product.product_url}"
        )
    return (
        f"ข้อมูล {label} ของ {product.name} จากหน้าสินค้าล่าสุดครับ\n"
        + "\n".join(evidence)
        + f"\n\nตรวจสอบข้อมูลล่าสุดอีกครั้ง: {product.product_url}"
    )


def is_cheaper_refinement(message: object | None) -> bool:
    return bool(_CHEAPER_RE.search(normalize_text(message)))


def is_alternative_request(message: object | None) -> bool:
    return bool(_ALTERNATIVE_RE.search(normalize_text(message)))


def cheaper_than(products: Sequence[Product]) -> float | None:
    prices = [product.price for product in products if product.price is not None]
    return min(prices) if prices else None


__all__ = [
    "cheaper_than",
    "comparison_queries",
    "find_named_product",
    "is_alternative_request",
    "is_cheaper_refinement",
    "is_product_question",
    "product_question_answer",
]
