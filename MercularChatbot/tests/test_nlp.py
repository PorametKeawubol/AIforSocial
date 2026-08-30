from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from evaluate_nlp import evaluate_cases, load_cases
from nlp import (
    INTENT_CONTACT,
    INTENT_GREETING,
    INTENT_HELP,
    INTENT_ORDER,
    INTENT_PROMOTION,
    INTENT_REFRESH,
    INTENT_SEARCH,
    INTENT_THANKS,
    INTENT_UNKNOWN,
    SORT_PRICE_ASC,
    ThaiCommandParser,
    category_aliases_for,
    normalize_text,
    parse_command,
)


@pytest.fixture(scope="module")
def parser() -> ThaiCommandParser:
    return ThaiCommandParser()


def test_normalization_preserves_prices_and_model_numbers() -> None:
    assert normalize_text("  WH-1000XM5 ราคา ๓,๕๐๐ บาท  ") == "wh-1000xm5 ราคา 3500 บาท"
    assert normalize_text("ราคา ≥ ๓๐๐๐") == "ราคา >= 3000"
    assert normalize_text("ดีมากกกก") == "ดีมาก"


@pytest.mark.parametrize(
    ("text", "intent"),
    (
        ("สวัสดีครับ", INTENT_GREETING),
        ("ช่วยอะไรได้บ้าง", INTENT_HELP),
        ("ขอบคุณมาก", INTENT_THANKS),
        ("ขอคุยกับแอดมิน", INTENT_CONTACT),
        ("วิธีสั่งซื้อทำยังไง", INTENT_ORDER),
        ("อยากสั่งซื้อสินค้า", INTENT_ORDER),
        ("สุ่มใหม่", INTENT_REFRESH),
        ("อื่นๆ", INTENT_REFRESH),
        ("ช่วยเหลือ", INTENT_HELP),
        ("ติดต่อ", INTENT_CONTACT),
        ("โอเค", INTENT_UNKNOWN),
        ("", INTENT_UNKNOWN),
    ),
)
def test_conversational_and_fallback_intents(
    parser: ThaiCommandParser,
    text: str,
    intent: str,
) -> None:
    parsed = parser.parse(text)
    assert parsed.intent == intent
    assert 0 <= parsed.confidence <= 1
    assert parsed.entities.query == ""


def test_thai_multi_condition_command_extracts_every_entity(parser: ThaiCommandParser) -> None:
    parsed = parser.parse("หูฟัง Sony หรือ JBL ไร้สาย งบ ๕k พร้อมส่ง ถูกสุด")

    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == "หูฟัง"
    assert parsed.brands == ("Sony", "JBL")
    assert parsed.min_price is None
    assert parsed.max_price == 5_000
    assert parsed.in_stock is True
    assert parsed.sort == SORT_PRICE_ASC
    assert parsed.query == "ไร้สาย"


def test_english_multi_condition_command(parser: ThaiCommandParser) -> None:
    parsed = parser.parse("mechanical keyboard Keychron budget 3.5k available now")

    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == "คีย์บอร์ด"
    assert parsed.brands == ("Keychron",)
    assert parsed.max_price == 3_500
    assert parsed.in_stock is True
    assert parsed.query == "mechanical"


@pytest.mark.parametrize(
    ("text", "category", "brand", "maximum"),
    (
        ("คีบอด logitec ไม่เกิน ๒พัน", "คีย์บอร์ด", "Logitech", 2_000),
        ("เม้าส์ razer งบ 1.5k", "เมาส์", "Razer", 1_500),
        ("hedphone senheiser ราคา ๙๐๐๐", "หูฟัง", "Sennheiser", 9_000),
        ("ปริ้นเตอร์ราคาไม่เกีน 3k", "เครื่องพิมพ์", None, 3_000),
    ),
)
def test_common_typos_and_fuzzy_aliases(
    parser: ThaiCommandParser,
    text: str,
    category: str,
    brand: str | None,
    maximum: float,
) -> None:
    parsed = parser.parse(text)
    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == category
    assert parsed.entities.brand == brand
    assert parsed.max_price == maximum


@pytest.mark.parametrize(
    "text",
    (
        "มือถือ",
        "หามือถือ",
        "สมาร์ทโฟน",
        "โทรศัพท์",
        "smartphone",
        "มือถึอ",
    ),
)
def test_phone_aliases_are_product_searches(parser: ThaiCommandParser, text: str) -> None:
    parsed = parser.parse(text)

    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == "โทรศัพท์"
    assert parsed.query == ""


@pytest.mark.parametrize(
    "text",
    ("คอม", "หาคอม", "คอมพิวเตอร์", "คอมตั้งโต๊ะ", "desktop", "pc"),
)
def test_computer_aliases_are_product_searches(parser: ThaiCommandParser, text: str) -> None:
    parsed = parser.parse(text)

    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == "คอมพิวเตอร์"
    assert parsed.query == ""


def test_requirement_typos_use_cases_and_promotions_are_structured_safely(
    parser: ThaiCommandParser,
) -> None:
    typo = parser.parse("หาเม้า logitehc ไม่เกิน 200O")
    fps = parser.parse("แนะนำหูฟังเล่น FPS งบ 3,000")
    valorant = parser.parse("อยากได้เมาส์สำหรับเล่น Valorant เน้นเบา ๆ")
    promotion = parser.parse("มีโปรโมชั้นอะไรบ้าง")

    assert typo.category == "เมาส์"
    assert typo.brands == ("Logitech",)
    assert typo.max_price == 2_000
    assert typo.query == ""
    assert fps.query == "เล่น fps"
    assert valorant.query == "เล่น valorant เน้นเบา ๆ"
    assert promotion.intent == INTENT_PROMOTION


@pytest.mark.parametrize(
    ("text", "minimum", "maximum"),
    (
        ("เมาส์ระหว่าง 1k ถึง 3k", 1_000, 3_000),
        ("เมาส์ราคา 1000 ถึง 3500", 1_000, 3_500),
        ("เมาส์ 1พัน-3พัน", 1_000, 3_000),
        ("เมาส์ขั้นต่ำ ๑,๕๐๐ บาท", 1_500, None),
        ("เมาส์ราคาไม่เกินสองพัน", None, 2_000),
        ("เมาส์ไม่เกิน ฿8000", None, 8_000),
    ),
)
def test_price_ranges_thai_digits_and_k_notation(
    parser: ThaiCommandParser,
    text: str,
    minimum: float | None,
    maximum: float | None,
) -> None:
    entities = parser.parse(text).entities
    assert entities.min_price == minimum
    assert entities.max_price == maximum


@pytest.mark.parametrize(
    ("text", "minimum", "maximum", "min_inclusive", "max_inclusive"),
    (
        ("ลำโพง มากกว่า 6990", 6_990, None, False, True),
        ("speaker over 6990", 6_990, None, False, True),
        ("ลำโพง ต่ำกว่า 6990", None, 6_990, True, False),
        ("speaker under 6990", None, 6_990, True, False),
        ("ลำโพงอย่างน้อย 6990", 6_990, None, True, True),
        ("speaker at most 6990", None, 6_990, True, True),
    ),
)
def test_strict_and_inclusive_price_boundaries_are_preserved(
    parser,
    text,
    minimum,
    maximum,
    min_inclusive,
    max_inclusive,
):
    entities = parser.parse(text).entities

    assert entities.min_price == minimum
    assert entities.max_price == maximum
    assert entities.min_price_inclusive is min_inclusive
    assert entities.max_price_inclusive is max_inclusive


def test_live_audit_regressions(parser: ThaiCommandParser) -> None:
    discovery = parser.parse("มีสินค้าอะไรแนะนำบ้าง")
    assert discovery.intent == INTENT_SEARCH
    assert discovery.query == ""

    for text in (
        "มีอะไรบ้าง",
        "มีอะไรแนะนำ",
        "มีอะไรน่าสนใจ",
        "แนะนำอะไรหน่อย",
        "แนะนำหน่อย",
    ):
        recommendation = parser.parse(text)
        assert recommendation.intent == INTENT_SEARCH
        assert recommendation.query == ""

    assert parser.parse("คำสั่งมีอะไรบ้าง").intent == INTENT_HELP

    ranged = parser.parse("คีย์บอร์ดราคา 1000 ถึง 3500 เรียงถูกสุด")
    assert ranged.entities.min_price == 1_000
    assert ranged.entities.max_price == 3_500
    assert ranged.entities.sort == SORT_PRICE_ASC
    assert ranged.entities.query == ""

    normal = parser.parse("หาหูฟัง Sony ไม่เกิน 3000")
    assert normal.entities.query == ""

    availability = parser.parse(
        "หาเมาส์ Logitech ไม่เกิน 2,000 บาท เอาเฉพาะของพร้อมส่ง"
    )
    assert availability.entities.in_stock is True
    assert availability.entities.max_price == 2_000
    assert availability.entities.query == ""


def test_dynamic_catalogue_values_keep_catalogue_spelling() -> None:
    parser = ThaiCommandParser(brands=["Nothing"], categories=["Custom DAC"])
    parsed = parser.parse("Custom DAC Nothing ราคาไม่เกิน 10k")

    assert parsed.category == "Custom DAC"
    assert parsed.brands == ("Nothing",)
    assert parsed.max_price == 10_000


def test_functional_api_and_entity_compatibility_aliases() -> None:
    parsed = parse_command("Sony headphones")
    assert parsed.intent == INTENT_SEARCH
    assert parsed.entities.brand == "Sony"
    assert parsed.entities.availability is None
    assert parsed.entities.sort_preference is None
    assert parsed.entities.residual_query == ""
    assert parsed.to_dict()["entities"]["brands"] == ["Sony"]


def test_ambiguous_number_is_not_mistaken_for_product_model(parser: ThaiCommandParser) -> None:
    assert parser.parse("12345").intent == INTENT_UNKNOWN
    model = parser.parse("WH-1000XM5")
    assert model.intent == INTENT_SEARCH
    assert model.query == "wh-1000xm5"


@pytest.mark.parametrize("text", ("this", "shipping status", "refreshing", "helpful"))
def test_ascii_intent_phrases_require_word_boundaries(
    parser: ThaiCommandParser,
    text: str,
) -> None:
    assert parser.parse(text).intent == INTENT_UNKNOWN


@pytest.mark.parametrize(
    ("text", "query"),
    (
        ("AirPods", "airpods"),
        ("Soundcore", "soundcore"),
        ("micro sd card", "micro sd card"),
        ("หา Soundcore", "soundcore"),
    ),
)
def test_arbitrary_product_and_model_terms_are_searchable(
    parser: ThaiCommandParser,
    text: str,
    query: str,
) -> None:
    parsed = parser.parse(text)
    assert parsed.intent == INTENT_SEARCH
    assert parsed.category is None
    assert parsed.query == query


def test_upward_price_and_compound_thai_number(parser: ThaiCommandParser) -> None:
    numeric = parser.parse("หูฟังราคา 5000 บาทขึ้นไป")
    assert numeric.min_price == 5_000
    assert numeric.max_price is None
    assert numeric.query == ""

    words = parser.parse("หูฟังราคาหนึ่งหมื่นห้าพันบาทขึ้นไป")
    assert words.min_price == 15_000
    assert words.max_price is None
    assert words.query == ""


def test_stock_and_brand_negation_are_not_inverted(parser: ThaiCommandParser) -> None:
    stock = parser.parse("หูฟังไม่เอาของหมด")
    assert stock.intent == INTENT_SEARCH
    assert stock.in_stock is True
    assert stock.query == ""

    exclusion = parser.parse("หูฟังไม่เอา Sony")
    assert exclusion.intent == INTENT_SEARCH
    assert exclusion.entities.brands == ()
    assert exclusion.entities.excluded_brands == ("Sony",)


@pytest.mark.parametrize(
    ("text", "query"),
    (
        ("ที่รองข้อมือเมาส์", "ที่รองข้อมือ"),
        ("แผ่นรองเมาส์", "แผ่นรอง"),
        ("mouse pad", "pad"),
    ),
)
def test_accessory_terms_survive_boundary_safe_filler_cleanup(
    parser: ThaiCommandParser,
    text: str,
    query: str,
) -> None:
    parsed = parser.parse(text)
    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == "เมาส์"
    assert parsed.query == query


def test_printer_alias_matches_live_snapshot_wording() -> None:
    parser = ThaiCommandParser(categories=["เครื่องปริ้น / หมึก"])
    parsed = parser.parse("หาเครื่องพิมพ์ไม่เกิน 5000")
    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == "เครื่องพิมพ์"
    assert parsed.max_price == 5_000


@pytest.mark.parametrize(
    ("text", "category", "brand", "minimum", "maximum"),
    (
        ("หาคีย์บอร์ด 1000-3500", "คีย์บอร์ด", None, 1_000, 3_500),
        ("หาลำโพง Marshall 9000-20000", "ลำโพง", "Marshall", 9_000, 20_000),
    ),
)
def test_bare_price_ranges_after_product_terms(
    parser: ThaiCommandParser,
    text: str,
    category: str,
    brand: str | None,
    minimum: float,
    maximum: float,
) -> None:
    parsed = parser.parse(text)
    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == category
    assert parsed.entities.brand == brand
    assert parsed.min_price == minimum
    assert parsed.max_price == maximum
    assert parsed.query == ""


@pytest.mark.parametrize(
    ("text", "category", "query"),
    (
        ("หาแผ่นรองเมาส์", "เมาส์", "แผ่นรอง"),
        ("หาmicro sd card", None, "micro sd card"),
        ("ขอรุ่น AirPods", None, "airpods"),
    ),
)
def test_attached_command_prefixes_leave_searchable_product_terms(
    parser: ThaiCommandParser,
    text: str,
    category: str | None,
    query: str,
) -> None:
    parsed = parser.parse(text)
    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == category
    assert parsed.query == query


def test_short_monitor_alias_stock_negation_and_compound_budget(
    parser: ThaiCommandParser,
) -> None:
    monitor = parser.parse("จอ")
    assert monitor.intent == INTENT_SEARCH
    assert monitor.category == "จอคอม"

    unavailable = parser.parse("หูฟังไม่พร้อมส่ง")
    assert unavailable.intent == INTENT_SEARCH
    assert unavailable.in_stock is False
    assert unavailable.query == ""

    budget = parser.parse("หูฟังสองพันห้าร้อยบาท")
    assert budget.intent == INTENT_SEARCH
    assert budget.category == "หูฟัง"
    assert budget.max_price == 2_500
    assert budget.query == ""


def test_category_alias_expansion_preserves_spacing_variants() -> None:
    aliases = category_aliases_for("สมาร์ทวอทช์")
    assert "smartwatch" in aliases
    assert "smart watch" in aliases


@pytest.mark.parametrize(
    ("text", "category", "query"),
    (
        ("computer accessories", "อุปกรณ์คอมพิวเตอร์", ""),
        ("fitness tracker", "สมาร์ทวอทช์", "fitness tracker"),
        ("UPS", "อุปกรณ์คอมพิวเตอร์", "ups"),
        ("toner", "เครื่องพิมพ์", "toner"),
        ("scanner", "เครื่องพิมพ์", "scanner"),
        ("flash drive", "อุปกรณ์เสริม", "flash drive"),
        ("USB hub", "อุปกรณ์เสริม", "usb hub"),
        ("conference camera", "เว็บแคม", "conference camera"),
        ("ขาไมค์", "ไมโครโฟน", "ขาไมค์"),
        ("earbuds", "หูฟัง", "earbuds"),
        ("เอียร์บัด", "หูฟัง", "เอียร์บัด"),
        ("soundbar", "ลำโพง", "soundbar"),
        ("ซาวด์บาร์", "ลำโพง", "ซาวด์บาร์"),
    ),
)
def test_live_subtypes_keep_hard_filter_residual(
    parser: ThaiCommandParser,
    text: str,
    category: str,
    query: str,
) -> None:
    parsed = parser.parse(text)
    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == category
    assert parsed.query == query


@pytest.mark.parametrize(
    ("text", "query"),
    (
        ("จอ 24-27 นิ้ว", "24-27 นิ้ว"),
        ("เมาส์ dpi 800-1600", "dpi 800-1600"),
        ("power bank 10000-20000 mAh", "10000-20000 mah"),
        ("หูฟัง 20-20000 Hz", "20-20000 hz"),
    ),
)
def test_specification_ranges_are_not_parsed_as_price_budgets(
    parser: ThaiCommandParser,
    text: str,
    query: str,
) -> None:
    parsed = parser.parse(text)
    assert parsed.intent == INTENT_SEARCH
    assert parsed.min_price is None
    assert parsed.max_price is None
    assert parsed.query == query


def test_refresh_word_inside_monitor_spec_remains_search(
    parser: ThaiCommandParser,
) -> None:
    parsed = parser.parse("refresh rate monitor 144hz")
    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == "จอคอม"
    assert parsed.query == "refresh rate 144hz"
    assert parser.parse("refresh").intent == INTENT_REFRESH


@pytest.mark.parametrize(
    ("operator", "minimum", "maximum", "inclusive"),
    (
        (">", 3_000, None, False),
        (">=", 3_000, None, True),
        ("≥", 3_000, None, True),
        ("<", None, 3_000, False),
        ("<=", None, 3_000, True),
        ("≤", None, 3_000, True),
    ),
)
def test_symbolic_price_comparators(
    parser: ThaiCommandParser,
    operator: str,
    minimum: float | None,
    maximum: float | None,
    inclusive: bool,
) -> None:
    parsed = parser.parse(f"หูฟังราคา {operator} 3000")
    assert parsed.intent == INTENT_SEARCH
    assert parsed.min_price == minimum
    assert parsed.max_price == maximum
    if minimum is not None:
        assert parsed.entities.min_price_inclusive is inclusive
    if maximum is not None:
        assert parsed.entities.max_price_inclusive is inclusive
    assert parsed.query == ""


def test_brand_label_is_structural_only_around_extracted_brand(
    parser: ThaiCommandParser,
) -> None:
    parsed = parser.parse("หาหูฟังแบรนด์ Xiaomi ไม่เกิน 3,000 บาท")
    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == "หูฟัง"
    assert parsed.brands == ("Xiaomi",)
    assert parsed.max_price == 3_000
    assert parsed.query == ""


@pytest.mark.parametrize(
    ("text", "category", "query"),
    (
        ("เครื่องสำรองไฟ", "อุปกรณ์คอมพิวเตอร์", "เครื่องสำรองไฟ"),
        ("กล้องประชุม", "เว็บแคม", "กล้องประชุม"),
        ("สแกนเนอร์", "เครื่องพิมพ์", "สแกนเนอร์"),
        ("แฟลชไดรฟ์", "อุปกรณ์เสริม", "แฟลชไดรฟ์"),
        ("watch strap", "สมาร์ทวอทช์", "watch strap"),
        ("pop filter", "ไมโครโฟน", "pop filter"),
        ("สายไมค์", "ไมโครโฟน", "สายไมค์"),
    ),
)
def test_final_live_aliases_preserve_subtype_queries(
    parser: ThaiCommandParser,
    text: str,
    category: str,
    query: str,
) -> None:
    parsed = parser.parse(text)
    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == category
    assert parsed.query == query


def test_watch_compatibility_is_not_a_manufacturer_filter(
    parser: ThaiCommandParser,
) -> None:
    strap = parser.parse("สาย apple watch")
    assert strap.category == "สมาร์ทวอทช์"
    assert strap.brands == ()
    assert strap.query == "สาย apple watch"

    device = parser.parse("Apple smartwatch")
    assert device.brands == ("Apple",)
    assert device.query == ""


def test_gaming_monitor_keeps_gaming_as_subtype(parser: ThaiCommandParser) -> None:
    parsed = parser.parse("gaming monitor")
    assert parsed.intent == INTENT_SEARCH
    assert parsed.category == "จอคอม"
    assert parsed.query == "gaming"

    thai = parser.parse("จอเกมมิ่ง")
    assert thai.category == "จอคอม"
    assert thai.query == "เกมมิ่ง"


def test_labelled_corpus_is_large_varied_and_above_rubric_gate() -> None:
    cases = load_cases(PROJECT_DIR / "data" / "nlp_evaluation.json")
    kinds = {case["kind"] for case in cases}
    assert len(cases) >= 50
    assert {"basic", "colloquial", "typo", "multi_condition", "ambiguous", "no_match"} <= kinds

    metrics, _ = evaluate_cases(cases)
    assert metrics.intent_accuracy > 0.85
    assert metrics.entity_field_accuracy > 0.85
    assert metrics.entity_f1 > 0.85
    assert metrics.passed


def test_parser_latency_has_large_margin_under_end_to_end_budget(parser: ThaiCommandParser) -> None:
    messages = [
        "หูฟัง Sony หรือ JBL ไร้สาย งบ ๕k พร้อมส่ง ถูกสุด",
        "คีย์บอร์ดราคา 1000 ถึง 3500 เรียงถูกสุด",
        "มีสินค้าอะไรแนะนำบ้าง",
        "ขอคุยกับแอดมิน",
    ]
    samples: list[float] = []
    for index in range(200):
        started = time.perf_counter()
        parser.parse(messages[index % len(messages)])
        samples.append((time.perf_counter() - started) * 1_000)
    p95 = sorted(samples)[int(len(samples) * 0.95) - 1]
    # The rubric allows 1.5–2 seconds end-to-end; 50 ms leaves ample webhook/render time.
    assert statistics.fmean(samples) < 25
    assert p95 < 50
