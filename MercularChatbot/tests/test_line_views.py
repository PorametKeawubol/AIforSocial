"""Offline contract tests for Mercular's LINE Messaging API views."""

from __future__ import annotations

import json

import pytest
from linebot.v3.messaging import FlexMessage, TextMessage

from line_views import (
    LINE_ACTION_LABEL_LIMIT,
    LINE_ALT_TEXT_LIMIT,
    LINE_POSTBACK_DATA_LIMIT,
    MAX_CAROUSEL_PRODUCTS,
    build_category_picker_message,
    build_category_picker_payload,
    build_postback_data,
    build_product_carousel_message,
    build_product_carousel_payload,
    build_promotion_bubble_payload,
    build_promotion_carousel_message,
    contact_message,
    data_unavailable_message,
    default_quick_reply_items,
    disclaimer_message,
    greeting_message,
    help_message,
    is_refresh_postback,
    no_results_message,
    normalize_https_url,
    parse_postback_data,
    parse_product_postback,
    payload_is_json_serializable,
    product_carousel_message,
    product_comparison_message,
    product_detail_message,
    refresh_message,
    text_with_quick_replies,
    truncate_text,
)
from catalog_navigation import CategoryMenu, CategoryOption
from models import Product
from promotions import Promotion


def product(number: int = 1, **overrides: object) -> Product:
    values: dict[str, object] = {
        "id": f"product-{number}",
        "sku": f"SKU-{number}",
        "name": f"หูฟังเกมมิ่งรุ่น {number}",
        "brand": "Mercular Test",
        "category": "หูฟังเกมมิ่ง",
        "category_path": ("เครื่องเสียง", "หูฟัง"),
        "price": 1_990.0 + number,
        "original_price": 2_490.0,
        "image_url": f"https://cdn.example.com/products/{number}.jpg",
        "product_url": f"https://www.mercular.com/product/{number}",
        "in_stock": True,
        "description": "เสียงชัด ใส่สบาย เหมาะกับการเล่นเกม",
    }
    values.update(overrides)
    return Product(**values)  # type: ignore[arg-type]


def promotion() -> Promotion:
    return Promotion(
        id="payday",
        title="PAYDAY ลดสูงสุด 8,000 บาท",
        summary="คูปองทุกหมวดและดีลเฉพาะสินค้า",
        image_url="https://cdn.example.com/promo.jpg",
        article_url="https://www.mercular.com/review-article/payday",
        starts_at="2026-08-24",
        ends_at="2026-08-31",
        discount_summary="ลดสูงสุด 8,000 บาท",
    )


def bubbles(payload: dict[str, object]) -> list[dict[str, object]]:
    contents = payload["contents"]
    assert isinstance(contents, dict)
    result = contents["contents"]
    assert isinstance(result, list)
    return result


def actions(bubble: dict[str, object]) -> list[dict[str, object]]:
    footer = bubble["footer"]
    assert isinstance(footer, dict)
    buttons = footer["contents"]
    assert isinstance(buttons, list)
    return [button["action"] for button in buttons]  # type: ignore[index]


def test_exactly_five_products_are_rendered_and_a_sixth_is_not() -> None:
    payload = build_product_carousel_payload([product(i) for i in range(1, 7)])

    assert payload is not None
    assert payload["type"] == "flex"
    assert payload["contents"]["type"] == "carousel"
    assert len(bubbles(payload)) == MAX_CAROUSEL_PRODUCTS == 5
    assert [
        parse_product_postback(actions(bubble)[0]["data"])
        for bubble in bubbles(payload)
    ] == [f"product-{i}" for i in range(1, 6)]

    # This is real UTF-8 JSON made entirely from primitive payload values.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert "หูฟังเกมมิ่ง" in encoded
    assert payload_is_json_serializable(payload)


def test_category_picker_paginates_choices_into_tappable_flex_buttons() -> None:
    options = tuple(
        CategoryOption(
            f"หมวดสินค้าที่ยาวมาก {index}",
            index * 10,
            f"เลือกหมวด: คอมพิวเตอร์ > หมวด {index}",
            True,
        )
        for index in range(1, 8)
    )
    menu = CategoryMenu(("คอมพิวเตอร์",), 280, options)

    payload = build_category_picker_payload(menu)
    message = build_category_picker_message(menu)

    assert isinstance(message, FlexMessage)
    assert payload_is_json_serializable(payload)
    assert len(bubbles(payload)) == 2
    rendered_buttons = [
        item
        for bubble in bubbles(payload)
        for item in bubble["body"]["contents"]  # type: ignore[index]
        if item["type"] == "button"  # type: ignore[index]
    ]
    assert len(rendered_buttons) == 7
    assert all(
        len(button["action"]["label"]) <= LINE_ACTION_LABEL_LIMIT
        and len(button["action"]["text"]) <= LINE_POSTBACK_DATA_LIMIT
        for button in rendered_buttons
    )


def test_duplicates_are_not_assumed_away_and_later_unique_products_fill_five() -> None:
    first = product(1)
    payload = build_product_carousel_payload(
        [first, first, product(2), product(2), product(3), product(4), product(5)]
    )

    assert payload is not None
    identifiers = [
        parse_product_postback(actions(bubble)[0]["data"])
        for bubble in bubbles(payload)
    ]
    assert identifiers == [f"product-{i}" for i in range(1, 6)]
    assert len(identifiers) == len(set(identifiers))


@pytest.mark.parametrize("count", [1, 2, 4])
def test_fewer_than_five_products_render_without_padding(count: int) -> None:
    payload = build_product_carousel_payload([product(i) for i in range(count)])

    assert payload is not None
    assert len(bubbles(payload)) == count


def test_zero_products_never_builds_an_invalid_empty_carousel() -> None:
    assert build_product_carousel_payload([]) is None
    assert product_carousel_message([]) is None

    fallback = build_product_carousel_message([])
    assert isinstance(fallback, TextMessage)
    assert "ยังไม่พบสินค้า" in fallback.text


def test_missing_optional_fields_remain_readable_and_omit_the_hero() -> None:
    payload = build_product_carousel_payload(
        [
            product(
                1,
                brand="",
                category="",
                price=None,
                original_price=None,
                image_url="",
                description="",
                in_stock=False,
            )
        ]
    )

    assert payload is not None
    bubble = bubbles(payload)[0]
    assert "hero" not in bubble
    body = bubble["body"]
    body_text = "\n".join(
        item["text"]
        for item in body["contents"]  # type: ignore[index]
        if item["type"] == "text"
    )
    assert "ไม่ระบุแบรนด์" in body_text
    assert "ไม่ระบุหมวดหมู่" in body_text
    assert "ตรวจสอบราคาที่เว็บไซต์" in body_text
    assert "สินค้าหมดชั่วคราว" in body_text

    action_types = [action["type"] for action in actions(bubble)]
    assert action_types == ["postback", "uri"]


@pytest.mark.parametrize(
    ("image_url", "has_hero"),
    [
        ("https://cdn.example.com/picture?id=1", True),
        ("http://cdn.example.com/picture.jpg", True),
        ("javascript:alert(1)", False),
        ("https://127.0.0.1/picture.jpg", False),
        ("https://user:password@cdn.example.com/picture.jpg", False),
        ("", False),
    ],
)
def test_only_public_https_images_reach_line(
    image_url: str,
    has_hero: bool,
) -> None:
    payload = build_product_carousel_payload([product(image_url=image_url)])

    assert payload is not None
    bubble = bubbles(payload)[0]
    assert ("hero" in bubble) is has_hero
    if has_hero:
        hero = bubble["hero"]
        assert hero["url"].startswith("https://")  # type: ignore[index,union-attr]
        assert hero["aspectRatio"] == "1:1"  # type: ignore[index,union-attr]
        assert hero["aspectMode"] == "cover"  # type: ignore[index,union-attr]


def test_non_mercular_product_url_never_becomes_a_uri_action() -> None:
    payload = build_product_carousel_payload(
        [product(product_url="https://evil.example/checkout")]
    )

    assert payload is not None
    bubble = bubbles(payload)[0]
    assert [action["type"] for action in actions(bubble)] == ["postback"]
    assert "action" not in bubble["hero"]  # type: ignore[operator]


def test_text_is_wrapped_and_truncated_with_bounded_alt_text() -> None:
    long_name = "ชุดหูฟังเกมมิ่งคุณภาพสูง" * 20
    long_brand = "แบรนด์ทดสอบ" * 20
    long_category = "อุปกรณ์เกมมิ่งและสตรีมมิ่ง" * 20
    payload = build_product_carousel_payload(
        [
            product(
                name=long_name,
                brand=long_brand,
                category=long_category,
            )
        ],
        alt_text="สินค้าแนะนำ" * 100,
    )

    assert payload is not None
    assert len(payload["altText"]) <= LINE_ALT_TEXT_LIMIT
    assert payload["altText"].endswith("…")
    body_items = bubbles(payload)[0]["body"]["contents"]  # type: ignore[index]
    text_items = [item for item in body_items if item["type"] == "text"]
    assert all(item["wrap"] is True for item in text_items)
    assert all("maxLines" in item for item in text_items)
    assert len(text_items[0]["text"]) <= 55
    assert len(text_items[1]["text"]) <= 90
    assert text_items[0]["text"].endswith("…")
    assert text_items[1]["text"].endswith("…")


def test_flex_sdk_wrapper_round_trips_the_pure_payload() -> None:
    products = [product(1), product(2)]
    payload = build_product_carousel_payload(products, alt_text="ผลการค้นหา")
    message = product_carousel_message(products, alt_text="ผลการค้นหา")

    assert isinstance(message, FlexMessage)
    assert message.to_dict() == payload
    assert payload["quickReply"]["items"][-1]["action"]["text"] == "สุ่มใหม่"
    json.dumps(message.to_dict(), ensure_ascii=False, allow_nan=False)


def test_promotion_card_has_only_safe_mercular_link_and_bounded_content() -> None:
    bubble = build_promotion_bubble_payload(promotion())
    message = build_promotion_carousel_message([promotion()])

    assert bubble["footer"]["contents"][0]["action"]["uri"].startswith(
        "https://www.mercular.com/review-article/"
    )
    assert "8,000" in bubble["body"]["contents"][2]["text"]
    assert isinstance(message, FlexMessage)
    assert len(message.to_dict()["contents"]["contents"]) == 1


def test_thai_uri_paths_are_utf8_percent_encoded_for_line() -> None:
    payload = build_product_carousel_payload(
        [product(product_url="https://www.mercular.com/หูฟัง-รุ่นใหม่?q=สีดำ")]
    )

    assert payload is not None
    website = actions(bubbles(payload)[0])[-1]
    assert website["uri"].startswith("https://www.mercular.com/%E0%B8%AB")
    assert "หูฟัง" not in website["uri"]
    assert "%E0%B8%AA%E0%B8%B5%E0%B8%94%E0%B8%B3" in website["uri"]


def test_product_actions_are_valid_and_postback_round_trips_exact_id() -> None:
    identifier = "sku/ไทย & รุ่น=หนึ่ง"
    payload = build_product_carousel_payload([product(id=identifier)])

    assert payload is not None
    detail, website = actions(bubbles(payload)[0])
    assert detail["type"] == "postback"
    assert len(detail["label"]) <= LINE_ACTION_LABEL_LIMIT
    assert len(detail["data"]) <= LINE_POSTBACK_DATA_LIMIT
    assert parse_product_postback(detail["data"]) == identifier
    assert website == {
        "type": "uri",
        "label": "ซื้อที่ Mercular",
        "uri": "https://www.mercular.com/product/1",
    }


def test_postback_parser_rejects_tampering_duplicates_and_unknown_actions() -> None:
    valid = build_postback_data("product_detail", product_id="p-123")

    assert parse_postback_data(valid) == {
        "action": "product_detail",
        "product_id": "p-123",
    }
    assert parse_product_postback(valid) == "p-123"
    assert parse_postback_data("action=product_detail&product_id=1&product_id=2") is None
    assert parse_postback_data("action=delete&product_id=1") is None
    assert parse_postback_data("action=product_detail") is None
    assert parse_postback_data("not-a-query-string") is None
    assert parse_postback_data("x" * 301) is None
    assert parse_product_postback('{"action":"product_detail","id":"legacy-1"}') == (
        "legacy-1"
    )
    assert parse_postback_data('{"action":"product_detail","id":"1","extra":true}') is None


def test_quick_replies_offer_nlp_examples_help_and_refresh() -> None:
    items = default_quick_reply_items()

    assert 7 == len(items) <= 13
    assert all(item["type"] == "action" for item in items)
    assert all(len(item["action"]["label"]) <= LINE_ACTION_LABEL_LIMIT for item in items)
    message_actions = [item["action"] for item in items if item["action"]["type"] == "message"]
    assert len(message_actions) == 7
    assert any(action["text"] == "เลือกหมวดสินค้า" for action in message_actions)
    assert any("ราคา" in action["text"] for action in message_actions)
    assert any(action["text"] == "ช่วยเหลือ" for action in message_actions)
    assert any("โปรโมชัน" in action["text"] for action in message_actions)

    refresh = items[-1]["action"]
    assert refresh == {"type": "message", "label": "สุ่มใหม่", "text": "สุ่มใหม่"}
    assert is_refresh_postback(build_postback_data("refresh"))
    assert not is_refresh_postback("action=refresh&unexpected=true")


def test_all_friendly_text_helpers_return_sdk_messages_with_valid_json() -> None:
    messages = [
        greeting_message(),
        help_message(),
        no_results_message("หูฟังราคา 1 บาท"),
        data_unavailable_message(),
        contact_message(),
        disclaimer_message(),
        refresh_message(),
        text_with_quick_replies("ข้อความทดสอบ"),
    ]

    for message in messages:
        assert isinstance(message, TextMessage)
        payload = message.to_dict()
        assert payload["type"] == "text"
        assert payload["text"]
        assert 1 <= len(payload["quickReply"]["items"]) <= 13
        json.dumps(payload, ensure_ascii=False, allow_nan=False)


def test_greeting_uses_configured_bot_name_safely() -> None:
    message = greeting_message("MercuMate")

    assert "ผม MercuMate" in message.text


def test_product_detail_is_readable_safe_and_contains_quick_replies() -> None:
    message = product_detail_message(
        product(
            in_stock=False,
            overview="จอเกมมิ่ง 165Hz",
            highlights=("พาเนล IPS", "รีเฟรชเรต 165Hz"),
            specifications=(("พาเนล", "IPS"), ("รีเฟรชเรต", "165Hz")),
            rating=4.8,
            review_count=5,
            warranty="ประกันศูนย์ 3 ปี",
        )
    )
    payload = message.to_dict()

    assert isinstance(message, TextMessage)
    assert "หูฟังเกมมิ่งรุ่น" in payload["text"]
    assert "Mercular Test" in payload["text"]
    assert "฿1,991" in payload["text"]
    assert "สินค้าหมดชั่วคราว" in payload["text"]
    assert "ภาพรวม: จอเกมมิ่ง 165Hz" in payload["text"]
    assert "พาเนล IPS" in payload["text"]
    assert "4.8/5 จาก 5 รีวิว" in payload["text"]
    assert "ประกันศูนย์ 3 ปี" in payload["text"]


def test_product_comparison_reports_price_and_only_known_spec_differences() -> None:
    first = product(
        1,
        name="Mouse Alpha",
        price=1990,
        specifications=(("น้ำหนัก", "59 กรัม"), ("DPI", "26000")),
    )
    second = product(
        2,
        name="Mouse Beta",
        price=2490,
        specifications=(("น้ำหนัก", "63 กรัม"), ("DPI", "26000")),
    )

    message = product_comparison_message(first, second)

    assert "ถูกกว่า ฿500" in message.text
    assert "น้ำหนัก" in message.text
    assert "59 กรัม" in message.text and "63 กรัม" in message.text
    assert "DPI" not in message.text


def test_unknown_stock_is_disclosed_without_claiming_available_or_sold_out() -> None:
    payload = product_detail_message(product(in_stock=None)).to_dict()

    assert "ไม่ทราบ" in payload["text"]
    assert "โปรดตรวจสอบที่เว็บไซต์" in payload["text"]
    assert "https://www.mercular.com/product/1" in payload["text"]
    assert payload["quickReply"]["items"]


def test_url_and_thai_text_validators_cover_unsafe_edge_cases() -> None:
    assert normalize_https_url("http://www.mercular.com/product/1", mercular_only=True) == (
        "https://www.mercular.com/product/1"
    )
    assert normalize_https_url("https://mercular.com", mercular_only=True) == (
        "https://mercular.com/"
    )
    assert normalize_https_url("https://not-mercular.example/product", mercular_only=True) == ""
    assert normalize_https_url("https://mercular.com.evil.example/product", mercular_only=True) == ""
    assert normalize_https_url("file:///tmp/picture.jpg") == ""
    assert normalize_https_url("https://localhost/picture.jpg") == ""
    assert normalize_https_url("https://10.0.0.1/picture.jpg") == ""

    result = truncate_text("  สวัสดี\xa0 ครับ  " * 20, 25)
    assert len(result) <= 25
    assert result.endswith("…")
