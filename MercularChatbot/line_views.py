"""LINE Messaging API v3 views for the Mercular shopping assistant.

The functions ending in ``_payload`` return ordinary dictionaries.  They are
easy to test and serialize without a LINE connection.  The message helpers
perform the small final conversion to the SDK's ``FlexMessage`` and
``TextMessage`` models.
"""

from __future__ import annotations

import ipaddress
import json
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from linebot.v3.messaging import FlexMessage, TextMessage

try:  # Support both ``python app.py`` and package imports in offline tests.
    from .catalog_navigation import ALL_CATEGORIES_COMMAND, CategoryMenu
    from .models import Product, clean_text
    from .nlp import normalize_text
    from .promotions import Promotion
except ImportError:  # pragma: no cover - exercised when app.py is run directly
    from catalog_navigation import ALL_CATEGORIES_COMMAND, CategoryMenu
    from models import Product, clean_text
    from nlp import normalize_text
    from promotions import Promotion


MAX_CAROUSEL_PRODUCTS = 5
LINE_ALT_TEXT_LIMIT = 400
LINE_TEXT_LIMIT = 5_000
LINE_ACTION_LABEL_LIMIT = 20
LINE_POSTBACK_DATA_LIMIT = 300
LINE_URI_LIMIT = 1_000
LINE_QUICK_REPLY_LIMIT = 13
CATEGORY_OPTIONS_PER_BUBBLE = 5

PRODUCT_DETAIL_ACTION = "product_detail"
REFRESH_ACTION = "refresh"
MERCULAR_HOME_URL = "https://www.mercular.com/"
MERCULAR_CONTACT_URL = "https://www.mercular.com/contents/contact"

_ACCENT_COLOR = "#1677FF"
_MUTED_COLOR = "#6B7280"
_SUCCESS_COLOR = "#16803A"
_OUT_OF_STOCK_COLOR = "#B42318"


def truncate_text(
    value: object | None,
    limit: int,
    *,
    fallback: str = "",
) -> str:
    """Collapse whitespace and truncate text without ending on a Thai mark."""

    if limit < 1:
        return ""
    text = " ".join(str(value or "").replace("\xa0", " ").split())
    if not text:
        text = " ".join(str(fallback or "").replace("\xa0", " ").split())
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"

    prefix = text[: limit - 1].rstrip()
    # A dangling combining vowel/tone mark is especially hard to read in Thai.
    while prefix and unicodedata.category(prefix[-1]).startswith("M"):
        prefix = prefix[:-1].rstrip()
    return f"{prefix}…" if prefix else "…"


def normalize_https_url(
    value: object | None,
    *,
    mercular_only: bool = False,
    max_length: int = LINE_URI_LIMIT,
) -> str:
    """Return a public HTTPS URL safe for LINE, or ``""`` when invalid.

    Mercular's model accepts HTTP source links because historical scrape data
    can contain them.  At the presentation boundary they are upgraded to HTTPS.
    Local/private hosts, embedded credentials, controls, and non-web schemes are
    rejected.  With ``mercular_only=True`` the host must be mercular.com or one
    of its subdomains.
    """

    candidate = str(value or "").strip()
    if (
        not candidate
        or len(candidate) > max_length
        or any(character.isspace() or ord(character) < 32 for character in candidate)
    ):
        return ""

    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        return ""
    if parsed.scheme.casefold() not in {"http", "https"} or not hostname:
        return ""
    if parsed.username is not None or parsed.password is not None:
        return ""

    host = hostname.rstrip(".").casefold()
    if not host or host == "localhost" or host.endswith((".localhost", ".local")):
        return ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Bare intranet names are not suitable for a public LINE message.
        if "." not in host:
            return ""
    else:
        if not address.is_global:
            return ""

    if mercular_only and host != "mercular.com" and not host.endswith(".mercular.com"):
        return ""
    if mercular_only and port not in (None, 443):
        return ""

    # Rebuild the authority from parsed pieces so a spoofed/user-info netloc is
    # never copied.  IPv6 literals need brackets in an absolute URL.
    rendered_host = f"[{host}]" if ":" in host else host
    authority = rendered_host if port is None else f"{rendered_host}:{port}"
    # LINE URI fields require UTF-8 percent encoding.  Keep URI delimiters and
    # existing percent escapes while encoding Thai/non-ASCII path/query text.
    safe_path = quote(
        parsed.path or "/",
        safe="/%:@-._~!$&'()*+,;=",
        encoding="utf-8",
        errors="strict",
    )
    safe_query = quote(
        parsed.query,
        safe="%=&:+,/?@-._~!$'()*;",
        encoding="utf-8",
        errors="strict",
    )
    result = urlunsplit(("https", authority, safe_path, safe_query, ""))
    return result if len(result) <= max_length else ""


def is_safe_https_url(value: object | None, *, mercular_only: bool = False) -> bool:
    """Whether ``value`` is already a valid public HTTPS URL."""

    candidate = str(value or "").strip()
    return candidate.casefold().startswith("https://") and bool(
        normalize_https_url(candidate, mercular_only=mercular_only)
    )


def build_postback_data(action: str, **parameters: object) -> str:
    """Build bounded query-string postback data accepted by LINE."""

    action_name = clean_text(action, limit=40)
    if action_name not in {PRODUCT_DETAIL_ACTION, REFRESH_ACTION}:
        raise ValueError("unsupported postback action")
    pairs: list[tuple[str, str]] = [("action", action_name)]
    for key, value in parameters.items():
        clean_key = clean_text(key, limit=40)
        clean_value = clean_text(value, limit=240)
        if clean_key and clean_value:
            pairs.append((clean_key, clean_value))
    data = urlencode(pairs)
    if len(data) > LINE_POSTBACK_DATA_LIMIT:
        raise ValueError("postback data exceeds LINE's 300-character limit")
    return data


def parse_postback_data(data: object | None) -> dict[str, str] | None:
    """Parse a postback generated here, rejecting duplicates and unknown keys."""

    raw = str(data or "")
    if not raw or len(raw) > LINE_POSTBACK_DATA_LIMIT:
        return None
    # Accept the compact JSON shape used by early versions of this project so
    # queued LINE deliveries remain usable across an application upgrade.
    if raw.startswith("{"):
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(decoded, dict) or decoded.get("action") != PRODUCT_DETAIL_ACTION:
            return None
        if set(decoded) not in (
            {"action", "id"},
            {"action", "product_id"},
        ):
            return None
        product_id = clean_text(
            decoded.get("product_id", decoded.get("id")),
            limit=240,
        )
        return (
            {"action": PRODUCT_DETAIL_ACTION, "product_id": product_id}
            if product_id
            else None
        )
    try:
        pairs = parse_qsl(
            raw,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=4,
        )
    except ValueError:
        return None
    if not pairs or len({key for key, _ in pairs}) != len(pairs):
        return None

    parsed = dict(pairs)
    action = parsed.get("action")
    if action == PRODUCT_DETAIL_ACTION:
        if set(parsed) != {"action", "product_id"}:
            return None
        product_id = clean_text(parsed.get("product_id"), limit=240)
        if not product_id:
            return None
        return {"action": action, "product_id": product_id}
    if action == REFRESH_ACTION and set(parsed) == {"action"}:
        return {"action": action}
    return None


def parse_product_postback(data: object | None) -> str | None:
    """Return the exact product ID in a valid detail postback."""

    parsed = parse_postback_data(data)
    if parsed and parsed["action"] == PRODUCT_DETAIL_ACTION:
        return parsed["product_id"]
    return None


def is_refresh_postback(data: object | None) -> bool:
    parsed = parse_postback_data(data)
    return bool(parsed and parsed["action"] == REFRESH_ACTION)


def _detail_identifier(product: Product) -> str:
    """Choose a repository identifier which fits in one LINE postback."""

    for identifier in (product.id, product.sku):
        cleaned = clean_text(identifier, limit=240)
        if not cleaned:
            continue
        try:
            build_postback_data(PRODUCT_DETAIL_ACTION, product_id=cleaned)
        except ValueError:
            continue
        return cleaned
    raise ValueError("product has no postback-safe id or sku")


def _product_body(product: Product) -> dict[str, Any]:
    brand = truncate_text(product.brand, 55, fallback="ไม่ระบุแบรนด์")
    category = truncate_text(product.category, 70, fallback="ไม่ระบุหมวดหมู่")
    name = truncate_text(product.name, 90, fallback="สินค้า Mercular")
    price = truncate_text(product.display_price, 45, fallback="ตรวจสอบราคาที่เว็บไซต์")
    if product.in_stock is True:
        stock_text, stock_color = "● พร้อมส่ง", _SUCCESS_COLOR
    elif product.in_stock is False:
        stock_text, stock_color = "● สินค้าหมดชั่วคราว", _OUT_OF_STOCK_COLOR
    else:
        stock_text, stock_color = "● ตรวจสอบสต็อกที่เว็บไซต์", _MUTED_COLOR

    return {
        "type": "box",
        "layout": "vertical",
        "spacing": "sm",
        "paddingAll": "18px",
        "contents": [
            {
                "type": "text",
                "text": brand,
                "size": "xs",
                "color": _ACCENT_COLOR,
                "weight": "bold",
                "wrap": True,
                "maxLines": 2,
            },
            {
                "type": "text",
                "text": name,
                "size": "md",
                "weight": "bold",
                "color": "#111827",
                "wrap": True,
                "maxLines": 3,
            },
            {
                "type": "text",
                "text": f"หมวดหมู่: {category}",
                "size": "xs",
                "color": _MUTED_COLOR,
                "wrap": True,
                "maxLines": 2,
            },
            {"type": "separator", "margin": "md"},
            {
                "type": "text",
                "text": price,
                "size": "lg",
                "weight": "bold",
                "color": _ACCENT_COLOR,
                "wrap": True,
                "maxLines": 2,
                "margin": "md",
            },
            {
                "type": "text",
                "text": stock_text,
                "size": "xs",
                "weight": "bold",
                "color": stock_color,
                "wrap": True,
                "maxLines": 1,
            },
        ],
    }


def build_product_bubble_payload(product: Product) -> dict[str, Any]:
    """Build one valid product bubble; unsafe optional media is omitted."""

    if not isinstance(product, Product):
        raise TypeError("product must be a Product")

    identifier = _detail_identifier(product)
    detail_data = build_postback_data(
        PRODUCT_DETAIL_ACTION,
        product_id=identifier,
    )
    footer_contents: list[dict[str, Any]] = [
        {
            "type": "button",
            "style": "secondary",
            "height": "sm",
            "action": {
                "type": "postback",
                "label": "ดูรายละเอียด",
                "data": detail_data,
                "displayText": truncate_text(
                    f"ดูรายละเอียด {product.name}",
                    LINE_POSTBACK_DATA_LIMIT,
                ),
            },
        }
    ]

    product_url = normalize_https_url(product.product_url, mercular_only=True)
    if product_url:
        footer_contents.append(
            {
                "type": "button",
                "style": "primary",
                "height": "sm",
                "color": _ACCENT_COLOR,
                "action": {
                    "type": "uri",
                    "label": "ซื้อที่ Mercular",
                    "uri": product_url,
                },
            }
        )

    bubble: dict[str, Any] = {
        "type": "bubble",
        "size": "kilo",
        "body": _product_body(product),
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "paddingAll": "14px",
            "contents": footer_contents,
        },
        "styles": {
            "footer": {"separator": True},
        },
    }

    image_url = normalize_https_url(product.image_url, max_length=2_000)
    if image_url:
        hero: dict[str, Any] = {
            "type": "image",
            "url": image_url,
            "size": "full",
            "aspectRatio": "1:1",
            "aspectMode": "cover",
            "animated": False,
        }
        if product_url:
            hero["action"] = {"type": "uri", "uri": product_url}
        # Insertion order is irrelevant to JSON, but placing the hero first
        # keeps snapshots and logs easy for people to scan.
        bubble = {"type": bubble.pop("type"), "hero": hero, **bubble}
    return bubble


def _unique_products(products: Iterable[Product]) -> list[Product]:
    selected: list[Product] = []
    seen: set[str] = set()
    for product in products:
        if not isinstance(product, Product):
            continue
        identifier = clean_text(product.id, limit=240)
        if not identifier or identifier in seen:
            continue
        try:
            _detail_identifier(product)
        except ValueError:
            continue
        seen.add(identifier)
        selected.append(product)
        if len(selected) == MAX_CAROUSEL_PRODUCTS:
            break
    return selected


def build_product_carousel_payload(
    products: Iterable[Product],
    *,
    alt_text: object | None = None,
) -> dict[str, Any] | None:
    """Build a Flex payload for zero to five unique products.

    ``None`` deliberately represents zero usable products: LINE rejects an
    empty carousel, so callers cannot accidentally send invalid Flex JSON.
    """

    selected = _unique_products(products)
    if not selected:
        return None
    fallback_alt = f"สินค้า Mercular แนะนำ {len(selected)} รายการ"
    safe_alt = truncate_text(alt_text, LINE_ALT_TEXT_LIMIT, fallback=fallback_alt)
    return {
        "type": "flex",
        "altText": safe_alt,
        "quickReply": {"items": default_quick_reply_items()},
        "contents": {
            "type": "carousel",
            "contents": [build_product_bubble_payload(product) for product in selected],
        },
    }


def product_carousel_message(
    products: Iterable[Product],
    *,
    alt_text: object | None = None,
) -> FlexMessage | None:
    """Convert a non-empty carousel payload to a LINE SDK v3 message."""

    payload = build_product_carousel_payload(products, alt_text=alt_text)
    if payload is None:
        return None
    return FlexMessage.from_dict(payload)


def build_product_carousel_message(
    products: Iterable[Product],
    *,
    alt_text: object | None = None,
) -> FlexMessage | TextMessage:
    """Return a carousel, or a friendly text fallback for an empty result."""

    message = product_carousel_message(products, alt_text=alt_text)
    return message if message is not None else no_results_message()


# Short integration-friendly alias used by the webhook.
build_product_carousel = build_product_carousel_message


def build_promotion_bubble_payload(promotion: Promotion) -> dict[str, Any]:
    """Build one link-only promotion card from verified snapshot fields."""

    if not isinstance(promotion, Promotion):
        raise TypeError("promotion must be a Promotion")
    article_url = normalize_https_url(promotion.article_url, mercular_only=True)
    if not article_url:
        raise ValueError("promotion has no safe Mercular article URL")
    contents: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "PROMOTION",
            "size": "xs",
            "weight": "bold",
            "color": _ACCENT_COLOR,
        },
        {
            "type": "text",
            "text": truncate_text(promotion.title, 120, fallback="โปรโมชัน Mercular"),
            "size": "md",
            "weight": "bold",
            "wrap": True,
            "maxLines": 4,
        },
    ]
    if promotion.discount_summary:
        contents.append(
            {
                "type": "text",
                "text": truncate_text(promotion.discount_summary, 100),
                "size": "sm",
                "weight": "bold",
                "color": _OUT_OF_STOCK_COLOR,
                "wrap": True,
                "maxLines": 3,
                "margin": "md",
            }
        )
    if promotion.starts_at or promotion.ends_at:
        period = " – ".join(
            part for part in (promotion.starts_at[:10], promotion.ends_at[:10]) if part
        )
        contents.append(
            {
                "type": "text",
                "text": f"ระยะเวลา: {period}",
                "size": "xs",
                "color": _MUTED_COLOR,
                "wrap": True,
                "maxLines": 2,
                "margin": "sm",
            }
        )
    if promotion.summary:
        contents.append(
            {
                "type": "text",
                "text": truncate_text(promotion.summary, 220),
                "size": "xs",
                "color": _MUTED_COLOR,
                "wrap": True,
                "maxLines": 5,
                "margin": "md",
            }
        )
    bubble: dict[str, Any] = {
        "type": "bubble",
        "size": "kilo",
        "body": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "18px",
            "contents": contents,
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "14px",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "height": "sm",
                    "color": _ACCENT_COLOR,
                    "action": {
                        "type": "uri",
                        "label": "ดูโปรโมชัน",
                        "uri": article_url,
                    },
                }
            ],
        },
        "styles": {"footer": {"separator": True}},
    }
    image_url = normalize_https_url(promotion.image_url, max_length=2_000)
    if image_url:
        bubble = {
            "type": "bubble",
            "hero": {
                "type": "image",
                "url": image_url,
                "size": "full",
                "aspectRatio": "20:13",
                "aspectMode": "cover",
                "action": {"type": "uri", "uri": article_url},
            },
            **{key: value for key, value in bubble.items() if key != "type"},
        }
    return bubble


def build_promotion_carousel_message(promotions: Iterable[Promotion]) -> FlexMessage | TextMessage:
    selected: list[Promotion] = []
    seen: set[str] = set()
    for promotion in promotions:
        if not isinstance(promotion, Promotion) or promotion.id in seen:
            continue
        seen.add(promotion.id)
        selected.append(promotion)
        if len(selected) == MAX_CAROUSEL_PRODUCTS:
            break
    if not selected:
        return promotion_unavailable_message()
    return FlexMessage.from_dict(
        {
            "type": "flex",
            "altText": truncate_text(
                f"โปรโมชัน Mercular ล่าสุด {len(selected)} รายการ",
                LINE_ALT_TEXT_LIMIT,
            ),
            "quickReply": {"items": default_quick_reply_items()},
            "contents": {
                "type": "carousel",
                "contents": [build_promotion_bubble_payload(item) for item in selected],
            },
        }
    )


def build_category_picker_payload(menu: CategoryMenu) -> dict[str, Any]:
    """Build a paginated Flex picker for one catalogue breadcrumb level."""

    if not isinstance(menu, CategoryMenu):
        raise TypeError("menu must be a CategoryMenu")
    if not menu.options:
        raise ValueError("category menu must contain at least one option")

    pages = [
        menu.options[index : index + CATEGORY_OPTIONS_PER_BUBBLE]
        for index in range(0, len(menu.options), CATEGORY_OPTIONS_PER_BUBBLE)
    ]
    bubbles: list[dict[str, Any]] = []
    for page_number, options in enumerate(pages, start=1):
        buttons = [
            {
                "type": "button",
                "style": "secondary",
                "height": "sm",
                "action": {
                    "type": "message",
                    "label": truncate_text(
                        f"{option.label} ({option.product_count})",
                        LINE_ACTION_LABEL_LIMIT,
                    ),
                    "text": truncate_text(
                        option.command,
                        LINE_POSTBACK_DATA_LIMIT,
                    ),
                },
            }
            for option in options
        ]
        bubbles.append(
            {
                "type": "bubble",
                "size": "kilo",
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": "#07162E",
                    "paddingAll": "18px",
                    "contents": [
                        {
                            "type": "text",
                            "text": truncate_text(menu.title, 80),
                            "color": "#FFFFFF",
                            "weight": "bold",
                            "size": "lg",
                            "wrap": True,
                            "maxLines": 2,
                        },
                        {
                            "type": "text",
                            "text": truncate_text(
                                f"{menu.breadcrumb} · {menu.product_count} รายการ",
                                120,
                            ),
                            "color": "#7FE9FF",
                            "size": "xs",
                            "wrap": True,
                            "maxLines": 2,
                        },
                    ],
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "spacing": "sm",
                    "paddingAll": "14px",
                    "contents": [
                        {
                            "type": "text",
                            "text": (
                                f"{menu.prompt} · หน้า {page_number}/{len(pages)}"
                            ),
                            "size": "xs",
                            "color": _MUTED_COLOR,
                            "wrap": True,
                        },
                        *buttons,
                    ],
                },
            }
        )
    return {
        "type": "flex",
        "altText": truncate_text(
            f"เลือกหมวดสินค้า {menu.breadcrumb}",
            LINE_ALT_TEXT_LIMIT,
        ),
        "quickReply": {"items": default_quick_reply_items(include_refresh=False)},
        "contents": {"type": "carousel", "contents": bubbles},
    }


def build_category_picker_message(menu: CategoryMenu) -> FlexMessage:
    """Convert a category menu to a LINE SDK Flex message."""

    return FlexMessage.from_dict(build_category_picker_payload(menu))


def default_quick_reply_items(*, include_refresh: bool = True) -> list[dict[str, Any]]:
    """Category/NLP shortcuts plus help and optional refresh actions."""

    examples = [
        ("สินค้าทั้งหมด", ALL_CATEGORIES_COMMAND),
        ("เมาส์ Logitech งบ 3k", "หาเมาส์เกมมิ่ง Logitech ราคาไม่เกิน 3,000 บาท"),
        ("หูฟัง Xiaomi", "หาหูฟังแบรนด์ Xiaomi ไม่เกิน 3,000 บาท"),
        ("คีย์บอร์ดพร้อมส่ง", "แนะนำคีย์บอร์ดเกมมิ่ง เอาเฉพาะพร้อมส่ง"),
        ("โปรโมชันล่าสุด", "มีโปรโมชันอะไรบ้าง"),
        ("ช่วยเหลือ", "ช่วยเหลือ"),
    ]
    items: list[dict[str, Any]] = [
        {
            "type": "action",
            "action": {
                "type": "message",
                "label": truncate_text(label, LINE_ACTION_LABEL_LIMIT),
                "text": truncate_text(command, LINE_POSTBACK_DATA_LIMIT),
            },
        }
        for label, command in examples
    ]
    if include_refresh:
        items.append(
            {
                "type": "action",
                "action": {
                    # A message action deliberately re-enters the NLP refresh
                    # intent, which can recover the user's remembered filters.
                    "type": "message",
                    "label": "สุ่มใหม่",
                    "text": "สุ่มใหม่",
                },
            }
        )
    return items[:LINE_QUICK_REPLY_LIMIT]


def build_text_message_payload(
    text: object | None,
    *,
    quick_reply_items: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a bounded text payload with optional Quick Reply dictionaries."""

    safe_text = truncate_text(
        text,
        LINE_TEXT_LIMIT,
        fallback="ขออภัยครับ กรุณาลองใหม่อีกครั้ง",
    )
    payload: dict[str, Any] = {"type": "text", "text": safe_text}
    if quick_reply_items:
        payload["quickReply"] = {
            "items": [dict(item) for item in quick_reply_items[:LINE_QUICK_REPLY_LIMIT]]
        }
    return payload


def text_with_quick_replies(
    text: object | None,
    *,
    include_refresh: bool = True,
    quick_reply_items: Sequence[dict[str, Any]] | None = None,
) -> TextMessage:
    """Wrap friendly text and the standard discovery shortcuts for LINE."""

    items = (
        list(quick_reply_items)
        if quick_reply_items is not None
        else default_quick_reply_items(include_refresh=include_refresh)
    )
    return TextMessage.from_dict(
        build_text_message_payload(text, quick_reply_items=items)
    )


def greeting_message(bot_name: object | None = "MercuMate") -> TextMessage:
    safe_name = truncate_text(bot_name, 40, fallback="MercuMate")
    return text_with_quick_replies(
        f"สวัสดีครับ 👋 ผม {safe_name} เป็นบอตสาธิตเพื่อการศึกษา "
        "(ไม่ใช่ช่องทางทางการ) "
        "ช่วยค้นหาสินค้า Mercular จากชื่อ หมวดหมู่ แบรนด์ "
        "งบประมาณ และสถานะพร้อมส่งได้ ลองแตะตัวอย่างด้านล่างหรือพิมพ์สิ่งที่หาได้เลยครับ",
        include_refresh=False,
    )


def help_message() -> TextMessage:
    return text_with_quick_replies(
        "วิธีค้นหา 🔎\n"
        "• เลือกหมวดและหมวดย่อยจาก Rich Menu ได้โดยไม่ต้องพิมพ์\n"
        "• ระบุสินค้า/หมวดหมู่ เช่น หูฟัง หรือ earbuds\n"
        "• เพิ่มแบรนด์และงบ เช่น Xiaomi ไม่เกิน 3,000 บาท\n"
        "• เพิ่มสี คุณสมบัติ ‘พร้อมส่ง’ หรือแบรนด์ที่ไม่เอาได้\n"
        "• พิมพ์ชื่อ 2 รุ่นเพื่อเปรียบเทียบ หรือถามสเปกหลังแตะ ‘ดูรายละเอียด’\n"
        "• พิมพ์ ‘ขอถูกกว่านี้’ เพื่อใช้เงื่อนไขเดิม หรือดูโปรโมชันล่าสุดได้\n"
        "ตัวอย่าง: หาเมาส์เกมมิ่ง Logitech งบ 3k พร้อมส่ง",
    )


def no_results_message(query: object | None = None) -> TextMessage:
    cleaned_query = truncate_text(query, 120)
    subject = f"สำหรับ “{cleaned_query}”" if cleaned_query else "ตามเงื่อนไขนี้"
    return text_with_quick_replies(
        f"ยังไม่พบสินค้า{subject}ครับ 😅 ลองเพิ่มงบ ตัดบางเงื่อนไข "
        "หรือเลือกตัวอย่างค้นหาด้านล่างได้เลย",
        include_refresh=False,
    )


def data_unavailable_message() -> TextMessage:
    return text_with_quick_replies(
        "ขออภัยครับ ตอนนี้ยังอ่านข้อมูลสินค้า Mercular ล่าสุดไม่ได้ 🛠️ "
        "ระบบยังไม่เดาราคาหรือสต็อกให้ กรุณาลองอีกครั้งในภายหลัง",
        include_refresh=False,
    )


def promotion_unavailable_message() -> TextMessage:
    return text_with_quick_replies(
        "ยังไม่มีข้อมูลโปรโมชันล่าสุดครับ "
        "ตรวจสอบโปรโมชันปัจจุบันได้ที่เว็บไซต์ Mercular โดยตรง",
        include_refresh=False,
    )


def contact_message(contact_url: object | None = MERCULAR_CONTACT_URL) -> TextMessage:
    safe_url = normalize_https_url(contact_url, mercular_only=True) or MERCULAR_HOME_URL
    return text_with_quick_replies(
        "หากต้องการสอบถามการสั่งซื้อ การชำระเงิน การจัดส่ง หรือการรับประกัน "
        f"กรุณาติดต่อ Mercular ผ่านช่องทางบนเว็บไซต์ทางการครับ\n{safe_url}",
        include_refresh=False,
    )


def disclaimer_message() -> TextMessage:
    return text_with_quick_replies(
        "หมายเหตุ: แชตบอตนี้เป็นโครงงานเพื่อการศึกษาและไม่ใช่ช่องทางทางการของ "
        "Mercular ราคา โปรโมชัน และสต็อกอาจเปลี่ยนแปลงได้ "
        "โปรดยืนยันบนหน้าสินค้าก่อนซื้อ และอย่าส่งข้อมูลบัตรหรือรหัสผ่านในแชตครับ",
        include_refresh=False,
    )


def refresh_message() -> TextMessage:
    return text_with_quick_replies(
        "แตะ ‘สุ่มใหม่’ เพื่อดูสินค้าอีกชุดที่ยังตรงกับเงื่อนไขล่าสุดครับ 🔄",
    )


def product_detail_payload(product: Product) -> dict[str, Any]:
    """Build a readable, bounded text detail for one catalog product."""

    if not isinstance(product, Product):
        raise TypeError("product must be a Product")
    brand = truncate_text(product.brand, 120, fallback="ไม่ระบุแบรนด์")
    category = truncate_text(product.category, 160, fallback="ไม่ระบุหมวดหมู่")
    description = truncate_text(product.description, 1_500)
    stock = (
        "พร้อมส่ง"
        if product.in_stock is True
        else "สินค้าหมดชั่วคราว"
        if product.in_stock is False
        else "ไม่ทราบ — โปรดตรวจสอบที่เว็บไซต์"
    )
    lines = [
        f"🛍️ {truncate_text(product.name, 250)}",
        f"🏷️ แบรนด์: {brand}",
        f"📂 หมวดหมู่: {category}",
        f"💰 ราคา: {truncate_text(product.display_price, 80)}",
        f"📦 สถานะ: {stock}",
    ]
    if description:
        lines.extend(("", f"รายละเอียด: {description}"))
    if product.overview:
        lines.extend(("", f"ภาพรวม: {truncate_text(product.overview, 600)}"))
    if product.rating is not None:
        review_suffix = (
            f" จาก {product.review_count:,} รีวิว"
            if product.review_count is not None
            else ""
        )
        lines.append(f"⭐ คะแนน: {product.rating:.1f}/5{review_suffix}")
    if product.highlights:
        lines.extend(
            (
                "",
                "คุณสมบัติเด่น:",
                *(
                    f"• {truncate_text(highlight, 220)}"
                    for highlight in product.highlights[:4]
                ),
            )
        )
    if product.specifications:
        lines.extend(
            (
                "",
                "สเปก:",
                *(
                    f"• {truncate_text(name, 120)}: {truncate_text(value, 220)}"
                    for name, value in product.specifications[:8]
                ),
            )
        )
    if product.warranty:
        lines.append(f"🛡️ ประกัน: {truncate_text(product.warranty, 500)}")
    product_url = normalize_https_url(product.product_url, mercular_only=True)
    if product_url:
        lines.extend(("", f"ดู/ซื้อสินค้าที่ Mercular: {product_url}"))
    lines.extend(("", "ราคาและสต็อกอาจเปลี่ยนแปลง โปรดยืนยันบนเว็บไซต์ก่อนซื้อ"))
    return build_text_message_payload(
        "\n".join(lines),
        quick_reply_items=default_quick_reply_items(),
    )


def product_detail_message(product: Product) -> TextMessage:
    """Return one SDK ``TextMessage`` suitable for a detail postback reply."""

    return TextMessage.from_dict(product_detail_payload(product))


def product_comparison_message(first: Product, second: Product) -> TextMessage:
    """Compare two catalog records without inferring missing specifications."""

    if not isinstance(first, Product) or not isinstance(second, Product):
        raise TypeError("comparison requires two Product values")

    def stock(product: Product) -> str:
        if product.in_stock is True:
            return "พร้อมส่ง"
        if product.in_stock is False:
            return "สินค้าหมด"
        return "ไม่ทราบ"

    lines = [
        "🔍 เปรียบเทียบจากข้อมูลล่าสุด",
        "",
        f"A: {truncate_text(first.name, 220)}",
        f"• ราคา {first.display_price} | {stock(first)}",
        f"B: {truncate_text(second.name, 220)}",
        f"• ราคา {second.display_price} | {stock(second)}",
    ]
    if first.price is not None and second.price is not None and first.price != second.price:
        cheaper = first if first.price < second.price else second
        lines.extend(
            (
                "",
                f"💰 {truncate_text(cheaper.name, 120)} ถูกกว่า "
                f"฿{abs(first.price - second.price):,.0f}",
            )
        )

    first_specs = {
        normalize_text(name): (name, value)
        for name, value in first.specifications
        if name and value
    }
    second_specs = {
        normalize_text(name): (name, value)
        for name, value in second.specifications
        if name and value
    }
    differences = []
    for key in first_specs.keys() & second_specs.keys():
        name, first_value = first_specs[key]
        second_value = second_specs[key][1]
        if normalize_text(first_value) != normalize_text(second_value):
            differences.append((name, first_value, second_value))
    if differences:
        lines.extend(("", "สเปกที่ต่างกัน:"))
        for name, first_value, second_value in differences[:8]:
            lines.append(
                f"• {truncate_text(name, 90)}: A {truncate_text(first_value, 140)} | "
                f"B {truncate_text(second_value, 140)}"
            )
    else:
        lines.extend(
            ("", "ยังไม่มีสเปกชื่อเดียวกันมากพอสำหรับสรุปความต่างเพิ่มเติมครับ")
        )
    lines.extend(
        (
            "",
            f"A: {first.product_url}",
            f"B: {second.product_url}",
            "ราคา สต็อก และโปรโมชันอาจเปลี่ยนแปลง โปรดตรวจสอบบนเว็บไซต์ก่อนซื้อ",
        )
    )
    return text_with_quick_replies(truncate_text("\n".join(lines), LINE_TEXT_LIMIT))


def payload_is_json_serializable(payload: dict[str, Any]) -> bool:
    """Small diagnostic helper used by health checks and offline tests."""

    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


__all__ = [
    "MAX_CAROUSEL_PRODUCTS",
    "PRODUCT_DETAIL_ACTION",
    "REFRESH_ACTION",
    "build_postback_data",
    "build_category_picker_message",
    "build_category_picker_payload",
    "build_product_bubble_payload",
    "build_product_carousel",
    "build_product_carousel_message",
    "build_product_carousel_payload",
    "build_promotion_bubble_payload",
    "build_promotion_carousel_message",
    "build_text_message_payload",
    "contact_message",
    "data_unavailable_message",
    "default_quick_reply_items",
    "disclaimer_message",
    "greeting_message",
    "help_message",
    "is_refresh_postback",
    "is_safe_https_url",
    "no_results_message",
    "normalize_https_url",
    "parse_postback_data",
    "parse_product_postback",
    "payload_is_json_serializable",
    "product_carousel_message",
    "product_detail_message",
    "product_detail_payload",
    "product_comparison_message",
    "promotion_unavailable_message",
    "refresh_message",
    "text_with_quick_replies",
    "truncate_text",
]
