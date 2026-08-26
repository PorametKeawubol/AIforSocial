"""Branded LINE message-type showcase for MercuMate.

The shopping flow remains the primary UX.  This module provides an explicit
"Message Lab" reached from the Rich Menu so every LINE message object can be
demonstrated one at a time without exceeding LINE's five-message reply limit.
"""

from __future__ import annotations

from collections.abc import Iterable

from linebot.v3.messaging import (
    AudioMessage,
    CouponMessage,
    FlexMessage,
    ImageMessage,
    ImagemapMessage,
    LocationMessage,
    StickerMessage,
    TemplateMessage,
    TextMessage,
    TextMessageV2,
    VideoMessage,
)

try:  # Support direct ``python app.py`` execution and package imports.
    from .line_views import normalize_https_url
except ImportError:  # pragma: no cover - direct execution path
    from line_views import normalize_https_url


SHOWCASE_PREFIX = "เดโม:"
SHOWCASE_TYPES = (
    "text",
    "text-v2",
    "sticker",
    "image",
    "video",
    "audio",
    "location",
    "coupon",
    "imagemap",
    "template",
    "flex",
)
IMAGEMAP_SIZES = (240, 300, 460, 700, 1040)
DEMO_AUDIO_DURATION_MS = 5_095

_NAVY = "#06152E"
_NAVY_LIGHT = "#0B244A"
_CYAN = "#35D7FF"
_BLUE = "#1677FF"
_WHITE = "#FFFFFF"
_MUTED = "#A9C7E8"


def _public_base(value: object | None) -> str:
    """Return a normalized public HTTPS origin without a trailing slash."""

    normalized = normalize_https_url(value, max_length=900)
    return normalized.rstrip("/") if normalized else ""


def _media_url(public_base_url: object | None, path: str) -> str:
    base = _public_base(public_base_url)
    if not base:
        return ""
    return f"{base}/{path.lstrip('/')}"


def showcase_command(kind: str) -> str:
    if kind not in SHOWCASE_TYPES:
        raise ValueError(f"unsupported showcase type: {kind}")
    return f"{SHOWCASE_PREFIX}{kind}"


def parse_showcase_command(text: object | None) -> str | None:
    """Parse only explicit Message Lab commands, leaving normal NLP untouched."""

    normalized = " ".join(str(text or "").strip().casefold().split())
    if normalized in {"เดโมข้อความ", "message lab", "message demo"}:
        return "hub"
    aliases = {
        "textv2": "text-v2",
        "text v2": "text-v2",
        "รูปภาพ": "image",
        "วิดีโอ": "video",
        "เสียง": "audio",
        "ตำแหน่ง": "location",
        "คูปอง": "coupon",
    }
    if normalized.startswith(SHOWCASE_PREFIX):
        requested = normalized[len(SHOWCASE_PREFIX) :].strip()
        requested = aliases.get(requested, requested)
        return requested if requested in SHOWCASE_TYPES else None
    return None


def _message_action(label: str, kind: str) -> dict[str, str]:
    return {
        "type": "message",
        "label": label,
        "text": showcase_command(kind),
    }


def _hub_bubble(
    eyebrow: str,
    title: str,
    description: str,
    actions: Iterable[tuple[str, str]],
    *,
    accent: str,
) -> dict[str, object]:
    buttons = [
        {
            "type": "button",
            "style": "primary",
            "height": "sm",
            "color": accent,
            "margin": "sm",
            "action": _message_action(label, kind),
        }
        for label, kind in actions
    ]
    return {
        "type": "bubble",
        "size": "kilo",
        "styles": {
            "header": {"backgroundColor": _NAVY_LIGHT},
            "body": {"backgroundColor": _NAVY},
            "footer": {"backgroundColor": _NAVY},
        },
        "header": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "18px",
            "contents": [
                {
                    "type": "text",
                    "text": eyebrow,
                    "size": "xs",
                    "weight": "bold",
                    "color": _CYAN,
                },
                {
                    "type": "text",
                    "text": title,
                    "size": "xl",
                    "weight": "bold",
                    "color": _WHITE,
                    "margin": "sm",
                    "wrap": True,
                },
            ],
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "18px",
            "contents": [
                {
                    "type": "text",
                    "text": description,
                    "size": "sm",
                    "color": _MUTED,
                    "wrap": True,
                }
            ],
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "14px",
            "contents": buttons,
        },
    }


def showcase_hub_message() -> FlexMessage:
    """Return a carousel that exposes all requested message objects."""

    payload = {
        "type": "flex",
        "altText": "MercuMate Message Lab — เลือกชนิดข้อความที่ต้องการทดลอง",
        "contents": {
            "type": "carousel",
            "contents": [
                _hub_bubble(
                    "01 / BASIC",
                    "ข้อความพื้นฐาน",
                    "ทดลองข้อความ ตัวแปร Emoji สติกเกอร์ และภาพแบรนด์",
                    (
                        ("Text", "text"),
                        ("Text v2", "text-v2"),
                        ("Sticker", "sticker"),
                        ("Image", "image"),
                    ),
                    accent=_BLUE,
                ),
                _hub_bubble(
                    "02 / MEDIA",
                    "สื่อและตำแหน่ง",
                    "ทดลองวิดีโอ เสียง และ Location message แบบปลอดภัย",
                    (
                        ("Video", "video"),
                        ("Audio", "audio"),
                        ("Location", "location"),
                    ),
                    accent="#086DD7",
                ),
                _hub_bubble(
                    "03 / INTERACTIVE",
                    "ข้อความโต้ตอบ",
                    "ทดลองคูปอง Imagemap Template และ Flex Message",
                    (
                        ("Coupon", "coupon"),
                        ("Imagemap", "imagemap"),
                        ("Template", "template"),
                        ("Flex", "flex"),
                    ),
                    accent="#0058B8",
                ),
            ],
        },
    }
    return FlexMessage.from_dict(payload)


def _missing_media_message() -> TextMessage:
    return TextMessage(
        text=(
            "ยังเปิดเดโมสื่อนี้ไม่ได้ เพราะ PUBLIC_BASE_URL ต้องเป็น HTTPS "
            "ที่ LINE เข้าถึงได้ กรุณาเปิด tunnel แล้วตั้งค่า URL ก่อนครับ"
        )
    )


def _coupon_demo_fallback() -> FlexMessage:
    return FlexMessage.from_dict(
        {
            "type": "flex",
            "altText": "ตัวอย่าง Coupon Message ของ MercuMate",
            "contents": {
                "type": "bubble",
                "size": "kilo",
                "styles": {
                    "header": {"backgroundColor": _NAVY_LIGHT},
                    "body": {"backgroundColor": _NAVY},
                    "footer": {"backgroundColor": _NAVY},
                },
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "paddingAll": "20px",
                    "contents": [
                        {
                            "type": "text",
                            "text": "COUPON PREVIEW",
                            "size": "xs",
                            "weight": "bold",
                            "color": _CYAN,
                        },
                        {
                            "type": "text",
                            "text": "MercuMate Demo",
                            "size": "xl",
                            "weight": "bold",
                            "color": _WHITE,
                            "margin": "sm",
                        },
                    ],
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "paddingAll": "20px",
                    "contents": [
                        {
                            "type": "text",
                            "text": "ไม่มีส่วนลดจริง",
                            "size": "xxl",
                            "weight": "bold",
                            "color": _CYAN,
                            "wrap": True,
                        },
                        {
                            "type": "text",
                            "text": (
                                "Coupon Message จริงต้องสร้าง Coupon ใน LINE OA "
                                "Manager และใส่ LINE_COUPON_ID ก่อน"
                            ),
                            "size": "sm",
                            "color": _MUTED,
                            "wrap": True,
                            "margin": "md",
                        },
                    ],
                },
                "footer": {
                    "type": "box",
                    "layout": "vertical",
                    "paddingAll": "16px",
                    "contents": [
                        {
                            "type": "button",
                            "style": "secondary",
                            "height": "sm",
                            "action": {
                                "type": "message",
                                "label": "กลับ Message Lab",
                                "text": "เดโมข้อความ",
                            },
                        }
                    ],
                },
            },
        }
    )


def _flex_demo(public_base_url: str) -> FlexMessage:
    hero_url = _media_url(public_base_url, "static/media/mercumate-cover.jpg")
    bubble: dict[str, object] = {
        "type": "bubble",
        "size": "mega",
        "styles": {
            "body": {"backgroundColor": _NAVY},
            "footer": {"backgroundColor": _NAVY},
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "22px",
            "contents": [
                {
                    "type": "text",
                    "text": "MERCUMATE FLEX",
                    "size": "xs",
                    "weight": "bold",
                    "color": _CYAN,
                },
                {
                    "type": "text",
                    "text": "Your Gear Assistant",
                    "size": "xl",
                    "weight": "bold",
                    "color": _WHITE,
                    "margin": "sm",
                },
                {
                    "type": "text",
                    "text": "Flex Message ปรับสี เลย์เอาต์ รูป และ Action ได้อิสระ",
                    "size": "sm",
                    "color": _MUTED,
                    "wrap": True,
                    "margin": "md",
                },
            ],
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "16px",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "height": "sm",
                    "color": _BLUE,
                    "action": {
                        "type": "message",
                        "label": "ค้นหาสินค้า",
                        "text": "มีสินค้าอะไรแนะนำบ้าง",
                    },
                },
                {
                    "type": "button",
                    "style": "secondary",
                    "height": "sm",
                    "margin": "sm",
                    "action": {
                        "type": "message",
                        "label": "กลับ Message Lab",
                        "text": "เดโมข้อความ",
                    },
                },
            ],
        },
    }
    if hero_url:
        bubble = {
            "type": "bubble",
            "size": "mega",
            "hero": {
                "type": "image",
                "url": hero_url,
                "size": "full",
                "aspectRatio": "8:3",
                "aspectMode": "cover",
            },
            **{key: value for key, value in bubble.items() if key not in {"type", "size"}},
        }
    return FlexMessage.from_dict(
        {
            "type": "flex",
            "altText": "ตัวอย่าง MercuMate Flex Message",
            "contents": bubble,
        }
    )


def build_showcase_message(
    kind: str,
    *,
    public_base_url: object | None = None,
    coupon_id: object | None = None,
) -> object:
    """Build one requested LINE message object with honest fallbacks."""

    if kind not in SHOWCASE_TYPES:
        raise ValueError(f"unsupported showcase type: {kind}")

    base = _public_base(public_base_url)
    if kind == "text":
        return TextMessage.from_dict(
            {
                "type": "text",
                "text": (
                    "💬 Text Message\n"
                    "MercuMate ช่วยค้นหา Gaming, Gadget และ IT & Tech "
                    "จากภาษาธรรมชาติได้ครับ"
                ),
                "quickReply": {
                    "items": [
                        {
                            "type": "action",
                            "action": {
                                "type": "message",
                                "label": "ลองค้นหา",
                                "text": "หาเมาส์เกมมิ่ง ไม่เกิน 3000 พร้อมส่ง",
                            },
                        },
                        {
                            "type": "action",
                            "action": {
                                "type": "message",
                                "label": "Message Lab",
                                "text": "เดโมข้อความ",
                            },
                        },
                    ]
                },
            }
        )
    if kind == "text-v2":
        return TextMessageV2.from_dict(
            {
                "type": "textV2",
                "text": "{sparkle} Text Message v2\nรองรับ substitution และ LINE Emoji แล้วครับ",
                "substitution": {
                    "sparkle": {
                        "type": "emoji",
                        "productId": "5ac1bfd5040ab15980c9b435",
                        "emojiId": "001",
                    }
                },
            }
        )
    if kind == "sticker":
        return StickerMessage(package_id="446", sticker_id="1988")
    if kind == "location":
        return LocationMessage(
            title="ตำแหน่งตัวอย่าง (Demo)",
            address="กรุงเทพมหานคร — ไม่ใช่หน้าร้าน Mercular",
            latitude=13.7563,
            longitude=100.5018,
        )
    if kind == "coupon":
        safe_coupon_id = str(coupon_id or "").strip()
        return (
            CouponMessage(coupon_id=safe_coupon_id, delivery_tag="mercumate-demo")
            if safe_coupon_id
            else _coupon_demo_fallback()
        )
    if kind == "flex":
        return _flex_demo(base)

    if not base:
        return _missing_media_message()
    if kind == "image":
        return ImageMessage(
            original_content_url=_media_url(base, "static/media/mercumate-cover.jpg"),
            preview_image_url=_media_url(
                base, "static/media/mercumate-cover-preview.jpg"
            ),
        )
    if kind == "video":
        return VideoMessage(
            original_content_url=_media_url(base, "static/media/mercumate-intro.mp4"),
            preview_image_url=_media_url(
                base, "static/media/mercumate-video-preview.jpg"
            ),
            tracking_id="mercumate-intro-v1",
        )
    if kind == "audio":
        return AudioMessage(
            original_content_url=_media_url(base, "static/media/mercumate-chime.mp3"),
            duration=DEMO_AUDIO_DURATION_MS,
        )
    if kind == "imagemap":
        return ImagemapMessage.from_dict(
            {
                "type": "imagemap",
                "baseUrl": _media_url(base, "media/imagemap/mercumate"),
                "altText": "MercuMate Imagemap — เลือก Gaming, Audio หรือ Gadget",
                "baseSize": {"width": 1040, "height": 520},
                "actions": [
                    {
                        "type": "message",
                        "text": "หาอุปกรณ์เกมมิ่ง ไม่เกิน 3000 พร้อมส่ง",
                        "area": {"x": 0, "y": 0, "width": 347, "height": 520},
                    },
                    {
                        "type": "message",
                        "text": "หาหูฟังไร้สาย ไม่เกิน 3000",
                        "area": {"x": 347, "y": 0, "width": 346, "height": 520},
                    },
                    {
                        "type": "message",
                        "text": "แนะนำอุปกรณ์คอมพิวเตอร์ ไม่เกิน 3000",
                        "area": {"x": 693, "y": 0, "width": 347, "height": 520},
                    },
                ],
            }
        )
    if kind == "template":
        return TemplateMessage.from_dict(
            {
                "type": "template",
                "altText": "MercuMate Template Message — เลือกหมวดสินค้า",
                "template": {
                    "type": "buttons",
                    "thumbnailImageUrl": _media_url(
                        base, "static/media/mercumate-cover-preview.jpg"
                    ),
                    "imageAspectRatio": "rectangle",
                    "imageSize": "cover",
                    "title": "MercuMate Gear Finder",
                    "text": "เลือกหมวดที่สนใจ แล้วให้ NLP ค้นหาสินค้า",
                    "actions": [
                        {
                            "type": "message",
                            "label": "Gaming",
                            "text": "หาอุปกรณ์เกมมิ่ง ไม่เกิน 3000 พร้อมส่ง",
                        },
                        {
                            "type": "message",
                            "label": "Audio",
                            "text": "หาหูฟังไร้สาย ไม่เกิน 3000",
                        },
                        {
                            "type": "message",
                            "label": "Gadget",
                            "text": "แนะนำอุปกรณ์คอมพิวเตอร์ ไม่เกิน 3000",
                        },
                    ],
                },
            }
        )
    raise AssertionError(f"unhandled showcase type: {kind}")


__all__ = [
    "DEMO_AUDIO_DURATION_MS",
    "IMAGEMAP_SIZES",
    "SHOWCASE_TYPES",
    "build_showcase_message",
    "parse_showcase_command",
    "showcase_command",
    "showcase_hub_message",
]
