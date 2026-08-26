"""Offline contracts for MercuMate's LINE Message Lab."""

from __future__ import annotations

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

from message_showcase import (
    DEMO_AUDIO_DURATION_MS,
    SHOWCASE_TYPES,
    build_showcase_message,
    parse_showcase_command,
    showcase_command,
    showcase_hub_message,
)


PUBLIC_BASE = "https://mercumate.example"


def test_showcase_hub_exposes_every_requested_type_once() -> None:
    message = showcase_hub_message()

    assert isinstance(message, FlexMessage)
    payload = message.to_dict()
    bubbles = payload["contents"]["contents"]
    commands = [
        button["action"]["text"]
        for bubble in bubbles
        for button in bubble["footer"]["contents"]
    ]
    assert commands == [showcase_command(kind) for kind in SHOWCASE_TYPES]
    assert len(bubbles) == 3


def test_explicit_showcase_commands_do_not_capture_normal_nlp_text() -> None:
    assert parse_showcase_command("เดโมข้อความ") == "hub"
    assert parse_showcase_command("เดโม:text v2") == "text-v2"
    assert parse_showcase_command("เดโม:รูปภาพ") == "image"
    assert parse_showcase_command("หาเมาส์เกมมิ่ง") is None
    assert parse_showcase_command("เดโม:unknown") is None


def test_every_line_message_model_is_constructed_with_required_fields() -> None:
    expected_types = {
        "text": TextMessage,
        "text-v2": TextMessageV2,
        "sticker": StickerMessage,
        "image": ImageMessage,
        "video": VideoMessage,
        "audio": AudioMessage,
        "location": LocationMessage,
        "coupon": CouponMessage,
        "imagemap": ImagemapMessage,
        "template": TemplateMessage,
        "flex": FlexMessage,
    }

    for kind, expected in expected_types.items():
        message = build_showcase_message(
            kind,
            public_base_url=PUBLIC_BASE,
            coupon_id="coupon-real-id",
        )
        assert isinstance(message, expected), kind
        payload = message.to_dict()
        assert payload["type"]


def test_media_messages_use_public_https_assets_and_real_duration() -> None:
    image = build_showcase_message("image", public_base_url=PUBLIC_BASE)
    video = build_showcase_message("video", public_base_url=PUBLIC_BASE)
    audio = build_showcase_message("audio", public_base_url=PUBLIC_BASE)

    assert image.original_content_url.startswith(f"{PUBLIC_BASE}/")
    assert image.preview_image_url.startswith(f"{PUBLIC_BASE}/")
    assert video.original_content_url.endswith(".mp4")
    assert video.preview_image_url.endswith(".jpg")
    assert audio.original_content_url.endswith(".mp3")
    assert audio.duration == DEMO_AUDIO_DURATION_MS > 0


def test_missing_public_origin_returns_honest_text_instead_of_invalid_media() -> None:
    for kind in ("image", "video", "audio", "imagemap", "template"):
        message = build_showcase_message(kind, public_base_url="http://localhost:5000")
        assert isinstance(message, TextMessage)
        assert "PUBLIC_BASE_URL" in message.text


def test_location_is_explicitly_a_demo_and_coordinates_are_valid() -> None:
    message = build_showcase_message("location")

    assert isinstance(message, LocationMessage)
    assert "Demo" in message.title
    assert -90 <= message.latitude <= 90
    assert -180 <= message.longitude <= 180
    assert "ไม่ใช่หน้าร้าน" in message.address


def test_coupon_requires_real_id_and_never_invents_a_discount() -> None:
    fallback = build_showcase_message("coupon", coupon_id="")
    real = build_showcase_message("coupon", coupon_id="coupon-real-id")

    assert isinstance(fallback, FlexMessage)
    fallback_payload = fallback.to_dict()
    assert "ไม่มีส่วนลดจริง" in str(fallback_payload)
    assert isinstance(real, CouponMessage)
    assert real.coupon_id == "coupon-real-id"


def test_imagemap_covers_1040_by_520_without_gaps_or_overlap() -> None:
    message = build_showcase_message("imagemap", public_base_url=PUBLIC_BASE)
    payload = message.to_dict()

    assert payload["baseUrl"] == f"{PUBLIC_BASE}/media/imagemap/mercumate"
    assert payload["baseSize"] == {"width": 1040, "height": 520}
    areas = [action["area"] for action in payload["actions"]]
    assert [(area["x"], area["width"]) for area in areas] == [
        (0, 347),
        (347, 346),
        (693, 347),
    ]
    assert all(area["y"] == 0 and area["height"] == 520 for area in areas)


def test_text_v2_uses_line_emoji_substitution() -> None:
    payload = build_showcase_message("text-v2").to_dict()

    assert payload["type"] == "textV2"
    assert "{sparkle}" in payload["text"]
    assert payload["substitution"]["sparkle"]["type"] == "emoji"

