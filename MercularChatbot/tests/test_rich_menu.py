"""Rich Menu geometry and asset constraints."""

from __future__ import annotations

from rich_menu import (
    RICH_MENU_ACTIONS,
    RICH_MENU_HEIGHT,
    RICH_MENU_IMAGE_PATH,
    RICH_MENU_MAX_IMAGE_BYTES,
    RICH_MENU_NAME,
    RICH_MENU_WIDTH,
    build_rich_menu_payload,
    build_rich_menu_request,
    validate_local_rich_menu_image,
)
from catalog_navigation import (
    COMPUTER_ROOT,
    GAMING_ROOT,
    MOBILE_ROOT,
    parse_category_navigation,
)
from nlp import INTENT_HELP, INTENT_PROMOTION, ThaiCommandParser


def _overlaps(left: dict[str, int], right: dict[str, int]) -> bool:
    return not (
        left["x"] + left["width"] <= right["x"]
        or right["x"] + right["width"] <= left["x"]
        or left["y"] + left["height"] <= right["y"]
        or right["y"] + right["height"] <= left["y"]
    )


def test_rich_menu_request_is_sdk_serializable_with_six_actions() -> None:
    request = build_rich_menu_request()
    payload = request.to_dict()

    assert payload["size"] == {"width": 2500, "height": 1686}
    assert payload["selected"] is True
    assert len(payload["chatBarText"]) <= 14
    assert len(payload["areas"]) == 6
    assert all(area["action"]["type"] == "message" for area in payload["areas"])
    assert RICH_MENU_NAME.endswith("v3")


def test_every_visible_action_sends_a_unique_supported_command() -> None:
    parser = ThaiCommandParser()
    expected_paths = {
        "all_products": (),
        "gaming": (GAMING_ROOT,),
        "computer": (COMPUTER_ROOT,),
        "mobile": (MOBILE_ROOT,),
    }

    assert [item.key for item in RICH_MENU_ACTIONS[:4]] == list(expected_paths)
    assert len({item.message for item in RICH_MENU_ACTIONS}) == len(RICH_MENU_ACTIONS)
    for item in RICH_MENU_ACTIONS[:4]:
        request = parse_category_navigation(item.message)
        assert request is not None
        assert request.path == expected_paths[item.key]
        assert request.show_products is False
    assert parser.parse(RICH_MENU_ACTIONS[4].message).intent == INTENT_PROMOTION
    assert parser.parse(RICH_MENU_ACTIONS[5].message).intent == INTENT_HELP
    assert [item.label for item in RICH_MENU_ACTIONS] == [
        "สินค้าทั้งหมด",
        "เกมมิ่ง",
        "คอมพิวเตอร์",
        "มือถือ/แท็บเล็ต",
        "โปรโมชัน",
        "ช่วยเหลือ",
    ]


def test_tap_areas_match_visible_panels_and_never_overlap() -> None:
    areas = [area["bounds"] for area in build_rich_menu_payload()["areas"]]

    for area in areas:
        assert area["x"] >= 0 and area["y"] >= 0
        assert area["x"] + area["width"] <= RICH_MENU_WIDTH
        assert area["y"] + area["height"] <= RICH_MENU_HEIGHT
    for index, area in enumerate(areas):
        assert all(not _overlaps(area, other) for other in areas[index + 1 :])


def test_final_rich_menu_asset_is_jpeg_and_below_line_limit() -> None:
    validate_local_rich_menu_image()

    assert RICH_MENU_IMAGE_PATH.read_bytes()[:3] == b"\xff\xd8\xff"
    assert 0 < RICH_MENU_IMAGE_PATH.stat().st_size <= RICH_MENU_MAX_IMAGE_BYTES
