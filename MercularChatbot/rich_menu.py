"""MercuMate Rich Menu definition shared by deployment and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from linebot.v3.messaging import RichMenuRequest


PROJECT_DIR = Path(__file__).resolve().parent
RICH_MENU_IMAGE_PATH = (
    PROJECT_DIR / "assets" / "rich_menu" / "mercumate-rich-menu-v1.jpg"
)
RICH_MENU_NAME = "MercuMate Main Menu v1"
RICH_MENU_CHAT_BAR_TEXT = "เมนู MercuMate"
RICH_MENU_WIDTH = 2_500
RICH_MENU_HEIGHT = 1_686
RICH_MENU_MAX_IMAGE_BYTES = 1_000_000

# Bounds follow the six luminous panels in the generated artwork.  The hero
# mascot/header remains decorative rather than pretending to be a seventh tap
# target.  Small gutters prevent adjacent actions from overlapping.
_PANELS = (
    (90, 435, 760, 540),
    (870, 435, 760, 540),
    (1_650, 435, 760, 540),
    (90, 995, 760, 545),
    (870, 995, 760, 545),
    (1_650, 995, 760, 545),
)

_ACTIONS = (
    ("ค้นหาสินค้า", "มีสินค้าอะไรแนะนำบ้าง"),
    ("เกมมิ่ง", "หาเมาส์เกมมิ่ง ไม่เกิน 3000 พร้อมส่ง"),
    ("ออดิโอ", "หาหูฟังไร้สาย ไม่เกิน 3000"),
    ("แก็ดเจ็ต", "แนะนำอุปกรณ์คอมพิวเตอร์ ไม่เกิน 3000"),
    ("เดโมข้อความ", "เดโมข้อความ"),
    ("ช่วยเหลือ", "ช่วยเหลือ"),
)


def build_rich_menu_payload() -> dict[str, Any]:
    """Return the exact Rich Menu object sent to LINE."""

    areas = []
    for (x, y, width, height), (label, text) in zip(
        _PANELS, _ACTIONS, strict=True
    ):
        areas.append(
            {
                "bounds": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                },
                "action": {
                    "type": "message",
                    "label": label,
                    "text": text,
                },
            }
        )
    return {
        "size": {"width": RICH_MENU_WIDTH, "height": RICH_MENU_HEIGHT},
        "selected": True,
        "name": RICH_MENU_NAME,
        "chatBarText": RICH_MENU_CHAT_BAR_TEXT,
        "areas": areas,
    }


def build_rich_menu_request() -> RichMenuRequest:
    return RichMenuRequest.from_dict(build_rich_menu_payload())


def validate_local_rich_menu_image(path: Path = RICH_MENU_IMAGE_PATH) -> None:
    """Fail early for mistakes LINE would reject before any account mutation."""

    if not path.is_file():
        raise FileNotFoundError(f"Rich Menu image not found: {path}")
    if path.suffix.casefold() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Rich Menu image must be JPEG or PNG")
    size = path.stat().st_size
    if size <= 0 or size > RICH_MENU_MAX_IMAGE_BYTES:
        raise ValueError(
            f"Rich Menu image must be 1-{RICH_MENU_MAX_IMAGE_BYTES} bytes; got {size}"
        )


__all__ = [
    "RICH_MENU_CHAT_BAR_TEXT",
    "RICH_MENU_HEIGHT",
    "RICH_MENU_IMAGE_PATH",
    "RICH_MENU_MAX_IMAGE_BYTES",
    "RICH_MENU_NAME",
    "RICH_MENU_WIDTH",
    "build_rich_menu_payload",
    "build_rich_menu_request",
    "validate_local_rich_menu_image",
]
