"""MercuMate Rich Menu definition shared by deployment and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from linebot.v3.messaging import RichMenuRequest

try:  # Support package imports and direct script execution.
    from .catalog_navigation import (
        ALL_CATEGORIES_COMMAND,
        COMPUTER_CATEGORIES_COMMAND,
        GAMING_CATEGORIES_COMMAND,
        MOBILE_CATEGORIES_COMMAND,
    )
except ImportError:  # pragma: no cover
    from catalog_navigation import (
        ALL_CATEGORIES_COMMAND,
        COMPUTER_CATEGORIES_COMMAND,
        GAMING_CATEGORIES_COMMAND,
        MOBILE_CATEGORIES_COMMAND,
    )


PROJECT_DIR = Path(__file__).resolve().parent
RICH_MENU_IMAGE_PATH = (
    PROJECT_DIR / "assets" / "rich_menu" / "mercumate-rich-menu-v3.jpg"
)
RICH_MENU_NAME = "MercuMate Main Menu v3"
RICH_MENU_CHAT_BAR_TEXT = "เมนู MercuMate"
RICH_MENU_WIDTH = 2_500
RICH_MENU_HEIGHT = 1_686
RICH_MENU_MAX_IMAGE_BYTES = 1_000_000


@dataclass(frozen=True, slots=True)
class RichMenuAction:
    """One visible panel and the natural-language command it sends to the bot."""

    key: str
    label: str
    message: str
    bounds: tuple[int, int, int, int]

    def to_area(self) -> dict[str, Any]:
        x, y, width, height = self.bounds
        return {
            "bounds": {"x": x, "y": y, "width": width, "height": height},
            "action": {
                "type": "message",
                "label": self.label,
                "text": self.message,
            },
        }


# Bounds follow the six luminous panels in the artwork. The mascot/header is
# decorative, and the gutters keep adjacent tap targets from overlapping.
RICH_MENU_ACTIONS = (
    RichMenuAction(
        "all_products",
        "สินค้าทั้งหมด",
        ALL_CATEGORIES_COMMAND,
        (90, 435, 760, 540),
    ),
    RichMenuAction(
        "gaming",
        "เกมมิ่ง",
        GAMING_CATEGORIES_COMMAND,
        (870, 435, 760, 540),
    ),
    RichMenuAction(
        "computer",
        "คอมพิวเตอร์",
        COMPUTER_CATEGORIES_COMMAND,
        (1_650, 435, 760, 540),
    ),
    RichMenuAction(
        "mobile",
        "มือถือ/แท็บเล็ต",
        MOBILE_CATEGORIES_COMMAND,
        (90, 995, 760, 545),
    ),
    RichMenuAction(
        "promotion",
        "โปรโมชัน",
        "มีโปรโมชันอะไรบ้าง",
        (870, 995, 760, 545),
    ),
    RichMenuAction(
        "help",
        "ช่วยเหลือ",
        "ช่วยเหลือ",
        (1_650, 995, 760, 545),
    ),
)


def build_rich_menu_payload() -> dict[str, Any]:
    """Return the exact Rich Menu object sent to LINE."""

    return {
        "size": {"width": RICH_MENU_WIDTH, "height": RICH_MENU_HEIGHT},
        "selected": True,
        "name": RICH_MENU_NAME,
        "chatBarText": RICH_MENU_CHAT_BAR_TEXT,
        "areas": [item.to_area() for item in RICH_MENU_ACTIONS],
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
    "RICH_MENU_ACTIONS",
    "RICH_MENU_WIDTH",
    "RichMenuAction",
    "build_rich_menu_payload",
    "build_rich_menu_request",
    "validate_local_rich_menu_image",
]
