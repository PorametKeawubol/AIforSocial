#!/usr/bin/env python3
"""Validate MercuMate category-picker payloads with LINE without sending them."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from linebot.v3.messaging import (  # noqa: E402
    ApiClient,
    Configuration,
    MessagingApi,
    ValidateMessageRequest,
)

from config import Settings  # noqa: E402
from catalog_navigation import (  # noqa: E402
    COMPUTER_ROOT,
    GAMING_ROOT,
    MOBILE_ROOT,
    build_category_menu,
)
from line_views import build_category_picker_message  # noqa: E402
from repository import ProductRepository  # noqa: E402


def main() -> int:
    settings = Settings.from_env()
    if not settings.line_channel_access_token:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN is not configured")

    named_messages = []
    products = ProductRepository(settings.snapshot_path, settings=settings).all()
    for name, path in (
        ("categories-all", ()),
        ("categories-gaming", (GAMING_ROOT,)),
        ("categories-computer", (COMPUTER_ROOT,)),
        ("categories-mobile", (MOBILE_ROOT,)),
    ):
        menu = build_category_menu(products, path)
        if menu is not None and menu.options:
            named_messages.append((name, build_category_picker_message(menu)))

    configuration = Configuration(access_token=settings.line_channel_access_token)
    with ApiClient(configuration) as api_client:
        messaging = MessagingApi(api_client)
        for start in range(0, len(named_messages), 5):
            batch = named_messages[start : start + 5]
            messaging.validate_reply(
                ValidateMessageRequest(messages=[message for _name, message in batch])
            )
            print("Validated: " + ", ".join(name for name, _message in batch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
