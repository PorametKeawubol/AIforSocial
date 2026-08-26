#!/usr/bin/env python3
"""Validate every MercuMate demo payload with LINE without sending it."""

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
from message_showcase import (  # noqa: E402
    SHOWCASE_TYPES,
    build_showcase_message,
    showcase_hub_message,
)


def main() -> int:
    settings = Settings.from_env()
    if not settings.line_channel_access_token:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN is not configured")
    if not settings.public_base_url:
        raise RuntimeError("PUBLIC_BASE_URL is not configured")

    named_messages = [("hub", showcase_hub_message())]
    named_messages.extend(
        (
            kind,
            build_showcase_message(
                kind,
                public_base_url=settings.public_base_url,
                coupon_id=settings.line_coupon_id,
            ),
        )
        for kind in SHOWCASE_TYPES
    )

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
