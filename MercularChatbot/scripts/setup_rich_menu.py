#!/usr/bin/env python3
"""Validate, create, upload, and select the MercuMate Rich Menu.

The script is idempotent by menu name.  It never deletes existing menus; pass
``--force-new`` only when intentionally publishing a new image for the same
version name.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from linebot.v3.messaging import (  # noqa: E402
    ApiClient,
    ApiException,
    Configuration,
    MessagingApi,
    MessagingApiBlob,
)

from config import Settings  # noqa: E402
from rich_menu import (  # noqa: E402
    RICH_MENU_HEIGHT,
    RICH_MENU_IMAGE_PATH,
    RICH_MENU_NAME,
    RICH_MENU_WIDTH,
    build_rich_menu_request,
    validate_local_rich_menu_image,
)


def _image_dimensions(path: Path) -> str:
    try:
        result = subprocess.run(
            ["identify", "-format", "%wx%h", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("ImageMagick identify is required to verify the asset") from exc
    return result.stdout.strip()


def publish_rich_menu(*, dry_run: bool = False, force_new: bool = False) -> str:
    settings = Settings.from_env()
    if not settings.line_channel_access_token:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN is not configured")

    validate_local_rich_menu_image()
    dimensions = _image_dimensions(RICH_MENU_IMAGE_PATH)
    expected_dimensions = f"{RICH_MENU_WIDTH}x{RICH_MENU_HEIGHT}"
    if dimensions != expected_dimensions:
        raise ValueError(
            f"Rich Menu image must be {expected_dimensions}; got {dimensions}"
        )

    request = build_rich_menu_request()
    configuration = Configuration(access_token=settings.line_channel_access_token)
    with ApiClient(configuration) as api_client:
        messaging = MessagingApi(api_client)
        blob = MessagingApiBlob(api_client)

        # LINE performs the authoritative object validation before any create.
        messaging.validate_rich_menu_object(request)
        if dry_run:
            return "validated-only"

        rich_menu_id = ""
        needs_image_upload = True
        if not force_new:
            listing = messaging.get_rich_menu_list()
            existing = next(
                (
                    menu
                    for menu in listing.richmenus
                    if menu.name == RICH_MENU_NAME
                    and menu.size.width == RICH_MENU_WIDTH
                    and menu.size.height == RICH_MENU_HEIGHT
                ),
                None,
            )
            if existing is not None:
                rich_menu_id = existing.rich_menu_id
                try:
                    blob.get_rich_menu_image(
                        rich_menu_id,
                        _request_timeout=(3.0, 20.0),
                    )
                except ApiException as exc:
                    if int(getattr(exc, "status", 0) or 0) != 404:
                        raise
                else:
                    needs_image_upload = False

        if not rich_menu_id:
            created = messaging.create_rich_menu(request)
            rich_menu_id = created.rich_menu_id
        if needs_image_upload:
            image_bytes = RICH_MENU_IMAGE_PATH.read_bytes()
            content_type = (
                "image/png"
                if RICH_MENU_IMAGE_PATH.suffix.casefold() == ".png"
                else "image/jpeg"
            )
            blob.set_rich_menu_image(
                rich_menu_id,
                image_bytes,
                _headers={"Content-Type": content_type},
                _request_timeout=(10.0, 60.0),
            )

        messaging.set_default_rich_menu(rich_menu_id)
        return rich_menu_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate with LINE but do not create or select a menu",
    )
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="create a new menu even when a matching version name exists",
    )
    args = parser.parse_args()
    rich_menu_id = publish_rich_menu(
        dry_run=args.dry_run,
        force_new=args.force_new,
    )
    print(f"MercuMate Rich Menu ready: {rich_menu_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
