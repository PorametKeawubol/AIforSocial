#!/usr/bin/env python3
"""Refresh Mercular promotion article cards outside the LINE webhook."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from promotions import PromotionError, PromotionScraper  # noqa: E402


def main() -> int:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        snapshot = PromotionScraper(settings).refresh()
    except (OSError, PromotionError) as error:
        logging.getLogger(__name__).error("Promotion sync failed: %s", error)
        return 1
    print(
        json.dumps(
            {
                "promotions": len(snapshot["promotions"]),
                "snapshot_path": str(settings.promotion_snapshot_path),
                "generated_at": snapshot["generated_at"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
