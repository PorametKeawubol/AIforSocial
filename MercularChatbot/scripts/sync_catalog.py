#!/usr/bin/env python3
"""Run an out-of-webhook Mercular category refresh and record daily history.

The default ``seed`` scope refreshes the small demonstration set configured in
``Settings``.  ``sitemap-leaves`` discovers public category URLs from Mercular's
published category sitemap, then fetches only leaf category pages.  Use that mode
only when you have permission to maintain a broader local product index.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import requests


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from models import Product  # noqa: E402
from price_history import PriceHistoryStore  # noqa: E402
from scraper import MercularScraper, MercularScraperError  # noqa: E402
from taxonomy import CategorySitemapClient, leaf_categories  # noqa: E402


LOGGER = logging.getLogger(__name__)


def refresh_catalog(
    settings: Settings,
    *,
    scope: str = "seed",
    max_products_per_category: int | None = None,
    record_history: bool = True,
) -> dict[str, object]:
    """Refresh one category scope, then save a daily price/stock observation."""

    active_settings = (
        replace(
            settings,
            max_products_per_category=max(1, int(max_products_per_category)),
        )
        if max_products_per_category is not None
        else settings
    )
    if scope == "seed":
        scraper = MercularScraper(active_settings)
    elif scope == "sitemap-leaves":
        sources = CategorySitemapClient(
            timeout_seconds=active_settings.request_timeout_seconds
        ).fetch()
        leaves = leaf_categories(sources)
        if not leaves:
            raise MercularScraperError("category sitemap contained no leaf categories")
        taxonomy = {source.url: source.path for source in sources}
        scraper = MercularScraper(
            active_settings,
            category_urls=(source.url for source in leaves),
            taxonomy_paths={source.url: source.path for source in leaves},
            category_taxonomy=taxonomy,
        )
    else:
        raise ValueError(f"unsupported sync scope: {scope!r}")

    snapshot = scraper.refresh(active_settings.snapshot_path)
    history_count = 0
    if record_history:
        products = [Product.from_dict(value) for value in snapshot["products"]]
        history_count = PriceHistoryStore(active_settings.price_history_path).record_snapshot(
            products,
            observed_at=snapshot["generated_at"],
        )
    return {
        "scope": scope,
        "categories_requested": snapshot["summary"]["categories_requested"],
        "products": len(snapshot["products"]),
        "history_observations": history_count,
        "snapshot_path": str(active_settings.snapshot_path),
        "history_path": str(active_settings.price_history_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("seed", "sitemap-leaves"),
        default="seed",
        help="seed is the demo category set; sitemap-leaves uses every public leaf category",
    )
    parser.add_argument(
        "--max-products-per-category",
        type=int,
        help="cap parsed products per category page (defaults to MAX_PRODUCTS_PER_CATEGORY)",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="refresh the catalog without writing price and stock observations",
    )
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        result = refresh_catalog(
            settings,
            scope=args.scope,
            max_products_per_category=args.max_products_per_category,
            record_history=not args.no_history,
        )
    except (OSError, ValueError, MercularScraperError, requests.RequestException) as error:
        LOGGER.error("Catalog sync failed: %s", error)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
