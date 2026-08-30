#!/usr/bin/env python3
"""Run an out-of-webhook Mercular category refresh.

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
from models import Product, utc_now_iso  # noqa: E402
from scraper import (  # noqa: E402
    MercularScraper,
    MercularScraperError,
    deduplicate_products,
    write_snapshot,
)
from taxonomy import CategorySitemapClient, leaf_categories  # noqa: E402


LOGGER = logging.getLogger(__name__)


# These are curated collections, not category-tree nodes from categories.xml.
# Keep their tags on a product even when it is also found in a normal category.
SPECIAL_COLLECTIONS = {
    "https://www.mercular.com/flash-sale": "collection:flash-sale",
    "https://www.mercular.com/new-arrival": "collection:new-arrival",
}
ALLOW_EMPTY_SPECIAL_COLLECTIONS = frozenset(
    {"https://www.mercular.com/new-arrival"}
)


def _with_special_collections(urls: Sequence[str]) -> tuple[str, ...]:
    """Append special collection pages without requesting any URL twice."""

    return tuple(dict.fromkeys((*urls, *SPECIAL_COLLECTIONS)))


def refresh_catalog(
    settings: Settings,
    *,
    scope: str = "seed",
    max_products_per_category: int | None = None,
) -> dict[str, object]:
    """Refresh one category scope and atomically replace its local snapshot."""

    active_settings = (
        replace(
            settings,
            max_products_per_category=max(1, int(max_products_per_category)),
        )
        if max_products_per_category is not None
        else settings
    )
    if scope == "seed":
        scraper = MercularScraper(
            active_settings,
            category_urls=_with_special_collections(active_settings.category_urls),
            collection_tags=SPECIAL_COLLECTIONS,
            allow_empty_urls=ALLOW_EMPTY_SPECIAL_COLLECTIONS,
        )
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
            category_urls=_with_special_collections(tuple(source.url for source in leaves)),
            taxonomy_paths={source.url: source.path for source in leaves},
            category_taxonomy=taxonomy,
            collection_tags=SPECIAL_COLLECTIONS,
            allow_empty_urls=ALLOW_EMPTY_SPECIAL_COLLECTIONS,
        )
    else:
        raise ValueError(f"unsupported sync scope: {scope!r}")

    snapshot = scraper.refresh(active_settings.snapshot_path)
    return {
        "scope": scope,
        "categories_requested": snapshot["summary"]["categories_requested"],
        "products": len(snapshot["products"]),
        "special_collections": list(SPECIAL_COLLECTIONS),
        "snapshot_path": str(active_settings.snapshot_path),
    }


def _merge_category_retry(
    existing: dict[str, object],
    retry: dict[str, object],
) -> dict[str, object]:
    """Merge a narrow failed-URL retry without dropping the good catalogue."""

    retry_categories = {
        str(item.get("url", "")): item
        for item in retry.get("categories", [])
        if isinstance(item, dict) and item.get("url")
    }
    retried_urls = set(retry_categories)
    old_categories = [
        item for item in existing.get("categories", []) if isinstance(item, dict)
    ]
    categories = [
        retry_categories.get(str(item.get("url", "")), item)
        for item in old_categories
    ]
    known_urls = {str(item.get("url", "")) for item in categories}
    categories.extend(
        item for url, item in retry_categories.items() if url not in known_urls
    )

    old_products = [
        Product.from_dict(item)
        for item in existing.get("products", [])
        if isinstance(item, dict)
    ]
    retry_products = [
        Product.from_dict(item)
        for item in retry.get("products", [])
        if isinstance(item, dict)
    ]
    # Existing rows stay authoritative so enriched detail fields are never lost.
    products = deduplicate_products((*old_products, *retry_products))
    errors = [
        item
        for item in existing.get("errors", [])
        if isinstance(item, dict) and str(item.get("url", "")) not in retried_urls
    ]
    errors.extend(item for item in retry.get("errors", []) if isinstance(item, dict))

    source = dict(existing.get("source", {}))
    source["last_failed_category_retry"] = {
        "retried_at": retry.get("generated_at") or utc_now_iso(),
        "categories_requested": len(retried_urls),
        "categories_resolved": sum(
            item.get("status") in {"ok", "empty"}
            for item in retry_categories.values()
        ),
    }
    merged = dict(existing)
    merged.update(
        {
            "generated_at": retry.get("generated_at") or utc_now_iso(),
            "source": source,
            "categories": categories,
            "errors": errors,
            "products": [product.to_dict() for product in products],
            "summary": {
                "categories_requested": len(categories),
                "categories_succeeded": sum(
                    item.get("status") in {"ok", "empty"} for item in categories
                ),
                "categories_failed": len(errors),
                "products": len(products),
            },
        }
    )
    return merged


def retry_failed_categories(
    settings: Settings,
) -> dict[str, object]:
    """Retry only failed category URLs and atomically merge their outcome."""

    snapshot_path = settings.snapshot_path.expanduser().resolve()
    existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise MercularScraperError("catalog snapshot root must be an object")
    failed_rows = [
        item
        for item in existing.get("categories", [])
        if isinstance(item, dict) and item.get("status") not in {"ok", "empty"}
    ]
    failed_urls = tuple(
        dict.fromkeys(
            str(item.get("url", "")) for item in failed_rows if item.get("url")
        )
    )
    if not failed_urls:
        return {
            "categories_retried": 0,
            "categories_resolved": 0,
            "categories_remaining": 0,
            "products": len(existing.get("products", [])),
            "snapshot_path": str(snapshot_path),
        }

    taxonomy_paths = {
        str(item["url"]): tuple(str(part) for part in item.get("taxonomy_path", []))
        for item in failed_rows
    }
    collection_tags = {
        str(item["url"]): str(item.get("collection_tag", ""))
        for item in failed_rows
        if item.get("collection_tag")
    }
    retry = MercularScraper(
        settings,
        category_urls=failed_urls,
        taxonomy_paths=taxonomy_paths,
        collection_tags=collection_tags,
    ).scrape()
    merged = _merge_category_retry(existing, retry)
    write_snapshot(merged, snapshot_path)
    resolved = sum(
        item.get("status") in {"ok", "empty"}
        for item in retry.get("categories", [])
        if isinstance(item, dict)
    )
    return {
        "categories_retried": len(failed_urls),
        "categories_resolved": resolved,
        "categories_remaining": len(failed_urls) - resolved,
        "products": len(merged["products"]),
        "snapshot_path": str(snapshot_path),
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
        "--retry-failed",
        action="store_true",
        help="retry only category URLs marked error/blocked in the current snapshot",
    )
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        result = (
            retry_failed_categories(settings)
            if args.retry_failed
            else refresh_catalog(
                settings,
                scope=args.scope,
                max_products_per_category=args.max_products_per_category,
            )
        )
    except (OSError, ValueError, MercularScraperError, requests.RequestException) as error:
        LOGGER.error("Catalog sync failed: %s", error)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
