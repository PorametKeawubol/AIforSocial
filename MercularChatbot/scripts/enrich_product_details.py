#!/usr/bin/env python3
"""Enrich the local Mercular snapshot from product-page Next.js payloads.

This command is intentionally separate from ``sync_catalog.py`` and the LINE
webhook.  It reads server-rendered HTML first, keeps Playwright as a fallback for
pages that need rendering, waits between navigations, and updates only records
whose product page was read successfully.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from detail_scraper import ProductDetailScraper, ProductDetailScraperError  # noqa: E402
from models import Product, utc_now_iso  # noqa: E402
from scraper import write_snapshot  # noqa: E402


LOGGER = logging.getLogger(__name__)


CATEGORY_SCOPE_ROOTS = {
    "computer": "คอมพิวเตอร์",
    "smartphone-tablet-acc": "Smartphone / Tablet / ACC",
}


def _normal_scopes(scopes: Sequence[str]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(scope for scope in scopes if scope))
    unknown = set(values) - set(CATEGORY_SCOPE_ROOTS)
    if unknown:
        raise ValueError(f"unsupported category scopes: {', '.join(sorted(unknown))}")
    return values


def _in_category_scopes(product: Product, scopes: Sequence[str]) -> bool:
    if not scopes:
        return True
    root = product.category_path[0].casefold() if product.category_path else ""
    return any(root == CATEGORY_SCOPE_ROOTS[scope].casefold() for scope in scopes)


def _load_snapshot(path: Path) -> tuple[dict[str, Any], list[Product]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductDetailScraperError(f"could not read catalog snapshot: {error}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("products"), list):
        raise ProductDetailScraperError("catalog snapshot has no products array")
    try:
        products = [Product.from_dict(item) for item in raw["products"]]
    except (TypeError, ValueError) as error:
        raise ProductDetailScraperError(f"catalog snapshot has invalid product data: {error}") from error
    return raw, products


def enrich_snapshot(
    settings: Settings,
    *,
    limit: int | None = None,
    product_urls: Sequence[str] = (),
    refresh_existing: bool = False,
    category_scopes: Sequence[str] = (),
    executable_path: str | None = None,
) -> dict[str, object]:
    """Enrich missing details, preserving every unvisited snapshot record exactly."""

    scopes = _normal_scopes(category_scopes)
    snapshot, products = _load_snapshot(settings.snapshot_path)
    if product_urls:
        requested_urls = set(product_urls)
        selected = [product for product in products if product.product_url in requested_urls]
        missing_urls = requested_urls - {product.product_url for product in selected}
        if missing_urls:
            raise ProductDetailScraperError(
                "product URLs are not present in the local snapshot: "
                + ", ".join(sorted(missing_urls))
            )
    else:
        selected = products
    selected = [product for product in selected if _in_category_scopes(product, scopes)]

    active_settings = replace(
        settings,
        playwright_executable_path=(
            executable_path
            if executable_path is not None
            else settings.playwright_executable_path
        ),
    )
    result = ProductDetailScraper(active_settings).enrich(
        selected,
        limit=limit,
        refresh_existing=refresh_existing,
    )
    original_by_id = {product.id: product for product in selected}
    enriched_by_id = {product.id: product for product in result.products}
    changed_ids = {
        product_id
        for product_id, original in original_by_id.items()
        if enriched_by_id[product_id] != original
    }

    if changed_ids:
        rows = snapshot["products"]
        assert isinstance(rows, list)
        snapshot["products"] = [
            enriched_by_id[Product.from_dict(row).id].to_dict()
            if isinstance(row, dict) and Product.from_dict(row).id in changed_ids
            else row
            for row in rows
        ]
        source = dict(snapshot.get("source", {}))
        source["product_detail_enrichment"] = {
            "method": "HTTP __NEXT_DATA__ with Playwright fallback",
            "updated_at": utc_now_iso(),
            "fields": [
                "overview",
                "highlights",
                "specifications",
                "rating",
                "review_count",
                "recommended_count",
                "warranty",
                "service_notes",
            ],
        }
        snapshot["source"] = source
        snapshot["detail_summary"] = {
            "last_run_at": utc_now_iso(),
            "requested": result.requested,
            "succeeded": result.succeeded,
            "failed": result.failed,
            "skipped": result.skipped,
            "updated_products": len(changed_ids),
            "errors": list(result.errors),
        }
        write_snapshot(snapshot, settings.snapshot_path)

    return {
        "requested": result.requested,
        "succeeded": result.succeeded,
        "failed": result.failed,
        "skipped": result.skipped,
        "updated_products": len(changed_ids),
        "rate_limited": result.rate_limited,
        "retry_after_seconds": result.retry_after_seconds,
        "category_scopes": list(scopes),
        "snapshot_path": str(settings.snapshot_path),
        "errors": list(result.errors),
    }


def enrich_until_complete(
    settings: Settings,
    *,
    batch_size: int = 5,
    cooldown_seconds: int = 300,
    between_batches_seconds: int = 300,
    category_scopes: Sequence[str] = (),
    executable_path: str | None = None,
) -> dict[str, object]:
    """Safely resume every missing product detail until none are eligible.

    A source-side 429 is a pause signal, not a failed data record.  Successful
    pages are checkpointed by :func:`enrich_snapshot`; this loop waits in at-most
    60-second increments before retrying the remaining pages.  Permanent page
    errors are remembered for this run so one 404 cannot prevent later products
    from being reached.
    """

    batch_size = max(1, int(batch_size))
    cooldown_seconds = max(60, int(cooldown_seconds))
    # HTTP mode already spaces individual page requests.  Callers that have
    # validated a safe delay can choose zero here to avoid an unnecessary idle
    # minute; HTTP 429 still always uses ``cooldown_seconds``.
    between_batches_seconds = max(0, int(between_batches_seconds))
    scopes = _normal_scopes(category_scopes)
    failed_urls: set[str] = set()
    totals = {"requested": 0, "succeeded": 0, "failed": 0, "updated_products": 0}
    rate_limit_count = 0

    while True:
        _snapshot, products = _load_snapshot(settings.snapshot_path)
        pending_urls = [
            product.product_url
            for product in products
            if (
                not product.detail_updated_at
                and product.product_url not in failed_urls
                and _in_category_scopes(product, scopes)
            )
        ]
        if not pending_urls:
            return {
                **totals,
                "remaining_unavailable": len(failed_urls),
                "rate_limit_count": rate_limit_count,
                "complete": True,
                "category_scopes": list(scopes),
                "snapshot_path": str(settings.snapshot_path),
            }

        result = enrich_snapshot(
            settings,
            product_urls=pending_urls[:batch_size],
            refresh_existing=True,
            category_scopes=scopes,
            executable_path=executable_path,
        )
        for key in totals:
            value = result.get(key, 0)
            totals[key] += int(value) if isinstance(value, int) else 0
        errors = result.get("errors", [])
        if isinstance(errors, list) and not bool(result.get("rate_limited")):
            failed_urls.update(
                str(error.get("url"))
                for error in errors
                if (
                    isinstance(error, dict)
                    and error.get("url")
                    and str(error.get("retryable", "false")).casefold() != "true"
                )
            )
        LOGGER.info(
            "detail progress: updated=%s succeeded=%s failed=%s remaining=%s",
            totals["updated_products"],
            totals["succeeded"],
            totals["failed"],
            len(pending_urls) - batch_size,
        )
        if not bool(result.get("rate_limited")):
            transient_only = bool(errors) and all(
                isinstance(error, dict)
                and str(error.get("retryable", "false")).casefold() == "true"
                for error in errors
            ) and not bool(result.get("succeeded"))
            pause_seconds = cooldown_seconds if transient_only else between_batches_seconds
            if pause_seconds:
                # Mercular may allow only a small burst before exposing HTTP 429.
                # Pause proactively after a successful checkpoint so the next batch
                # does not need to probe that limit.
                LOGGER.info(
                    "detail batch checkpointed; pausing for %s seconds before the next batch",
                    pause_seconds,
                )
                remaining = pause_seconds
                while remaining:
                    interval = min(60, remaining)
                    time.sleep(interval)
                    remaining -= interval
            continue

        rate_limit_count += 1
        retry_after = result.get("retry_after_seconds")
        wait_seconds = max(
            cooldown_seconds,
            int(retry_after) if isinstance(retry_after, int) else 0,
        )
        LOGGER.warning(
            "Mercular rate-limited product details; pausing for %s seconds before retrying",
            wait_seconds,
        )
        remaining = wait_seconds
        while remaining:
            interval = min(60, remaining)
            time.sleep(interval)
            remaining -= interval


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        help="maximum eligible products to render; omit to process all missing details",
    )
    parser.add_argument(
        "--product-url",
        action="append",
        default=[],
        help="enrich one product URL already present in the local snapshot (repeatable)",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="also refresh products that already have detail_updated_at",
    )
    parser.add_argument(
        "--executable-path",
        help="optional system Chromium path; otherwise use `playwright install chromium`",
    )
    parser.add_argument(
        "--until-complete",
        action="store_true",
        help="resume missing details in safe batches, backing off automatically on HTTP 429",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="pages per safe resume batch when --until-complete is used (default: 5)",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=300,
        help="minimum pause after HTTP 429 in --until-complete mode (default: 300)",
    )
    parser.add_argument(
        "--between-batches-seconds",
        type=int,
        default=300,
        help="proactive pause between successful resume batches; 0 is allowed (default: 300)",
    )
    parser.add_argument(
        "--category-scope",
        choices=tuple(CATEGORY_SCOPE_ROOTS),
        action="append",
        default=[],
        help="limit to one main catalog root; repeat for multiple roots",
    )
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.until_complete:
            if args.limit is not None or args.product_url or args.refresh_existing:
                parser.error(
                    "--until-complete cannot be combined with --limit, --product-url, or --refresh-existing"
                )
            result = enrich_until_complete(
                settings,
                batch_size=args.batch_size,
                cooldown_seconds=args.cooldown_seconds,
                between_batches_seconds=args.between_batches_seconds,
                category_scopes=args.category_scope,
                executable_path=args.executable_path,
            )
        else:
            result = enrich_snapshot(
                settings,
                limit=args.limit,
                product_urls=args.product_url,
                refresh_existing=args.refresh_existing,
                category_scopes=args.category_scope,
                executable_path=args.executable_path,
            )
    except (OSError, ValueError, ProductDetailScraperError) as error:
        LOGGER.error("Product detail enrichment failed: %s", error)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
