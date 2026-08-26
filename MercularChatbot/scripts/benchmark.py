#!/usr/bin/env python3
"""Benchmark the local command → retrieval → Flex-rendering path."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import Settings  # noqa: E402
from line_views import build_product_carousel_payload  # noqa: E402
from models import Product  # noqa: E402
from nlp import ThaiCommandParser  # noqa: E402
from recommender import ProductRecommender  # noqa: E402
from repository import ProductRepository  # noqa: E402


COMMANDS = (
    "หาหูฟังไม่เกิน 3000 พร้อมส่ง",
    "เม้าเกมมิ่ง logitec งบ 3k",
    "คีย์บอร์ดราคา 1000 ถึง 3500 เรียงถูกสุด",
    "ลำโพง Marshall มากกว่า 9000 แต่ไม่เกิน 20000",
    "มีสินค้าอะไรแนะนำบ้าง",
)


def _synthetic_catalog(size: int = 250) -> list[Product]:
    categories = ("หูฟัง", "เมาส์", "คีย์บอร์ด", "ลำโพง", "จอคอม")
    brands = ("Sony", "Logitech", "Keychron", "JBL", "ASUS")
    products: list[Product] = []
    for index in range(size):
        category = categories[index % len(categories)]
        brand = brands[index % len(brands)]
        products.append(
            Product(
                id=str(index + 1),
                sku=f"DEMO-{index + 1}",
                name=f"{category} {brand} Gaming Wireless รุ่น {index + 1}",
                brand=brand,
                category=category,
                category_path=("Gadget", category),
                price=float(500 + (index % 75) * 100),
                original_price=float(700 + (index % 75) * 100),
                image_url=f"https://example.com/products/{index + 1}.jpg",
                product_url=f"https://www.mercular.com/demo-product-{index + 1}",
                in_stock=index % 7 != 0,
                tags=("gaming", "wireless"),
                source_url="https://www.mercular.com/",
                scraped_at="2026-08-24T00:00:00+00:00",
            )
        )
    return products


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[position]


def run_benchmark(iterations: int = 100) -> dict[str, float | int | str]:
    settings = Settings.from_env()
    products: list[Product] = []
    if settings.snapshot_path.exists():
        products = ProductRepository(settings.snapshot_path).all()
    catalog_source = "snapshot" if products else "synthetic"
    if not products:
        products = _synthetic_catalog()

    parser = ThaiCommandParser(
        brands={product.brand for product in products},
        categories={product.category for product in products},
    )
    recommender = ProductRecommender(history_size=20)

    # Warm caches and imports before measuring the same path the webhook uses.
    warm_command = parser.parse(COMMANDS[0])
    warm_products = recommender.recommend(products, warm_command, user_id="warmup")
    build_product_carousel_payload(warm_products)

    durations_ms: list[float] = []
    for index in range(max(1, iterations)):
        started = time.perf_counter()
        parsed = parser.parse(COMMANDS[index % len(COMMANDS)])
        selected = recommender.recommend(
            products,
            parsed,
            user_id=f"benchmark-{index % 25}",
            top_k=5,
        )
        build_product_carousel_payload(selected)
        durations_ms.append((time.perf_counter() - started) * 1_000)

    return {
        "catalog_source": catalog_source,
        "catalog_products": len(products),
        "iterations": len(durations_ms),
        "mean_ms": round(statistics.fmean(durations_ms), 3),
        "p50_ms": round(percentile(durations_ms, 0.50), 3),
        "p95_ms": round(percentile(durations_ms, 0.95), 3),
        "max_ms": round(max(durations_ms), 3),
        "target_ms": 1_500.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args(argv)
    results = run_benchmark(args.iterations)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if float(results["p95_ms"]) <= float(results["target_ms"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
