"""Durable daily observations for MercuMate product price and stock history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from typing import Iterable

try:  # Package import.
    from .models import Product, clean_text, utc_now_iso
except ImportError:  # pragma: no cover - direct script import.
    from models import Product, clean_text, utc_now_iso


@dataclass(frozen=True, slots=True)
class PriceObservation:
    product_id: str
    observed_at: str
    price: float | None
    original_price: float | None
    discount_amount: float | None
    in_stock: bool | None


def _canonical_timestamp(value: object | None) -> tuple[str, str]:
    text = clean_text(value or utc_now_iso(), limit=80)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC).replace(microsecond=0)
    return parsed.isoformat(), parsed.date().isoformat()


class PriceHistoryStore:
    """SQLite store with one idempotent observation per product per UTC date."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS product_observations (
                    product_id TEXT NOT NULL,
                    observed_date TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    sku TEXT NOT NULL,
                    name TEXT NOT NULL,
                    price REAL,
                    original_price REAL,
                    discount_amount REAL,
                    in_stock INTEGER,
                    source_url TEXT NOT NULL,
                    product_url TEXT NOT NULL,
                    PRIMARY KEY (product_id, observed_date)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_product_observations_product_time
                ON product_observations (product_id, observed_at)
                """
            )

    def record_snapshot(
        self,
        products: Iterable[Product],
        *,
        observed_at: object | None = None,
    ) -> int:
        """Upsert a day's price, discount, and stock state for every product."""

        timestamp, observed_date = _canonical_timestamp(observed_at)
        rows = []
        for product in products:
            discount = (
                max(0.0, product.original_price - product.price)
                if product.price is not None and product.original_price is not None
                else None
            )
            rows.append(
                (
                    product.id,
                    observed_date,
                    timestamp,
                    product.sku,
                    product.name,
                    product.price,
                    product.original_price,
                    discount,
                    None if product.in_stock is None else int(product.in_stock),
                    product.source_url,
                    product.product_url,
                )
            )
        if not rows:
            return 0
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO product_observations (
                    product_id, observed_date, observed_at, sku, name, price,
                    original_price, discount_amount, in_stock, source_url, product_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_id, observed_date) DO UPDATE SET
                    observed_at = excluded.observed_at,
                    sku = excluded.sku,
                    name = excluded.name,
                    price = excluded.price,
                    original_price = excluded.original_price,
                    discount_amount = excluded.discount_amount,
                    in_stock = excluded.in_stock,
                    source_url = excluded.source_url,
                    product_url = excluded.product_url
                """,
                rows,
            )
        return len(rows)

    def observations(self, product_id: object) -> list[PriceObservation]:
        """Return chronological price observations for one product."""

        identifier = clean_text(product_id, limit=120)
        if not identifier:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT product_id, observed_at, price, original_price, discount_amount, in_stock
                FROM product_observations
                WHERE product_id = ?
                ORDER BY observed_at ASC
                """,
                (identifier,),
            ).fetchall()
        return [
            PriceObservation(
                product_id=row["product_id"],
                observed_at=row["observed_at"],
                price=row["price"],
                original_price=row["original_price"],
                discount_amount=row["discount_amount"],
                in_stock=None if row["in_stock"] is None else bool(row["in_stock"]),
            )
            for row in rows
        ]


__all__ = ["PriceHistoryStore", "PriceObservation"]
