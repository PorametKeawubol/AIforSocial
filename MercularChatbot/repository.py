"""Thread-safe, last-known-good access to a Mercular product snapshot."""

from __future__ import annotations

import copy
import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

try:
    from .config import Settings
    from .models import Product, clean_text
    from .scraper import SCHEMA_VERSION
except ImportError:  # pragma: no cover - allows running modules from this folder.
    from config import Settings
    from models import Product, clean_text
    from scraper import SCHEMA_VERSION


LOGGER = logging.getLogger(__name__)


class SnapshotError(RuntimeError):
    """Raised internally when a snapshot is absent, corrupt, or incompatible."""


def _parse_datetime(value: object) -> datetime | None:
    text = clean_text(value, limit=100)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class ProductRepository:
    """Load products atomically and retain good data after a bad reload.

    Public reads are safe from concurrent request threads.  With
    ``auto_reload=True`` (the default), each read performs a cheap file-stat
    check and adopts a new complete snapshot when one appears.
    """

    def __init__(
        self,
        snapshot_path: str | Path | Settings | None = None,
        *,
        settings: Settings | None = None,
        stale_after_hours: int | float | None = None,
        auto_reload: bool = True,
    ) -> None:
        if isinstance(snapshot_path, Settings):
            if settings is not None:
                raise TypeError("pass Settings either positionally or by keyword, not both")
            settings = snapshot_path
            snapshot_path = None
        settings = settings or Settings.from_env()
        self.snapshot_path = Path(snapshot_path or settings.snapshot_path).expanduser().resolve()
        self.stale_after = timedelta(
            hours=max(
                0.0,
                float(
                    settings.stale_after_hours
                    if stale_after_hours is None
                    else stale_after_hours
                ),
            )
        )
        self.auto_reload = auto_reload
        self._lock = threading.RLock()
        self._products: tuple[Product, ...] = ()
        self._by_id: dict[str, Product] = {}
        self._by_sku: dict[str, Product] = {}
        self._metadata: dict[str, Any] = {}
        self._generated_at: datetime | None = None
        self._loaded_at: datetime | None = None
        self._loaded_signature: tuple[int, int] | None = None
        self._observed_signature: tuple[int, int] | None = None
        self._last_error = ""
        self.reload()

    @staticmethod
    def _signature(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    @staticmethod
    def _decode(raw: object) -> tuple[tuple[Product, ...], dict[str, Any], datetime | None]:
        if not isinstance(raw, Mapping):
            raise SnapshotError("snapshot root must be a JSON object")
        schema_version = raw.get("schema_version")
        if schema_version not in (SCHEMA_VERSION, str(SCHEMA_VERSION)):
            raise SnapshotError(f"unsupported snapshot schema_version: {schema_version!r}")
        values = raw.get("products")
        if not isinstance(values, list):
            raise SnapshotError("snapshot products must be a list")

        products: list[Product] = []
        seen_ids: set[str] = set()
        seen_skus: set[str] = set()
        seen_urls: set[str] = set()
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise SnapshotError(f"product at index {index} is not an object")
            try:
                product = Product.from_dict(value)
            except (TypeError, ValueError) as exc:
                raise SnapshotError(f"invalid product at index {index}: {exc}") from exc
            identity_id = product.id.casefold()
            identity_sku = product.sku.casefold()
            identity_url = product.product_url.casefold()
            if (
                identity_id in seen_ids
                or (identity_sku and identity_sku in seen_skus)
                or identity_url in seen_urls
            ):
                continue
            seen_ids.add(identity_id)
            if identity_sku:
                seen_skus.add(identity_sku)
            seen_urls.add(identity_url)
            products.append(product)

        metadata = copy.deepcopy({key: value for key, value in raw.items() if key != "products"})
        generated_at = _parse_datetime(raw.get("generated_at"))
        return tuple(products), metadata, generated_at

    def reload(self) -> bool:
        """Attempt a reload; return ``False`` and preserve prior state on error."""

        try:
            signature = self._signature(self.snapshot_path)
        except OSError as exc:
            with self._lock:
                self._observed_signature = None
                self._last_error = f"snapshot unavailable: {exc}"
            LOGGER.warning("Could not stat Mercular snapshot %s: %s", self.snapshot_path, exc)
            return False

        try:
            with self.snapshot_path.open("r", encoding="utf-8") as source:
                raw = json.load(source)
            products, metadata, generated_at = self._decode(raw)
        except (OSError, UnicodeError, json.JSONDecodeError, SnapshotError) as exc:
            with self._lock:
                self._observed_signature = signature
                self._last_error = f"snapshot reload rejected: {exc}"
            LOGGER.warning(
                "Rejected Mercular snapshot %s; retaining last-known-good data: %s",
                self.snapshot_path,
                exc,
            )
            return False

        by_id: dict[str, Product] = {}
        by_sku: dict[str, Product] = {}
        for product in products:
            by_id[product.id.casefold()] = product
            if product.sku:
                by_sku[product.sku.casefold()] = product

        with self._lock:
            self._products = products
            self._by_id = by_id
            self._by_sku = by_sku
            self._metadata = metadata
            self._generated_at = generated_at
            self._loaded_at = datetime.now(UTC)
            self._loaded_signature = signature
            self._observed_signature = signature
            self._last_error = ""
        return True

    def reload_if_changed(self) -> bool:
        """Reload only after a new file signature is observed."""

        try:
            signature = self._signature(self.snapshot_path)
        except OSError as exc:
            with self._lock:
                self._last_error = f"snapshot unavailable: {exc}"
            return False
        with self._lock:
            if signature == self._observed_signature:
                return False
        return self.reload()

    def load(self) -> bool:
        """Explicit initial-load alias; identical to a safe :meth:`reload`."""

        return self.reload()

    def _maybe_reload(self) -> None:
        if self.auto_reload:
            self.reload_if_changed()

    def all(self) -> list[Product]:
        """Return a stable copy of all products in snapshot order."""

        self._maybe_reload()
        with self._lock:
            return list(self._products)

    def get(self, identifier: object) -> Product | None:
        """Find by Product.id, falling back to SKU for LINE postbacks."""

        self._maybe_reload()
        key = clean_text(identifier, limit=120).casefold()
        if not key:
            return None
        with self._lock:
            return self._by_id.get(key) or self._by_sku.get(key)

    def get_by_sku(self, sku: object) -> Product | None:
        self._maybe_reload()
        key = clean_text(sku, limit=120).casefold()
        if not key:
            return None
        with self._lock:
            return self._by_sku.get(key)

    def brands(self) -> tuple[str, ...]:
        """Return distinct display brands sorted case-insensitively."""

        self._maybe_reload()
        with self._lock:
            values: dict[str, str] = {}
            for product in self._products:
                if product.brand:
                    values.setdefault(product.brand.casefold(), product.brand)
        return tuple(sorted(values.values(), key=str.casefold))

    def categories(self) -> tuple[str, ...]:
        """Return distinct leaf categories sorted case-insensitively."""

        self._maybe_reload()
        with self._lock:
            values: dict[str, str] = {}
            for product in self._products:
                category = product.category or (
                    product.category_path[-1] if product.category_path else ""
                )
                if category:
                    values.setdefault(category.casefold(), category)
        return tuple(sorted(values.values(), key=str.casefold))

    def __len__(self) -> int:
        self._maybe_reload()
        with self._lock:
            return len(self._products)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a deep copy so callers cannot mutate repository state."""

        self._maybe_reload()
        with self._lock:
            return copy.deepcopy(self._metadata)

    @property
    def is_ready(self) -> bool:
        """Whether a complete, non-empty catalog has loaded successfully."""

        self._maybe_reload()
        with self._lock:
            return self._loaded_signature is not None and bool(self._products)

    @property
    def ready(self) -> bool:
        return self.is_ready

    @property
    def readiness(self) -> bool:
        return self.is_ready

    @property
    def is_stale(self) -> bool:
        """Whether data is absent, undated, or older than the configured age."""

        self._maybe_reload()
        with self._lock:
            generated_at = self._generated_at
            ready = self._loaded_signature is not None and bool(self._products)
            stale_after = self.stale_after
        if not ready or generated_at is None:
            return True
        return datetime.now(UTC) - generated_at > stale_after

    @property
    def stale(self) -> bool:
        return self.is_stale

    @property
    def staleness(self) -> bool:
        return self.is_stale

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def status(self) -> dict[str, Any]:
        """Return operational diagnostics without leaking mutable internals."""

        self._maybe_reload()
        with self._lock:
            count = len(self._products)
            loaded_at = self._loaded_at.isoformat() if self._loaded_at else ""
            generated_at = self._generated_at.isoformat() if self._generated_at else ""
            error = self._last_error
            ready = self._loaded_signature is not None and bool(self._products)
        return {
            "ready": ready,
            "stale": self.is_stale,
            "products": count,
            "generated_at": generated_at,
            "loaded_at": loaded_at,
            "last_error": error,
            "snapshot_path": str(self.snapshot_path),
        }


# Short alias for callers that prefer the domain-specific name.
MercularRepository = ProductRepository


__all__ = ["MercularRepository", "ProductRepository", "SnapshotError"]
