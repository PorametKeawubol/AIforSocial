"""Domain models shared by the Mercular chatbot components."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse, urlunparse


def clean_text(value: object | None, *, limit: int = 2_000) -> str:
    """Collapse whitespace and cap untrusted website/chat text."""

    text = " ".join(str(value or "").replace("\xa0", " ").split())
    return text[:limit]


def https_url(value: object | None) -> str:
    """Return a normalized public HTTPS URL or an empty string."""

    candidate = clean_text(value, limit=2_048)
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    scheme = "https" if parsed.scheme == "http" else parsed.scheme
    return urlunparse((scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class Product:
    """A cleaned, serializable Mercular product record."""

    id: str
    sku: str
    name: str
    brand: str
    category: str
    category_path: tuple[str, ...]
    price: float | None
    original_price: float | None
    image_url: str
    product_url: str
    in_stock: bool | None
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    source_url: str = ""
    scraped_at: str = ""

    def __post_init__(self) -> None:
        if not clean_text(self.id):
            raise ValueError("product id is required")
        if not clean_text(self.name):
            raise ValueError("product name is required")
        if not https_url(self.product_url):
            raise ValueError("product_url must be an absolute HTTP(S) URL")
        for field_name in ("price", "original_price"):
            value = getattr(self, field_name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{field_name} must be a finite, non-negative number")

    @property
    def display_price(self) -> str:
        if self.price is None:
            return "ตรวจสอบราคาที่เว็บไซต์"
        return f"฿{self.price:,.0f}"

    @property
    def search_text(self) -> str:
        return " ".join(
            part
            for part in (
                self.name,
                self.brand,
                self.category,
                *self.category_path,
                *self.tags,
                self.description,
            )
            if part
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["category_path"] = list(self.category_path)
        result["tags"] = list(self.tags)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Product":
        def number_or_none(raw: object) -> float | None:
            if raw in (None, ""):
                return None
            try:
                parsed = float(str(raw).replace(",", ""))
            except (TypeError, ValueError):
                return None
            if not math.isfinite(parsed):
                raise ValueError("price fields must be finite numbers")
            return parsed

        def string_sequence(raw: object, field_name: str) -> tuple[str, ...]:
            if raw in (None, ""):
                return ()
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"{field_name} must be an array of strings")
            if any(not isinstance(item, str) for item in raw):
                raise ValueError(f"{field_name} must contain only strings")
            return tuple(
                dict.fromkeys(
                    cleaned
                    for item in raw
                    if (cleaned := clean_text(item, limit=200))
                )
            )

        def bool_or_none(raw: object) -> bool | None:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)) and raw in {0, 1}:
                return bool(raw)
            text = clean_text(raw, limit=30).casefold()
            if text in {"true", "1", "yes", "in_stock", "instock", "available"}:
                return True
            if text in {"false", "0", "no", "out_of_stock", "outofstock", "soldout"}:
                return False
            return None

        return cls(
            id=clean_text(value.get("id"), limit=120),
            sku=clean_text(value.get("sku"), limit=120),
            name=clean_text(value.get("name"), limit=500),
            brand=clean_text(value.get("brand"), limit=200),
            category=clean_text(value.get("category"), limit=200),
            category_path=string_sequence(value.get("category_path", []), "category_path"),
            price=number_or_none(value.get("price")),
            original_price=number_or_none(value.get("original_price")),
            image_url=https_url(value.get("image_url")),
            product_url=https_url(value.get("product_url")),
            in_stock=bool_or_none(value.get("in_stock")),
            description=clean_text(value.get("description")),
            tags=string_sequence(value.get("tags", []), "tags"),
            source_url=https_url(value.get("source_url")),
            scraped_at=clean_text(value.get("scraped_at"), limit=80),
        )


__all__ = ["Product", "clean_text", "https_url", "utc_now_iso"]
