"""Resilient, polite scraper for Mercular's public category pages.

Mercular currently renders category products into Next.js ``__NEXT_DATA__``.
That documented-in-code path is the primary parser below.  JSON-LD and
semantic HTML cards are deliberately retained as fallbacks so a front-end
deployment does not turn a small markup change into a total data outage.

The module performs no crawling beyond the category URLs explicitly present
in :class:`config.Settings` (plus ``/robots.txt`` when enabled).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import unquote, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag

try:  # Support both ``python -m MercularChatbot.scraper`` and ``python scraper.py``.
    from .config import Settings
    from .models import Product, clean_text, https_url, utc_now_iso
except ImportError:  # pragma: no cover - exercised by the documented CLI form.
    from config import Settings
    from models import Product, clean_text, https_url, utc_now_iso


LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MERCULAR_ORIGIN = "https://www.mercular.com"
MERCULAR_HOST = "www.mercular.com"
CRAWLER_NAME = "MercularSocialChatbot"
USER_AGENT = (
    "MercularSocialChatbot/1.0 "
    "(educational product-index refresh; +https://www.mercular.com/)"
)
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
MAX_REDIRECTS = 3

_MONEY_RE = re.compile(r"-?[0-9]+(?:[,.][0-9]{3})*(?:[.][0-9]+)?")
_INVALID_PRODUCT_NAMES = frozenset(
    {
        "#ref!",
        "#n/a",
        "#value!",
        "#name?",
        "n/a",
        "na",
        "null",
        "none",
        "undefined",
        "-",
        "—",
    }
)
_UNAMBIGUOUS_BRAND_TOKENS = (
    "AOC",
    "Acer",
    "ASUS",
    "DELL",
    "Elgato",
    "HP",
    "JBL",
    "LG",
    "Logitech",
    "MSI",
    "Philips",
    "Razer",
    "Samsung",
    "Sony",
    "ViewSonic",
    "Xiaomi",
)
_OUT_OF_STOCK_RE = re.compile(
    r"sold[\s_-]*out|out[\s_-]*of[\s_-]*stock|สินค้าหมด|หมดชั่วคราว",
    re.IGNORECASE,
)
_IN_STOCK_RE = re.compile(
    r"(?<!out[\s_/-])in[\s_-]*stock|available|พร้อมส่ง|มีสินค้า",
    re.IGNORECASE,
)


def _valid_product_name(value: object) -> bool:
    """Reject source placeholders that are non-empty but not product data."""

    name = clean_text(value, limit=500)
    return bool(name) and name.casefold() not in _INVALID_PRODUCT_NAMES


def _contains_brand_token(value: str, brand: str) -> bool:
    normalized = unquote(value).casefold()
    token = brand.casefold()
    return bool(
        re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized)
    )


def _reconcile_brand(name: str, brand: str, product_url: str) -> str:
    """Repair an obvious upstream brand/name/slug conflict conservatively.

    A correction is made only when exactly one known brand occurs as a complete
    token in *both* the product name and canonical slug, while the supplied
    brand occurs in neither.  This avoids guessing from a single noisy field.
    """

    identity_tokens = tuple(dict.fromkeys((brand, *_UNAMBIGUOUS_BRAND_TOKENS)))
    name_tokens = {
        candidate.casefold()
        for candidate in identity_tokens
        if candidate and _contains_brand_token(name, candidate)
    }
    url_tokens = {
        candidate.casefold()
        for candidate in identity_tokens
        if candidate and _contains_brand_token(product_url, candidate)
    }
    if name_tokens and url_tokens and name_tokens.isdisjoint(url_tokens):
        raise ValueError(
            "product name and canonical URL contain conflicting brand identities"
        )

    if brand and (
        _contains_brand_token(name, brand)
        or _contains_brand_token(product_url, brand)
    ):
        return brand
    candidates = [
        candidate
        for candidate in _UNAMBIGUOUS_BRAND_TOKENS
        if _contains_brand_token(name, candidate)
        and _contains_brand_token(product_url, candidate)
    ]
    if len(candidates) == 1:
        corrected = candidates[0]
        LOGGER.warning(
            "Corrected conflicting Mercular brand %r to %r for %s",
            brand,
            corrected,
            product_url,
        )
        return corrected
    return brand


class MercularScraperError(RuntimeError):
    """Base exception for a refresh that cannot safely complete."""


class InvalidCategoryURLError(MercularScraperError):
    """Raised when a configured URL is outside the allowed public origin."""


class RobotsDeniedError(MercularScraperError):
    """Raised when robots.txt does not permit a configured page."""


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, "", [], {}):
            return mapping[key]
    return None


def _localized_text(value: object) -> str:
    """Read Thai-first localized strings without exposing raw object reprs."""

    if isinstance(value, str) or isinstance(value, (int, float)):
        return clean_text(value)
    if isinstance(value, Mapping):
        for key in ("th", "name", "title", "label", "value", "en"):
            result = _localized_text(value.get(key))
            if result:
                return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            result = _localized_text(item)
            if result:
                return result
    return ""


def _number(value: object) -> float | None:
    """Return a finite, non-negative number from common price representations."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Mapping):
        for key in ("value", "amount", "price", "current", "salePrice", "minPrice"):
            parsed = _number(value.get(key))
            if parsed is not None:
                return parsed
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        match = _MONEY_RE.search(clean_text(value, limit=100).replace(",", ""))
        if not match:
            return None
        try:
            parsed = float(match.group(0))
        except ValueError:
            return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = clean_text(value, limit=30).casefold()
    if text in {"1", "true", "yes", "available", "instock", "in stock"}:
        return True
    if text in {"0", "false", "no", "soldout", "sold out", "outofstock", "out of stock"}:
        return False
    return default


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _snapshot_product_count(path: Path) -> int:
    """Best-effort count used only to protect a last-known-good snapshot."""

    try:
        with path.open(encoding="utf-8") as source:
            raw = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return 0
    products = raw.get("products") if isinstance(raw, Mapping) else None
    return len(products) if isinstance(products, list) else 0


def _iter_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        candidate = clean_text(value)
        if candidate:
            yield candidate
    elif isinstance(value, Mapping):
        preferred = _first_value(
            value, "url", "src", "imageUrl", "contentUrl", "image", "path"
        )
        if preferred is not None:
            yield from _iter_strings(preferred)
        else:
            for child in value.values():
                yield from _iter_strings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _iter_strings(child)


def _absolute_url(value: object, base_url: str) -> str:
    for candidate in _iter_strings(value):
        if candidate.startswith("//"):
            candidate = "https:" + candidate
        normalized = https_url(urljoin(base_url, candidate))
        if normalized:
            return normalized
    return ""


def _product_url(value: object, base_url: str) -> str:
    """Normalize a product link and reject links outside Mercular."""

    normalized = _absolute_url(value, base_url)
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    if (parsed.hostname or "").casefold() != MERCULAR_HOST:
        return ""
    return urlunparse(("https", MERCULAR_HOST, parsed.path, parsed.params, parsed.query, ""))


def _stable_id(product_url: str, name: str) -> str:
    material = f"{product_url}\x1f{name.casefold()}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:20]


def _category_name_from_url(url: str) -> str:
    slug = urlparse(url).path.strip("/").split("/")[-1]
    return clean_text(slug.replace("-", " ").title(), limit=200)


def _category_path(*values: object) -> tuple[str, ...]:
    pieces: list[str] = []
    for value in values:
        candidates: Iterable[object]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            candidates = value
        else:
            candidates = (value,)
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                candidate = _first_value(candidate, "title", "name", "label", "category")
            text = _localized_text(candidate)
            if text and text.casefold() not in {item.casefold() for item in pieces}:
                pieces.append(text[:200])
    return tuple(pieces)


def _tag_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        value = (value,)
    result: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            item = _first_value(item, "title", "name", "label", "text")
        text = _localized_text(item)
        if text and text.casefold() not in {entry.casefold() for entry in result}:
            result.append(text[:200])
    return tuple(result)


def _next_product(
    raw: Mapping[str, Any],
    *,
    base_url: str,
    source_url: str,
    scraped_at: str,
    category_title: object = None,
    breadcrumbs: object = None,
) -> Product | None:
    """Normalize one product from Mercular's current or nearby Next.js shape."""

    name = _localized_text(_first_value(raw, "title", "name", "productName"))
    slug_or_url = _first_value(raw, "url", "productUrl", "href", "link", "slug")
    product_url = _product_url(slug_or_url, base_url)
    if not _valid_product_name(name) or not product_url:
        return None

    product_id = clean_text(_first_value(raw, "id", "productId", "_id"), limit=120)
    sku = clean_text(_first_value(raw, "sku", "SKU", "productSku"), limit=120)
    product_id = product_id or sku or _stable_id(product_url, name)

    brand_raw = raw.get("brand")
    if isinstance(brand_raw, Mapping):
        brand = _localized_text(_first_value(brand_raw, "title", "name"))
    else:
        brand = _localized_text(brand_raw)
    brand = _reconcile_brand(name, brand, product_url)

    path = _category_path(breadcrumbs, category_title, raw.get("category"))
    category = _localized_text(category_title) or (path[-1] if path else "")
    current = _number(_first_value(raw, "discount", "salePrice", "currentPrice"))
    regular = _number(_first_value(raw, "price", "normalPrice", "originalPrice"))
    price = current if current is not None else regular
    original_price = regular

    image = _absolute_url(
        _first_value(raw, "mainImg", "mainImage", "image", "imageUrl", "thumbnail"),
        base_url,
    )
    sold_out = _first_value(raw, "soldout", "soldOut", "isSoldOut")
    if sold_out is not None:
        in_stock = not _bool(sold_out, default=False)
    else:
        available = _first_value(raw, "inStock", "available", "isAvailable")
        in_stock = _bool(available, default=False) if available is not None else None

    description = _localized_text(
        _first_value(raw, "description", "shortDescription", "subtitle")
    )
    tags = _tag_names((*_sequence(raw.get("tags")), *_sequence(raw.get("badges"))))
    return Product(
        id=product_id,
        sku=sku,
        name=name,
        brand=brand,
        category=category,
        category_path=path,
        price=price,
        original_price=original_price,
        image_url=image,
        product_url=product_url,
        in_stock=in_stock,
        description=description,
        tags=tags,
        source_url=source_url,
        scraped_at=scraped_at,
    )


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _walk_next_product_groups(node: object) -> Iterator[tuple[list[Any], object, object]]:
    """Find product lists in an unfamiliar Next payload while retaining context."""

    if isinstance(node, Mapping):
        category_title = _first_value(node, "categoryTitle", "categoryName", "title")
        breadcrumbs = _first_value(node, "breadcrumb", "breadcrumbs")
        for key in ("products", "productList", "items", "results"):
            children = node.get(key)
            if isinstance(children, list) and any(
                isinstance(item, Mapping)
                and _first_value(item, "slug", "productUrl", "url", "href")
                and _first_value(item, "title", "name", "productName")
                for item in children
            ):
                yield children, category_title, breadcrumbs
        for child in node.values():
            yield from _walk_next_product_groups(child)
    elif isinstance(node, list):
        for child in node:
            yield from _walk_next_product_groups(child)


def _schema_products(node: object) -> Iterator[Mapping[str, Any]]:
    if isinstance(node, Mapping):
        type_value = node.get("@type")
        types = type_value if isinstance(type_value, list) else (type_value,)
        if any(clean_text(value).casefold() == "product" for value in types):
            yield node
        for child in node.values():
            yield from _schema_products(child)
    elif isinstance(node, list):
        for child in node:
            yield from _schema_products(child)


def _json_ld_product(
    raw: Mapping[str, Any],
    *,
    base_url: str,
    source_url: str,
    scraped_at: str,
    breadcrumb_path: tuple[str, ...],
) -> Product | None:
    name = _localized_text(raw.get("name"))
    offers_raw = raw.get("offers")
    offer_candidates = [
        item for item in _sequence(offers_raw) if isinstance(item, Mapping)
    ]
    offers = next(
        (
            item
            for item in offer_candidates
            if _number(_first_value(item, "price", "lowPrice", "salePrice"))
            is not None
        ),
        (
            offer_candidates[0]
            if offer_candidates
            else offers_raw
            if isinstance(offers_raw, Mapping)
            else {}
        ),
    )
    offers = _as_mapping(offers)
    product_url = _product_url(
        _first_value(raw, "url", "@id") or _first_value(offers, "url"), base_url
    )
    if not _valid_product_name(name) or not product_url:
        return None

    sku = clean_text(_first_value(raw, "sku", "mpn", "gtin", "productID"), limit=120)
    product_id = clean_text(_first_value(raw, "productID", "@id", "id"), limit=120)
    if product_id.startswith("http"):
        product_id = ""
    brand_raw = raw.get("brand")
    brand = (
        _localized_text(_first_value(brand_raw, "name", "title"))
        if isinstance(brand_raw, Mapping)
        else _localized_text(brand_raw)
    )
    category_value = raw.get("category")
    path = _category_path(breadcrumb_path, category_value)
    category = _localized_text(category_value) or (path[-1] if path else "")
    price = _number(_first_value(offers, "price", "lowPrice", "salePrice"))
    original_price = _number(
        _first_value(raw, "originalPrice", "highPrice") or offers.get("highPrice")
    )
    availability = clean_text(offers.get("availability"), limit=200)
    if _OUT_OF_STOCK_RE.search(availability):
        in_stock = False
    elif _IN_STOCK_RE.search(availability):
        in_stock = True
    else:
        in_stock = None
    image = _absolute_url(raw.get("image"), base_url)
    return Product(
        id=product_id or sku or _stable_id(product_url, name),
        sku=sku,
        name=name,
        brand=brand,
        category=category,
        category_path=path,
        price=price,
        original_price=original_price,
        image_url=image,
        product_url=product_url,
        in_stock=in_stock,
        description=_localized_text(raw.get("description")),
        tags=(),
        source_url=source_url,
        scraped_at=scraped_at,
    )


def _html_breadcrumbs(soup: BeautifulSoup) -> tuple[str, ...]:
    selectors = (
        "nav[aria-label*='breadcrumb' i] a",
        "[class*='breadcrumb' i] a",
        "[itemtype*='BreadcrumbList'] [itemprop='name']",
    )
    for selector in selectors:
        path = _category_path([node.get_text(" ", strip=True) for node in soup.select(selector)])
        if path:
            return path
    return ()


def _attr(node: Tag, *names: str) -> str:
    for name in names:
        value = node.get(name)
        if isinstance(value, list):
            value = " ".join(str(item) for item in value)
        text = clean_text(value, limit=2_048)
        if text:
            return text
    return ""


def _select_text(node: Tag, selectors: Sequence[str]) -> str:
    for selector in selectors:
        match = node.select_one(selector)
        if match:
            content = _attr(match, "content") or match.get_text(" ", strip=True)
            text = clean_text(content)
            if text:
                return text
    return ""


def _html_card_product(
    card: Tag,
    *,
    base_url: str,
    source_url: str,
    scraped_at: str,
    breadcrumb_path: tuple[str, ...],
) -> Product | None:
    link = card if card.name == "a" and card.get("href") else card.select_one("a[href]")
    link_target = _attr(card, "data-url", "data-href") or (_attr(link, "href") if link else "")
    product_url = _product_url(link_target, base_url)
    if not product_url:
        return None

    name = _attr(card, "data-product-name", "data-name") or _select_text(
        card,
        (
            "[itemprop='name']",
            "[data-product-name]",
            "[class*='product-title' i]",
            "[class*='product-name' i]",
            "[class~='title' i]",
            "h1",
            "h2",
            "h3",
            "h4",
        ),
    )
    image_node = card.select_one("img")
    if not name and image_node:
        name = _attr(image_node, "alt", "title")
    if not _valid_product_name(name):
        return None

    sku = _attr(card, "data-sku", "sku")
    product_id = _attr(card, "data-product-id", "data-id", "id") or sku
    current_text = _attr(card, "data-price", "data-sale-price") or _select_text(
        card,
        (
            "[itemprop='price']",
            "[data-price]",
            "[class*='sale-price' i]",
            "[class*='current-price' i]",
            "[class~='price' i]",
        ),
    )
    price = _number(current_text)
    original_price = _number(
        _attr(card, "data-original-price")
        or _select_text(card, ("[class*='original-price' i]", "del", "s"))
    )
    if price is None:
        card_text = card.get_text(" ", strip=True)
        # Do not mistake model numbers (for example, "iPhone 15") for a
        # missing price. Unscoped text is a price only with a currency cue.
        if re.search(r"฿|\bTHB\b|บาท", card_text, re.IGNORECASE):
            price = _number(card_text)

    image_value = ""
    if image_node:
        image_value = _attr(image_node, "src", "data-src", "data-lazy-src")
        if not image_value:
            image_value = _attr(image_node, "srcset").split(",")[0].strip().split(" ")[0]
    image = _absolute_url(image_value, base_url)
    brand = _attr(card, "data-brand") or _select_text(
        card, ("[itemprop='brand']", "[class*='brand' i]")
    )
    category = _attr(card, "data-category") or (breadcrumb_path[-1] if breadcrumb_path else "")
    card_state = " ".join(
        (_attr(card, "class", "data-stock", "data-availability"), card.get_text(" ", strip=True))
    )
    return Product(
        id=product_id or _stable_id(product_url, name),
        sku=sku,
        name=name,
        brand=brand,
        category=category,
        category_path=_category_path(breadcrumb_path, category),
        price=price,
        original_price=original_price,
        image_url=image,
        product_url=product_url,
        in_stock=(
            False
            if _OUT_OF_STOCK_RE.search(card_state)
            else True
            if _IN_STOCK_RE.search(card_state)
            else None
        ),
        description=_select_text(card, ("[itemprop='description']", "[class*='description' i]")),
        tags=(),
        source_url=source_url,
        scraped_at=scraped_at,
    )


def _identity_keys(product: Product) -> set[tuple[str, str]]:
    # Next.js may expose a literal Thai slug while rendered anchors contain
    # the percent-encoded equivalent.  They are the same product identity.
    keys = {("url", unquote(product.product_url).casefold())}
    if product.id:
        keys.add(("id", product.id.casefold()))
    if product.sku:
        keys.add(("sku", product.sku.casefold()))
    return keys


def _merge_products(primary: Product, supplement: Product) -> Product:
    """Fill holes from a duplicate without replacing authoritative SSR data."""

    stock_values = (primary.in_stock, supplement.in_stock)
    merged_stock = (
        False
        if False in stock_values
        else True
        if True in stock_values
        else None
    )

    return replace(
        primary,
        sku=primary.sku or supplement.sku,
        brand=primary.brand or supplement.brand,
        category=primary.category or supplement.category,
        category_path=_category_path(primary.category_path, supplement.category_path),
        price=primary.price if primary.price is not None else supplement.price,
        original_price=(
            primary.original_price
            if primary.original_price is not None
            else supplement.original_price
        ),
        image_url=primary.image_url or supplement.image_url,
        in_stock=merged_stock,
        description=primary.description or supplement.description,
        tags=_tag_names((*primary.tags, *supplement.tags)),
    )


def deduplicate_products(products: Iterable[Product]) -> list[Product]:
    """Dedupe by any stable SKU, id, or canonical product URL."""

    items = list(products)
    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Preserve the earliest record as the authoritative merge base.
        if left_root < right_root:
            parents[right_root] = left_root
        else:
            parents[left_root] = right_root

    key_owner: dict[tuple[str, str], int] = {}
    for index, product in enumerate(items):
        for key in _identity_keys(product):
            owner = key_owner.setdefault(key, index)
            union(index, owner)

    groups: dict[int, list[int]] = {}
    for index in range(len(items)):
        groups.setdefault(find(index), []).append(index)

    result: list[Product] = []
    for indices in sorted(groups.values(), key=lambda values: values[0]):
        merged = items[indices[0]]
        for index in indices[1:]:
            merged = _merge_products(merged, items[index])
        result.append(merged)
    return result


def parse_html(
    html: str | bytes,
    source_url: str,
    *,
    scraped_at: str | None = None,
) -> list[Product]:
    """Extract clean products from one Mercular category page.

    Malformed scripts/cards are ignored independently.  At minimum a product
    needs a name and a canonical Mercular URL; price and image may be absent.
    """

    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    source_url = _product_url(source_url, MERCULAR_ORIGIN)
    if not source_url:
        raise InvalidCategoryURLError("source_url must be on https://www.mercular.com")
    scraped_at = scraped_at or utc_now_iso()
    soup = BeautifulSoup(html or "", "html.parser")
    fallback_next_products: list[Product] = []

    next_node = soup.find("script", id="__NEXT_DATA__")
    if next_node:
        try:
            payload = json.loads(next_node.string or next_node.get_text() or "{}")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring malformed __NEXT_DATA__ from %s: %s", source_url, exc)
        else:
            primary: object = payload
            for key in ("props", "pageProps", "pageProps", "productPageProps"):
                primary = primary.get(key, {}) if isinstance(primary, Mapping) else {}
            canonical_products: list[Product] = []
            if isinstance(primary, Mapping) and isinstance(primary.get("products"), list):
                for raw in primary["products"]:
                    if not isinstance(raw, Mapping):
                        continue
                    try:
                        product = _next_product(
                            raw,
                            base_url=source_url,
                            source_url=source_url,
                            scraped_at=scraped_at,
                            category_title=primary.get("categoryTitle"),
                            breadcrumbs=primary.get("breadcrumb"),
                        )
                    except (TypeError, ValueError) as exc:
                        LOGGER.warning(
                            "Skipping malformed canonical Next.js product from %s: %s",
                            source_url,
                            exc,
                        )
                        continue
                    if product:
                        canonical_products.append(product)
            canonical_products = deduplicate_products(canonical_products)
            if canonical_products:
                # The exact category list is authoritative.  Other Product
                # markup on a real category page includes recommendations.
                return canonical_products

            # If the canonical path moved, treat recursively discovered Next
            # records as one of the tolerant fallbacks and allow JSON-LD/HTML
            # to supplement it.
            for raw_items, category_title, breadcrumbs in _walk_next_product_groups(payload):
                for raw in raw_items:
                    if not isinstance(raw, Mapping):
                        continue
                    try:
                        product = _next_product(
                            raw,
                            base_url=source_url,
                            source_url=source_url,
                            scraped_at=scraped_at,
                            category_title=category_title,
                            breadcrumbs=breadcrumbs,
                        )
                    except (TypeError, ValueError) as exc:
                        LOGGER.warning("Skipping malformed Next.js product from %s: %s", source_url, exc)
                        continue
                    if product:
                        fallback_next_products.append(product)

    breadcrumbs = _html_breadcrumbs(soup)
    schema_products: list[Product] = []
    for script in soup.select("script[type='application/ld+json']"):
        try:
            payload = json.loads(script.string or script.get_text() or "null")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring malformed JSON-LD from %s: %s", source_url, exc)
            continue
        for raw in _schema_products(payload):
            try:
                product = _json_ld_product(
                    raw,
                    base_url=source_url,
                    source_url=source_url,
                    scraped_at=scraped_at,
                    breadcrumb_path=breadcrumbs,
                )
            except (TypeError, ValueError) as exc:
                LOGGER.warning("Skipping malformed JSON-LD product from %s: %s", source_url, exc)
                continue
            if product:
                schema_products.append(product)

    card_selectors = (
        "[itemscope][itemtype*='Product' i]",
        "[data-product-id]",
        "[data-sku]",
        "[data-testid*='product-card' i]",
        "[class*='product-card' i]",
        "article[class*='product' i]",
        "li[class*='product' i]",
    )
    cards: list[Tag] = []
    seen_nodes: set[int] = set()
    for selector in card_selectors:
        for card in soup.select(selector):
            node_id = id(card)
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                cards.append(card)

    # Last-resort semantic cards: a linked block with both an image and a
    # heading/name.  This survives CSS module/class renames.
    for link in soup.select("a[href]"):
        if id(link) in seen_nodes or not link.select_one("img"):
            continue
        href = _attr(link, "href")
        if not _product_url(href, source_url):
            continue
        if link.select_one("h1, h2, h3, h4, [itemprop='name']"):
            seen_nodes.add(id(link))
            cards.append(link)

    html_products: list[Product] = []
    for card in cards:
        try:
            product = _html_card_product(
                card,
                base_url=source_url,
                source_url=source_url,
                scraped_at=scraped_at,
                breadcrumb_path=breadcrumbs,
            )
        except (TypeError, ValueError) as exc:
            LOGGER.warning("Skipping malformed HTML product from %s: %s", source_url, exc)
            continue
        if product:
            html_products.append(product)

    return deduplicate_products(
        (*fallback_next_products, *schema_products, *html_products)
    )


def _validate_category_url(url: object) -> str:
    candidate = clean_text(url, limit=2_048)
    parsed = urlparse(candidate)
    if (
        parsed.scheme.casefold() != "https"
        or (parsed.hostname or "").casefold() != MERCULAR_HOST
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or not parsed.path.strip("/")
    ):
        raise InvalidCategoryURLError(
            f"category URL is not an HTTPS {MERCULAR_HOST} page: {clean_text(url)!r}"
        )
    return urlunparse(("https", MERCULAR_HOST, parsed.path.rstrip("/") or "/", "", parsed.query, ""))


class MercularScraper:
    """Fetch configured Mercular category pages and build a JSON snapshot."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.category_urls = tuple(
            dict.fromkeys(_validate_category_url(url) for url in self.settings.category_urls)
        )
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "th,en;q=0.8",
            }
        )
        self._sleep = sleeper
        self._monotonic = monotonic
        self._last_request_at: float | None = None
        self._rate_lock = threading.Lock()
        self._robots: RobotFileParser | None = None
        self._robots_error: Exception | None = None

    def _allowed_request_url(self, url: str, *, robots: bool = False) -> str:
        candidate = clean_text(url, limit=2_048)
        parsed = urlparse(candidate)
        if (
            parsed.scheme.casefold() != "https"
            or (parsed.hostname or "").casefold() != MERCULAR_HOST
            or parsed.username
            or parsed.password
            or parsed.port not in (None, 443)
        ):
            raise InvalidCategoryURLError(f"refusing network request outside {MERCULAR_HOST}")
        canonical_path = parsed.path if robots else parsed.path.rstrip("/") or "/"
        canonical = urlunparse(
            ("https", MERCULAR_HOST, canonical_path, "", parsed.query, "")
        )
        if robots:
            if parsed.path != "/robots.txt":
                raise InvalidCategoryURLError("robots request must target /robots.txt")
        elif canonical not in self.category_urls:
            raise InvalidCategoryURLError("refusing category URL not configured in Settings")
        return canonical

    def _redirect_url(self, current_url: str, location: object, *, robots: bool) -> str:
        """Resolve one redirect without allowing requests outside the approved scope."""

        raw_location = clean_text(location, limit=2_048)
        if not raw_location:
            raise InvalidCategoryURLError("Mercular redirect is missing Location")
        target = urljoin(current_url, raw_location)
        # This validates HTTPS, host, port, credentials, and exact configured
        # category/robots scope before the redirected request is sent.
        self._allowed_request_url(target, robots=robots)
        parsed = urlparse(target)
        return urlunparse(
            ("https", MERCULAR_HOST, parsed.path or "/", "", parsed.query, "")
        )

    def _rate_limit(self) -> None:
        with self._rate_lock:
            now = self._monotonic()
            if self._last_request_at is not None:
                remaining = self.settings.scrape_delay_seconds - (now - self._last_request_at)
                if remaining > 0:
                    self._sleep(remaining)
                    now = self._monotonic()
            self._last_request_at = now

    def _request(self, url: str, *, robots: bool = False) -> requests.Response:
        url = self._allowed_request_url(url, robots=robots)
        attempts = max(1, min(10, self.settings.request_retries + 1))
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                current_url = url
                for redirect_count in range(MAX_REDIRECTS + 1):
                    self._rate_limit()
                    response = self.session.get(
                        current_url,
                        timeout=self.settings.request_timeout_seconds,
                        allow_redirects=False,
                    )
                    observed_url = getattr(response, "url", "") or current_url
                    self._allowed_request_url(observed_url, robots=robots)
                    if response.status_code not in REDIRECT_STATUS_CODES:
                        break
                    if redirect_count >= MAX_REDIRECTS:
                        raise requests.TooManyRedirects(
                            f"Mercular exceeded {MAX_REDIRECTS} redirects"
                        )
                    current_url = self._redirect_url(
                        current_url,
                        response.headers.get("Location"),
                        robots=robots,
                    )
                if response.status_code in RETRYABLE_STATUS_CODES and attempt + 1 < attempts:
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        retry_delay = float(retry_after) if retry_after.strip() else None
                    except ValueError:
                        retry_delay = None
                    backoff = (
                        min(60.0, max(0.0, retry_delay))
                        if retry_delay is not None
                        else 0.5 * (2**attempt)
                    )
                    self._sleep(backoff)
                    continue
                response.raise_for_status()
                return response
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                self._sleep(0.5 * (2**attempt))
            except requests.RequestException as exc:
                last_error = exc
                break
        assert last_error is not None
        raise last_error

    def _robots_parser(self) -> RobotFileParser:
        if self._robots is not None:
            return self._robots
        if self._robots_error is not None:
            raise self._robots_error
        robots_url = f"{MERCULAR_ORIGIN}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            try:
                response = self._request(robots_url, robots=True)
            except requests.HTTPError as exc:
                status = getattr(exc.response, "status_code", 0)
                if status in {401, 403}:
                    parser.disallow_all = True
                elif 400 <= status < 500:
                    # RFC 9309: an unavailable robots file in the 4xx class
                    # means the crawler may access resources (401/403 remain
                    # explicit access denials for conservative compatibility).
                    parser.allow_all = True
                else:
                    raise
            else:
                parser.parse(response.text.splitlines())
        except Exception as exc:
            # Avoid hammering robots.txt once per category after a fully
            # retried transient failure; all category summaries still receive
            # an isolated error entry.
            self._robots_error = exc
            raise
        self._robots = parser
        return parser

    def _verify_allowed(self, url: str) -> None:
        if not self.settings.verify_robots:
            return
        if not self._robots_parser().can_fetch(USER_AGENT, url):
            raise RobotsDeniedError(f"robots.txt disallows {url}")

    def fetch_page(self, url: str) -> str:
        """Fetch one configured category with timeout/retries and robots checks."""

        url = self._allowed_request_url(_validate_category_url(url))
        self._verify_allowed(url)
        response = self._request(url)
        # Requests normally gets this right, but explicitly decode with a
        # safe replacement policy for malformed upstream bytes.
        if getattr(response, "encoding", None):
            return response.text
        return response.content.decode(getattr(response, "apparent_encoding", None) or "utf-8", errors="replace")

    @staticmethod
    def parse_html(
        html: str | bytes,
        source_url: str,
        *,
        scraped_at: str | None = None,
    ) -> list[Product]:
        """Class-level convenience wrapper around :func:`parse_html`."""

        return parse_html(html, source_url, scraped_at=scraped_at)

    def scrape(self) -> dict[str, Any]:
        """Collect all categories, isolating every page-level failure."""

        generated_at = utc_now_iso()
        collected: list[Product] = []
        categories: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        for url in self.category_urls:
            try:
                html = self.fetch_page(url)
                parsed_products = parse_html(html, url, scraped_at=generated_at)
                page_products = parsed_products[: self.settings.max_products_per_category]
                if not page_products:
                    raise MercularScraperError(
                        "category page contained no valid product records"
                    )
                before = len(deduplicate_products(collected))
                collected = deduplicate_products((*collected, *page_products))
                added = len(collected) - before
                categories.append(
                    {
                        "url": url,
                        "category": (
                            page_products[0].category
                            if page_products and page_products[0].category
                            else _category_name_from_url(url)
                        ),
                        "status": "ok",
                        "products_found": len(parsed_products),
                        "products_kept": len(page_products),
                        "products_added": added,
                        "error": "",
                    }
                )
            except Exception as exc:  # One category must never stop the others.
                LOGGER.warning("Mercular category failed (%s): %s", url, exc)
                error = {
                    "url": url,
                    "type": type(exc).__name__,
                    "message": clean_text(exc, limit=500),
                }
                errors.append(error)
                categories.append(
                    {
                        "url": url,
                        "category": _category_name_from_url(url),
                        "status": "blocked" if isinstance(exc, RobotsDeniedError) else "error",
                        "products_found": 0,
                        "products_kept": 0,
                        "products_added": 0,
                        "error": error["message"],
                    }
                )

        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "source": {
                "name": "Mercular",
                "website": f"{MERCULAR_ORIGIN}/",
                "attribution": "Product information from public Mercular category pages.",
                "category_urls": list(self.category_urls),
                "crawler": USER_AGENT,
                "robots_checked": self.settings.verify_robots,
            },
            "summary": {
                "categories_requested": len(self.category_urls),
                "categories_succeeded": sum(item["status"] == "ok" for item in categories),
                "categories_failed": len(errors),
                "products": len(collected),
            },
            "categories": categories,
            "errors": errors,
            "products": [product.to_dict() for product in collected],
        }

    def refresh(self, output_path: str | Path | None = None) -> dict[str, Any]:
        """Scrape and atomically replace the snapshot when it contains products."""

        snapshot = self.scrape()
        if not snapshot["products"]:
            raise MercularScraperError(
                "refresh produced no valid products; the last-known-good snapshot was preserved"
            )
        summary = snapshot["summary"]
        requested = max(1, int(summary["categories_requested"]))
        success_ratio = int(summary["categories_succeeded"]) / requested
        try:
            minimum_success_ratio = min(
                1.0,
                max(0.0, float(os.getenv("SCRAPER_MIN_SUCCESS_RATIO", "0.60"))),
            )
        except ValueError:
            minimum_success_ratio = 0.60
        if success_ratio < minimum_success_ratio:
            raise MercularScraperError(
                "refresh quality gate failed: "
                f"{summary['categories_succeeded']}/{requested} categories succeeded "
                f"({success_ratio:.0%} < {minimum_success_ratio:.0%}); "
                "the last-known-good snapshot was preserved"
            )

        destination = Path(output_path or self.settings.snapshot_path).expanduser().resolve()
        # A successful HTTP response with far fewer records commonly signals a
        # changed source layout.  Require an explicit override for planned scope
        # reductions so a weak partial crawl cannot destroy good local data.
        if destination.exists() and not _truthy_env("SCRAPER_ALLOW_SHRINK"):
            previous_count = _snapshot_product_count(destination)
            if previous_count:
                try:
                    minimum_retention = min(
                        1.0,
                        max(
                            0.0,
                            float(os.getenv("SCRAPER_MIN_RETENTION_RATIO", "0.50")),
                        ),
                    )
                except ValueError:
                    minimum_retention = 0.50
                retention = len(snapshot["products"]) / previous_count
                if retention < minimum_retention:
                    raise MercularScraperError(
                        "refresh quality gate failed: product count fell from "
                        f"{previous_count} to {len(snapshot['products'])} "
                        f"({retention:.0%} < {minimum_retention:.0%}); "
                        "the last-known-good snapshot was preserved"
                    )

        write_snapshot(snapshot, destination)
        return snapshot


def write_snapshot(snapshot: Mapping[str, Any], output_path: str | Path) -> Path:
    """Durably write JSON in the destination directory, then atomically replace."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(snapshot, temporary, ensure_ascii=False, indent=2, sort_keys=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        # Best effort: make the directory entry durable where supported.
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            pass
        else:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return destination
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _offline_snapshot(html_path: Path, source_url: str) -> dict[str, Any]:
    generated_at = utc_now_iso()
    products = parse_html(html_path.read_bytes(), source_url, scraped_at=generated_at)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "source": {
            "name": "Mercular",
            "website": f"{MERCULAR_ORIGIN}/",
            "attribution": f"Parsed from local HTML fixture {html_path.name}.",
            "category_urls": [source_url],
            "crawler": USER_AGENT,
            "robots_checked": False,
        },
        "summary": {
            "categories_requested": 1,
            "categories_succeeded": 1,
            "categories_failed": 0,
            "products": len(products),
        },
        "categories": [
            {
                "url": source_url,
                "category": products[0].category if products else _category_name_from_url(source_url),
                "status": "ok",
                "products_found": len(products),
                "products_kept": len(products),
                "products_added": len(products),
                "error": "",
            }
        ],
        "errors": [],
        "products": [product.to_dict() for product in products],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--refresh", action="store_true", help="fetch all configured categories")
    mode.add_argument("--input-html", type=Path, help="parse a local HTML file without network access")
    parser.add_argument("--output", type=Path, help="snapshot path (defaults to Settings)")
    parser.add_argument(
        "--source-url",
        default=f"{MERCULAR_ORIGIN}/audio",
        help="Mercular category URL represented by --input-html",
    )
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    output = args.output or settings.snapshot_path
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.input_html:
            snapshot = _offline_snapshot(args.input_html, _validate_category_url(args.source_url))
            if not snapshot["products"]:
                raise MercularScraperError("the HTML file contains no valid Mercular products")
            write_snapshot(snapshot, output)
        else:
            snapshot = MercularScraper(settings).refresh(output)
    except (OSError, MercularScraperError, requests.RequestException) as exc:
        LOGGER.error("Snapshot refresh failed: %s", exc)
        return 1
    LOGGER.info("Wrote %s products to %s", len(snapshot["products"]), output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CRAWLER_NAME",
    "InvalidCategoryURLError",
    "MercularScraper",
    "MercularScraperError",
    "RobotsDeniedError",
    "SCHEMA_VERSION",
    "USER_AGENT",
    "deduplicate_products",
    "main",
    "parse_html",
    "write_snapshot",
]
