"""Mercular promotion article snapshots, scraping, and read-only access."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
import logging
from pathlib import Path
import re
import threading
from typing import Any, Iterable, Mapping
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
import requests

try:
    from .config import Settings
    from .models import clean_text, https_url, utc_now_iso
    from .scraper import USER_AGENT, write_snapshot
except ImportError:  # pragma: no cover - direct script support.
    from config import Settings
    from models import clean_text, https_url, utc_now_iso
    from scraper import USER_AGENT, write_snapshot


LOGGER = logging.getLogger(__name__)
PROMOTION_SCHEMA_VERSION = 1
PROMOTION_CATEGORY_URL = "https://www.mercular.com/category-review-article/promotion"
_ARTICLE_PATH_RE = re.compile(r"^/review-article/[a-z0-9][a-z0-9-]*$", re.IGNORECASE)


class PromotionError(RuntimeError):
    """Promotion source or snapshot could not be read safely."""


@dataclass(frozen=True, slots=True)
class Promotion:
    """One compact promotion article suitable for a LINE card."""

    id: str
    title: str
    summary: str
    image_url: str
    article_url: str
    published_at: str = ""
    starts_at: str = ""
    ends_at: str = ""
    discount_summary: str = ""
    scraped_at: str = ""

    def __post_init__(self) -> None:
        if not clean_text(self.id, limit=160):
            raise ValueError("promotion id is required")
        if not clean_text(self.title, limit=500):
            raise ValueError("promotion title is required")
        parsed = urlparse(self.article_url)
        if (
            parsed.scheme != "https"
            or (parsed.hostname or "").casefold() not in {"mercular.com", "www.mercular.com"}
            or not _ARTICLE_PATH_RE.fullmatch(parsed.path.rstrip("/"))
        ):
            raise ValueError("article_url must be a Mercular review article")
        for field_name in ("published_at", "starts_at", "ends_at"):
            value = clean_text(getattr(self, field_name), limit=40)
            if value:
                try:
                    date.fromisoformat(value[:10])
                except ValueError as error:
                    raise ValueError(f"{field_name} must start with an ISO date") from error

    def is_current(self, on_date: date | None = None) -> bool:
        """Treat unknown campaign dates as current, never invent an expiry."""

        today = on_date or date.today()
        starts = date.fromisoformat(self.starts_at[:10]) if self.starts_at else None
        ends = date.fromisoformat(self.ends_at[:10]) if self.ends_at else None
        return not ((starts and starts > today) or (ends and ends < today))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Promotion":
        return cls(
            id=clean_text(value.get("id"), limit=160),
            title=clean_text(value.get("title"), limit=500),
            summary=clean_text(value.get("summary"), limit=1_000),
            image_url=https_url(value.get("image_url")),
            article_url=https_url(value.get("article_url")),
            published_at=clean_text(value.get("published_at"), limit=40),
            starts_at=clean_text(value.get("starts_at"), limit=40),
            ends_at=clean_text(value.get("ends_at"), limit=40),
            discount_summary=clean_text(value.get("discount_summary"), limit=500),
            scraped_at=clean_text(value.get("scraped_at"), limit=80),
        )


def _article_url(value: object, source_url: str) -> str:
    candidate = clean_text(value, limit=2_048)
    if not candidate:
        return ""
    parsed = urlparse(urljoin(source_url, candidate))
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").casefold() not in {"mercular.com", "www.mercular.com"}
        or not _ARTICLE_PATH_RE.fullmatch(parsed.path.rstrip("/"))
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        return ""
    return urlunparse(("https", "www.mercular.com", parsed.path.rstrip("/"), "", "", ""))


def _first_text(value: object, keys: Iterable[str]) -> str:
    if not isinstance(value, Mapping):
        return ""
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, Mapping):
            raw = raw.get("th", raw.get("title", raw.get("name", "")))
        text = clean_text(raw, limit=1_000)
        if text and not text.startswith(("http://", "https://")):
            return text
    return ""


def _first_url(value: object, keys: Iterable[str]) -> str:
    if not isinstance(value, Mapping):
        return ""
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, Mapping):
            raw = raw.get("url", raw.get("src", raw.get("path", "")))
        if isinstance(raw, list) and raw:
            raw = raw[0]
        url = https_url(raw)
        if url:
            return url
    return ""


def _iso_date(value: object) -> str:
    # ISO datetimes continue immediately with ``T`` (a word character), so ``\b``
    # is not a valid right boundary here.
    match = re.search(
        r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)",
        clean_text(value, limit=100),
    )
    return match.group(1) if match else ""


def _mapping_candidates(value: object) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _mapping_candidates(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mapping_candidates(child)


def _promotion_from_mapping(
    value: Mapping[str, Any], source_url: str, scraped_at: str
) -> Promotion | None:
    article_url = ""
    for key in ("url", "href", "link", "articleUrl", "canonicalUrl", "slug"):
        article_url = _article_url(value.get(key), source_url)
        if article_url:
            break
    if not article_url:
        for raw in value.values():
            if isinstance(raw, str) and "/review-article/" in raw:
                article_url = _article_url(raw, source_url)
                if article_url:
                    break
    if not article_url:
        return None
    title = _first_text(value, ("title", "articleTitle", "name", "heading", "seoTitle"))
    if not title:
        return None
    slug = urlparse(article_url).path.rsplit("/", 1)[-1]
    return Promotion(
        id=slug,
        title=title,
        summary=_first_text(
            value,
            ("shortDescription", "summary", "excerpt", "description", "seoDescription"),
        ),
        image_url=_first_url(
            value,
            ("thumbnail", "thumbnailUrl", "coverImage", "image", "imageUrl", "banner"),
        ),
        article_url=article_url,
        published_at=_iso_date(
            _first_text(
                value,
                ("publishedAt", "publishDate", "publishedDate", "createdAt", "date"),
            )
        ),
        scraped_at=scraped_at,
    )


def _promotion_from_anchor(anchor: Any, source_url: str, scraped_at: str) -> Promotion | None:
    article_url = _article_url(anchor.get("href"), source_url)
    if not article_url:
        return None
    container = anchor.find_parent(("article", "li")) or anchor.parent
    heading = container.find(("h1", "h2", "h3", "h4")) if container else None
    title = clean_text(
        heading.get_text(" ", strip=True) if heading else anchor.get_text(" ", strip=True),
        limit=500,
    )
    if not title or title.casefold() in {"อ่านต่อ", "read more", "ดูรายละเอียด"}:
        return None
    paragraph = container.find("p") if container else None
    image = container.find("img") if container else anchor.find("img")
    image_url = ""
    if image is not None:
        image_url = https_url(image.get("src") or image.get("data-src"))
    slug = urlparse(article_url).path.rsplit("/", 1)[-1]
    return Promotion(
        id=slug,
        title=title,
        summary=clean_text(paragraph.get_text(" ", strip=True) if paragraph else "", limit=1_000),
        image_url=image_url,
        article_url=article_url,
        scraped_at=scraped_at,
    )


def parse_promotion_html(
    html: str | bytes,
    source_url: str = PROMOTION_CATEGORY_URL,
    *,
    scraped_at: str | None = None,
) -> tuple[Promotion, ...]:
    """Extract unique article cards from SSR Next.js data and rendered anchors."""

    observed_at = scraped_at or utc_now_iso()
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[Promotion] = []
    node = soup.select_one("script#__NEXT_DATA__")
    if node is not None:
        try:
            payload = json.loads(node.get_text() or "{}")
        except json.JSONDecodeError:
            payload = {}
        for mapping in _mapping_candidates(payload):
            promotion = _promotion_from_mapping(mapping, source_url, observed_at)
            if promotion is not None:
                candidates.append(promotion)
    for anchor in soup.select('a[href*="/review-article/"]'):
        promotion = _promotion_from_anchor(anchor, source_url, observed_at)
        if promotion is not None:
            candidates.append(promotion)

    merged: dict[str, Promotion] = {}
    for promotion in candidates:
        previous = merged.get(promotion.article_url)
        if previous is None:
            merged[promotion.article_url] = promotion
            continue
        merged[promotion.article_url] = Promotion(
            id=previous.id,
            title=max((previous.title, promotion.title), key=len),
            summary=max((previous.summary, promotion.summary), key=len),
            image_url=previous.image_url or promotion.image_url,
            article_url=previous.article_url,
            published_at=previous.published_at or promotion.published_at,
            scraped_at=observed_at,
        )
    return tuple(merged.values())


class PromotionScraper:
    """Fetch one public promotion category page outside the webhook."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.session = session or requests.Session()

    def refresh(self, output_path: str | Path | None = None) -> dict[str, Any]:
        source_url = self.settings.promotion_category_url
        if source_url != PROMOTION_CATEGORY_URL:
            parsed = urlparse(source_url)
            if (
                parsed.scheme != "https"
                or (parsed.hostname or "").casefold() not in {"mercular.com", "www.mercular.com"}
                or parsed.path.rstrip("/") != "/category-review-article/promotion"
            ):
                raise PromotionError("promotion category URL is outside the approved Mercular path")
        try:
            response = self.session.get(
                source_url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "th-TH,th;q=0.9,en;q=0.7",
                },
                timeout=self.settings.request_timeout_seconds,
            )
        except requests.RequestException as error:
            raise PromotionError(f"promotion page request failed: {error}") from error
        if response.status_code == 429:
            raise PromotionError("Mercular returned HTTP 429; keep the previous promotion snapshot")
        if response.status_code >= 400:
            raise PromotionError(f"promotion page returned HTTP {response.status_code}")

        generated_at = utc_now_iso()
        promotions = parse_promotion_html(
            response.content,
            source_url,
            scraped_at=generated_at,
        )
        if not promotions:
            raise PromotionError(
                "promotion parser found no article cards; previous snapshot was preserved"
            )
        destination = Path(output_path or self.settings.promotion_snapshot_path)
        previous_by_url: dict[str, Promotion] = {}
        try:
            raw = json.loads(destination.read_text(encoding="utf-8"))
            previous_by_url = {
                item.article_url: item
                for value in raw.get("promotions", [])
                if isinstance(value, Mapping)
                for item in (Promotion.from_dict(value),)
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        preserved = [
            Promotion(
                **{
                    **promotion.to_dict(),
                    "starts_at": previous_by_url.get(promotion.article_url, promotion).starts_at,
                    "ends_at": previous_by_url.get(promotion.article_url, promotion).ends_at,
                    "discount_summary": previous_by_url.get(
                        promotion.article_url, promotion
                    ).discount_summary,
                }
            )
            for promotion in promotions
        ]
        snapshot = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "source": {
                "name": "Mercular promotion articles",
                "url": source_url,
                "crawler": USER_AGENT,
            },
            "summary": {"promotions": len(preserved)},
            "promotions": [promotion.to_dict() for promotion in preserved],
        }
        write_snapshot(snapshot, destination)
        return snapshot


class PromotionRepository:
    """Auto-reloading, last-known-good promotion snapshot reader."""

    def __init__(self, path: str | Path | None = None, *, settings: Settings | None = None) -> None:
        settings = settings or Settings.from_env()
        self.path = Path(path or settings.promotion_snapshot_path).expanduser().resolve()
        self._lock = threading.RLock()
        self._signature: tuple[int, int] | None = None
        self._promotions: tuple[Promotion, ...] = ()
        self._last_error = ""
        self.reload()

    def reload(self) -> bool:
        try:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if raw.get("schema_version") != PROMOTION_SCHEMA_VERSION:
                raise PromotionError("unsupported promotion snapshot schema")
            promotions = tuple(
                Promotion.from_dict(value)
                for value in raw.get("promotions", [])
                if isinstance(value, Mapping)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError, PromotionError) as error:
            with self._lock:
                self._last_error = str(error)
            return False
        with self._lock:
            self._signature = signature
            self._promotions = promotions
            self._last_error = ""
        return True

    def _reload_if_changed(self) -> None:
        try:
            stat = self.path.stat()
        except OSError:
            return
        signature = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            unchanged = signature == self._signature
        if not unchanged:
            self.reload()

    def current(self, *, on_date: date | None = None, limit: int = 5) -> list[Promotion]:
        self._reload_if_changed()
        with self._lock:
            values = [promotion for promotion in self._promotions if promotion.is_current(on_date)]
        values.sort(key=lambda item: (item.published_at, item.id), reverse=True)
        return values[: max(0, min(5, int(limit)))]

    def all(self) -> list[Promotion]:
        self._reload_if_changed()
        with self._lock:
            return list(self._promotions)

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error


__all__ = [
    "PROMOTION_CATEGORY_URL",
    "PROMOTION_SCHEMA_VERSION",
    "Promotion",
    "PromotionError",
    "PromotionRepository",
    "PromotionScraper",
    "parse_promotion_html",
]
