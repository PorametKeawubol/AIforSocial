"""Discover Mercular's public category hierarchy from its published sitemap.

This module only reads the category sitemap and normalises its public category
URLs.  It deliberately does not use the site's ``/browse`` route or search API.
The daily sync command chooses leaf URLs so a parent category is not fetched in
addition to each of its child categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse, urlunparse
from xml.etree import ElementTree

import requests


MERCULAR_HOST = "www.mercular.com"
MERCULAR_CATEGORY_SITEMAP_URL = "https://www.mercular.com/sitemaps/categories.xml"
SITEMAP_USER_AGENT = "MercularSocialChatbot/1.0 (educational category taxonomy refresh)"
_SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"


@dataclass(frozen=True, slots=True)
class CategorySource:
    """One canonical public category URL and its slash-separated hierarchy."""

    url: str
    path: tuple[str, ...]


def category_source_from_url(value: object) -> CategorySource | None:
    """Return a safe category URL, excluding sitemap placeholder routes."""

    parsed = urlparse(str(value or "").strip())
    if (
        parsed.scheme.casefold() != "https"
        or (parsed.hostname or "").casefold() != MERCULAR_HOST
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        return None
    raw_parts = tuple(part for part in parsed.path.split("/") if part)
    # URLs such as ``/smart-gadget/-/`` are sitemap placeholders, not a real
    # category in the taxonomy.  Keeping them would duplicate their parent.
    if (
        not raw_parts
        or raw_parts[0].casefold() == "browse"
        or any(part == "-" or part.startswith("--") for part in raw_parts)
    ):
        return None
    url = urlunparse(("https", MERCULAR_HOST, "/" + "/".join(raw_parts), "", "", ""))
    return CategorySource(url=url, path=raw_parts)


def parse_category_sitemap(xml: str | bytes) -> tuple[CategorySource, ...]:
    """Parse, validate, and de-duplicate category URLs from sitemap XML."""

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as error:
        raise ValueError("category sitemap is not valid XML") from error
    sources: dict[str, CategorySource] = {}
    for element in root.findall(f"{{{_SITEMAP_NAMESPACE}}}url/{{{_SITEMAP_NAMESPACE}}}loc"):
        source = category_source_from_url(element.text)
        if source is not None:
            sources.setdefault(source.url, source)
    return tuple(sorted(sources.values(), key=lambda source: (source.path, source.url)))


def leaf_categories(sources: Iterable[CategorySource]) -> tuple[CategorySource, ...]:
    """Keep categories that have no deeper category URL beneath them."""

    unique = {source.url: source for source in sources}
    values = tuple(unique.values())
    leaves = [
        source
        for source in values
        if not any(
            other.path[: len(source.path)] == source.path
            and len(other.path) > len(source.path)
            for other in values
        )
    ]
    return tuple(sorted(leaves, key=lambda source: (source.path, source.url)))


class CategorySitemapClient:
    """Fetch the public category sitemap with bounded timeouts."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        sitemap_url: str = MERCULAR_CATEGORY_SITEMAP_URL,
        timeout_seconds: float = 15.0,
    ) -> None:
        parsed = urlparse(sitemap_url)
        if (
            parsed.scheme.casefold() != "https"
            or (parsed.hostname or "").casefold() != MERCULAR_HOST
            or parsed.username
            or parsed.password
            or parsed.port not in (None, 443)
            or not parsed.path.startswith("/sitemaps/")
        ):
            raise ValueError("sitemap_url must be an HTTPS URL on www.mercular.com")
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": SITEMAP_USER_AGENT,
                "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1",
                "Accept-Language": "th,en;q=0.8",
            }
        )
        self.sitemap_url = sitemap_url
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def fetch(self) -> tuple[CategorySource, ...]:
        response = self.session.get(
            self.sitemap_url,
            timeout=self.timeout_seconds,
            allow_redirects=False,
        )
        response.raise_for_status()
        return parse_category_sitemap(response.content)


__all__ = [
    "CategorySitemapClient",
    "CategorySource",
    "MERCULAR_CATEGORY_SITEMAP_URL",
    "SITEMAP_USER_AGENT",
    "category_source_from_url",
    "leaf_categories",
    "parse_category_sitemap",
]
