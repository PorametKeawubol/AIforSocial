"""Collect KFC Thailand menu and promotion data into a local JSON snapshot.

KFC Thailand is a client-rendered web application.  Its public page first
loads a public configuration document and then reads the live ordering
catalog plus promotion content, so looking only at the initial HTML often
returns no menu cards.  This module follows those public data paths and keeps
an HTML/Selenium fallback for future page changes.

No KFC configuration token is stored in this project: the public
configuration is read afresh from KFC on every refresh and is never written
to the JSON snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


load_dotenv()
LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "kfc_menu.json"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)
PRICE_PATTERN = re.compile(
    r"(?:฿|THB[ \t]*|บาท[ \t]*)([0-9][0-9,]*(?:[.][0-9]{1,2})?)"
    r"|([0-9][0-9,]*(?:[.][0-9]{1,2})?)[ \t]*(?:บาท|฿)"
    r"|([0-9][0-9,]*)[ \t]*[.][ \t]*-",
    re.IGNORECASE,
)
# The current KFC catalog exposes these items through the public Contentful
# ``item`` type with legacy English names, while customers search with the
# Thai display names.  Their Contentful records provide descriptions but not
# a channel-independent price, so no price is manufactured for them.
SUPPLEMENTAL_ITEM_MENUS = {
    "volcano wing": {
        "name": "วิงซ์ภูเขาไฟระเบิด",
        "aliases": ("Volcano WingZ", "Volcano Wing", "วิงซ์ภูเขาไฟระเบิด"),
    },
    "volcano wingz": {
        "name": "วิงซ์ภูเขาไฟระเบิด",
        "aliases": ("Volcano WingZ", "Volcano Wing", "วิงซ์ภูเขาไฟระเบิด"),
    },
}
# KFC's public web app currently exposes this Thai pickup catalog.  These are
# identifiers, not credentials; environment overrides make the scraper easy
# to update if KFC rolls over to a new catalog.
DEFAULT_OLO_TENANT_ID = "59dhhptudcn7hk1ogssvsb4cujvbcnh6o"
DEFAULT_OLO_CATALOG_NAME = "KFCThailandMenu-12197-web-pickup-th"


class KfcScraperError(RuntimeError):
    """Raised when a usable KFC snapshot cannot be collected."""


@dataclass(frozen=True)
class Source:
    """One public KFC page used for source attribution."""

    kind: str
    url: str


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _clean_text(value: object | None) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split())


def _public_image_url(value: object | None) -> str:
    """Return a cache-safe public image URL from a localized KFC field.

    KFC's live catalog stores ``imageName`` as a list of localized values.
    LINE requires an absolute HTTPS URL and the worksheet recommends removing
    query strings before sending the URL, so normalize both cases here.
    """

    candidates: list[str] = []

    def collect(node: object) -> None:
        if isinstance(node, str):
            candidate = node.strip()
            if candidate:
                candidates.append(candidate)
        elif isinstance(node, dict):
            for key in ("value", "url", "src", "imageUrl"):
                if key in node:
                    collect(node[key])
        elif isinstance(node, list):
            for child in node:
                collect(child)

    collect(value)
    for candidate in candidates:
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        normalized = urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.params, "", "")
        )
        if normalized:
            return normalized
    return ""


def _first_price(value: str) -> str:
    """Return a canonical Thai-baht price from human-readable text."""

    match = PRICE_PATTERN.search(value or "")
    if not match:
        return ""
    number = (match.group(1) or match.group(2) or match.group(3) or "").replace(",", "")
    try:
        amount = float(number)
    except ValueError:
        return ""
    return f"฿{amount:,.2f}"


def _rich_text_segments(value: object) -> list[str]:
    """Flatten Contentful rich text while preserving its text-node order."""

    segments: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("nodeType") == "text":
                text = _clean_text(node.get("value"))
                if text:
                    segments.append(text)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return segments


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _stable_id(
    kind: str, name: str, description: str, source_url: str, details: str = ""
) -> str:
    material = "\x1f".join((kind, name, description, source_url, details)).encode("utf-8")
    return hashlib.sha1(material).hexdigest()[:16]


def _deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for record in records:
        name = _clean_text(record.get("name"))
        if not name:
            continue
        key = (
            str(record.get("kind", "menu")),
            name.casefold(),
            _clean_text(record.get("description")).casefold(),
            json.dumps(record.get("components", []), ensure_ascii=False, sort_keys=True),
            json.dumps(record.get("choices", []), ensure_ascii=False, sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        record = dict(record)
        record["name"] = name
        record["description"] = _clean_text(record.get("description"))
        record["category"] = _clean_text(record.get("category"))
        record["aliases"] = list(
            dict.fromkeys(
                alias
                for alias in (_clean_text(alias) for alias in record.get("aliases", []))
                if alias
            )
        )
        result.append(record)
    return result


class KfcScraper:
    """Fetch public KFC content and normalize it for question answering."""

    def __init__(
        self,
        config_url: str | None = None,
        menu_url: str | None = None,
        promotion_url: str | None = None,
        locale: str | None = None,
        timeout_seconds: int | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config_url = (
            config_url
            or os.getenv("KFC_CONFIG_URL")
            or "https://www.kfc.co.th/api/v2/configs/th"
        )
        self.menu_url = (
            menu_url or os.getenv("KFC_MENU_URL") or "https://www.kfc.co.th/menu/meals"
        )
        self.promotion_url = (
            promotion_url
            or os.getenv("KFC_PROMOTION_URL")
            or "https://www.kfc.co.th/promos-rewards"
        )
        self.locale = locale or os.getenv("KFC_LOCALE", "th-TH")
        # Include KFC's live menu catalog by default; it can be disabled only
        # for an explicitly promotion-only deployment.
        self.include_menus = _truthy(os.getenv("KFC_INCLUDE_MENUS"), default=True)
        self.timeout_seconds = max(
            5, int(timeout_seconds or os.getenv("SCRAPER_TIMEOUT_SECONDS", "25"))
        )
        self.session = session or requests.Session()
        # ``requests.Session`` already supplies a Python User-Agent, so use
        # update rather than setdefault.  KFC's edge sometimes serves a slow
        # anti-bot path to that default header.
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/json, text/plain, */*",
            }
        )

    def _get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        response = self.session.get(
            url, params=params, headers=headers, timeout=self.timeout_seconds
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise KfcScraperError(f"KFC returned non-JSON data from {url}") from exc
        if not isinstance(payload, dict):
            raise KfcScraperError(f"KFC returned an unexpected JSON shape from {url}")
        return payload

    def _public_content_endpoint(
        self, config: dict[str, Any] | None = None
    ) -> tuple[str, str]:
        """Read public KFC config and return Contentful endpoint/token in memory."""

        config = config or self._get_json(self.config_url)
        space = _clean_text(config.get("CONTENTFUL_SPACE"))
        environment = _clean_text(config.get("CONTENTFUL_ENVIRONMENT")) or "master"
        access_token = _clean_text(config.get("CONTENTFUL_ACCESS_TOKEN"))
        if not all((space, access_token)):
            raise KfcScraperError("KFC public content configuration is incomplete")
        endpoint = f"https://cdn.contentful.com/spaces/{space}/environments/{environment}/entries"
        return endpoint, access_token

    def _contentful_entries(
        self,
        endpoint: str,
        access_token: str,
        content_type: str,
        *,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all entries of one public Contentful type without persisting keys."""

        entries: list[dict[str, Any]] = []
        skip = 0
        limit = 1000
        while True:
            params: dict[str, Any] = {
                "access_token": access_token,
                "content_type": content_type,
                "locale": self.locale,
                "include": 2,
                "limit": limit,
                "skip": skip,
            }
            if query:
                params["query"] = query
            payload = self._get_json(endpoint, params=params)
            page = payload.get("items", [])
            if not isinstance(page, list):
                raise KfcScraperError("KFC content response does not contain an item list")
            entries.extend(item for item in page if isinstance(item, dict))
            total = int(payload.get("total", len(entries)) or 0)
            skip += len(page)
            if not page or skip >= total:
                return entries

    @staticmethod
    def _localized_text(value: object, locale: str = "th-TH") -> str:
        """Return a human-readable value from KFC's localized OLO fields."""

        if isinstance(value, str):
            return _clean_text(value)
        if isinstance(value, dict):
            return _clean_text(value.get("value") or value.get("name"))
        if not isinstance(value, list):
            return ""

        fallback = ""
        for entry in value:
            if isinstance(entry, str):
                fallback = fallback or _clean_text(entry)
                continue
            if not isinstance(entry, dict):
                continue
            text = _clean_text(entry.get("value") or entry.get("name"))
            if not text:
                continue
            language = _clean_text(entry.get("lang") or entry.get("locale"))
            if language.casefold() == locale.casefold():
                return text
            fallback = fallback or text
        return fallback

    def _guest_access_token(self, config: dict[str, Any]) -> str:
        """Get a short-lived public guest token without persisting it."""

        token_url = _clean_text(config.get("GUEST_TOKEN_URL"))
        client_id = _clean_text(config.get("GUEST_CLIENT_ID_WEB"))
        grant_type = _clean_text(config.get("GUEST_GRANT_TYPE"))
        scope = _clean_text(config.get("GUEST_SCOPE"))
        if not all((token_url, client_id, grant_type, scope)):
            raise KfcScraperError("KFC public guest-token configuration is incomplete")
        try:
            response = self.session.post(
                token_url,
                data={
                    "client_id": client_id,
                    "grant_type": grant_type,
                    "scope": scope,
                },
                headers={"Accept": "application/json"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise KfcScraperError("Could not obtain KFC's public guest access token") from exc
        access_token = _clean_text(payload.get("access_token")) if isinstance(payload, dict) else ""
        if not access_token:
            raise KfcScraperError("KFC public guest token response is incomplete")
        return access_token

    def _olo_catalog(self, config: dict[str, Any]) -> dict[str, Any]:
        """Fetch KFC's current public Thai ordering catalog in memory."""

        base_url = _clean_text(config.get("TENANT_OS_ENDPOINT_BASE_URL")).rstrip("/")
        api_key = _clean_text(config.get("KFC_APP_API_KEY") or config.get("APP_API_KEY"))
        tenant_id = _clean_text(os.getenv("KFC_OLO_TENANT_ID")) or DEFAULT_OLO_TENANT_ID
        catalog_name = (
            _clean_text(os.getenv("KFC_OLO_CATALOG_NAME")) or DEFAULT_OLO_CATALOG_NAME
        )
        if not all((base_url, api_key, tenant_id, catalog_name)):
            raise KfcScraperError("KFC public ordering-catalog configuration is incomplete")
        guest_token = self._guest_access_token(config)
        return self._get_json(
            f"{base_url}/v1/catalogs/{tenant_id}/{catalog_name}",
            headers={
                "Accept": "application/json",
                "Accept-Language": self.locale,
                "Authorization": f"Bearer {guest_token}",
                "x-api-key": api_key,
                "x-tenant-id": tenant_id,
            },
        )

    @staticmethod
    def _olo_price(item: dict[str, Any]) -> str:
        """Convert the OLO catalog's minor-unit price into Thai baht."""

        availability = item.get("availability")
        if not isinstance(availability, list):
            return ""
        for option in availability:
            if not isinstance(option, dict):
                continue
            raw_price = option.get("price")
            try:
                minor_units = float(str(raw_price).replace(",", ""))
            except (TypeError, ValueError):
                continue
            if minor_units > 0:
                return f"฿{minor_units / 100:,.2f}"
        return ""

    def _olo_description(self, item: dict[str, Any]) -> str:
        """Return KFC's own non-composition description for an OLO menu item."""

        name = self._localized_text(item.get("dname"), self.locale) or _clean_text(
            item.get("name")
        )
        detail = ""
        for field in ("longDescription", "shortDescription"):
            candidate = self._localized_text(item.get(field), self.locale)
            if candidate and candidate.casefold() != name.casefold():
                detail = candidate
                break

        return _clean_text(detail)

    def _olo_contents_and_choices(
        self, item: dict[str, Any]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Separate fixed contents from the selectable OLO modifier options."""

        components: list[str] = []
        choices: list[dict[str, Any]] = []
        modifier_groups = item.get("modgrpIds")
        if isinstance(modifier_groups, list):
            for group in modifier_groups:
                if not isinstance(group, dict):
                    continue
                group_name = self._localized_text(group.get("dname"), self.locale)
                group_name = group_name or _clean_text(group.get("name"))
                if not group_name:
                    continue
                options: list[str] = []
                modifiers = group.get("modifiers")
                if isinstance(modifiers, list):
                    for modifier in modifiers:
                        if not isinstance(modifier, dict):
                            continue
                        option_name = self._localized_text(
                            modifier.get("dname"), self.locale
                        )
                        option_name = option_name or _clean_text(modifier.get("name"))
                        if option_name:
                            options.append(option_name)
                options = list(dict.fromkeys(options))
                # A group with exactly one option bearing its own name is a
                # fixed component; otherwise the customer can choose from its
                # listed options (including a one-option delivery drink).
                if not options or (
                    len(options) == 1 and options[0].casefold() == group_name.casefold()
                ):
                    components.append(group_name)
                else:
                    choices.append({"group": group_name, "options": options})
        return list(dict.fromkeys(components)), choices

    @staticmethod
    def _olo_items(catalog: object) -> Iterable[dict[str, Any]]:
        """Yield catalog items while avoiding unrelated modifier metadata."""

        if isinstance(catalog, list):
            for node in catalog:
                yield from KfcScraper._olo_items(node)
            return
        if not isinstance(catalog, dict):
            return
        availability = catalog.get("availability")
        if (
            isinstance(availability, list)
            and catalog.get("id")
            and (catalog.get("name") or catalog.get("dname"))
        ):
            yield catalog
        for key in ("categories", "products", "items"):
            child = catalog.get(key)
            if isinstance(child, (dict, list)):
                yield from KfcScraper._olo_items(child)

    def _menu_from_olo_catalog(
        self, catalog: dict[str, Any], scraped_at: str
    ) -> list[dict[str, Any]]:
        """Normalize current, visible OLO items into menu records."""

        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in self._olo_items(catalog):
            item_id = _clean_text(item.get("id"))
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            if item.get("isHidden") is True or item.get("isCategoryHidden") is True:
                continue
            name = self._localized_text(item.get("dname"), self.locale) or _clean_text(
                item.get("name")
            )
            price = self._olo_price(item)
            if not name or not price:
                continue
            aliases = [name, _clean_text(item.get("name"))]
            components, choices = self._olo_contents_and_choices(item)
            records.append(
                self._record(
                    kind="menu",
                    name=name,
                    price=price,
                    description=self._olo_description(item),
                    category="เมนู KFC",
                    source_url=self.menu_url,
                    aliases=aliases,
                    components=components,
                    choices=choices,
                    image_url=_public_image_url(item.get("imageName")),
                    scraped_at=scraped_at,
                )
            )
        return _deduplicate(records)

    def _record(
        self,
        *,
        kind: str,
        name: str,
        description: str = "",
        price: str = "",
        category: str = "",
        source_url: str,
        aliases: Iterable[str] = (),
        components: Iterable[str] = (),
        choices: Iterable[dict[str, Any]] = (),
        image_url: str = "",
        scraped_at: str,
    ) -> dict[str, Any]:
        name = _clean_text(name)
        description = _clean_text(description)
        normalized_components = list(
            dict.fromkeys(
                component
                for component in (_clean_text(value) for value in components)
                if component
            )
        )
        normalized_choices: list[dict[str, Any]] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            group = _clean_text(choice.get("group"))
            raw_options = choice.get("options", [])
            if not isinstance(raw_options, list):
                raw_options = []
            options = list(
                dict.fromkeys(
                    option
                    for option in (_clean_text(value) for value in raw_options)
                    if option
                )
            )
            if group and options:
                normalized_choices.append({"group": group, "options": options})
        detail_fingerprint = json.dumps(
            {"components": normalized_components, "choices": normalized_choices},
            ensure_ascii=False,
            sort_keys=True,
        )
        return {
            "id": _stable_id(kind, name, description, source_url, detail_fingerprint),
            "kind": kind,
            "name": name,
            "price": price,
            "description": description,
            "category": _clean_text(category),
            "aliases": list(aliases),
            "components": normalized_components,
            "choices": normalized_choices,
            "image_url": _public_image_url(image_url),
            "url": source_url,
            "scraped_at": scraped_at,
        }

    def _menu_from_contentful(
        self, entries: Iterable[dict[str, Any]], scraped_at: str
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for entry in entries:
            fields = entry.get("fields", {})
            if not isinstance(fields, dict):
                continue
            segments = _rich_text_segments(fields.get("body", {}))
            if not segments:
                continue
            price = _first_price(" ".join(segments))
            # Brand-page content also contains non-menu editorial cards.  A
            # displayed price is the reliable way to identify an actual item.
            if not price:
                continue
            name = segments[0]
            description_parts = [
                segment for segment in segments[1:] if _first_price(segment) != price
            ]
            description = "\n".join(description_parts)
            english_name = _clean_text(fields.get("name"))
            records.append(
                self._record(
                    kind="menu",
                    name=name,
                    price=price,
                    description=description,
                    category="เมนู KFC",
                    source_url=self.menu_url,
                    aliases=(english_name, name),
                    scraped_at=scraped_at,
                )
            )
        return records

    def _supplemental_menu_from_contentful(
        self, entries: Iterable[dict[str, Any]], scraped_at: str
    ) -> list[dict[str, Any]]:
        """Normalize known current items stored under Contentful's ``item`` type."""

        records: list[dict[str, Any]] = []
        for entry in entries:
            fields = entry.get("fields", {})
            if not isinstance(fields, dict):
                continue
            item_name = _clean_text(fields.get("itemName"))
            base_name = re.sub(r"\s*\([^)]*\)\s*$", "", item_name).casefold()
            spec = SUPPLEMENTAL_ITEM_MENUS.get(base_name)
            if not spec:
                continue
            description = _clean_text(
                fields.get("longDescription") or fields.get("shortDescription")
            )
            if not description:
                continue
            records.append(
                self._record(
                    kind="menu",
                    name=str(spec["name"]),
                    description=description,
                    category="เมนู KFC",
                    source_url=self.menu_url,
                    aliases=(item_name, *spec["aliases"]),
                    scraped_at=scraped_at,
                )
            )
        return _deduplicate(records)

    @staticmethod
    def _promotion_is_within_date_range(fields: dict[str, Any], now: datetime) -> bool:
        start = _parse_datetime(_clean_text(fields.get("startDate")))
        end = _parse_datetime(_clean_text(fields.get("endDate")))
        return not ((start and start > now) or (end and end < now))

    def _promotion_from_contentful(
        self,
        entries: Iterable[dict[str, Any]],
        scraped_at: str,
        *,
        dated: bool,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        now = datetime.now(UTC)
        for entry in entries:
            fields = entry.get("fields", {})
            if not isinstance(fields, dict):
                continue
            if dated and not self._promotion_is_within_date_range(fields, now):
                continue
            name = _clean_text(fields.get("headline") or fields.get("displayTitle"))
            if not name:
                continue
            description_parts = [
                _clean_text(fields.get("subHeadline")),
                _clean_text(fields.get("description")),
                _clean_text(fields.get("noteLine")),
            ]
            days = fields.get("applicableDays")
            if isinstance(days, list) and days:
                description_parts.append("ใช้ได้: " + ", ".join(map(str, days)))
            start = _clean_text(fields.get("startDate"))
            end = _clean_text(fields.get("endDate"))
            if start or end:
                description_parts.append(f"ช่วงโปรโมชัน: {start or '-'} ถึง {end or '-'}")
            destination = _clean_text(fields.get("onClick"))
            source_url = (
                urljoin("https://www.kfc.co.th/", destination)
                if destination
                else self.promotion_url
            )
            description = "\n".join(part for part in description_parts if part)
            records.append(
                self._record(
                    kind="promotion",
                    name=name,
                    price=_first_price(" ".join([name, description])),
                    description=description,
                    category="โปรโมชัน KFC",
                    source_url=source_url,
                    aliases=(name, _clean_text(fields.get("subHeadline"))),
                    scraped_at=scraped_at,
                )
            )
        return records

    def _scrape_public_dynamic_content(self, scraped_at: str) -> list[dict[str, Any]]:
        config = self._get_json(self.config_url)
        records: list[dict[str, Any]] = []
        if self.include_menus:
            try:
                records.extend(
                    self._menu_from_olo_catalog(self._olo_catalog(config), scraped_at)
                )
            except Exception as exc:
                # Contentful remains a useful compatibility fallback if the
                # live ordering service changes its public catalog contract.
                LOGGER.warning("KFC live ordering catalog failed: %s", exc)

        try:
            endpoint, access_token = self._public_content_endpoint(config)
            if self.include_menus and not any(
                record.get("kind") == "menu" for record in records
            ):
                menu_entries = self._contentful_entries(
                    endpoint, access_token, "brandPageProduct"
                )
                supplemental_item_entries = self._contentful_entries(
                    endpoint, access_token, "item", query="Volcano Wing"
                )
                records.extend(self._menu_from_contentful(menu_entries, scraped_at))
                records.extend(
                    self._supplemental_menu_from_contentful(
                        supplemental_item_entries, scraped_at
                    )
                )
            hero_entries = self._contentful_entries(endpoint, access_token, "marketingHero")
            limited_offer_entries = self._contentful_entries(
                endpoint, access_token, "limitedTimeOffers"
            )
            records.extend(
                self._promotion_from_contentful(hero_entries, scraped_at, dated=True)
            )
            records.extend(
                self._promotion_from_contentful(
                    limited_offer_entries, scraped_at, dated=False
                )
            )
        except Exception as exc:
            LOGGER.warning("KFC Contentful content failed: %s", exc)
        return _deduplicate(records)

    def _download_html(self, url: str) -> str:
        headers = {"Accept": "text/html,application/xhtml+xml"}
        response = self.session.get(url, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.text

    def _records_from_json_ld(
        self, soup: BeautifulSoup, source: Source, scraped_at: str
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        def visit(value: object) -> None:
            if isinstance(value, list):
                for child in value:
                    visit(child)
            elif isinstance(value, dict):
                item_type = value.get("@type", "")
                item_types = {item_type} if isinstance(item_type, str) else set(item_type)
                if item_types & {"Product", "MenuItem", "Offer"}:
                    name = _clean_text(value.get("name"))
                    description = _clean_text(value.get("description"))
                    offers = value.get("offers", {})
                    if isinstance(offers, list):
                        offers = offers[0] if offers else {}
                    raw_price = ""
                    if isinstance(offers, dict):
                        raw_price = _clean_text(offers.get("price"))
                    price = _first_price(raw_price) or (
                        f"฿{float(raw_price):,.2f}" if raw_price.replace(".", "", 1).isdigit() else ""
                    )
                    if name and (source.kind == "promotion" or price):
                        records.append(
                            self._record(
                                kind=source.kind,
                                name=name,
                                price=price,
                                description=description,
                                source_url=source.url,
                                aliases=(name,),
                                scraped_at=scraped_at,
                            )
                        )
                for child in value.values():
                    visit(child)

        for script in soup.select('script[type="application/ld+json"]'):
            try:
                visit(json.loads(script.get_text()))
            except (TypeError, ValueError):
                continue
        return records

    def parse_html(self, html: str, source: Source, scraped_at: str) -> list[dict[str, Any]]:
        """Parse a conventional KFC page when public dynamic content is unavailable."""

        soup = BeautifulSoup(html, "html.parser")
        records = self._records_from_json_ld(soup, source, scraped_at)
        selectors = (
            "article",
            "[class*='product-card']",
            "[class*='ProductCard']",
            "[class*='menu-card']",
            "[class*='MenuCard']",
            "[class*='offer-card']",
            "[class*='OfferCard']",
            "[class*='promotion-card']",
            "[class*='PromotionCard']",
        )
        for card in soup.select(",".join(selectors)):
            title = card.select_one(
                "h1,h2,h3,h4,[class*='title'],[class*='Title'],[class*='name'],[class*='Name']"
            )
            name = _clean_text(title.get_text(" ", strip=True) if title else "")
            if not name or len(name) > 160:
                continue
            text = _clean_text(card.get_text(" ", strip=True))
            price = _first_price(text)
            if source.kind == "menu" and not price:
                continue
            description_element = card.select_one(
                "p,[class*='description'],[class*='Description']"
            )
            description = _clean_text(
                description_element.get_text(" ", strip=True)
                if description_element
                else text.replace(name, "", 1)
            )
            link = card.select_one("a[href]")
            source_url = urljoin(source.url, link.get("href")) if link else source.url
            records.append(
                self._record(
                    kind=source.kind,
                    name=name,
                    price=price,
                    description=description,
                    source_url=source_url,
                    aliases=(name,),
                    scraped_at=scraped_at,
                )
            )
        return _deduplicate(records)

    @staticmethod
    def _image_records_from_html(
        html: str, source_url: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Extract product cards from the rendered KFC page.

        The class names used by the worksheet are included first, followed by
        lazy-image attributes used by newer builds of the KFC web app.  This
        keeps the browser path useful if the public catalog changes shape.
        """

        soup = BeautifulSoup(html, "html.parser")
        image_selector = (
            "img[class*='menu-product-image'],"
            "img[class*='product-image'],"
            "img[data-src],img[data-lazy-src],img[srcset]"
        )
        title_selector = (
            "h1,h2,h3,h4,h5,"
            "[class*='menu-product-header'],[class*='menu-product-title'],"
            "[class*='product-title'],[class*='menu-product-name'],"
            "[class*='product-name']"
        )
        records: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for image in soup.select(image_selector):
            raw_url = (
                image.get("src")
                or image.get("data-src")
                or image.get("data-lazy-src")
                or image.get("srcset", "").split(",", 1)[0].strip().split(" ", 1)[0]
            )
            image_url = _public_image_url(urljoin(source_url, raw_url))
            if not image_url:
                continue

            name = ""
            card = image
            for parent in [image, *list(image.parents)[:8]]:
                title = parent.select_one(title_selector)
                candidate = _clean_text(title.get_text(" ", strip=True) if title else "")
                if candidate and len(candidate) <= 160:
                    name = candidate
                    card = parent
                    break
            if not name:
                name = _clean_text(image.get("alt"))
            if not name or name.casefold() in {"kfc", "logo", "image"}:
                continue

            key = (name.casefold(), image_url)
            if key in seen:
                continue
            seen.add(key)
            link = card.select_one("a[href]") if hasattr(card, "select_one") else None
            item_url = urljoin(source_url, link.get("href")) if link else source_url
            text = _clean_text(card.get_text(" ", strip=True))
            records.append(
                {
                    "name": name,
                    "image_url": image_url,
                    "price": _first_price(text),
                    "description": text.replace(name, "", 1).strip(),
                    "url": item_url,
                }
            )
            if len(records) >= max_results:
                break
        return records

    def scrape_menu_images(self, max_results: int | None = None) -> list[dict[str, Any]]:
        """Return current KFC meal names and public image URLs.

        The live ordering catalog is the fast path and already contains the
        same Contentful image URLs displayed by the menu page.  Selenium is
        retained as the worksheet-compatible fallback for pages that expose
        images only after client-side rendering.
        """

        limit = max(1, int(max_results or os.getenv("MAX_RESULTS", "10")))
        records: list[dict[str, Any]] = []
        try:
            config = self._get_json(self.config_url)
            catalog = self._olo_catalog(config)
            seen: set[tuple[str, str]] = set()
            for item in self._olo_items(catalog):
                if item.get("isHidden") is True or item.get("isCategoryHidden") is True:
                    continue
                category_urls = item.get("categoryUrl")
                if category_urls:
                    if not isinstance(category_urls, list):
                        category_urls = [category_urls]
                    normalized_categories = {
                        _clean_text(value).strip("/").casefold()
                        for value in category_urls
                    }
                    if "meals" not in normalized_categories:
                        continue
                name = self._localized_text(item.get("dname"), self.locale) or _clean_text(
                    item.get("name")
                )
                image_url = _public_image_url(item.get("imageName"))
                if not name or not image_url:
                    continue
                components, choices = self._olo_contents_and_choices(item)
                key = (name.casefold(), image_url)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "id": _clean_text(item.get("id")),
                        "name": name,
                        "image_url": image_url,
                        "price": self._olo_price(item),
                        "description": self._olo_description(item),
                        "components": components,
                        "choices": choices,
                        "url": self.menu_url,
                    }
                )
                if len(records) >= limit:
                    break
        except Exception as exc:
            LOGGER.warning("KFC catalog image extraction failed: %s", exc)

        if records:
            return records

        try:
            html = self._render_with_selenium(self.menu_url)
            records = self._image_records_from_html(html, self.menu_url, limit)
            if records:
                return records
            raise KfcScraperError("Rendered KFC page did not contain meal images")
        except Exception as exc:
            LOGGER.warning("KFC Selenium image extraction failed: %s", exc)
            raise KfcScraperError(
                "No KFC meal images were collected. Check network access and retry."
            ) from exc

    def _render_with_selenium(self, url: str) -> str:
        """Render a fallback page only when the public dynamic path is unavailable."""

        try:
            from selenium import webdriver
            from selenium.common.exceptions import TimeoutException, WebDriverException
        except ImportError as exc:  # pragma: no cover - dependency setup error
            raise KfcScraperError("Selenium is not installed for browser fallback") from exc

        options = webdriver.ChromeOptions()
        options.page_load_strategy = "eager"
        if _truthy(os.getenv("HEADLESS"), default=True):
            options.add_argument("--headless=new")
        for argument in (
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--window-size=1440,1400",
            "--lang=th-TH",
        ):
            options.add_argument(argument)
        chrome_binary = _clean_text(os.getenv("CHROME_BINARY"))
        if chrome_binary:
            options.binary_location = chrome_binary

        if _truthy(os.getenv("CHROMEDRIVER_AUTO_INSTALL"), default=True):
            try:
                import chromedriver_autoinstaller

                chromedriver_autoinstaller.install()
            except Exception as exc:  # Selenium Manager may still locate a driver.
                LOGGER.debug("ChromeDriver auto-install was skipped: %s", exc)

        try:
            driver = webdriver.Chrome(options=options)
        except WebDriverException as exc:
            raise KfcScraperError("Could not start Chrome for KFC fallback scraping") from exc
        try:
            driver.set_page_load_timeout(self.timeout_seconds)
            try:
                driver.get(url)
            except TimeoutException:
                driver.execute_script("window.stop();")
            time.sleep(max(0, float(os.getenv("PAGE_SETTLE_SECONDS", "5"))))
            html = driver.page_source
        finally:
            driver.quit()

        if _truthy(os.getenv("SAVE_HTML"), default=False):
            output_dir = PROJECT_DIR / os.getenv("HTML_OUTPUT_DIR", "artifacts")
            output_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
            (output_dir / f"kfc-{digest}.html").write_text(html, encoding="utf-8")
        return html

    def _scrape_html_fallback(self, scraped_at: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        sources = [Source("promotion", self.promotion_url)]
        if self.include_menus:
            sources.insert(0, Source("menu", self.menu_url))
        for source in sources:
            try:
                html = self._download_html(source.url)
                parsed = self.parse_html(html, source, scraped_at)
                if not parsed and _truthy(
                    os.getenv("SCRAPER_USE_SELENIUM_FALLBACK"), default=True
                ):
                    parsed = self.parse_html(
                        self._render_with_selenium(source.url), source, scraped_at
                    )
                records.extend(parsed)
            except Exception as exc:
                LOGGER.warning("HTML fallback failed for %s: %s", source.url, exc)
        return _deduplicate(records)

    def scrape(self) -> list[dict[str, Any]]:
        """Return current public menu and promotion records."""

        scraped_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        try:
            records = self._scrape_public_dynamic_content(scraped_at)
        except Exception as exc:
            LOGGER.warning("KFC dynamic source failed; trying HTML fallback: %s", exc)
            records = []
        if not records:
            records = self._scrape_html_fallback(scraped_at)
        if not records:
            raise KfcScraperError(
                "No KFC menu or promotion records were collected. Check network access and retry."
            )
        return _deduplicate(records)

    def refresh(self, destination: Path = DATA_FILE) -> dict[str, Any]:
        """Refresh the local snapshot atomically and return the payload."""

        items = self.scrape()
        now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        payload = {
            "schema_version": 1,
            "generated_at": now,
            "scope": "menus_and_promotions" if self.include_menus else "promotions",
            "sources": (
                [self.menu_url, self.promotion_url]
                if self.include_menus
                else [self.promotion_url]
            ),
            "items": items,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
            ) as temporary:
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
        return payload


def load_snapshot(path: Path = DATA_FILE) -> dict[str, Any]:
    """Load the current snapshot, accepting the old list-only handout format too."""

    if not path.exists():
        return {"schema_version": 1, "items": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KfcScraperError(f"Could not read {path}") from exc
    if isinstance(payload, list):
        return {"schema_version": 1, "items": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("items", []), list):
        raise KfcScraperError(f"{path} has an invalid snapshot format")
    return payload


def search_menu(question: str) -> str:
    """Compatibility entry point used by the worksheet's webhook example."""

    from qa_engine import KfcQuestionAnswerer

    return KfcQuestionAnswerer().answer(question)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Refresh and query KFC menu data")
    parser.add_argument("--refresh", action="store_true", help="fetch a fresh KFC snapshot")
    parser.add_argument(
        "--build-index", action="store_true", help="build the multilingual BERT embedding cache"
    )
    parser.add_argument("--ask", metavar="QUESTION", help="ask the local KFC QA engine")
    args = parser.parse_args()

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    if args.refresh or (not args.build_index and not args.ask):
        snapshot = KfcScraper().refresh()
        counts = {
            kind: sum(1 for item in snapshot["items"] if item.get("kind") == kind)
            for kind in ("menu", "promotion")
        }
        print(
            f"Saved {len(snapshot['items'])} records "
            f"({counts['menu']} menu, {counts['promotion']} promotion) to {DATA_FILE}"
        )
    if args.build_index:
        from qa_engine import KfcQuestionAnswerer

        KfcQuestionAnswerer().build_index()
        print("Built multilingual BERT index")
    if args.ask:
        from qa_engine import KfcQuestionAnswerer

        print(KfcQuestionAnswerer().answer(args.ask))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
