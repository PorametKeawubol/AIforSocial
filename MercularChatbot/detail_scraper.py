"""Hybrid enrichment for structured Mercular product-page facts.

The category crawler intentionally stays lightweight.  This module is a separate,
opt-in product-page job that opens one headless browser outside the LINE webhook,
reads the server-rendered Next.js payload with a lightweight HTTP request first.  It
uses Playwright only for a page whose HTML does not contain that payload, and keeps
only compact product facts: highlights, specifications, rating/review counts,
warranty, and service notes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
import re
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import requests

try:
    from .config import Settings
    from .models import Product, clean_text, utc_now_iso
    from .scraper import MERCULAR_HOST, USER_AGENT
except ImportError:  # pragma: no cover - direct script execution support.
    from config import Settings
    from models import Product, clean_text, utc_now_iso
    from scraper import MERCULAR_HOST, USER_AGENT


LOGGER = logging.getLogger(__name__)
_COUNT_RE = re.compile(r"\b([0-9][0-9,]*)\s*(?:คน)?\s*รีวิว")
_RATING_RE = re.compile(r"\b([0-5](?:[.,][0-9])?)\s*(?:จาก\s*)?([0-9][0-9,]*)\s*รีวิว")
_RECOMMENDED_RE = re.compile(r"\b([0-9][0-9,]*)\s*คน\s*แนะนำให้ซื้อ")


class ProductDetailScraperError(RuntimeError):
    """Raised when a product page cannot be safely rendered or parsed."""


class ProductDetailRateLimitedError(ProductDetailScraperError):
    """Raised when Mercular asks this job to pause product-page requests."""

    def __init__(self, retry_after_seconds: int | None = None) -> None:
        self.retry_after_seconds = retry_after_seconds
        wait = (
            f"; retry after {retry_after_seconds} seconds"
            if retry_after_seconds is not None
            else ""
        )
        super().__init__(f"Mercular returned HTTP 429 Too Many Requests{wait}")


class ProductDetailPayloadUnavailableError(ProductDetailScraperError):
    """The server HTML did not expose a usable Next.js payload.

    This is deliberately separate from an HTTP or parsing failure: hybrid mode can
    use Playwright as a rendering fallback for this one product without making the
    normal detail path depend on a browser.
    """


class ProductDetailHTTPError(ProductDetailScraperError):
    """An HTTP product page response that is not usable for enrichment."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"product page returned HTTP {status_code}")


class ProductDetailTransportError(ProductDetailScraperError):
    """A temporary DNS, timeout, or connection failure."""


def _retryable_error(error: BaseException) -> bool:
    if isinstance(
        error,
        (ProductDetailRateLimitedError, ProductDetailTransportError),
    ):
        return True
    if isinstance(error, ProductDetailHTTPError):
        return error.status_code in {408, 425, 429} or error.status_code >= 500
    return type(error).__name__ in {"Error", "TimeoutError", "PlaywrightError"}


@dataclass(frozen=True, slots=True)
class DetailRefreshResult:
    """One bounded detail-enrichment result without mutating the input list."""

    products: tuple[Product, ...]
    requested: int
    succeeded: int
    failed: int
    skipped: int
    errors: tuple[dict[str, str], ...]
    rate_limited: bool = False
    retry_after_seconds: int | None = None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object, *, limit: int = 2_000) -> str:
    if isinstance(value, Mapping):
        for key in ("th", "name", "title", "value", "en"):
            text = _text(value.get(key), limit=limit)
            if text:
                return text
        return ""
    return clean_text(value, limit=limit)


def _html_text(value: object, *, limit: int = 1_200) -> str:
    raw = clean_text(value, limit=20_000)
    if not raw:
        return ""
    return clean_text(BeautifulSoup(raw, "html.parser").get_text(" ", strip=True), limit=limit)


def _number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _score(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    return result if 0 <= result <= 5 else None


def _retry_after_seconds(value: object) -> int | None:
    try:
        seconds = int(float(clean_text(value, limit=40)))
    except (TypeError, ValueError):
        return None
    return min(86_400, max(1, seconds))


def _page_props(payload: object) -> Mapping[str, Any]:
    """Read the stable product-page object inside Next.js page props."""

    current: object = payload
    for key in ("props", "pageProps", "pageProps"):
        current = _mapping(current).get(key, {})
    return _mapping(current)


def _unique_strings(values: Iterable[object], *, limit: int = 200) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value, limit=limit)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return tuple(result)


def _overview(props: Mapping[str, Any]) -> str:
    """Use the compact editorial summary only; never copy a full long review."""

    product_content = _html_text(props.get("productContent"), limit=1_000)
    if product_content:
        return product_content
    best_of = _mapping(props.get("bestOf"))
    return _text(best_of.get("title"), limit=500)


def _rating_from_rendered_text(rendered_text: object) -> tuple[float | None, int | None, int | None]:
    text = clean_text(rendered_text, limit=20_000)
    rating: float | None = None
    review_count: int | None = None
    recommended_count: int | None = None
    rating_match = _RATING_RE.search(text)
    if rating_match:
        rating = _score(rating_match.group(1))
        review_count = _number(rating_match.group(2))
    if review_count is None:
        count_match = _COUNT_RE.search(text)
        if count_match:
            review_count = _number(count_match.group(1))
    recommended_match = _RECOMMENDED_RE.search(text)
    if recommended_match:
        recommended_count = _number(recommended_match.group(1))
    return rating, review_count, recommended_count


def enrich_product(
    product: Product,
    page_payload: object,
    *,
    rendered_text: object = "",
    updated_at: str | None = None,
) -> Product:
    """Merge one rendered product-page payload into an existing catalog record."""

    props = _page_props(page_payload)
    if not props:
        raise ProductDetailScraperError("rendered page is missing Next.js product props")

    highlights = _unique_strings(_mapping_or_sequence(props.get("keyHighlightItems")))
    specifications: list[tuple[str, str]] = []
    seen_specs: set[tuple[str, str]] = set()
    for raw in _mapping_or_sequence(props.get("productSpec")):
        item = _mapping(raw)
        name = _text(item.get("title"), limit=200)
        value = _text(item.get("desc"), limit=500)
        pair = (name, value)
        if name and value and pair not in seen_specs:
            seen_specs.add(pair)
            specifications.append(pair)

    warranty = ""
    service_notes: list[str] = []
    for option in _mapping_or_sequence(props.get("options")):
        for perk in _mapping_or_sequence(_mapping(option).get("perks")):
            item = _mapping(perk)
            kind = _text(item.get("type"), limit=100).casefold()
            title = _text(item.get("title"), limit=200)
            detail = _text(item.get("detail"), limit=1_000)
            note = ": ".join(part for part in (title, detail) if part)
            if kind == "warranty":
                warranty = detail or title
            elif note:
                service_notes.append(note)

    rating, review_count, recommended_count = _rating_from_rendered_text(rendered_text)
    return replace(
        product,
        overview=_overview(props) or product.overview,
        highlights=highlights or product.highlights,
        specifications=tuple(specifications) or product.specifications,
        rating=rating if rating is not None else product.rating,
        review_count=review_count if review_count is not None else product.review_count,
        recommended_count=(
            recommended_count
            if recommended_count is not None
            else product.recommended_count
        ),
        warranty=warranty or product.warranty,
        service_notes=_unique_strings(service_notes, limit=1_000) or product.service_notes,
        detail_updated_at=updated_at or utc_now_iso(),
    )


def _mapping_or_sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


class ProductDetailScraper:
    """Use one Playwright browser to enrich a bounded set of product URLs."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.session = session or requests.Session()
        self._sleep = sleeper
        self._monotonic = monotonic
        self._last_request_at: float | None = None

    @staticmethod
    def _validated_product_url(value: str) -> str:
        parsed = urlparse(clean_text(value, limit=2_048))
        if (
            parsed.scheme != "https"
            or (parsed.hostname or "").casefold() != MERCULAR_HOST
            or parsed.username
            or parsed.password
            or parsed.port not in (None, 443)
            or not parsed.path.strip("/")
        ):
            raise ProductDetailScraperError(
                f"product URL is not an HTTPS {MERCULAR_HOST} page"
            )
        return f"https://{MERCULAR_HOST}{parsed.path.rstrip('/')}"

    @classmethod
    def _root_slug_url(cls, value: str) -> str:
        """Return Mercular's root-slug fallback for legacy nested product URLs."""

        validated = cls._validated_product_url(value)
        slug = urlparse(validated).path.rstrip("/").rsplit("/", 1)[-1]
        return f"https://{MERCULAR_HOST}/{slug}"

    def _rate_limit(self) -> None:
        now = self._monotonic()
        if self._last_request_at is not None:
            remaining = self.settings.detail_scrape_delay_seconds - (now - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._monotonic()

    def _fetch_http_payload(self, product_url: str) -> tuple[str, object, str]:
        """Fetch the already server-rendered Next.js product payload.

        Every response URL is validated, including redirect hops, before any HTML
        is parsed.  We make one ordinary request using the crawler's declared user
        agent; this is faster than starting Chromium and does not bypass source
        rate limits.
        """

        self._rate_limit()
        try:
            response = self.session.get(
                product_url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "th-TH,th;q=0.9,en;q=0.7",
                },
                timeout=self.settings.detail_scrape_timeout_seconds,
                allow_redirects=True,
            )
        except requests.RequestException as error:
            raise ProductDetailTransportError(
                f"product page request failed: {error}"
            ) from error

        try:
            for redirect in getattr(response, "history", ()):
                self._validated_product_url(str(redirect.url))
            final_url = self._validated_product_url(str(response.url))
        except (AttributeError, TypeError, ValueError, ProductDetailScraperError) as error:
            raise ProductDetailScraperError(f"product page redirected unsafely: {error}") from error

        status_code = int(response.status_code)
        if status_code == 429:
            raise ProductDetailRateLimitedError(
                _retry_after_seconds(response.headers.get("retry-after"))
            )
        if status_code >= 400 or 300 <= status_code < 400:
            raise ProductDetailHTTPError(status_code)

        payload_node = BeautifulSoup(response.content, "html.parser").select_one(
            "script#__NEXT_DATA__"
        )
        if payload_node is None:
            raise ProductDetailPayloadUnavailableError(
                "server HTML has no __NEXT_DATA__ payload"
            )
        try:
            payload = json.loads(payload_node.get_text() or "{}")
        except json.JSONDecodeError as error:
            raise ProductDetailPayloadUnavailableError(
                "server HTML has an invalid __NEXT_DATA__ payload"
            ) from error
        if not _page_props(payload):
            raise ProductDetailPayloadUnavailableError(
                "server HTML has no Next.js product props"
            )
        rendered_text = BeautifulSoup(response.content, "html.parser").get_text(" ", strip=True)
        return final_url, payload, rendered_text

    def _enrich_with_http(self, product: Product) -> Product:
        product_url = self._validated_product_url(product.product_url)
        try:
            final_url, payload, rendered_text = self._fetch_http_payload(product_url)
        except ProductDetailHTTPError as error:
            root_slug_url = self._root_slug_url(product_url)
            if error.status_code != 404 or root_slug_url == product_url:
                raise
            final_url, payload, rendered_text = self._fetch_http_payload(root_slug_url)
        return enrich_product(
            replace(product, product_url=final_url),
            payload,
            rendered_text=rendered_text,
        )

    def _enrich_with_playwright(
        self,
        products: Iterable[Product],
        *,
        limit: int | None = None,
        refresh_existing: bool = False,
    ) -> DetailRefreshResult:
        """Render selected product pages and return merged products plus errors."""

        all_products = tuple(products)
        selected = tuple(
            product
            for product in all_products
            if refresh_existing or not product.detail_updated_at
        )
        if limit is not None:
            selected = selected[: max(0, int(limit))]
        selected_ids = {product.id for product in selected}
        merged = {product.id: product for product in all_products}
        errors: list[dict[str, str]] = []
        succeeded = 0
        attempted = 0
        rate_limited = False
        retry_after_seconds: int | None = None

        if not selected:
            return DetailRefreshResult(
                products=all_products,
                requested=0,
                succeeded=0,
                failed=0,
                skipped=len(all_products),
                errors=(),
            )

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as error:  # pragma: no cover - dependency is declared.
            raise ProductDetailScraperError(
                "Playwright is not installed; run pip install -r requirements.txt and playwright install chromium"
            ) from error

        try:
            with sync_playwright() as playwright:
                launch_options: dict[str, Any] = {"headless": True}
                if self.settings.playwright_executable_path:
                    launch_options["executable_path"] = self.settings.playwright_executable_path
                    launch_options["args"] = ["--no-sandbox"]
                browser = playwright.chromium.launch(**launch_options)
                try:
                    context = browser.new_context(
                        user_agent=USER_AGENT,
                        locale="th-TH",
                        viewport={"width": 1440, "height": 1200},
                    )
                    # The product payload is in server-rendered HTML.  Blocking
                    # decoration assets reduces bandwidth while preserving that data.
                    context.route(
                        "**/*",
                        lambda route: (
                            route.abort()
                            if route.request.resource_type in {"image", "media", "font"}
                            else route.continue_()
                        ),
                    )
                    page = context.new_page()
                    for product in selected:
                        try:
                            attempted += 1
                            product_url = self._validated_product_url(product.product_url)
                            self._rate_limit()
                            response = page.goto(
                                product_url,
                                wait_until="domcontentloaded",
                                timeout=int(self.settings.detail_scrape_timeout_seconds * 1_000),
                            )
                            if response is not None and response.status == 404:
                                root_slug_url = self._root_slug_url(product_url)
                                if root_slug_url != product_url:
                                    self._rate_limit()
                                    response = page.goto(
                                        root_slug_url,
                                        wait_until="domcontentloaded",
                                        timeout=int(
                                            self.settings.detail_scrape_timeout_seconds * 1_000
                                        ),
                                    )
                            if response is not None and response.status >= 400:
                                if response.status == 429:
                                    raise ProductDetailRateLimitedError(
                                        _retry_after_seconds(response.headers.get("retry-after"))
                                    )
                                raise ProductDetailScraperError(
                                    f"product page returned HTTP {response.status}"
                                )
                            payload_node = page.locator("#__NEXT_DATA__")
                            payload_node.wait_for(
                                state="attached",
                                timeout=int(self.settings.detail_scrape_timeout_seconds * 1_000),
                            )
                            payload = json.loads(payload_node.text_content() or "{}")
                            merged[product.id] = enrich_product(
                                replace(
                                    product,
                                    product_url=self._validated_product_url(page.url),
                                ),
                                payload,
                                rendered_text=page.locator("body").inner_text(),
                            )
                            succeeded += 1
                        except (PlaywrightError, ValueError, ProductDetailScraperError) as error:
                            LOGGER.warning("Mercular product detail failed (%s): %s", product.product_url, error)
                            errors.append(
                                {
                                    "product_id": product.id,
                                    "url": product.product_url,
                                    "type": type(error).__name__,
                                    "message": clean_text(error, limit=500),
                                    "retryable": str(_retryable_error(error)).lower(),
                                }
                            )
                            if isinstance(error, ProductDetailRateLimitedError):
                                rate_limited = True
                                retry_after_seconds = error.retry_after_seconds
                                break
                    context.close()
                finally:
                    browser.close()
        except PlaywrightError as error:
            raise ProductDetailScraperError(f"Playwright could not start Chromium: {error}") from error

        return DetailRefreshResult(
            products=tuple(merged[product.id] for product in all_products),
            requested=attempted,
            succeeded=succeeded,
            failed=len(errors),
            skipped=len(all_products) - attempted,
            errors=tuple(errors),
            rate_limited=rate_limited,
            retry_after_seconds=retry_after_seconds,
        )

    def enrich(
        self,
        products: Iterable[Product],
        *,
        limit: int | None = None,
        refresh_existing: bool = False,
    ) -> DetailRefreshResult:
        """Enrich selected records, preferring server HTML over a browser.

        The HTTP route is deliberately sequential and goes through the same
        configured delay as the browser route.  Its speed benefit comes from not
        launching Chromium or executing client JavaScript for pages that already
        publish their data in ``__NEXT_DATA__``.
        """

        if self.settings.detail_scrape_mode == "playwright":
            return self._enrich_with_playwright(
                products,
                limit=limit,
                refresh_existing=refresh_existing,
            )

        all_products = tuple(products)
        selected = tuple(
            product
            for product in all_products
            if refresh_existing or not product.detail_updated_at
        )
        if limit is not None:
            selected = selected[: max(0, int(limit))]
        merged = {product.id: product for product in all_products}
        errors: list[dict[str, str]] = []
        render_fallback: list[Product] = []
        succeeded = 0
        attempted = 0
        rate_limited = False
        retry_after_seconds: int | None = None

        for product in selected:
            attempted += 1
            try:
                merged[product.id] = self._enrich_with_http(product)
                succeeded += 1
            except ProductDetailPayloadUnavailableError:
                render_fallback.append(product)
            except ProductDetailScraperError as error:
                LOGGER.warning("Mercular product detail failed (%s): %s", product.product_url, error)
                errors.append(
                    {
                        "product_id": product.id,
                        "url": product.product_url,
                        "type": type(error).__name__,
                        "message": clean_text(error, limit=500),
                        "retryable": str(_retryable_error(error)).lower(),
                    }
                )
                if isinstance(error, ProductDetailRateLimitedError):
                    rate_limited = True
                    retry_after_seconds = error.retry_after_seconds
                    break

        # Do not add browser traffic after the source asks us to slow down.  The
        # checkpointing runner will resume these still-missing records later.
        if render_fallback and not rate_limited:
            fallback_result = self._enrich_with_playwright(
                render_fallback,
                refresh_existing=True,
            )
            merged.update({product.id: product for product in fallback_result.products})
            succeeded += fallback_result.succeeded
            errors.extend(fallback_result.errors)
            rate_limited = fallback_result.rate_limited
            retry_after_seconds = fallback_result.retry_after_seconds

        return DetailRefreshResult(
            products=tuple(merged[product.id] for product in all_products),
            requested=attempted,
            succeeded=succeeded,
            failed=len(errors),
            skipped=len(all_products) - attempted,
            errors=tuple(errors),
            rate_limited=rate_limited,
            retry_after_seconds=retry_after_seconds,
        )


__all__ = [
    "DetailRefreshResult",
    "ProductDetailScraper",
    "ProductDetailScraperError",
    "ProductDetailPayloadUnavailableError",
    "ProductDetailRateLimitedError",
    "ProductDetailTransportError",
    "enrich_product",
]
