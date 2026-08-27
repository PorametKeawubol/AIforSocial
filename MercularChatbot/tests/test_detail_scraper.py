from models import Product
import json
import requests

from detail_scraper import (
    ProductDetailPayloadUnavailableError,
    ProductDetailRateLimitedError,
    ProductDetailScraper,
    ProductDetailScraperError,
    enrich_product,
)
from scripts.enrich_product_details import _in_category_scopes, enrich_until_complete


def _product(**changes):
    values = {
        "id": "107694",
        "sku": "2046128000001",
        "name": "MONITOR HP OMEN 24",
        "brand": "HP",
        "category": "จอคอม",
        "category_path": ("คอมพิวเตอร์", "จอคอม"),
        "price": 3590.0,
        "original_price": 3590.0,
        "image_url": "",
        "product_url": "https://www.mercular.com/hp-omen-24-23-8-ips-fhd-gaming-monitor-165hz",
        "in_stock": False,
    }
    values.update(changes)
    return Product(**values)


def _payload():
    return {
        "props": {
            "pageProps": {
                "pageProps": {
                    "keyHighlightItems": ["จอขนาด 24 นิ้ว", "รีเฟรชเรต 165Hz"],
                    "productSpec": [
                        {"title": {"th": "ขนาดหน้าจอ"}, "desc": {"th": "23.8 นิ้ว"}},
                        {"title": {"th": "พาเนล"}, "desc": {"th": "IPS"}},
                    ],
                    "bestOf": {"title": "จอเกมมิ่งคุ้มค่า"},
                    "options": [
                        {
                            "perks": [
                                {
                                    "type": "warranty",
                                    "title": "ประกันศูนย์",
                                    "detail": "ประกัน 3 ปี",
                                },
                                {
                                    "type": "freeShipping",
                                    "title": "ส่งฟรี",
                                    "detail": "เมื่อซื้อครบ 500 บาท",
                                },
                            ]
                        }
                    ],
                }
            }
        }
    }


class _Response:
    def __init__(self, status_code, content, *, url=None, headers=None):
        self.status_code = status_code
        self.content = content
        self.url = url or _product().product_url
        self.headers = headers or {}
        self.history = ()


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class _TimeoutSession:
    def get(self, _url, **_kwargs):
        raise requests.Timeout("temporary DNS/connection timeout")


def test_hybrid_mode_reads_server_rendered_next_data_without_playwright():
    page = (
        '<html><body>4.8 จาก 5 รีวิว 4 คนแนะนำให้ซื้อ'
        '<script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(_payload(), ensure_ascii=False)
        + "</script></body></html>"
    ).encode()
    session = _Session([_Response(200, page)])
    scraper = ProductDetailScraper(session=session, sleeper=lambda _seconds: None)

    result = scraper.enrich([_product()])

    assert result.succeeded == 1
    assert result.failed == 0
    assert result.products[0].highlights == ("จอขนาด 24 นิ้ว", "รีเฟรชเรต 165Hz")
    assert result.products[0].rating == 4.8
    assert len(session.calls) == 1
    assert session.calls[0][1]["headers"]["User-Agent"]


def test_http_payload_unavailable_is_a_distinct_playwright_fallback_signal():
    scraper = ProductDetailScraper(
        session=_Session([_Response(200, b"<html><body>loading</body></html>")]),
        sleeper=lambda _seconds: None,
    )

    try:
        scraper._enrich_with_http(_product())
    except ProductDetailPayloadUnavailableError as error:
        assert "__NEXT_DATA__" in str(error)
    else:  # pragma: no cover - keeps the expected fallback contract explicit.
        raise AssertionError("missing payload must use the browser fallback")


def test_http_429_preserves_source_retry_after_signal():
    scraper = ProductDetailScraper(
        session=_Session([_Response(429, b"", headers={"retry-after": "120"})]),
        sleeper=lambda _seconds: None,
    )

    try:
        scraper._enrich_with_http(_product())
    except ProductDetailRateLimitedError as error:
        assert error.retry_after_seconds == 120
    else:  # pragma: no cover - makes source-safe behavior obvious.
        raise AssertionError("HTTP 429 must stop detail requests")


def test_temporary_transport_failure_is_marked_retryable_for_resume_runner():
    scraper = ProductDetailScraper(
        session=_TimeoutSession(),
        sleeper=lambda _seconds: None,
    )

    result = scraper.enrich([_product()])

    assert result.failed == 1
    assert result.errors[0]["retryable"] == "true"


def test_enrich_product_keeps_compact_structured_product_page_facts():
    result = enrich_product(
        _product(),
        _payload(),
        rendered_text="4.8 จาก 5 รีวิว 4 คนแนะนำให้ซื้อสินค้าชิ้นนี้",
        updated_at="2026-08-27T13:00:00+00:00",
    )

    assert result.highlights == ("จอขนาด 24 นิ้ว", "รีเฟรชเรต 165Hz")
    assert result.specifications == (("ขนาดหน้าจอ", "23.8 นิ้ว"), ("พาเนล", "IPS"))
    assert result.overview == "จอเกมมิ่งคุ้มค่า"
    assert result.rating == 4.8
    assert result.review_count == 5
    assert result.recommended_count == 4
    assert result.warranty == "ประกัน 3 ปี"
    assert result.service_notes == ("ส่งฟรี: เมื่อซื้อครบ 500 บาท",)
    assert result.detail_updated_at == "2026-08-27T13:00:00+00:00"
    assert "165Hz" in result.search_text


def test_enrich_product_rejects_non_product_next_payload():
    try:
        enrich_product(_product(), {})
    except ProductDetailScraperError as error:
        assert "Next.js" in str(error)
    else:  # pragma: no cover - makes the error expectation readable.
        raise AssertionError("invalid page payload must be rejected")


def test_root_slug_fallback_normalizes_legacy_nested_product_url():
    assert ProductDetailScraper._root_slug_url(
        "https://www.mercular.com/accessories/cable/cable-displayport-to-displayport-1m-ugreen-25903"
    ) == "https://www.mercular.com/cable-displayport-to-displayport-1m-ugreen-25903"


def test_safe_resume_requires_positive_batch_and_minimum_cooldown(monkeypatch, tmp_path):
    """The public loop is bounded by input validation before it touches Playwright."""

    from config import Settings

    settings = Settings(
        line_channel_secret="",
        line_channel_access_token="",
        snapshot_path=tmp_path / "missing.json",
        port=5000,
        log_level="INFO",
        top_k=5,
        history_ttl_seconds=60,
        history_size=1,
        request_timeout_seconds=5,
        request_retries=0,
        scrape_delay_seconds=1,
        max_products_per_category=1,
        stale_after_hours=24,
        category_urls=(),
        verify_robots=False,
    )
    monkeypatch.setattr(
        "scripts.enrich_product_details._load_snapshot",
        lambda _path: ({}, []),
    )

    result = enrich_until_complete(settings, batch_size=0, cooldown_seconds=0)

    assert result["complete"] is True
    assert result["remaining_unavailable"] == 0


def test_category_scopes_select_only_requested_main_catalog_roots():
    computer = _product(category_path=("คอมพิวเตอร์", "จอคอม"))
    mobile = _product(id="mobile", category_path=("Smartphone / Tablet / ACC", "มือถือ"))
    audio = _product(id="audio", category_path=("หูฟัง/ลำโพง", "หูฟัง"))

    scopes = ("computer", "smartphone-tablet-acc")

    assert _in_category_scopes(computer, scopes) is True
    assert _in_category_scopes(mobile, scopes) is True
    assert _in_category_scopes(audio, scopes) is False
