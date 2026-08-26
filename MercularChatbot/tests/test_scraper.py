import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import requests

from config import Settings
from models import Product
from scraper import (
    InvalidCategoryURLError,
    MercularScraper,
    MercularScraperError,
    RobotsDeniedError,
    USER_AGENT,
    deduplicate_products,
    parse_html,
    write_snapshot,
)


FIXTURES = Path(__file__).parent / "fixtures"
AUDIO_URL = "https://www.mercular.com/audio"


def _settings(tmp_path, **changes):
    values = {
        "line_channel_secret": "",
        "line_channel_access_token": "",
        "snapshot_path": tmp_path / "mercular.json",
        "port": 5000,
        "log_level": "INFO",
        "top_k": 5,
        "history_ttl_seconds": 1800,
        "history_size": 20,
        "request_timeout_seconds": 7.5,
        "request_retries": 1,
        "scrape_delay_seconds": 0,
        "max_products_per_category": 20,
        "stale_after_hours": 24,
        "category_urls": (AUDIO_URL,),
        "verify_robots": False,
    }
    values.update(changes)
    return Settings(**values)


def _product(**changes):
    values = {
        "id": "p-1",
        "sku": "SKU-1",
        "name": "Test Product",
        "brand": "Test",
        "category": "Audio",
        "category_path": ("Audio",),
        "price": 1000.0,
        "original_price": 1200.0,
        "image_url": "https://cdn.example.com/p-1.jpg",
        "product_url": "https://www.mercular.com/test-product",
        "in_stock": True,
        "source_url": AUDIO_URL,
        "scraped_at": "2026-08-24T00:00:00+00:00",
    }
    values.update(changes)
    return Product(**values)


class FakeResponse:
    def __init__(self, text="", status=200, url=AUDIO_URL, headers=None):
        self.text = text
        self.content = text.encode()
        self.status_code = status
        self.url = url
        self.headers = headers or {}
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    def __init__(self, outcomes):
        self.headers = {}
        self.outcomes = {url: list(values) for url, values in outcomes.items()}
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes[url].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_current_next_data_extracts_complete_fields_and_skips_malformed_item():
    products = parse_html((FIXTURES / "next_category.html").read_text(), AUDIO_URL)

    assert len(products) == 2
    sony = products[0]
    assert sony.id == "1001"
    assert sony.sku == "SONY-WH1000XM5-B"
    assert sony.name == "Sony WH-1000XM5 Wireless Headphones"
    assert sony.brand == "Sony"
    assert sony.category == "เครื่องเสียง"
    assert sony.category_path == ("หน้าหลัก", "เครื่องเสียง")
    assert sony.price == 12990.0
    assert sony.original_price == 14990.0
    assert sony.image_url == "https://cdn.example.com/sony-xm5.jpg?width=640"
    assert sony.product_url == "https://www.mercular.com/sony-wh-1000xm5-wireless-headphones"
    assert sony.in_stock is True
    assert sony.tags == ("ไร้สาย", "ส่งฟรี")


def test_missing_price_and_image_are_kept_but_missing_name_or_url_is_not():
    products = parse_html((FIXTURES / "next_category.html").read_bytes(), AUDIO_URL)

    missing = next(product for product in products if product.sku == "NO-PRICE-IMAGE")
    assert missing.price is None
    assert missing.original_price is None
    assert missing.image_url == ""
    assert missing.in_stock is False
    assert all(product.sku != "BROKEN" for product in products)


def test_spreadsheet_error_placeholder_is_cleaned_out_of_catalog():
    html = """
    <script id="__NEXT_DATA__" type="application/json">
      {"props":{"pageProps":{"pageProps":{"productPageProps":{
        "categoryTitle":"พาวเวอร์แบงค์","breadcrumb":[],"products":[
          {"id":"bad","sku":"bad","title":"#REF!","slug":"-ref-",
           "price":100,"soldout":false}
        ]
      }}}}}
    </script>
    """

    assert parse_html(html, AUDIO_URL) == []


def test_obvious_brand_name_slug_conflict_is_corrected_conservatively():
    html = """
    <script id="__NEXT_DATA__" type="application/json">
      {"props":{"pageProps":{"pageProps":{"productPageProps":{
        "categoryTitle":"จอคอมพิวเตอร์","breadcrumb":[],"products":[
          {"id":"monitor-1","sku":"monitor-1",
           "title":"MONITOR 24.5 MSI PRO MP251",
           "slug":"monitor-24-5-msi-pro-mp251","brand":{"title":"AOC"},
           "price":3990,"soldout":false}
        ]
      }}}}}
    </script>
    """

    products = parse_html(html, AUDIO_URL)

    assert len(products) == 1
    assert products[0].brand == "MSI"


def test_disjoint_name_and_slug_brand_identities_are_rejected():
    html = """
    <script id="__NEXT_DATA__" type="application/json">
      {"props":{"pageProps":{"pageProps":{"productPageProps":{
        "categoryTitle":"กล้อง Webcam","breadcrumb":[],"products":[
          {"id":"bad-link","sku":"bad-link",
           "title":"Webcam ELGATO FACECAM 4K",
           "slug":"webcam-logitech-mx-brio-1","brand":{"title":"Elgato"},
           "price":7990,"soldout":false}
        ]
      }}}}}
    </script>
    """

    assert parse_html(html, AUDIO_URL) == []


def test_changed_next_shape_aggregates_tolerant_json_ld_and_html_fallbacks():
    products = parse_html((FIXTURES / "fallbacks.html").read_text(), "https://www.mercular.com/gaming")
    by_sku = {product.sku: product for product in products if product.sku}

    assert len(products) == 4
    assert by_sku["KEY-CHANGED-1"].price == 2990.0
    assert by_sku["KEY-CHANGED-1"].original_price == 3490.0
    assert by_sku["KEY-CHANGED-1"].brand == "KeyBrand"
    assert by_sku["MOUSE-SCHEMA-1"].price == 1590.0
    assert by_sku["CARD-2"].in_stock is False
    assert any(product.name == "Semantic Headset" for product in products)


def test_json_ld_is_used_when_next_data_has_no_valid_category_products():
    html = """
    <script id="__NEXT_DATA__" type="application/json">{"props":{"pageProps":{}}}</script>
    <script type="application/ld+json">
      {"@context":"https://schema.org","@type":"Product","productID":"schema-1",
       "sku":"MOUSE-SCHEMA-1","name":"Schema Gaming Mouse","url":"/schema-gaming-mouse",
       "image":["https://cdn.example.com/mouse.png"],"brand":{"name":"MouseBrand"},
       "description":"เมาส์น้ำหนักเบา","offers":{"price":"1,590.00","availability":"InStock"}}
    </script>
    """

    products = parse_html(html, "https://www.mercular.com/gaming")

    assert len(products) == 1
    assert products[0].sku == "MOUSE-SCHEMA-1"
    assert products[0].price == 1590.0
    assert products[0].description == "เมาส์น้ำหนักเบา"


def test_changed_html_selectors_are_last_fallback_and_nav_images_are_ignored():
    products = parse_html(
        (FIXTURES / "html_cards.html").read_text(),
        "https://www.mercular.com/gaming",
    )
    by_sku = {product.sku: product for product in products if product.sku}

    assert len(products) == 2
    assert by_sku["CARD-2"].image_url == "https://www.mercular.com/images/card-speaker.jpg"
    assert by_sku["CARD-2"].in_stock is False
    semantic = next(product for product in products if product.name == "Semantic Headset")
    assert semantic.price == 2490.0
    assert semantic.product_url == "https://www.mercular.com/semantic-headset"


def test_primary_next_products_take_priority_over_unrelated_json_ld_and_cards():
    primary = (FIXTURES / "next_category.html").read_text()
    extras = """
      <script type="application/ld+json">
        {"@type":"Product","name":"Recommendation","sku":"REC-1",
         "url":"/recommendation","offers":{"price":99}}
      </script>
      <a href="/promotion"><img src="/promo.jpg"><h3>Promotion tile</h3></a>
    """
    html = primary.replace("</body>", extras + "</body>")

    products = parse_html(html, AUDIO_URL)

    assert len(products) == 2
    assert {product.sku for product in products} == {"SONY-WH1000XM5-B", "NO-PRICE-IMAGE"}


def test_malformed_next_data_does_not_hide_valid_json_ld():
    html = """
    <script id="__NEXT_DATA__" type="application/json">{nope</script>
    <script type="application/ld+json">
      {"@type":"Product","name":"Fallback","sku":"FB-1",
       "url":"/fallback","offers":{"price":"499","availability":"OutOfStock"}}
    </script>
    """

    products = parse_html(html, AUDIO_URL)

    assert [product.sku for product in products] == ["FB-1"]
    assert products[0].price == 499.0
    assert products[0].image_url == ""
    assert products[0].in_stock is False


def test_dedupe_uses_any_sku_id_or_url_and_fills_missing_values():
    first = _product(price=None, image_url="")
    same_sku = _product(
        id="different-id",
        product_url="https://www.mercular.com/different-url",
        price=900,
        image_url="https://cdn.example.com/fill.jpg",
    )
    same_url = _product(id="another", sku="ANOTHER", price=800)

    products = deduplicate_products([first, same_sku, same_url])

    assert len(products) == 1
    assert products[0].id == "p-1"
    assert products[0].price == 900
    assert products[0].image_url == "https://cdn.example.com/fill.jpg"


def test_dedupe_treats_percent_encoded_and_unicode_paths_as_same_url():
    unicode_product = _product(product_url="https://www.mercular.com/หูฟัง-test")
    encoded_product = _product(
        id="other",
        sku="OTHER",
        product_url="https://www.mercular.com/%E0%B8%AB%E0%B8%B9%E0%B8%9F%E0%B8%B1%E0%B8%87-test",
    )

    assert len(deduplicate_products([unicode_product, encoded_product])) == 1


@pytest.mark.parametrize(("supplement", "expected"), ((False, False), (True, True), (None, None)))
def test_dedupe_fills_unknown_stock_with_explicit_duplicate_state(supplement, expected):
    primary = _product(in_stock=None)
    duplicate = _product(id="other", in_stock=supplement)

    assert deduplicate_products([primary, duplicate])[0].in_stock is expected


def test_dedupe_merges_transitive_identity_matches():
    first = _product(id="id-a", sku="sku-a", product_url="https://www.mercular.com/a")
    second = _product(id="id-b", sku="sku-b", product_url="https://www.mercular.com/b")
    bridge = _product(id="id-a", sku="sku-b", product_url="https://www.mercular.com/c")

    products = deduplicate_products([first, second, bridge])

    assert len(products) == 1
    assert products[0].id == "id-a"


def test_scrape_isolates_timeout_http_error_and_keeps_successful_category(tmp_path):
    html = (FIXTURES / "next_category.html").read_text()
    urls = (
        "https://www.mercular.com/audio",
        "https://www.mercular.com/gaming",
        "https://www.mercular.com/computer",
    )
    session = FakeSession(
        {
            urls[0]: [requests.Timeout("slow"), requests.Timeout("still slow")],
            urls[1]: [FakeResponse(status=404, url=urls[1])],
            urls[2]: [FakeResponse(html, url=urls[2])],
        }
    )
    settings = _settings(tmp_path, category_urls=urls, request_retries=1)
    scraper = MercularScraper(settings, session=session, sleeper=lambda _seconds: None)

    snapshot = scraper.scrape()

    assert snapshot["summary"] == {
        "categories_requested": 3,
        "categories_succeeded": 1,
        "categories_failed": 2,
        "products": 2,
    }
    assert [item["status"] for item in snapshot["categories"]] == ["error", "error", "ok"]
    assert {error["type"] for error in snapshot["errors"]} == {"Timeout", "HTTPError"}
    assert all(call[1]["timeout"] == 7.5 for call in session.calls)
    assert session.headers["User-Agent"] == USER_AGENT


def test_retryable_http_status_retries_then_succeeds(tmp_path):
    html = (FIXTURES / "next_category.html").read_text()
    session = FakeSession(
        {
            AUDIO_URL: [
                FakeResponse(status=503, url=AUDIO_URL),
                FakeResponse(html, url=AUDIO_URL),
            ]
        }
    )
    scraper = MercularScraper(
        _settings(tmp_path, request_retries=1),
        session=session,
        sleeper=lambda _seconds: None,
    )

    assert len(scraper.scrape()["products"]) == 2
    assert len(session.calls) == 2


def test_rate_limit_spaces_network_requests(tmp_path):
    html = (FIXTURES / "next_category.html").read_text()
    second_url = "https://www.mercular.com/gaming"
    session = FakeSession(
        {
            AUDIO_URL: [FakeResponse(html, url=AUDIO_URL)],
            second_url: [FakeResponse(html, url=second_url)],
        }
    )
    now = [100.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    scraper = MercularScraper(
        _settings(
            tmp_path,
            category_urls=(AUDIO_URL, second_url),
            request_retries=0,
            scrape_delay_seconds=1.25,
        ),
        session=session,
        sleeper=sleep,
        monotonic=lambda: now[0],
    )

    scraper.scrape()

    assert sleeps == [1.25]
    assert len(session.calls) == 2


def test_robots_check_blocks_category_without_requesting_it(tmp_path):
    robots_url = "https://www.mercular.com/robots.txt"
    session = FakeSession(
        {
            robots_url: [
                FakeResponse(
                    "User-agent: MercularSocialChatbot\nDisallow: /audio\n",
                    url=robots_url,
                )
            ]
        }
    )
    scraper = MercularScraper(
        _settings(tmp_path, verify_robots=True, request_retries=0),
        session=session,
        sleeper=lambda _seconds: None,
    )

    snapshot = scraper.scrape()

    assert snapshot["products"] == []
    assert snapshot["categories"][0]["status"] == "blocked"
    assert snapshot["errors"][0]["type"] == "RobotsDeniedError"
    assert [url for url, _kwargs in session.calls] == [robots_url]


def test_missing_robots_file_allows_configured_category(tmp_path):
    robots_url = "https://www.mercular.com/robots.txt"
    html = (FIXTURES / "next_category.html").read_text()
    session = FakeSession(
        {
            robots_url: [FakeResponse(status=404, url=robots_url)],
            AUDIO_URL: [FakeResponse(html, url=AUDIO_URL)],
        }
    )
    scraper = MercularScraper(
        _settings(tmp_path, verify_robots=True, request_retries=0),
        session=session,
        sleeper=lambda _seconds: None,
    )

    snapshot = scraper.scrape()

    assert snapshot["summary"]["products"] == 2
    assert snapshot["errors"] == []


def test_refuses_unconfigured_or_non_mercular_network_targets(tmp_path):
    with pytest.raises(InvalidCategoryURLError):
        MercularScraper(
            _settings(tmp_path, category_urls=("https://evil.example/audio",)),
            session=FakeSession({}),
        )

    scraper = MercularScraper(_settings(tmp_path), session=FakeSession({}))
    with pytest.raises(InvalidCategoryURLError):
        scraper.fetch_page("https://www.mercular.com/gaming")
    with pytest.raises(InvalidCategoryURLError):
        scraper.fetch_page("http://www.mercular.com/audio")


def test_redirect_is_validated_before_following_off_origin_target(tmp_path):
    session = FakeSession(
        {
            AUDIO_URL: [
                FakeResponse(
                    status=302,
                    url=AUDIO_URL,
                    headers={"Location": "http://127.0.0.1/private"},
                )
            ]
        }
    )
    scraper = MercularScraper(
        _settings(tmp_path, request_retries=0),
        session=session,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(InvalidCategoryURLError):
        scraper.fetch_page(AUDIO_URL)

    assert [url for url, _kwargs in session.calls] == [AUDIO_URL]
    assert session.calls[0][1]["allow_redirects"] is False


def test_same_scope_https_redirect_is_followed_with_a_hard_limit(tmp_path):
    redirected = f"{AUDIO_URL}/"
    html = (FIXTURES / "next_category.html").read_text()
    session = FakeSession(
        {
            AUDIO_URL: [
                FakeResponse(
                    status=301,
                    url=AUDIO_URL,
                    headers={"Location": redirected},
                )
            ],
            redirected: [FakeResponse(html, url=redirected)],
        }
    )
    scraper = MercularScraper(
        _settings(tmp_path, request_retries=0),
        session=session,
        sleeper=lambda _seconds: None,
    )

    assert len(scraper.fetch_page(AUDIO_URL)) > 0
    assert [url for url, _kwargs in session.calls] == [AUDIO_URL, redirected]
    assert all(not kwargs["allow_redirects"] for _url, kwargs in session.calls)


def test_write_snapshot_replaces_atomically_and_cleans_temporary_file(tmp_path, monkeypatch):
    destination = tmp_path / "nested" / "snapshot.json"
    destination.parent.mkdir()
    destination.write_text('{"old": true}', encoding="utf-8")
    observed = {}
    real_replace = os.replace

    def checking_replace(source, target):
        observed["old_during_replace"] = Path(target).read_text(encoding="utf-8")
        observed["temporary_json"] = json.loads(Path(source).read_text(encoding="utf-8"))
        real_replace(source, target)

    monkeypatch.setattr("scraper.os.replace", checking_replace)
    snapshot = {"schema_version": 1, "generated_at": "now", "products": []}

    result = write_snapshot(snapshot, destination)

    assert result == destination.resolve()
    assert observed["old_during_replace"] == '{"old": true}'
    assert observed["temporary_json"] == snapshot
    assert json.loads(destination.read_text(encoding="utf-8")) == snapshot
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_refresh_preserves_existing_snapshot_when_every_page_fails(tmp_path):
    target = tmp_path / "snapshot.json"
    target.write_text('{"known": "good"}', encoding="utf-8")
    session = FakeSession({AUDIO_URL: [requests.Timeout("offline")]})
    scraper = MercularScraper(
        _settings(tmp_path, snapshot_path=target, request_retries=0),
        session=session,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(MercularScraperError):
        scraper.refresh()

    assert target.read_text(encoding="utf-8") == '{"known": "good"}'


def test_zero_product_page_is_recorded_as_failure_not_false_success(tmp_path):
    session = FakeSession(
        {AUDIO_URL: [FakeResponse("<html><body>maintenance</body></html>", url=AUDIO_URL)]}
    )
    scraper = MercularScraper(
        _settings(tmp_path, request_retries=0),
        session=session,
        sleeper=lambda _seconds: None,
    )

    snapshot = scraper.scrape()

    assert snapshot["summary"] == {
        "categories_requested": 1,
        "categories_succeeded": 0,
        "categories_failed": 1,
        "products": 0,
    }
    assert snapshot["categories"][0]["status"] == "error"
    assert snapshot["errors"][0]["type"] == "MercularScraperError"


def test_partial_refresh_quality_gate_preserves_last_known_good(tmp_path):
    target = tmp_path / "snapshot.json"
    previous = '{"known": "good"}'
    target.write_text(previous, encoding="utf-8")
    html = (FIXTURES / "next_category.html").read_text()
    urls = (
        AUDIO_URL,
        "https://www.mercular.com/gaming",
        "https://www.mercular.com/computer",
    )
    session = FakeSession(
        {
            urls[0]: [FakeResponse(html, url=urls[0])],
            urls[1]: [requests.Timeout("offline")],
            urls[2]: [requests.Timeout("offline")],
        }
    )
    scraper = MercularScraper(
        _settings(
            tmp_path,
            snapshot_path=target,
            category_urls=urls,
            request_retries=0,
        ),
        session=session,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(MercularScraperError, match="quality gate"):
        scraper.refresh()

    assert target.read_text(encoding="utf-8") == previous


def test_large_product_count_drop_requires_explicit_override(tmp_path, monkeypatch):
    target = tmp_path / "snapshot.json"
    old_snapshot = {
        "schema_version": 1,
        "generated_at": "2026-08-24T00:00:00+00:00",
        "products": [{"id": str(index)} for index in range(10)],
    }
    target.write_text(json.dumps(old_snapshot), encoding="utf-8")
    html = (FIXTURES / "next_category.html").read_text()
    scraper = MercularScraper(
        _settings(tmp_path, snapshot_path=target, request_retries=0),
        session=FakeSession({AUDIO_URL: [FakeResponse(html, url=AUDIO_URL)]}),
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(MercularScraperError, match="product count fell"):
        scraper.refresh()

    assert json.loads(target.read_text()) == old_snapshot

    monkeypatch.setenv("SCRAPER_ALLOW_SHRINK", "true")
    scraper = MercularScraper(
        _settings(tmp_path, snapshot_path=target, request_retries=0),
        session=FakeSession({AUDIO_URL: [FakeResponse(html, url=AUDIO_URL)]}),
        sleeper=lambda _seconds: None,
    )
    refreshed = scraper.refresh()
    assert len(refreshed["products"]) == 2
