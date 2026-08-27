from dataclasses import replace
from datetime import date
import json

from config import Settings
from promotions import (
    PROMOTION_CATEGORY_URL,
    Promotion,
    PromotionRepository,
    PromotionScraper,
    parse_promotion_html,
)


def _promotion(**changes):
    values = {
        "id": "payday-aug-2026",
        "title": "PAYDAY ลดสูงสุด 8,000 บาท",
        "summary": "คูปองทุกหมวดและดีลหูฟัง",
        "image_url": "https://cdn.example.com/payday.jpg",
        "article_url": "https://www.mercular.com/review-article/payday-aug-2026",
        "published_at": "2026-08-24",
        "starts_at": "2026-08-24",
        "ends_at": "2026-08-31",
        "discount_summary": "ลดสูงสุด 8,000 บาท",
        "scraped_at": "2026-08-27T00:00:00+07:00",
    }
    values.update(changes)
    return Promotion(**values)


def _html():
    payload = {
        "props": {
            "pageProps": {
                "articles": [
                    {
                        "title": "PAYDAY ลดสูงสุด 8,000 บาท",
                        "url": "/review-article/payday-aug-2026?tracking=ignored",
                        "shortDescription": "คูปองทุกหมวดและดีลหูฟัง",
                        "thumbnailUrl": "https://cdn.example.com/payday.jpg",
                        "publishedAt": "2026-08-24T10:00:00+07:00",
                    }
                ]
            }
        }
    }
    return (
        '<html><body><article><a href="/review-article/payday-aug-2026">'
        "<h2>PAYDAY ลดสูงสุด 8,000 บาท</h2></a></article>"
        '<script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(payload, ensure_ascii=False)
        + "</script></body></html>"
    ).encode()


def test_parse_promotion_html_merges_next_payload_and_dom_duplicates():
    promotions = parse_promotion_html(
        _html(),
        scraped_at="2026-08-27T01:00:00+00:00",
    )

    assert len(promotions) == 1
    assert promotions[0].id == "payday-aug-2026"
    assert promotions[0].summary == "คูปองทุกหมวดและดีลหูฟัง"
    assert promotions[0].published_at == "2026-08-24"
    assert "?" not in promotions[0].article_url


def test_promotion_current_dates_do_not_present_expired_campaigns():
    promotion = _promotion()

    assert promotion.is_current(date(2026, 8, 27)) is True
    assert promotion.is_current(date(2026, 9, 1)) is False
    assert _promotion(starts_at="", ends_at="").is_current(date(2030, 1, 1)) is True


class _Response:
    status_code = 200
    content = _html()


class _Session:
    def get(self, url, **kwargs):
        assert url == PROMOTION_CATEGORY_URL
        assert kwargs["headers"]["User-Agent"]
        return _Response()


def test_scraper_writes_snapshot_and_repository_auto_reads_it(tmp_path):
    path = tmp_path / "promotions.json"
    settings = replace(
        Settings.from_env(),
        promotion_snapshot_path=path,
        promotion_category_url=PROMOTION_CATEGORY_URL,
    )

    snapshot = PromotionScraper(settings, session=_Session()).refresh()
    repository = PromotionRepository(path, settings=settings)

    assert snapshot["summary"]["promotions"] == 1
    assert repository.all()[0].title.startswith("PAYDAY")


def test_checked_in_snapshot_contains_current_verified_promotion():
    repository = PromotionRepository(Settings.from_env().promotion_snapshot_path)

    current = repository.current(on_date=date(2026, 8, 27))

    assert current
    assert current[0].ends_at == "2026-08-31"
