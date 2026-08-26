import base64
from dataclasses import replace
import hashlib
import hmac
import json
from pathlib import Path
from types import SimpleNamespace

from linebot.v3.messaging import (
    ApiException,
    FlexMessage,
    MessagingApi,
    TextMessage,
    VideoMessage,
)

from app import RecentQueries, RecentWebhookEvents, create_app
from config import Settings
from models import Product


def _settings(secret="test-secret", token="test-token"):
    return Settings(
        line_channel_secret=secret,
        line_channel_access_token=token,
        snapshot_path=Path("unused.json"),
        port=5000,
        log_level="INFO",
        top_k=5,
        history_ttl_seconds=1800,
        history_size=20,
        request_timeout_seconds=10,
        request_retries=2,
        scrape_delay_seconds=1,
        max_products_per_category=20,
        stale_after_hours=24,
        category_urls=("https://www.mercular.com/audio",),
        verify_robots=True,
    )


def _product(index=1):
    return Product(
        id=str(index),
        sku=f"SKU-{index}",
        name=f"หูฟัง Test {index}",
        brand="Test",
        category="หูฟัง",
        category_path=("Audio", "หูฟัง"),
        price=1000.0 + index,
        original_price=1200.0 + index,
        image_url=f"https://example.com/{index}.jpg",
        product_url=f"https://www.mercular.com/product-{index}",
        in_stock=True,
        source_url="https://www.mercular.com/audio",
        scraped_at="2026-08-24T00:00:00+00:00",
    )


class FakeRepository:
    def __init__(self, products=()):
        self.products = list(products)
        self.is_ready = bool(self.products)
        self.is_stale = False

    def all(self):
        return list(self.products)

    def get(self, product_id):
        return next((p for p in self.products if p.id == product_id), None)

    def brands(self):
        return {p.brand for p in self.products}

    def categories(self):
        return {p.category for p in self.products}


class FakeParser:
    def __init__(self):
        self.inputs = []

    def parse(self, text):
        self.inputs.append(text)
        mapping = {
            "สวัสดี": "greeting",
            "ช่วยด้วย": "help",
            "สุ่มใหม่": "refresh",
            "???": "unknown",
        }
        return SimpleNamespace(
            intent=mapping.get(text, "product_search"),
            confidence=0.99,
            entities=SimpleNamespace(query=text),
        )


class FakeRecommender:
    def __init__(self):
        self.calls = []

    def recommend(self, products, parsed, **kwargs):
        self.calls.append((list(products), parsed, kwargs))
        return list(products)[: kwargs.get("top_k", 5)]


def _signature(secret, body):
    return base64.b64encode(
        hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    ).decode()


def _event_body(text="หาหูฟัง", event_id="evt-1", reply_token="reply-1"):
    return json.dumps(
        {
            "destination": "Udestination",
            "events": [
                {
                    "type": "message",
                    "mode": "active",
                    "timestamp": 1787500000000,
                    "source": {"type": "user", "userId": "Uuser"},
                    "webhookEventId": event_id,
                    "deliveryContext": {"isRedelivery": False},
                    "replyToken": reply_token,
                    "message": {
                        "type": "text",
                        "id": "message-1",
                        "quoteToken": "quote-token",
                        "text": text,
                    },
                }
            ],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _post(client, body, secret="test-secret"):
    return client.post(
        "/callback",
        data=body,
        headers={
            "X-Line-Signature": _signature(secret, body),
            "Content-Type": "application/json",
        },
    )


def test_health_and_readiness_report_local_state_without_external_calls():
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda *_: None,
    )

    health = app.test_client().get("/healthz")
    ready = app.test_client().get("/readyz")

    assert health.status_code == 200
    assert health.get_json() == {
        "status": "ok",
        "line_configured": True,
        "catalog_ready": True,
    }
    assert ready.status_code == 200
    assert ready.get_json()["status"] == "ready"


def test_readiness_fails_when_credentials_or_catalog_are_missing():
    app = create_app(
        _settings(secret="", token=""),
        repository=FakeRepository(),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda *_: None,
    )

    assert app.test_client().get("/healthz").status_code == 200
    assert app.test_client().get("/readyz").status_code == 503
    assert app.test_client().post("/callback").status_code == 503


def test_callback_rejects_missing_and_invalid_signatures():
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda *_: None,
    )
    client = app.test_client()

    assert client.post("/callback", data="{}").status_code == 400
    assert client.post(
        "/callback", data="{}", headers={"X-Line-Signature": "invalid"}
    ).status_code == 400


def test_product_query_sends_summary_and_exact_five_card_carousel():
    products = [_product(i) for i in range(1, 7)]
    replies = []
    parser = FakeParser()
    recommender = FakeRecommender()
    app = create_app(
        _settings(),
        repository=FakeRepository(products),
        parser=parser,
        recommender=recommender,
        reply_sender=lambda token, messages: replies.append((token, messages)),
    )

    response = _post(app.test_client(), _event_body())

    assert response.status_code == 200
    assert parser.inputs == ["หาหูฟัง"]
    assert recommender.calls[0][2]["user_id"] == "Uuser"
    assert replies[0][0] == "reply-1"
    assert isinstance(replies[0][1], list)
    assert isinstance(replies[0][1][0], TextMessage)
    assert isinstance(replies[0][1][1], FlexMessage)
    flex_payload = replies[0][1][1].to_dict()
    assert len(flex_payload["contents"]["contents"]) == 5
    # LINE only renders Quick Replies attached to the final message object.
    assert flex_payload["quickReply"]["items"][-1]["action"]["text"] == "สุ่มใหม่"


def test_configured_three_card_result_reports_three_not_five():
    settings = replace(_settings(), top_k=3)
    replies = []
    app = create_app(
        settings,
        repository=FakeRepository([_product(i) for i in range(1, 5)]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda token, messages: replies.append((token, messages)),
    )

    assert _post(app.test_client(), _event_body()).status_code == 200
    summary, carousel = replies[0][1]
    assert "สุ่มสินค้า 3 รายการ" in summary.text
    assert len(carousel.to_dict()["contents"]["contents"]) == 3


def test_duplicate_line_delivery_is_answered_only_once():
    replies = []
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda token, message: replies.append((token, message)),
    )
    body = _event_body()

    assert _post(app.test_client(), body).status_code == 200
    assert _post(app.test_client(), body).status_code == 200
    assert len(replies) == 1


def test_failed_reply_returns_500_and_releases_event_for_redelivery():
    attempts = []
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda token, message: attempts.append(token) or False,
    )
    body = _event_body()

    assert _post(app.test_client(), body).status_code == 500
    assert _post(app.test_client(), body).status_code == 500
    assert attempts == ["reply-1", "reply-1"]


def test_real_sender_network_exception_returns_500_for_line_redelivery(monkeypatch):
    attempts = []

    def fail_network(_self, _request, **kwargs):
        attempts.append(kwargs.get("_request_timeout"))
        raise RuntimeError("network down")

    monkeypatch.setattr(MessagingApi, "reply_message", fail_network)
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
    )

    assert _post(app.test_client(), _event_body()).status_code == 500
    assert attempts == [(2.0, 5.0)]


def test_permanent_rich_payload_rejection_tries_plain_text_once(monkeypatch):
    outgoing = []

    def reject_rich_then_accept_text(_self, request, **_kwargs):
        outgoing.append(request.messages)
        if len(outgoing) == 1:
            raise ApiException(status=400, reason="invalid flex")
        return None

    monkeypatch.setattr(MessagingApi, "reply_message", reject_rich_then_accept_text)
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
    )

    assert _post(app.test_client(), _event_body()).status_code == 200
    assert len(outgoing) == 2
    assert len(outgoing[0]) == 2
    assert len(outgoing[1]) == 1
    assert isinstance(outgoing[1][0], TextMessage)


def test_retryable_line_api_rejection_returns_500_without_payload_retry(monkeypatch):
    attempts = []

    def unavailable(_self, _request, **_kwargs):
        attempts.append(1)
        raise ApiException(status=503, reason="unavailable")

    monkeypatch.setattr(MessagingApi, "reply_message", unavailable)
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
    )

    assert _post(app.test_client(), _event_body()).status_code == 500
    assert attempts == [1]


def test_oversized_webhook_is_rejected_before_body_processing():
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda *_: None,
    )
    app.config["MAX_CONTENT_LENGTH"] = 100
    body = _event_body() + (" " * 101)

    assert _post(app.test_client(), body).status_code == 413


def test_standby_event_is_acknowledged_without_attempting_a_reply():
    replies = []
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda token, message: replies.append((token, message)),
    )
    event = json.loads(_event_body())
    event["events"][0]["mode"] = "standby"
    event["events"][0].pop("replyToken")
    body = json.dumps(event, separators=(",", ":"), ensure_ascii=False)

    assert _post(app.test_client(), body).status_code == 200
    assert replies == []


def test_refresh_reuses_the_users_previous_filters():
    replies = []
    recommender = FakeRecommender()
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=recommender,
        reply_sender=lambda token, message: replies.append((token, message)),
    )
    client = app.test_client()

    _post(client, _event_body("หาหูฟัง", "evt-search", "reply-search"))
    _post(client, _event_body("สุ่มใหม่", "evt-refresh", "reply-refresh"))

    assert len(recommender.calls) == 2
    assert recommender.calls[0][1] is recommender.calls[1][1]


def test_refresh_without_previous_query_returns_guidance_not_results():
    replies = []
    recommender = FakeRecommender()
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=recommender,
        reply_sender=lambda token, message: replies.append((token, message)),
    )

    _post(app.test_client(), _event_body("สุ่มใหม่"))

    assert recommender.calls == []
    assert isinstance(replies[0][1], TextMessage)
    assert "ยังไม่มีคำค้น" in replies[0][1].text


def test_empty_catalog_returns_data_unavailable_fallback():
    replies = []
    app = create_app(
        _settings(),
        repository=FakeRepository(),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda token, message: replies.append((token, message)),
    )

    assert _post(app.test_client(), _event_body()).status_code == 200
    assert isinstance(replies[0][1], TextMessage)
    assert "ข้อมูลสินค้า" in replies[0][1].text


def test_message_lab_bypasses_nlp_and_returns_showcase_hub():
    replies = []
    parser = FakeParser()
    app = create_app(
        replace(_settings(), public_base_url="https://mercumate.example"),
        repository=FakeRepository([_product()]),
        parser=parser,
        recommender=FakeRecommender(),
        reply_sender=lambda token, message: replies.append((token, message)),
    )

    response = _post(app.test_client(), _event_body("เดโมข้อความ"))

    assert response.status_code == 200
    assert parser.inputs == []
    assert isinstance(replies[0][1], FlexMessage)
    assert len(replies[0][1].to_dict()["contents"]["contents"]) == 3


def test_message_lab_can_send_public_video_message():
    replies = []
    app = create_app(
        replace(_settings(), public_base_url="https://mercumate.example"),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda token, message: replies.append((token, message)),
    )

    response = _post(
        app.test_client(),
        _event_body("เดโม:video", event_id="evt-video", reply_token="reply-video"),
    )

    assert response.status_code == 200
    assert isinstance(replies[0][1], VideoMessage)
    assert replies[0][1].original_content_url.startswith("https://")


def test_imagemap_route_serves_only_required_sizes_as_jpeg():
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda *_: None,
    )
    client = app.test_client()

    response = client.get("/media/imagemap/mercumate/1040")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.data[:3] == b"\xff\xd8\xff"
    assert client.get("/media/imagemap/mercumate/999").status_code == 404


def test_detail_postback_looks_up_product_and_returns_safe_detail_message():
    replies = []
    app = create_app(
        _settings(),
        repository=FakeRepository([_product(7)]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda token, message: replies.append((token, message)),
    )
    body = json.dumps(
        {
            "destination": "Udestination",
            "events": [
                {
                    "type": "postback",
                    "mode": "active",
                    "timestamp": 1787500000000,
                    "source": {"type": "user", "userId": "Uuser"},
                    "webhookEventId": "evt-detail",
                    "deliveryContext": {"isRedelivery": False},
                    "replyToken": "reply-detail",
                    "postback": {"data": '{"action":"product_detail","id":"7"}'},
                }
            ],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )

    assert _post(app.test_client(), body).status_code == 200
    assert replies[0][0] == "reply-detail"
    outgoing = replies[0][1]
    if isinstance(outgoing, list):
        assert any("หูฟัง Test 7" in getattr(item, "text", "") for item in outgoing)
    else:
        assert "หูฟัง Test 7" in outgoing.text


def test_recent_event_and_query_caches_expire_and_failed_event_can_be_released():
    now = [0.0]
    events = RecentWebhookEvents(ttl_seconds=10, clock=lambda: now[0])
    queries = RecentQueries(ttl_seconds=10, clock=lambda: now[0])
    parsed = object()

    assert events.claim("one") is True
    assert events.claim("one") is False
    events.release("one")
    assert events.claim("one") is True
    queries.remember("user", parsed)
    assert queries.get("user") is parsed
    now[0] = 11
    assert events.claim("one") is True
    assert queries.get("user") is None
