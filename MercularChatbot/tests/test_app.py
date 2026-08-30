import base64
from dataclasses import replace
import hashlib
import hmac
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from linebot.v3.messaging import (
    ApiException,
    FlexMessage,
    MessagingApi,
    TextMessage,
)

from app import _recommendation_summary, create_app
from config import Settings
from models import Product
from nlp import SORT_PRICE_ASC, SORT_PRICE_DESC, ThaiCommandParser
from promotions import Promotion
from session_state import RecentQueries, RecentWebhookEvents


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
            "มีโปรโมชันอะไรบ้าง": "promotion",
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


class FakePromotionRepository:
    def current(self, **_kwargs):
        return [
            Promotion(
                id="payday",
                title="PAYDAY ลดสูงสุด 8,000 บาท",
                summary="คูปองล่าสุด",
                image_url="",
                article_url="https://www.mercular.com/review-article/payday",
                starts_at="2026-08-24",
                ends_at="2026-08-31",
            )
        ]


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


def _postback_body(product_id, event_id="evt-detail", reply_token="reply-detail"):
    return json.dumps(
        {
            "destination": "Udestination",
            "events": [
                {
                    "type": "postback",
                    "mode": "active",
                    "timestamp": 1787500000000,
                    "source": {"type": "user", "userId": "Uuser"},
                    "webhookEventId": event_id,
                    "deliveryContext": {"isRedelivery": False},
                    "replyToken": reply_token,
                    "postback": {"data": f'action=product_detail&product_id={product_id}'},
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


def test_promotion_intent_reads_promotion_snapshot_without_product_search():
    replies = []
    recommender = FakeRecommender()
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=recommender,
        promotion_repository=FakePromotionRepository(),
        reply_sender=lambda token, message: replies.append((token, message)),
    )

    response = _post(
        app.test_client(),
        _event_body("มีโปรโมชันอะไรบ้าง", event_id="evt-promotion"),
    )

    assert response.status_code == 200
    assert recommender.calls == []
    assert isinstance(replies[0][1], FlexMessage)
    assert "PAYDAY" in json.dumps(replies[0][1].to_dict(), ensure_ascii=False)


def test_dynamic_parser_is_reused_until_catalogue_vocabulary_changes(monkeypatch):
    import app as app_module

    real_parser = app_module.ThaiCommandParser
    builds = []

    class CountingParser:
        def __init__(self, *, brands, categories):
            builds.append((tuple(brands), tuple(categories)))
            self.delegate = real_parser(brands=brands, categories=categories)

        def parse(self, text):
            return self.delegate.parse(text)

    monkeypatch.setattr(app_module, "ThaiCommandParser", CountingParser)
    repository = FakeRepository([_product()])
    app = app_module.create_app(
        replace(_settings(), nlp_backend="rules"),
        repository=repository,
        recommender=FakeRecommender(),
        reply_sender=lambda *_: None,
    )
    client = app.test_client()

    assert _post(client, _event_body("หาหูฟัง", event_id="parser-cache-1")).status_code == 200
    assert _post(client, _event_body("หาหูฟัง", event_id="parser-cache-2")).status_code == 200
    assert len(builds) == 1

    repository.products.append(
        replace(_product(2), category="ลำโพง", category_path=("Audio", "ลำโพง"))
    )
    assert _post(client, _event_body("หาลำโพง", event_id="parser-cache-3")).status_code == 200
    assert len(builds) == 2


def test_product_list_question_opens_category_picker_instead_of_unknown_reply() -> None:
    replies = []
    parser = FakeParser()
    recommender = FakeRecommender()
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=parser,
        recommender=recommender,
        reply_sender=lambda token, message: replies.append((token, message)),
    )

    response = _post(
        app.test_client(),
        _event_body("มีสินค้าอะไรบ้าง", event_id="evt-category-picker"),
    )

    assert response.status_code == 200
    assert parser.inputs == []
    assert recommender.calls == []
    assert isinstance(replies[0][1], FlexMessage)
    assert "เลือกหมวดสินค้า" in replies[0][1].alt_text


def test_category_leaf_selection_runs_top_five_product_recommendation() -> None:
    replies = []
    recommender = FakeRecommender()
    app = create_app(
        _settings(),
        repository=FakeRepository([_product(index) for index in range(1, 7)]),
        parser=ThaiCommandParser(),
        recommender=recommender,
        reply_sender=lambda token, message: replies.append((token, message)),
    )

    response = _post(
        app.test_client(),
        _event_body(
            "ดูสินค้าหมวด: เกมมิ่ง > เมาส์เกมมิ่ง",
            event_id="evt-category-products",
        ),
    )

    assert response.status_code == 200
    assert recommender.calls[0][1].entities.category_path == (
        "เกมมิ่ง",
        "เมาส์เกมมิ่ง",
    )
    assert recommender.calls[0][1].entities.query == ""
    assert isinstance(replies[0][1], list)
    assert isinstance(replies[0][1][1], FlexMessage)


def test_explicit_comparison_uses_catalog_data_before_nlp_routing():
    replies = []
    parser = FakeParser()
    products = [
        replace(
            _product(1),
            name="MOUSE ALPHA X1",
            price=1990,
            specifications=(("น้ำหนัก", "59 กรัม"),),
        ),
        replace(
            _product(2),
            name="MOUSE BETA Z2",
            price=2490,
            specifications=(("น้ำหนัก", "63 กรัม"),),
        ),
    ]
    app = create_app(
        _settings(),
        repository=FakeRepository(products),
        parser=parser,
        recommender=FakeRecommender(),
        reply_sender=lambda token, message: replies.append((token, message)),
    )

    response = _post(
        app.test_client(),
        _event_body("Alpha X1 กับ Beta Z2 ต่างกันยังไง", event_id="evt-compare"),
    )

    assert response.status_code == 200
    assert parser.inputs == []
    assert "ถูกกว่า ฿500" in replies[0][1].text


def test_detail_postback_sets_context_for_followup_spec_question():
    replies = []
    product = replace(
        _product(7),
        specifications=(("การเชื่อมต่อ", "Bluetooth 5.2"),),
    )
    app = create_app(
        _settings(),
        repository=FakeRepository([product]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda token, message: replies.append((token, message)),
    )
    client = app.test_client()

    _post(client, _postback_body("7"))
    response = _post(
        client,
        _event_body(
            "ตัวนี้ใช้ Bluetooth ได้ไหม",
            event_id="evt-question",
            reply_token="reply-question",
        ),
    )

    assert response.status_code == 200
    assert "Bluetooth 5.2" in replies[-1][1].text


def test_cheaper_followup_reuses_filters_and_sets_strict_lower_ceiling():
    products = [_product(i) for i in range(1, 7)]
    recommender = FakeRecommender()
    parser = ThaiCommandParser(brands=["Logitech"], categories=["หูฟัง"])
    app = create_app(
        _settings(),
        repository=FakeRepository(products),
        parser=parser,
        recommender=recommender,
        reply_sender=lambda *_args: None,
    )
    client = app.test_client()

    _post(client, _event_body("หาหูฟัง งบ 3000", "evt-first", "reply-first"))
    response = _post(
        client,
        _event_body("ขอถูกกว่านี้", "evt-cheaper", "reply-cheaper"),
    )

    assert response.status_code == 200
    followup = recommender.calls[-1][1]
    assert followup.entities.category == "หูฟัง"
    assert followup.entities.max_price == 1001
    assert followup.entities.max_price_inclusive is False
    assert followup.entities.sort == SORT_PRICE_ASC


def test_brand_only_followup_merges_with_previous_category_and_budget():
    recommender = FakeRecommender()
    parser = ThaiCommandParser(brands=["Logitech"], categories=["หูฟัง"])
    app = create_app(
        _settings(),
        repository=FakeRepository([_product(i) for i in range(1, 4)]),
        parser=parser,
        recommender=recommender,
        reply_sender=lambda *_args: None,
    )
    client = app.test_client()

    _post(client, _event_body("หาหูฟัง งบ 3000", "evt-base", "reply-base"))
    _post(
        client,
        _event_body("เอา Logitech อย่างเดียว", "evt-brand", "reply-brand"),
    )

    followup = recommender.calls[-1][1]
    assert followup.entities.category == "หูฟัง"
    assert followup.entities.max_price == 3000
    assert followup.entities.brands == ("Logitech",)


def test_similar_but_cheaper_request_uses_the_opened_product_as_reference():
    recommender = FakeRecommender()
    source = replace(_product(7), category="เมาส์", price=3500)
    app = create_app(
        _settings(),
        repository=FakeRepository([source, _product(1)]),
        parser=ThaiCommandParser(categories=["เมาส์"]),
        recommender=recommender,
        reply_sender=lambda *_args: None,
    )
    client = app.test_client()

    _post(client, _postback_body("7", "evt-focus", "reply-focus"))
    response = _post(
        client,
        _event_body(
            "มีตัวคล้าย ๆ ตัวนี้แต่ถูกกว่าไหม",
            "evt-alternative",
            "reply-alternative",
        ),
    )

    assert response.status_code == 200
    command = recommender.calls[-1][1]
    assert command.entities.category == "เมาส์"
    assert command.entities.max_price == 3500
    assert command.entities.max_price_inclusive is False


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
    assert "แนะนำสินค้า 3 รายการ" in summary.text
    assert "จากข้อมูลที่มี" in summary.text
    assert "สุ่มสินค้า" not in summary.text
    assert "snapshot" not in summary.text
    assert len(carousel.to_dict()["contents"]["contents"]) == 3


def test_most_expensive_summary_describes_recommendation_and_ordering():
    summary = _recommendation_summary(5, 5, SORT_PRICE_DESC)

    assert "แนะนำสินค้า 5 รายการ" in summary
    assert "จากข้อมูลที่มี" in summary
    assert "เรียงราคาจากสูงไปต่ำ" in summary
    assert "สุ่มสินค้า" not in summary
    assert "snapshot" not in summary


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


def test_one_to_one_message_starts_five_second_loading_animation() -> None:
    loading_started = Event()
    calls = []

    def record_loading(chat_id, seconds):
        calls.append((chat_id, seconds))
        loading_started.set()

    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda *_: None,
        loading_sender=record_loading,
    )

    assert _post(app.test_client(), _event_body()).status_code == 200
    assert loading_started.wait(timeout=1)
    assert calls == [("Uuser", 5)]


def test_loading_animation_is_not_requested_for_group_chat() -> None:
    calls = []
    payload = json.loads(_event_body())
    payload["events"][0]["source"] = {
        "type": "group",
        "groupId": "Ggroup",
        "userId": "Uuser",
    }
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda *_: None,
        loading_sender=lambda *args: calls.append(args),
    )

    assert _post(app.test_client(), body).status_code == 200
    assert calls == []


def test_loading_animation_failure_never_blocks_reply() -> None:
    attempted = Event()
    replies = []

    def fail_loading(_chat_id, _seconds):
        attempted.set()
        raise RuntimeError("loading endpoint unavailable")

    app = create_app(
        _settings(),
        repository=FakeRepository([_product()]),
        parser=FakeParser(),
        recommender=FakeRecommender(),
        reply_sender=lambda token, message: replies.append((token, message)),
        loading_sender=fail_loading,
    )

    assert _post(app.test_client(), _event_body()).status_code == 200
    assert attempted.wait(timeout=1)
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
