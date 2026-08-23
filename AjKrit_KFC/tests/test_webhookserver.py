import base64
import hashlib
import hmac
import json

from linebot.v3.messaging import FlexMessage, ImageMessage, TextMessage

from intent_classifier import INTENT_LABELS, IntentClassifier, IntentPrediction
from webhookserver import Settings, build_kfc_menu_search_reply, create_app


class FakeAnswerer:
    def __init__(self):
        self.questions = []

    def answer(self, question):
        self.questions.append(question)
        return f"ตอบ: {question}"


class FakeIntentClassifier:
    def __init__(self, intent="greeting"):
        self.intent = intent

    def detect(self, text):
        return IntentPrediction(
            intent=self.intent,
            score=0.99,
            scores={intent: (0.99 if intent == self.intent else 0.1) for intent in INTENT_LABELS},
            backend="test",
            threshold=0.60,
        )


class FakeMenuSearchAnswerer(FakeAnswerer):
    def __init__(self, items):
        super().__init__()
        self.items = items
        self.searches = []

    def search_items(self, question, **kwargs):
        self.searches.append(question)
        limit = max(1, int(kwargs.get("limit", 2)))
        return [
            (max(0.01, 0.91 - index * 0.01), item)
            for index, item in enumerate(self.items[:limit])
        ]

    def find_item(self, item_key, **_kwargs):
        return next((item for item in self.items if item["id"] == item_key), None)


def _signature(secret, body):
    return base64.b64encode(
        hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    ).decode()


def test_health_endpoint_does_not_call_line_api():
    app = create_app(
        settings=Settings("secret", "token"),
        answerer=FakeAnswerer(),
        reply_sender=lambda token, text: None,
    )

    response = app.test_client().get("/healthz")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "line_configured": True}


def test_callback_verifies_signature_and_sends_a_text_reply():
    secret = "test-channel-secret"
    replies = []
    answerer = FakeAnswerer()
    app = create_app(
        settings=Settings(secret, "token"),
        answerer=answerer,
        intent_classifier=IntentClassifier(semantic_enabled=False),
        reply_sender=lambda token, text: replies.append((token, text)),
    )
    body = json.dumps(
        {
            "destination": "Udestination",
            "events": [
                {
                    "type": "message",
                    "mode": "active",
                    "timestamp": 1700000000000,
                    "source": {"type": "user", "userId": "Uuser"},
                    "webhookEventId": "01H00000000000000000000000",
                    "deliveryContext": {"isRedelivery": False},
                    "replyToken": "reply-token",
                    "message": {
                        "type": "text",
                        "id": "1",
                        "quoteToken": "quote-token",
                        "text": "เดอะบอกซ์ ซิกเนเจอร์ คืออะไร",
                    },
                }
            ],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )

    response = app.test_client().post(
        "/callback",
        data=body,
        headers={
            "X-Line-Signature": _signature(secret, body),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200
    assert answerer.questions == ["เดอะบอกซ์ ซิกเนเจอร์ คืออะไร"]
    assert replies == [("reply-token", "ตอบ: เดอะบอกซ์ ซิกเนเจอร์ คืออะไร")]


def test_callback_rejects_invalid_signature():
    app = create_app(
        settings=Settings("secret", "token"),
        answerer=FakeAnswerer(),
        reply_sender=lambda token, text: None,
    )

    response = app.test_client().post(
        "/callback",
        data='{"events":[]}',
        headers={"X-Line-Signature": "not-valid", "Content-Type": "application/json"},
    )

    assert response.status_code == 400


def test_greeting_intent_gets_conversational_reply_without_qa_lookup():
    secret = "test-channel-secret"
    replies = []
    answerer = FakeAnswerer()
    app = create_app(
        settings=Settings(secret, "token"),
        answerer=answerer,
        intent_classifier=FakeIntentClassifier(),
        reply_sender=lambda token, text: replies.append((token, text)),
    )
    body = json.dumps(
        {
            "destination": "Udestination",
            "events": [
                {
                    "type": "message",
                    "mode": "active",
                    "timestamp": 1700000000000,
                    "source": {"type": "user", "userId": "Uuser"},
                    "webhookEventId": "01H00000000000000000000002",
                    "deliveryContext": {"isRedelivery": False},
                    "replyToken": "greeting-reply-token",
                    "message": {
                        "type": "text",
                        "id": "3",
                        "quoteToken": "quote-token",
                        "text": "สวัสดี",
                    },
                }
            ],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )

    response = app.test_client().post(
        "/callback",
        data=body,
        headers={
            "X-Line-Signature": _signature(secret, body),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200
    assert answerer.questions == []
    assert replies[0][0] == "greeting-reply-token"
    assert "สวัสดีครับ" in replies[0][1]


def test_menu_command_replies_with_a_kfc_flex_carousel():
    secret = "test-channel-secret"
    replies = []
    app = create_app(
        settings=Settings(secret, "token"),
        answerer=FakeAnswerer(),
        menu_image_provider=lambda: [
            {
                "name": "เดอะบอกซ์ ซิกเนเจอร์",
                "price": "฿179.00",
                "image_url": "https://images.ctfassets.net/kfc/box.png",
                "url": "https://www.kfc.co.th/menu/meals",
            }
        ],
        reply_sender=lambda token, message: replies.append((token, message)),
    )
    body = json.dumps(
        {
            "destination": "Udestination",
            "events": [
                {
                    "type": "message",
                    "mode": "active",
                    "timestamp": 1700000000000,
                    "source": {"type": "user", "userId": "Uuser"},
                    "webhookEventId": "01H00000000000000000000001",
                    "deliveryContext": {"isRedelivery": False},
                    "replyToken": "menu-reply-token",
                    "message": {
                        "type": "text",
                        "id": "2",
                        "quoteToken": "quote-token",
                        "text": "menu",
                    },
                }
            ],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )

    response = app.test_client().post(
        "/callback",
        data=body,
        headers={
            "X-Line-Signature": _signature(secret, body),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200
    assert len(replies) == 1
    assert replies[0][0] == "menu-reply-token"
    assert isinstance(replies[0][1], list)
    assert "เมนู KFC ทั้งหมด 1 รายการ" in replies[0][1][0]
    assert isinstance(replies[0][1][1], FlexMessage)
    payload = replies[0][1][1].to_dict()
    assert payload["altText"] == "เมนู KFC"
    assert payload["contents"]["type"] == "carousel"
    assert payload["contents"]["contents"][0]["hero"]["url"].endswith("box.png")
    action = payload["contents"]["contents"][0]["footer"]["contents"][0]["action"]
    assert action["type"] == "postback"
    assert action["data"].startswith("menu_detail=")


def test_menu_postback_replies_with_image_and_full_detail_text():
    secret = "test-channel-secret"
    replies = []
    menu_item = {
        "id": "CAT41-TEST",
        "name": "ชุดอิ่มคุ้มไก่ไม่มีกระดูก",
        "price": "฿99.00",
        "image_url": "https://images.ctfassets.net/kfc/boneless.png",
        "components": ["ไก่ไม่มีกระดูกใหม่ 2 ชิ้น"],
        "choices": [
            {
                "group": "เลือกเครื่องเคียง",
                "options": ["เฟรนช์ฟรายส์ (ปกติ)", "มันบด"],
            }
        ],
        "url": "https://www.kfc.co.th/menu/meals",
    }
    app = create_app(
        settings=Settings(secret, "token"),
        answerer=FakeAnswerer(),
        menu_image_provider=lambda: [menu_item],
        reply_sender=lambda token, message: replies.append((token, message)),
    )
    body = json.dumps(
        {
            "destination": "Udestination",
            "events": [
                {
                    "type": "postback",
                    "mode": "active",
                    "timestamp": 1700000000000,
                    "source": {"type": "user", "userId": "Uuser"},
                    "webhookEventId": "01H00000000000000000000005",
                    "deliveryContext": {"isRedelivery": False},
                    "replyToken": "detail-reply-token",
                    "postback": {"data": "menu_detail=CAT41-TEST"},
                }
            ],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )

    response = app.test_client().post(
        "/callback",
        data=body,
        headers={
            "X-Line-Signature": _signature(secret, body),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200
    assert len(replies) == 1
    assert replies[0][0] == "detail-reply-token"
    assert isinstance(replies[0][1], list)
    assert isinstance(replies[0][1][0], ImageMessage)
    assert isinstance(replies[0][1][1], TextMessage)
    detail_text = replies[0][1][1].text
    assert "ชุดอิ่มคุ้มไก่ไม่มีกระดูก" in detail_text
    assert "📋 รายการย่อย:" in detail_text
    assert "📦 ประกอบด้วย:" in detail_text
    assert "🎛️ เลือกได้:" in detail_text
    assert "  - มันบด" in detail_text


def test_menu_synonym_fills_the_carousel_with_up_to_twelve_results():
    secret = "test-channel-secret"
    replies = []

    def unavailable_image_provider():
        raise RuntimeError("temporary scraper outage")

    menu_items = [
        {
            "id": "zinger-burger",
            "name": "ซิงเกอร์ เบอร์เกอร์",
            "price": "฿89.00",
            "image_url": "https://images.ctfassets.net/kfc/zinger.png",
            "components": ["ไก่ซิงเกอร์ 1 ชิ้น"],
            "url": "https://www.kfc.co.th/menu/meals",
        },
        {
            "id": "zinger-set",
            "name": "ชุดซิงเกอร์เบอร์เกอร์",
            "price": "฿129.00",
            "image_url": "https://images.ctfassets.net/kfc/zinger-set.png",
            "components": ["ซิงเกอร์ เบอร์เกอร์ 1 ชิ้น", "เฟรนช์ฟรายส์ 1 ที่"],
            "url": "https://www.kfc.co.th/menu/meals",
        },
    ]
    menu_items.extend(
        {
            "id": f"menu-{index}",
            "name": f"เมนูทดสอบ {index}",
            "price": f"฿{index * 10}.00",
            "image_url": f"https://images.ctfassets.net/kfc/menu-{index}.png",
            "url": "https://www.kfc.co.th/menu/meals",
        }
        for index in range(3, 14)
    )
    answerer = FakeMenuSearchAnswerer(menu_items)
    app = create_app(
        settings=Settings(secret, "token"),
        answerer=answerer,
        # The intent double intentionally says greeting.  The menu synonym
        # itself must still route to the two-card carousel.
        intent_classifier=FakeIntentClassifier(),
        menu_image_provider=unavailable_image_provider,
        reply_sender=lambda token, message: replies.append((token, message)),
    )
    body = json.dumps(
        {
            "destination": "Udestination",
            "events": [
                {
                    "type": "message",
                    "mode": "active",
                    "timestamp": 1700000000000,
                    "source": {"type": "user", "userId": "Uuser"},
                    "webhookEventId": "01H00000000000000000000006",
                    "deliveryContext": {"isRedelivery": False},
                    "replyToken": "menu-search-reply-token",
                    "message": {
                        "type": "text",
                        "id": "4",
                        "quoteToken": "quote-token",
                        "text": "เมนูอาหาร",
                    },
                }
            ],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )

    response = app.test_client().post(
        "/callback",
        data=body,
        headers={
            "X-Line-Signature": _signature(secret, body),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200
    assert answerer.questions == []
    assert answerer.searches == ["เมนู"]
    assert replies[0][0] == "menu-search-reply-token"
    assert isinstance(replies[0][1], list)
    assert "เมนู KFC ทั้งหมด 13 รายการ" in replies[0][1][0]
    assert "แสดง 12 รายการแรกใน Carousel" in replies[0][1][0]
    assert isinstance(replies[0][1][1], FlexMessage)
    carousel = replies[0][1][1].to_dict()
    assert carousel["altText"] == "เมนู KFC"
    assert len(carousel["contents"]["contents"]) == 12
    assert carousel["contents"]["contents"][-1]["body"]["contents"][0]["text"] == "เมนูทดสอบ 12"


def test_partial_menu_name_reports_all_unique_matches_up_to_carousel_capacity():
    menu_items = [
        {
            "id": f"joy-menu-{index}",
            "name": f"เมนูใจ {index}",
            "price": f"฿{index * 10}.00",
            "image_url": f"https://images.ctfassets.net/kfc/joy-{index}.png",
            "url": "https://www.kfc.co.th/menu/meals",
        }
        for index in range(1, 14)
    ]
    # The scraper can yield a repeated product record.  It must not consume a
    # second Carousel card or inflate the result count.
    menu_items.append({**menu_items[0], "id": "joy-menu-duplicate"})

    reply = build_kfc_menu_search_reply(
        "ใจ", FakeMenuSearchAnswerer(menu_items), image_items=[]
    )

    assert isinstance(reply, list)
    summary, carousel = reply
    assert "พบเมนูใกล้เคียง 13 รายการ" in summary
    assert "พบทั้งหมด 13 รายการ แต่ Carousel แสดงได้สูงสุด 12 รายการ" in summary
    assert isinstance(carousel, FlexMessage)
    assert len(carousel.to_dict()["contents"]["contents"]) == 12


def test_partial_typo_routes_to_menu_carousel_and_postback_uses_catalog_detail():
    secret = "test-channel-secret"
    replies = []
    menu_items = [
        {
            "id": "zinger-burger",
            "name": "ซิงเกอร์ เบอร์เกอร์",
            "price": "฿89.00",
            "image_url": "https://images.ctfassets.net/kfc/zinger.png",
            "components": ["ไก่ซิงเกอร์ 1 ชิ้น"],
            "choices": [
                {"group": "เลือกเครื่องเคียง", "options": ["เฟรนช์ฟรายส์", "มันบด"]}
            ],
            "url": "https://www.kfc.co.th/menu/meals",
        },
        {
            "id": "zinger-set",
            "name": "ชุดซิงเกอร์เบอร์เกอร์",
            "price": "฿129.00",
            "image_url": "https://images.ctfassets.net/kfc/zinger-set.png",
            "components": ["ซิงเกอร์ เบอร์เกอร์ 1 ชิ้น", "เฟรนช์ฟรายส์ 1 ที่"],
            "url": "https://www.kfc.co.th/menu/meals",
        },
    ]
    answerer = FakeMenuSearchAnswerer(menu_items)
    app = create_app(
        settings=Settings(secret, "token"),
        answerer=answerer,
        intent_classifier=FakeIntentClassifier(),
        # Neither search result is in the image-provider cache.  The postback
        # must therefore use answerer.find_item.
        menu_image_provider=lambda: [],
        reply_sender=lambda token, message: replies.append((token, message)),
    )
    text_body = json.dumps(
        {
            "destination": "Udestination",
            "events": [
                {
                    "type": "message",
                    "mode": "active",
                    "timestamp": 1700000000000,
                    "source": {"type": "user", "userId": "Uuser"},
                    "webhookEventId": "01H00000000000000000000007",
                    "deliveryContext": {"isRedelivery": False},
                    "replyToken": "typo-search-reply-token",
                    "message": {
                        "type": "text",
                        "id": "5",
                        "quoteToken": "quote-token",
                        "text": "ซิงเกอ เบอเกอร์",
                    },
                }
            ],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )

    text_response = app.test_client().post(
        "/callback",
        data=text_body,
        headers={
            "X-Line-Signature": _signature(secret, text_body),
            "Content-Type": "application/json",
        },
    )

    assert text_response.status_code == 200
    assert answerer.searches == ["ซิงเกอ เบอเกอร์", "ซิงเกอ เบอเกอร์"]
    assert isinstance(replies[0][1][1], FlexMessage)

    postback_body = json.dumps(
        {
            "destination": "Udestination",
            "events": [
                {
                    "type": "postback",
                    "mode": "active",
                    "timestamp": 1700000000000,
                    "source": {"type": "user", "userId": "Uuser"},
                    "webhookEventId": "01H00000000000000000000008",
                    "deliveryContext": {"isRedelivery": False},
                    "replyToken": "catalog-detail-reply-token",
                    "postback": {"data": "menu_detail=zinger-burger"},
                }
            ],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )

    postback_response = app.test_client().post(
        "/callback",
        data=postback_body,
        headers={
            "X-Line-Signature": _signature(secret, postback_body),
            "Content-Type": "application/json",
        },
    )

    assert postback_response.status_code == 200
    assert isinstance(replies[1][1][0], ImageMessage)
    assert "📋 รายการย่อย:" in replies[1][1][1].text
    assert "  - มันบด" in replies[1][1][1].text
