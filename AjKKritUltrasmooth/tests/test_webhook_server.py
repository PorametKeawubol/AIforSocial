import base64
import hashlib
import hmac
import json

from advice_scraper import Branch
from webhook_server import Settings, create_app, format_branch_results


def test_format_branch_results_includes_a_name_and_one_search_link():
    message = format_branch_results(
        "หาดใหญ่",
        [Branch("Advice หาดใหญ่", "https://www.advice.co.th/branch/hatyai")],
    )

    assert "Advice หาดใหญ่" in message
    assert "https://www.advice.co.th/wheretobuy/search?keyword=" in message
    assert "https://www.advice.co.th/branch/hatyai" not in message


def test_format_branch_results_lists_every_branch_before_one_link_preview():
    branches = [
        Branch("Advice Ranot", "https://example.test/1"),
        Branch("Advice U054", "https://example.test/2"),
        Branch("Advice U055", "https://example.test/3"),
        Branch("Advice U096", "https://example.test/4"),
    ]

    message = format_branch_results("สงขลา", branches)

    assert all(branch.name in message for branch in branches)
    assert sum(1 for line in message.splitlines() if line[:1].isdigit() and ". " in line) == 4
    assert message.count("https://") == 1


def test_health_endpoint_does_not_require_line_api_call():
    app = create_app(
        settings=Settings("secret", "token"),
        searcher=object(),
        reply_sender=lambda token, text: None,
    )

    response = app.test_client().get("/healthz")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "line_configured": True}


def test_callback_accepts_a_valid_empty_line_webhook():
    secret = "test-channel-secret"
    body = json.dumps(
        {"destination": "Utest", "events": []}, separators=(",", ":")
    )
    signature = base64.b64encode(
        hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    ).decode()
    app = create_app(
        settings=Settings(secret, "token"),
        searcher=object(),
        reply_sender=lambda token, text: None,
    )

    response = app.test_client().post(
        "/callback",
        data=body,
        headers={"X-Line-Signature": signature, "Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert response.text == "OK"


def test_callback_dispatches_text_event_to_searcher_and_reply_sender():
    secret = "test-channel-secret"
    replies = []

    class FakeSearcher:
        def search(self, keyword):
            assert keyword == "หาดใหญ่"
            return [Branch("Advice หาดใหญ่", "https://example.test/branch")]

    app = create_app(
        settings=Settings(secret, "token"),
        searcher=FakeSearcher(),
        reply_sender=lambda token, text: replies.append((token, text)),
    )
    payload = {
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
                    "text": "หาดใหญ่",
                },
            }
        ],
    }
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    signature = base64.b64encode(
        hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    ).decode()

    response = app.test_client().post(
        "/callback",
        data=body,
        headers={"X-Line-Signature": signature, "Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert replies == [
        (
            "reply-token",
            'พบสาขา Advice สำหรับ "หาดใหญ่" จำนวน 1 สาขา\n\n'
            "1. Advice หาดใหญ่\n\n"
            "ค้นหารายละเอียดบน Advice: "
            "https://www.advice.co.th/wheretobuy/search?keyword=%E0%B8%AB%E0%B8%B2%E0%B8%94%E0%B9%83%E0%B8%AB%E0%B8%8D%E0%B9%88",
        )
    ]


def test_callback_ignores_redelivery_of_the_same_webhook_event():
    secret = "test-channel-secret"
    replies = []

    class FakeSearcher:
        calls = 0

        def search(self, keyword):
            self.calls += 1
            return [Branch(f"Advice {keyword}", "https://example.test/branch")]

    searcher = FakeSearcher()
    app = create_app(
        settings=Settings(secret, "token"),
        searcher=searcher,
        reply_sender=lambda token, text: replies.append((token, text)),
    )
    payload = {
        "destination": "Udestination",
        "events": [
            {
                "type": "message",
                "mode": "active",
                "timestamp": 1700000000000,
                "source": {"type": "user", "userId": "Uuser"},
                "webhookEventId": "01H00000000000000000000001",
                "deliveryContext": {"isRedelivery": True},
                "replyToken": "reply-token",
                "message": {
                    "type": "text",
                    "id": "2",
                    "quoteToken": "quote-token",
                    "text": "หาดใหญ่",
                },
            }
        ],
    }
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    signature = base64.b64encode(
        hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    ).decode()
    headers = {"X-Line-Signature": signature, "Content-Type": "application/json"}

    first = app.test_client().post("/callback", data=body, headers=headers)
    second = app.test_client().post("/callback", data=body, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert searcher.calls == 1
    assert len(replies) == 1
