"""Flask webhook for a LINE Advice branch-search chatbot."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import quote

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    ApiException,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from advice_scraper import AdviceBranchSearcher, Branch, build_searcher_from_env


load_dotenv()
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    channel_secret: str
    access_token: str
    port: int = 5000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            channel_secret=(
                os.getenv("LINE_CHANNEL_SECRET") or os.getenv("CHANNEL_SECRET") or ""
            ).strip(),
            access_token=(
                os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
                or os.getenv("CHANNEL_ACCESS_TOKEN")
                or ""
            ).strip(),
            port=int(os.getenv("PORT", "5000")),
        )

    @property
    def configured(self) -> bool:
        return bool(self.channel_secret and self.access_token)


class RecentWebhookEvents:
    """Remember processed webhook IDs so LINE redelivery cannot reply twice."""

    def __init__(self, ttl_seconds: float = 600, max_entries: int = 1_000) -> None:
        self.ttl_seconds = max(1, float(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._events: dict[str, float] = {}
        self._lock = threading.Lock()

    def claim(self, event_id: str) -> bool:
        """Return whether this is the first delivery of ``event_id``."""

        now = time.monotonic()
        with self._lock:
            for known_event_id, expires_at in list(self._events.items()):
                if expires_at <= now:
                    del self._events[known_event_id]
            if event_id in self._events:
                return False
            while len(self._events) >= self.max_entries:
                oldest_event_id = next(iter(self._events))
                del self._events[oldest_event_id]
            self._events[event_id] = now + self.ttl_seconds
            return True


HELP_TEXT = (
    "พิมพ์ชื่อจังหวัด อำเภอ หรือพื้นที่ที่ต้องการค้นหาได้เลย\n"
    "ตัวอย่าง: หาดใหญ่"
)

ADVICE_SEARCH_PAGE_URL = "https://www.advice.co.th/wheretobuy/search"


def format_branch_results(keyword: str, branches: list[Branch], max_chars: int = 4500) -> str:
    """Format scraper results without one link preview per branch in LINE."""

    keyword = " ".join(keyword.split())
    if not branches:
        return f'ไม่พบสาขา Advice สำหรับ "{keyword}"\nลองค้นหาด้วยชื่อจังหวัดหรืออำเภออื่นได้เลย'

    lines = [f'พบสาขา Advice สำหรับ "{keyword}" จำนวน {len(branches)} สาขา', ""]
    for index, branch in enumerate(branches, start=1):
        line = f"{index}. {branch.name}"
        candidate = "\n".join(lines + [line])
        if len(candidate) > max_chars:
            lines.append("…แสดงผลเท่าที่ส่งได้ กรุณาระบุพื้นที่ให้แคบลง")
            break
        lines.append(line)

    # Advice's per-branch UI links contain a long percent-encoded branch name.
    # Sending one for every result makes LINE render stacked previews that can
    # visually hide later numbered items.  Keep one useful search link instead.
    search_link = f"{ADVICE_SEARCH_PAGE_URL}?keyword={quote(keyword, safe='')}"
    detail_line = f"ค้นหารายละเอียดบน Advice: {search_link}"
    if len("\n".join(lines + ["", detail_line])) <= max_chars:
        lines.extend(["", detail_line])
    return "\n".join(lines)


def create_app(
    settings: Settings | None = None,
    searcher: AdviceBranchSearcher | None = None,
    reply_sender: Callable[[str, str], None] | None = None,
) -> Flask:
    """Create the Flask app.

    ``searcher`` and ``reply_sender`` are injectable so the webhook can be
    tested without opening a browser or calling LINE's API.
    """

    settings = settings or Settings.from_env()
    app = Flask(__name__)
    app.config["LINE_CONFIGURED"] = settings.configured
    app.config["ADVICE_SEARCHER"] = searcher or build_searcher_from_env()
    app.config["WEBHOOK_EVENT_DEDUPLICATOR"] = RecentWebhookEvents(
        ttl_seconds=float(os.getenv("WEBHOOK_EVENT_TTL_SECONDS", "600")),
        max_entries=int(os.getenv("WEBHOOK_EVENT_MAX_ENTRIES", "1000")),
    )

    # Keep import/startup useful for health checks even before .env is filled
    # in, but never accept webhook traffic without real credentials.
    handler = WebhookHandler(settings.channel_secret or "missing-channel-secret")

    def send_reply(reply_token: str, message: str) -> bool:
        if reply_sender is not None:
            reply_sender(reply_token, message)
            return True
        if not settings.access_token:
            raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN is not configured")

        configuration = Configuration(access_token=settings.access_token)
        try:
            with ApiClient(configuration) as api_client:
                MessagingApi(api_client).reply_message(
                    ReplyMessageRequest(
                        reply_token=reply_token,
                        messages=[TextMessage(text=message)],
                    )
                )
        except ApiException as exc:
            # Reply tokens are single-use and short-lived.  Returning 200
            # prevents LINE from redelivering an already-expired event.
            LOGGER.warning(
                "Could not reply to LINE event (status=%s, reason=%s)",
                getattr(exc, "status", None),
                getattr(exc, "reason", None),
            )
            return False
        return True

    @handler.add(MessageEvent, message=TextMessageContent)
    def handle_text_message(event: MessageEvent) -> None:
        event_id = getattr(event, "webhook_event_id", "") or ""
        deduplicator: RecentWebhookEvents = app.config["WEBHOOK_EVENT_DEDUPLICATOR"]
        if event_id and not deduplicator.claim(event_id):
            LOGGER.info("Ignoring duplicate LINE webhook event %s", event_id)
            return

        text = (event.message.text or "").strip()
        if not text or text.lower() in {"help", "/help", "ช่วยเหลือ"}:
            send_reply(event.reply_token, HELP_TEXT)
            return

        try:
            branches = app.config["ADVICE_SEARCHER"].search(text)
            response = format_branch_results(text, branches)
        except ValueError as exc:
            response = str(exc)
        except Exception:
            LOGGER.exception("Advice branch search failed")
            response = "ขออภัย ระบบค้นหาสาขาขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง"

        send_reply(event.reply_token, response)

    @app.get("/")
    def index():
        return "LINE Advice branch chatbot is running"

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "line_configured": settings.configured})

    @app.post("/callback")
    def callback():
        if not settings.configured:
            LOGGER.error("Webhook credentials are not configured")
            abort(500, description="LINE credentials are not configured")

        signature = request.headers.get("X-Line-Signature")
        if not signature:
            abort(400, description="Missing X-Line-Signature header")

        # Read the exact raw body before parsing it.  LINE's signature is
        # calculated over these bytes/text without any JSON reformatting.
        body = request.get_data(as_text=True)
        LOGGER.info("Received LINE webhook (%d bytes)", len(body.encode("utf-8")))
        try:
            handler.handle(body, signature)
        except InvalidSignatureError:
            LOGGER.warning("Invalid LINE webhook signature")
            abort(400, description="Invalid X-Line-Signature")
        return "OK"

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    runtime_settings = Settings.from_env()
    app.run(host="0.0.0.0", port=runtime_settings.port, debug=False)
