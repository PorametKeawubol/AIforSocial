"""Flask webhook for the Mercular LINE shopping-assistant demo."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from flask import Flask, abort, jsonify, request, send_from_directory
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
from linebot.v3.webhooks import MessageEvent, PostbackEvent, TextMessageContent

try:  # Package import for gunicorn/``python -m MercularChatbot.app``.
    from .config import Settings
    from .line_views import (
        build_product_carousel_message,
        contact_message,
        data_unavailable_message,
        greeting_message,
        help_message,
        no_results_message,
        parse_product_postback,
        product_detail_message,
        text_with_quick_replies,
    )
    from .nlp import SORT_NEWEST, SORT_POPULAR, ThaiCommandParser
    from .message_showcase import (
        IMAGEMAP_SIZES,
        build_showcase_message,
        parse_showcase_command,
        showcase_hub_message,
    )
    from .recommender import ProductRecommender
    from .repository import ProductRepository
except ImportError:  # pragma: no cover - direct execution from this folder.
    from config import Settings
    from line_views import (
        build_product_carousel_message,
        contact_message,
        data_unavailable_message,
        greeting_message,
        help_message,
        no_results_message,
        parse_product_postback,
        product_detail_message,
        text_with_quick_replies,
    )
    from nlp import SORT_NEWEST, SORT_POPULAR, ThaiCommandParser
    from message_showcase import (
        IMAGEMAP_SIZES,
        build_showcase_message,
        parse_showcase_command,
        showcase_hub_message,
    )
    from recommender import ProductRecommender
    from repository import ProductRepository


LOGGER = logging.getLogger(__name__)


class ReplyDeliveryError(RuntimeError):
    """Raised so the webhook returns non-2xx and LINE can redeliver."""


def _env_number(name: str, default: float, *, integer: bool = False) -> float | int:
    """Read optional webhook tuning without letting malformed env break startup."""

    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    value = max(1.0, value)
    return int(value) if integer else value


def _event_is_standby(event: object) -> bool:
    mode = getattr(event, "mode", "active")
    # LINE SDK v3 models this as an Enum; older versions may expose a string.
    value = getattr(mode, "value", mode)
    return str(value).casefold() == "standby"


def _retryable_line_error(exc: ApiException) -> bool:
    try:
        status = int(getattr(exc, "status", 0) or 0)
    except (TypeError, ValueError):
        status = 0
    return status == 0 or status in {408, 425, 429} or status >= 500


class RecentWebhookEvents:
    """Bounded in-memory protection against duplicate LINE deliveries."""

    def __init__(
        self,
        ttl_seconds: float = 600,
        max_entries: int = 2_000,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._events: dict[str, float] = {}
        self._lock = threading.Lock()

    def claim(self, event_id: str) -> bool:
        if not event_id:
            return True
        now = self._clock()
        with self._lock:
            expired = [key for key, expiry in self._events.items() if expiry <= now]
            for key in expired:
                self._events.pop(key, None)
            if event_id in self._events:
                return False
            while len(self._events) >= self.max_entries:
                self._events.pop(next(iter(self._events)))
            self._events[event_id] = now + self.ttl_seconds
            return True

    def release(self, event_id: str) -> None:
        """Allow LINE redelivery when no reply could be submitted."""

        if not event_id:
            return
        with self._lock:
            self._events.pop(event_id, None)


class RecentQueries:
    """Remember a user's last product query so “สุ่มใหม่” keeps its filters."""

    def __init__(
        self,
        ttl_seconds: float = 1_800,
        max_entries: int = 2_000,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._queries: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def remember(self, user_id: str, parsed_command: Any) -> None:
        if not user_id:
            return
        now = self._clock()
        with self._lock:
            self._prune(now)
            while len(self._queries) >= self.max_entries:
                self._queries.pop(next(iter(self._queries)))
            self._queries[user_id] = (now + self.ttl_seconds, parsed_command)

    def get(self, user_id: str) -> Any | None:
        if not user_id:
            return None
        now = self._clock()
        with self._lock:
            self._prune(now)
            value = self._queries.get(user_id)
            return value[1] if value else None

    def _prune(self, now: float) -> None:
        for key, (expiry, _value) in list(self._queries.items()):
            if expiry <= now:
                self._queries.pop(key, None)


def _configured(settings: Settings) -> bool:
    return bool(settings.line_channel_secret and settings.line_channel_access_token)


def _event_user_key(event: object) -> str:
    source = getattr(event, "source", None)
    for attribute in ("user_id", "group_id", "room_id"):
        value = getattr(source, attribute, None)
        if value:
            return str(value)
    return "anonymous"


def _repository_flag(repository: object, name: str, default: bool = False) -> bool:
    value = getattr(repository, name, default)
    try:
        return bool(value() if callable(value) else value)
    except Exception:
        return default


def _intent_reply(intent: str, *, bot_name: str = "MercuMate") -> object | None:
    if intent == "greeting":
        return greeting_message(bot_name)
    if intent == "help":
        return help_message()
    if intent == "thanks":
        return text_with_quick_replies(
            "ยินดีครับ 😊 ถ้าอยากดูสินค้าอื่น ลองระบุประเภท แบรนด์ และงบได้เลย"
        )
    if intent == "contact":
        return contact_message()
    if intent == "order":
        return text_with_quick_replies(
            "บอตสาธิตนี้ไม่รับคำสั่งซื้อหรือข้อมูลชำระเงินครับ "
            "กดปุ่ม “ซื้อที่ Mercular” บนการ์ดเพื่อยืนยันราคา สต็อก และสั่งซื้อบนเว็บไซต์ทางการ"
        )
    if intent == "unknown":
        return text_with_quick_replies(
            "ผมยังไม่แน่ใจว่าต้องการสินค้าชนิดไหนครับ ลองบอกประเภท แบรนด์ หรืองบ เช่น “หูฟัง Xiaomi ไม่เกิน 3000”"
        )
    return None


def create_app(
    settings: Settings | None = None,
    repository: ProductRepository | None = None,
    parser: ThaiCommandParser | None = None,
    recommender: ProductRecommender | None = None,
    reply_sender: Callable[[str, object | list[object]], Any] | None = None,
) -> Flask:
    """Create an injectable Flask app for production and offline tests."""

    settings = settings or Settings.from_env()
    catalog = repository or ProductRepository(settings.snapshot_path, settings=settings)
    command_parser = parser or ThaiCommandParser(
        brands=catalog.brands(), categories=catalog.categories()
    )
    dynamic_catalog_parser = parser is None
    selector = recommender or ProductRecommender(
        history_ttl_seconds=settings.history_ttl_seconds,
        history_size=settings.history_size,
    )

    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=_env_number(
            "WEBHOOK_MAX_BODY_BYTES", 1_000_000, integer=True
        ),
        LINE_CONFIGURED=_configured(settings),
        PRODUCT_REPOSITORY=catalog,
        COMMAND_PARSER=command_parser,
        PRODUCT_RECOMMENDER=selector,
        WEBHOOK_EVENT_DEDUPLICATOR=RecentWebhookEvents(
            ttl_seconds=_env_number("WEBHOOK_EVENT_TTL_SECONDS", 600),
            max_entries=_env_number(
                "WEBHOOK_EVENT_MAX_ENTRIES", 2_000, integer=True
            ),
        ),
        RECENT_QUERIES=RecentQueries(
            ttl_seconds=settings.history_ttl_seconds,
            max_entries=2_000,
        ),
    )

    # A placeholder keeps health endpoints available before credentials are set;
    # /callback still refuses traffic until both credentials exist.
    handler = WebhookHandler(settings.line_channel_secret or "missing-channel-secret")
    line_connect_timeout = float(
        _env_number("LINE_REPLY_CONNECT_TIMEOUT_SECONDS", 2.0)
    )
    line_read_timeout = float(_env_number("LINE_REPLY_READ_TIMEOUT_SECONDS", 5.0))

    def send_reply(reply_token: str, message: object | list[object]) -> bool:
        messages = message if isinstance(message, list) else [message]
        outgoing = [
            TextMessage(text=str(item)[:4_500]) if isinstance(item, str) else item
            for item in messages
            if item is not None
        ]
        if not outgoing:
            outgoing = [data_unavailable_message()]
        if reply_sender is not None:
            try:
                result = reply_sender(
                    reply_token, outgoing if len(outgoing) > 1 else outgoing[0]
                )
            except Exception as exc:
                raise ReplyDeliveryError("injected LINE reply sender failed") from exc
            if result is False:
                raise ReplyDeliveryError("injected LINE reply sender rejected reply")
            return True
        if not settings.line_channel_access_token:
            raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN is not configured")
        configuration = Configuration(access_token=settings.line_channel_access_token)

        def submit(candidate_messages: list[object]) -> None:
            with ApiClient(configuration) as api_client:
                response = MessagingApi(api_client).reply_message(
                    ReplyMessageRequest(
                        reply_token=reply_token,
                        messages=candidate_messages,
                    ),
                    _request_timeout=(line_connect_timeout, line_read_timeout),
                )
                # Keep enough evidence to distinguish a successful LINE API
                # acceptance from a local webhook-only 200, without recording
                # message text, user IDs, reply tokens, or other personal data.
                sent_messages = getattr(response, "sent_messages", None) or []
                LOGGER.info(
                    "LINE accepted reply (requested_messages=%d, sent_messages=%d)",
                    len(candidate_messages),
                    len(sent_messages),
                )

        try:
            submit(outgoing)
        except ApiException as exc:
            LOGGER.warning(
                "LINE reply failed (status=%s, reason=%s)",
                getattr(exc, "status", None),
                getattr(exc, "reason", None),
            )
            if _retryable_line_error(exc):
                raise ReplyDeliveryError("LINE reply failed transiently") from exc

            # A permanent 4xx can be a malformed rich payload.  Give the user a
            # valid minimal response once, but do not ask LINE to redeliver a
            # permanently invalid reply token forever.
            fallback = TextMessage(
                text="ขออภัย แสดงการ์ดสินค้าไม่ได้ กรุณาลองค้นหาใหม่อีกครั้ง"
            )
            if len(outgoing) == 1 and isinstance(outgoing[0], TextMessage):
                LOGGER.error("Permanent LINE rejection for a plain-text reply; acknowledging")
                return True
            try:
                submit([fallback])
            except ApiException as fallback_exc:
                if _retryable_line_error(fallback_exc):
                    raise ReplyDeliveryError(
                        "LINE plain-text fallback failed transiently"
                    ) from fallback_exc
                LOGGER.error(
                    "Permanent LINE rejection for plain-text fallback; acknowledging "
                    "(status=%s)",
                    getattr(fallback_exc, "status", None),
                )
            except Exception as fallback_exc:
                raise ReplyDeliveryError(
                    "LINE plain-text fallback failed at the network layer"
                ) from fallback_exc
        except Exception as exc:
            raise ReplyDeliveryError("LINE reply failed at the network layer") from exc
        return True

    @handler.add(PostbackEvent)
    def handle_postback(event: PostbackEvent) -> None:
        if _event_is_standby(event):
            LOGGER.info("Ignoring LINE postback received in standby mode")
            return
        event_id = getattr(event, "webhook_event_id", "") or ""
        deduplicator: RecentWebhookEvents = app.config["WEBHOOK_EVENT_DEDUPLICATOR"]
        if not deduplicator.claim(event_id):
            LOGGER.info("Ignoring duplicate LINE postback event")
            return
        sent = False
        try:
            product_id = parse_product_postback(getattr(event.postback, "data", ""))
            product = catalog.get(product_id) if product_id else None
            if product is None:
                response = text_with_quick_replies(
                    "ไม่พบสินค้านี้ใน snapshot ล่าสุดครับ ลองค้นหาใหม่อีกครั้ง"
                )
            else:
                response = product_detail_message(product)
            sent = send_reply(event.reply_token, response)
        except ReplyDeliveryError:
            raise
        except Exception:
            LOGGER.exception("Mercular detail postback failed")
            try:
                sent = send_reply(
                    event.reply_token,
                    text_with_quick_replies(
                        "ขออภัย โหลดรายละเอียดสินค้าไม่ได้ชั่วคราว กรุณาลองใหม่อีกครั้ง"
                    ),
                )
            except ReplyDeliveryError:
                raise
            except Exception:
                LOGGER.exception("Mercular fallback reply failed")
        finally:
            if not sent:
                deduplicator.release(event_id)

    @handler.add(MessageEvent, message=TextMessageContent)
    def handle_text(event: MessageEvent) -> None:
        if _event_is_standby(event):
            LOGGER.info("Ignoring LINE message received in standby mode")
            return
        event_id = getattr(event, "webhook_event_id", "") or ""
        deduplicator: RecentWebhookEvents = app.config["WEBHOOK_EVENT_DEDUPLICATOR"]
        if not deduplicator.claim(event_id):
            LOGGER.info("Ignoring duplicate LINE message event")
            return
        sent = False
        try:
            # LINE text can be unexpectedly large; cap work and do not log it.
            text = (getattr(event.message, "text", "") or "").strip()[:2_000]
            user_id = _event_user_key(event)
            showcase_kind = parse_showcase_command(text)
            if showcase_kind is not None:
                response = (
                    showcase_hub_message()
                    if showcase_kind == "hub"
                    else build_showcase_message(
                        showcase_kind,
                        public_base_url=settings.public_base_url,
                        coupon_id=settings.line_coupon_id,
                    )
                )
                sent = send_reply(event.reply_token, response)
                return
            # The repository auto-reloads an atomically replaced snapshot.  Rebuild
            # the lightweight parser vocabulary too, so newly scraped brands and
            # categories work without restarting the webhook process.
            active_parser = (
                ThaiCommandParser(
                    brands=catalog.brands(), categories=catalog.categories()
                )
                if dynamic_catalog_parser
                else command_parser
            )
            parsed = active_parser.parse(text)
            intent = getattr(parsed, "intent", "unknown")

            direct_reply = _intent_reply(intent, bot_name=settings.bot_name)
            if direct_reply is not None:
                sent = send_reply(event.reply_token, direct_reply)
                return

            if intent == "refresh":
                remembered = app.config["RECENT_QUERIES"].get(user_id)
                if remembered is None:
                    sent = send_reply(
                        event.reply_token,
                        text_with_quick_replies(
                            "ยังไม่มีคำค้นก่อนหน้าสำหรับสุ่มใหม่ครับ ลองค้นหาสินค้าก่อน"
                        ),
                    )
                    return
                parsed = remembered
            elif intent in {"product_search", "search"}:
                app.config["RECENT_QUERIES"].remember(user_id, parsed)
            else:
                sent = send_reply(event.reply_token, help_message())
                return

            products = catalog.all()
            if not products:
                sent = send_reply(event.reply_token, data_unavailable_message())
                return
            selected = selector.recommend(
                products,
                parsed,
                user_id=user_id,
                top_k=settings.top_k,
            )
            if not selected:
                sent = send_reply(event.reply_token, no_results_message())
                return

            count = len(selected)
            qualifier = (
                f"สุ่มสินค้า {count} รายการที่ตรงเงื่อนไขให้แล้วครับ"
                if count == settings.top_k
                else f"พบสินค้าที่ตรงเงื่อนไข {count} รายการ จึงแสดงทั้งหมดโดยไม่เติมสินค้าผิดเงื่อนไขครับ"
            )
            summary = text_with_quick_replies(
                f"{qualifier}\nราคาและสต็อกเป็นข้อมูลจาก snapshot โปรดตรวจสอบบนเว็บไซต์ก่อนซื้อ"
                + (
                    "\nsnapshot ไม่มีตัวเลขยอดขายหรือวันวางจำหน่าย "
                    "จึงจัดชุดนี้ตามความเกี่ยวข้องของข้อมูลที่มีครับ"
                    if getattr(getattr(parsed, "entities", None), "sort", None)
                    in {SORT_NEWEST, SORT_POPULAR}
                    else ""
                )
            )
            carousel = build_product_carousel_message(selected)
            sent = send_reply(event.reply_token, [summary, carousel])
        except ReplyDeliveryError:
            raise
        except Exception:
            LOGGER.exception("Mercular command handling failed")
            try:
                sent = send_reply(
                    event.reply_token,
                    text_with_quick_replies(
                        "ขออภัย ระบบค้นหาสินค้าขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง"
                    ),
                )
            except ReplyDeliveryError:
                raise
            except Exception:
                LOGGER.exception("Mercular fallback reply failed")
        finally:
            if not sent:
                deduplicator.release(event_id)

    @app.get("/")
    def index():
        return f"{settings.bot_name} — Mercular LINE shopping-assistant demo is running"

    @app.get("/healthz")
    def healthz():
        return jsonify(
            {
                "status": "ok",
                "line_configured": _configured(settings),
                "catalog_ready": _repository_flag(catalog, "is_ready"),
            }
        )

    @app.get("/readyz")
    def readyz():
        ready = _configured(settings) and _repository_flag(catalog, "is_ready")
        payload = {
            "status": "ready" if ready else "not_ready",
            "line_configured": _configured(settings),
            "catalog_ready": _repository_flag(catalog, "is_ready"),
            "catalog_stale": _repository_flag(catalog, "is_stale", True),
        }
        return jsonify(payload), 200 if ready else 503

    @app.get("/media/imagemap/mercumate/<int:size>")
    def mercumate_imagemap(size: int):
        """Serve the five extensionless image URLs required by LINE Imagemap."""

        if size not in IMAGEMAP_SIZES:
            abort(404)
        return send_from_directory(
            os.path.join(app.root_path, "static", "imagemap", "mercumate"),
            f"{size}.jpg",
            mimetype="image/jpeg",
            max_age=86_400,
        )

    @app.post("/")
    @app.post("/callback")
    def callback():
        if not _configured(settings):
            abort(503, description="LINE credentials are not configured")
        signature = request.headers.get("X-Line-Signature")
        if not signature:
            abort(400, description="Missing X-Line-Signature header")
        body = request.get_data(as_text=True)
        try:
            handler.handle(body, signature)
        except InvalidSignatureError:
            LOGGER.warning("Invalid LINE webhook signature")
            abort(400, description="Invalid X-Line-Signature")
        except ReplyDeliveryError:
            LOGGER.error("LINE reply was not delivered; requesting webhook redelivery")
            abort(500, description="LINE reply delivery failed")
        return "OK"

    return app


app = create_app()


if __name__ == "__main__":
    runtime_settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, runtime_settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app.run(host="0.0.0.0", port=runtime_settings.port, debug=False)
