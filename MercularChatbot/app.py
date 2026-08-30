"""Flask webhook for the Mercular LINE shopping-assistant demo."""

from __future__ import annotations

from dataclasses import replace
import logging
import math
import os
from threading import RLock, Thread
import time
from collections.abc import Callable
from typing import Any

from flask import Flask, abort, jsonify, request
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    ApiException,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    ShowLoadingAnimationRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, PostbackEvent, TextMessageContent

try:  # Package import for gunicorn/``python -m MercularChatbot.app``.
    from .bert_nlp import PhayaThaiBertCommandParser, PhayaThaiBertIntentClassifier
    from .catalog_navigation import (
        build_category_menu,
        parse_category_navigation,
    )
    from .config import Settings
    from .conversation import (
        cheaper_than,
        comparison_queries,
        find_named_product,
        is_alternative_request,
        is_cheaper_refinement,
        is_product_question,
        product_question_answer,
    )
    from .line_views import (
        build_category_picker_message,
        build_product_carousel_message,
        build_promotion_carousel_message,
        contact_message,
        data_unavailable_message,
        greeting_message,
        help_message,
        no_results_message,
        parse_product_postback,
        product_detail_message,
        product_comparison_message,
        promotion_unavailable_message,
        text_with_quick_replies,
    )
    from .nlp import (
        INTENT_CONTACT,
        INTENT_GREETING,
        INTENT_HELP,
        INTENT_ORDER,
        INTENT_PROMOTION,
        INTENT_REFRESH,
        INTENT_SEARCH,
        INTENT_THANKS,
        INTENT_UNKNOWN,
        SORT_NEWEST,
        SORT_POPULAR,
        SORT_DISCOUNT,
        SORT_PRICE_ASC,
        SORT_PRICE_DESC,
        CommandEntities,
        ParsedCommand,
        ThaiCommandParser,
        normalize_text,
    )
    from .recommender import ProductRecommender
    from .repository import ProductRepository
    from .promotions import PromotionRepository
    from .session_state import RecentProducts, RecentQueries, RecentWebhookEvents
except ImportError:  # pragma: no cover - direct execution from this folder.
    from bert_nlp import PhayaThaiBertCommandParser, PhayaThaiBertIntentClassifier
    from catalog_navigation import (
        build_category_menu,
        parse_category_navigation,
    )
    from config import Settings
    from conversation import (
        cheaper_than,
        comparison_queries,
        find_named_product,
        is_alternative_request,
        is_cheaper_refinement,
        is_product_question,
        product_question_answer,
    )
    from line_views import (
        build_category_picker_message,
        build_product_carousel_message,
        build_promotion_carousel_message,
        contact_message,
        data_unavailable_message,
        greeting_message,
        help_message,
        no_results_message,
        parse_product_postback,
        product_detail_message,
        product_comparison_message,
        promotion_unavailable_message,
        text_with_quick_replies,
    )
    from nlp import (
        INTENT_CONTACT,
        INTENT_GREETING,
        INTENT_HELP,
        INTENT_ORDER,
        INTENT_PROMOTION,
        INTENT_REFRESH,
        INTENT_SEARCH,
        INTENT_THANKS,
        INTENT_UNKNOWN,
        SORT_NEWEST,
        SORT_POPULAR,
        SORT_DISCOUNT,
        SORT_PRICE_ASC,
        SORT_PRICE_DESC,
        CommandEntities,
        ParsedCommand,
        ThaiCommandParser,
        normalize_text,
    )
    from recommender import ProductRecommender
    from repository import ProductRepository
    from promotions import PromotionRepository
    from session_state import RecentProducts, RecentQueries, RecentWebhookEvents


LOGGER = logging.getLogger(__name__)
SEARCH_INTENTS = frozenset({INTENT_SEARCH, "product_search"})


def _recommendation_summary(count: int, requested_count: int, sort: str | None) -> str:
    """Describe the recommendation without exposing storage implementation details."""

    sort_descriptions = {
        SORT_PRICE_ASC: " โดยเรียงราคาจากต่ำไปสูง",
        SORT_PRICE_DESC: " โดยเรียงราคาจากสูงไปต่ำ",
        SORT_DISCOUNT: " โดยเรียงส่วนลดจากมากไปน้อย",
    }
    ordering = sort_descriptions.get(sort, "")
    if count == requested_count:
        lead = f"แนะนำสินค้า {count} รายการที่ตรงเงื่อนไขจากข้อมูลที่มี{ordering}ครับ"
    else:
        lead = (
            f"จากข้อมูลที่มี พบสินค้าที่ตรงเงื่อนไข {count} รายการ "
            f"จึงแนะนำทั้งหมดโดยไม่เติมสินค้าที่ไม่ตรงเงื่อนไข{ordering}ครับ"
        )
    freshness_note = "ราคาและสต็อกอาจเปลี่ยนแปลง โปรดตรวจสอบบนเว็บไซต์ก่อนซื้อ"
    if sort in {SORT_NEWEST, SORT_POPULAR}:
        return (
            f"{lead}\n{freshness_note}\n"
            "ข้อมูลที่มีไม่มีตัวเลขยอดขายหรือวันวางจำหน่าย "
            "จึงจัดลำดับตามความเกี่ยวข้องครับ"
        )
    return f"{lead}\n{freshness_note}"


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


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


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


def _configured(settings: Settings) -> bool:
    return bool(settings.line_channel_secret and settings.line_channel_access_token)


def _event_user_key(event: object) -> str:
    source = getattr(event, "source", None)
    for attribute in ("user_id", "group_id", "room_id"):
        value = getattr(source, attribute, None)
        if value:
            return str(value)
    return "anonymous"


def _event_direct_user_id(event: object) -> str:
    """Return a user ID only for a one-on-one chat supported by LINE loading."""

    source = getattr(event, "source", None)
    source_type = getattr(source, "type", "")
    source_type = getattr(source_type, "value", source_type)
    if str(source_type).casefold() != "user":
        return ""
    return str(getattr(source, "user_id", "") or "")


def _repository_flag(repository: object, name: str, default: bool = False) -> bool:
    value = getattr(repository, name, default)
    try:
        return bool(value() if callable(value) else value)
    except Exception:
        return default


def _intent_reply(intent: str, *, bot_name: str = "MercuMate") -> object | None:
    if intent == INTENT_GREETING:
        return greeting_message(bot_name)
    if intent == INTENT_HELP:
        return help_message()
    if intent == INTENT_THANKS:
        return text_with_quick_replies(
            "ยินดีครับ 😊 ถ้าอยากดูสินค้าอื่น ลองระบุประเภท แบรนด์ และงบได้เลย"
        )
    if intent == INTENT_CONTACT:
        return contact_message()
    if intent == INTENT_ORDER:
        return text_with_quick_replies(
            "บอตสาธิตนี้ไม่รับคำสั่งซื้อหรือข้อมูลชำระเงินครับ "
            "กดปุ่ม “ซื้อที่ Mercular” บนการ์ดเพื่อยืนยันราคา สต็อก และสั่งซื้อบนเว็บไซต์ทางการ"
        )
    if intent == INTENT_UNKNOWN:
        return text_with_quick_replies(
            "ผมยังไม่แน่ใจว่าต้องการสินค้าชนิดไหนครับ ลองบอกประเภท แบรนด์ หรืองบ เช่น “หูฟัง Xiaomi ไม่เกิน 3000”"
        )
    return None


def create_app(
    settings: Settings | None = None,
    repository: ProductRepository | None = None,
    parser: ThaiCommandParser | None = None,
    recommender: ProductRecommender | None = None,
    promotion_repository: PromotionRepository | None = None,
    reply_sender: Callable[[str, object | list[object]], Any] | None = None,
    loading_sender: Callable[[str, int], Any] | None = None,
) -> Flask:
    """Create an injectable Flask app for production and offline tests."""

    settings = settings or Settings.from_env()
    catalog = repository or ProductRepository(settings.snapshot_path, settings=settings)
    promotion_catalog = promotion_repository or PromotionRepository(settings=settings)
    phayathaibert_classifier = (
        PhayaThaiBertIntentClassifier(
            settings.phayathaibert_model_name,
            local_files_only=settings.phayathaibert_local_files_only,
        )
        if parser is None and settings.nlp_backend == "phayathaibert"
        else None
    )

    def catalog_vocabulary() -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (
            tuple(sorted(catalog.brands(), key=str.casefold)),
            tuple(sorted(catalog.categories(), key=str.casefold)),
        )

    def build_command_parser(
        vocabulary: tuple[tuple[str, ...], tuple[str, ...]],
    ) -> ThaiCommandParser | PhayaThaiBertCommandParser:
        brands, categories = vocabulary
        if settings.nlp_backend == "phayathaibert":
            return PhayaThaiBertCommandParser(
                brands=brands,
                categories=categories,
                min_confidence=settings.phayathaibert_min_confidence,
                classifier=phayathaibert_classifier,
            )
        return ThaiCommandParser(brands=brands, categories=categories)

    parser_vocabulary = catalog_vocabulary() if parser is None else ((), ())
    command_parser = parser or build_command_parser(parser_vocabulary)
    dynamic_catalog_parser = parser is None
    parser_lock = RLock()
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
        PROMOTION_REPOSITORY=promotion_catalog,
        PHAYATHAIBERT_CLASSIFIER=phayathaibert_classifier,
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
        RECENT_PRODUCTS=RecentProducts(
            ttl_seconds=settings.history_ttl_seconds,
            max_entries=2_000,
        ),
    )

    def current_command_parser() -> ThaiCommandParser | PhayaThaiBertCommandParser:
        """Reuse the parser until an atomically reloaded catalogue changes its vocabulary."""

        nonlocal command_parser, parser_vocabulary
        if not dynamic_catalog_parser:
            return command_parser
        current_vocabulary = catalog_vocabulary()
        if current_vocabulary == parser_vocabulary:
            return command_parser
        with parser_lock:
            if current_vocabulary != parser_vocabulary:
                command_parser = build_command_parser(current_vocabulary)
                parser_vocabulary = current_vocabulary
                app.config["COMMAND_PARSER"] = command_parser
        return command_parser

    # A placeholder keeps health endpoints available before credentials are set;
    # /callback still refuses traffic until both credentials exist.
    handler = WebhookHandler(settings.line_channel_secret or "missing-channel-secret")
    line_connect_timeout = float(
        _env_number("LINE_REPLY_CONNECT_TIMEOUT_SECONDS", 2.0)
    )
    line_read_timeout = float(_env_number("LINE_REPLY_READ_TIMEOUT_SECONDS", 5.0))
    line_api_client = (
        ApiClient(Configuration(access_token=settings.line_channel_access_token))
        if settings.line_channel_access_token and reply_sender is None
        else None
    )
    line_messaging_api = MessagingApi(line_api_client) if line_api_client else None
    loading_api_client = (
        ApiClient(Configuration(access_token=settings.line_channel_access_token))
        if settings.line_channel_access_token and reply_sender is None
        else None
    )
    loading_messaging_api = (
        MessagingApi(loading_api_client) if loading_api_client else None
    )
    loading_animation_enabled = _env_enabled(
        "LINE_LOADING_ANIMATION_ENABLED", True
    )

    def start_loading_animation(event: object) -> None:
        """Start LINE's indicator without adding another network wait to the reply."""

        if not loading_animation_enabled:
            return
        chat_id = _event_direct_user_id(event)
        if not chat_id or (loading_sender is None and loading_messaging_api is None):
            return

        def submit() -> None:
            started = time.perf_counter()
            try:
                if loading_sender is not None:
                    loading_sender(chat_id, 5)
                else:
                    assert loading_messaging_api is not None
                    loading_messaging_api.show_loading_animation(
                        ShowLoadingAnimationRequest(
                            chatId=chat_id,
                            loadingSeconds=5,
                        ),
                        _request_timeout=(line_connect_timeout, line_read_timeout),
                    )
            except Exception as exc:
                # Loading is cosmetic.  It must never suppress or delay the answer.
                LOGGER.info("LINE loading animation unavailable: %s", type(exc).__name__)
                return
            LOGGER.info(
                "LINE loading animation accepted (api_ms=%.1f)",
                (time.perf_counter() - started) * 1_000,
            )

        Thread(
            target=submit,
            name="line-loading-animation",
            daemon=True,
        ).start()

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
        if line_messaging_api is None:
            raise RuntimeError("LINE Messaging API client is unavailable")

        def submit(candidate_messages: list[object]) -> None:
            started = time.perf_counter()
            response = line_messaging_api.reply_message(
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
                "LINE accepted reply (requested_messages=%d, sent_messages=%d, api_ms=%.1f)",
                len(candidate_messages),
                len(sent_messages),
                (time.perf_counter() - started) * 1_000,
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
        start_loading_animation(event)
        sent = False
        try:
            product_id = parse_product_postback(getattr(event.postback, "data", ""))
            product = catalog.get(product_id) if product_id else None
            if product is None:
                response = text_with_quick_replies(
                    "ไม่พบสินค้านี้ในข้อมูลล่าสุดครับ ลองค้นหาใหม่อีกครั้ง"
                )
            else:
                app.config["RECENT_PRODUCTS"].focus(
                    _event_user_key(event), product.id
                )
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
        start_loading_animation(event)
        sent = False
        try:
            # LINE text can be unexpectedly large; cap work and do not log it.
            text = (getattr(event.message, "text", "") or "").strip()[:2_000]
            user_id = _event_user_key(event)
            products = catalog.all()
            navigation = parse_category_navigation(text)
            if navigation is not None and not navigation.show_products:
                menu = build_category_menu(products, navigation.path)
                response = (
                    build_category_picker_message(menu)
                    if menu is not None and menu.options
                    else text_with_quick_replies(
                        "ยังไม่มีสินค้าในหมวดนี้จากข้อมูลล่าสุดครับ"
                    )
                )
                sent = send_reply(event.reply_token, response)
                return

            compare_sides = comparison_queries(text)
            if compare_sides is not None:
                first = find_named_product(products, compare_sides[0])
                second = find_named_product(products, compare_sides[1])
                if first is None or second is None:
                    missing = [
                        query
                        for query, product in zip(compare_sides, (first, second))
                        if product is None
                    ]
                    sent = send_reply(
                        event.reply_token,
                        text_with_quick_replies(
                            "ยังเปรียบเทียบไม่ได้ เพราะไม่พบชื่อรุ่นแบบเจาะจงในรายการสินค้าปัจจุบัน: "
                            + ", ".join(f"“{item}”" for item in missing)
                            + "\nหากระบุเพียงประเภทหรือแบรนด์ที่มีหลายรุ่น "
                            "ลองพิมพ์ชื่อรุ่นเต็มตามหน้า Mercular แล้วเทียบอีกครั้งครับ "
                            "ระบบจะไม่เลือกรุ่นชื่อคล้ายกันมาแทนเพื่อป้องกันการเปรียบเทียบผิดรุ่น"
                        ),
                    )
                    return
                sent = send_reply(
                    event.reply_token,
                    product_comparison_message(first, second),
                )
                return

            recent_products: RecentProducts = app.config["RECENT_PRODUCTS"]
            recent_ids, focused_id = recent_products.get(user_id)
            if is_product_question(text):
                focused = catalog.get(focused_id) if focused_id else None
                response_text = (
                    product_question_answer(focused, text)
                    if focused is not None
                    else "ยังไม่ทราบว่าหมายถึงสินค้าตัวไหนครับ แตะ “ดูรายละเอียด” "
                    "บนการ์ดสินค้าก่อน แล้วถามว่า ‘ตัวนี้ใช้ Bluetooth ได้ไหม’ ได้เลย"
                )
                sent = send_reply(
                    event.reply_token,
                    text_with_quick_replies(response_text),
                )
                return
            # The repository auto-reloads an atomically replaced snapshot.  Rebuild
            # the lightweight parser vocabulary too, so newly scraped brands and
            # categories work without restarting the webhook process.
            if navigation is not None and navigation.show_products:
                parsed = ParsedCommand(
                    INTENT_SEARCH,
                    1.0,
                    CommandEntities(category_path=navigation.path),
                    text,
                    normalize_text(text),
                )
            else:
                parsed = current_command_parser().parse(text)

            if is_alternative_request(text):
                focused = catalog.get(focused_id) if focused_id else None
                if focused is None:
                    sent = send_reply(
                        event.reply_token,
                        text_with_quick_replies(
                            "ยังไม่ทราบว่าหมายถึงตัวไหนครับ แตะ “ดูรายละเอียด” "
                            "ของสินค้าต้นแบบก่อน แล้วขอตัวคล้ายกันที่ถูกกว่าได้เลย"
                        ),
                    )
                    return
                parsed = ParsedCommand(
                    INTENT_SEARCH,
                    1.0,
                    CommandEntities(
                        category=focused.category,
                        max_price=focused.price,
                        max_price_inclusive=False,
                        sort=SORT_PRICE_ASC,
                    ),
                    text,
                    normalize_text(text),
                )
            elif is_cheaper_refinement(text):
                remembered = app.config["RECENT_QUERIES"].get(user_id)
                shown = [
                    product
                    for identifier in recent_ids
                    if (product := catalog.get(identifier)) is not None
                ]
                ceiling = cheaper_than(shown)
                if remembered is None or ceiling is None:
                    sent = send_reply(
                        event.reply_token,
                        text_with_quick_replies(
                            "ยังไม่มีผลค้นหาก่อนหน้าให้ลดงบครับ ลองค้นหาสินค้าก่อน"
                        ),
                    )
                    return
                parsed = replace(
                    remembered,
                    entities=replace(
                        remembered.entities,
                        max_price=ceiling,
                        max_price_inclusive=False,
                        sort=SORT_PRICE_ASC,
                    ),
                    raw_text=text,
                    normalized_text=normalize_text(text),
                )
            elif (
                getattr(parsed, "intent", "") in SEARCH_INTENTS
                and "อย่างเดียว" in normalize_text(text)
                and getattr(parsed.entities, "brands", ())
            ):
                remembered = app.config["RECENT_QUERIES"].get(user_id)
                if remembered is not None:
                    parsed = replace(
                        parsed,
                        entities=replace(
                            remembered.entities,
                            brands=parsed.entities.brands,
                            excluded_brands=parsed.entities.excluded_brands,
                            query=parsed.entities.query or remembered.entities.query,
                        ),
                    )
            intent = getattr(parsed, "intent", INTENT_UNKNOWN)

            if intent == INTENT_PROMOTION:
                promotions = promotion_catalog.current(limit=5)
                response = (
                    build_promotion_carousel_message(promotions)
                    if promotions
                    else promotion_unavailable_message()
                )
                sent = send_reply(event.reply_token, response)
                return

            direct_reply = _intent_reply(intent, bot_name=settings.bot_name)
            if direct_reply is not None:
                sent = send_reply(event.reply_token, direct_reply)
                return

            if intent == INTENT_REFRESH:
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
            elif intent in SEARCH_INTENTS:
                app.config["RECENT_QUERIES"].remember(user_id, parsed)
            else:
                sent = send_reply(event.reply_token, help_message())
                return

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
                sent = send_reply(event.reply_token, no_results_message(text))
                return
            recent_products.remember_results(user_id, selected)

            summary = text_with_quick_replies(
                _recommendation_summary(
                    len(selected),
                    settings.top_k,
                    getattr(getattr(parsed, "entities", None), "sort", None),
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
    classifier = app.config.get("PHAYATHAIBERT_CLASSIFIER")
    if classifier is not None:
        started = time.perf_counter()
        try:
            classifier.warm_up()
        except PhayaThaiBertUnavailable as error:
            LOGGER.warning("PhayaThaiBERT warm-up failed; rules remain available: %s", error)
        else:
            LOGGER.info(
                "PhayaThaiBERT warmed before accepting LINE traffic (warmup_ms=%.1f)",
                (time.perf_counter() - started) * 1_000,
            )
    app.run(host="0.0.0.0", port=runtime_settings.port, debug=False)
