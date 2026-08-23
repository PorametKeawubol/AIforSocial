"""Flask webhook for the LINE KFC menu and promotion chatbot."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import quote, unquote, urlparse

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    ApiException,
    Configuration,
    FlexContainer,
    FlexMessage,
    ImageMessage,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, PostbackEvent, TextMessageContent

from intent_classifier import IntentClassifier
from qa_engine import KfcQuestionAnswerer
from scraper import KfcScraper, KfcScraperError


load_dotenv()
LOGGER = logging.getLogger(__name__)
DEFAULT_MENU_URL = "https://www.kfc.co.th/menu/meals"
DEFAULT_STORE_LOCATOR_URL = "https://www.kfc.co.th/store-locator"
DEFAULT_KFC_HOME_URL = "https://www.kfc.co.th/"
MENU_SEARCH_CAROUSEL_LIMIT = 12
MENU_SEARCH_CANDIDATE_LIMIT = 1_000
# LINE Flex Carousel accepts at most 12 bubbles.  Both menu overviews and
# partial-name searches use every available slot after relevance filtering.
MENU_OVERVIEW_CAROUSEL_LIMIT = 12
MENU_OVERVIEW_MAX_RESULTS = max(
    MENU_OVERVIEW_CAROUSEL_LIMIT,
    int(os.getenv("MENU_OVERVIEW_MAX_RESULTS", "1000")),
)
MENU_DISCOVERY_PHRASES = (
    "menu",
    "เมนู",
    "รายการอาหาร",
    "เมนูอาหาร",
    "รายการของกิน",
    "มีอะไรขาย",
    "มีอะไรให้เลือก",
    "ขอดูเมนู",
    "เปิดดูเมนู",
)
MENU_CONFLICT_PHRASES = (
    "โปร",
    "โปรโมชั่น",
    "โปรโมชัน",
    "สั่ง",
    "สาขา",
    "ร้านอยู่",
    "ใกล้ฉัน",
)


def _normalise_menu_text(value: object | None) -> str:
    text = " ".join(str(value or "").replace("\xa0", " ").split()).casefold()
    return re.sub(r"[^\w\sก-๙]", " ", text)


def _compact_menu_key(value: object | None) -> str:
    return re.sub(r"\s+", "", _normalise_menu_text(value))


def _looks_like_menu_request(text: str) -> bool:
    """Recognise menu wording even if the intent model is unavailable."""

    normalized = _normalise_menu_text(text)
    if not normalized or any(phrase in normalized for phrase in MENU_CONFLICT_PHRASES):
        return False
    return any(phrase in normalized for phrase in MENU_DISCOVERY_PHRASES)


def _is_menu_overview_request(text: str, answerer: object) -> bool:
    """Recognise a request for every menu page, not a named-menu search."""

    classifier = getattr(answerer, "is_menu_discovery_query", None)
    if callable(classifier):
        try:
            return bool(classifier(text))
        except Exception:
            LOGGER.exception("KFC menu overview detection failed")

    # Small test doubles and the worksheet's older answerer do not expose the
    # semantic helper.  Keep their exact generic phrases working too.
    return _compact_menu_key(text) in {
        "menu",
        "เมนู",
        "รายการอาหาร",
        "เมนูอาหาร",
        "รายการของกิน",
        "มีอะไรขาย",
        "มีอะไรให้เลือก",
        "มีเมนูอะไรบ้าง",
        "ขอเมนู",
        "ขอดูเมนู",
        "เปิดดูเมนู",
    }


def _reply_for_intent(intent: str) -> str | None:
    """Return a conversational reply for intents not handled by QA search."""

    if intent == "greeting":
        return (
            "สวัสดีครับ 🍗\n"
            "ผมช่วยค้นหาเมนูและโปรโมชัน KFC ได้\n"
            "ลองพิมพ์ “มีเมนูอะไรบ้าง” หรือ “โปรวันนี้” ได้เลยครับ"
        )
    if intent == "thanks":
        return "ยินดีครับ 😊 หากต้องการดูเมนูหรือโปรโมชัน KFC ถามได้เลยครับ"
    if intent == "location":
        location_url = _https_url(
            os.getenv("KFC_STORE_LOCATOR_URL"), DEFAULT_STORE_LOCATOR_URL
        )
        return f"ค้นหาสาขา KFC ใกล้คุณได้ที่นี่ครับ 📍\n{location_url}"
    if intent == "order":
        order_url = _https_url(os.getenv("KFC_ORDER_URL"), DEFAULT_KFC_HOME_URL)
        return (
            "สามารถดูเมนูและสั่งอาหารผ่านเว็บไซต์ KFC ได้เลยครับ 🍗\n"
            f"{order_url}"
        )
    return None


def _https_url(value: object | None, fallback: str = "") -> str:
    """Accept only absolute HTTPS URLs suitable for a LINE Flex payload."""

    candidate = str(value or "").strip()
    parsed = urlparse(candidate)
    if parsed.scheme != "https" or not parsed.netloc:
        return fallback
    return candidate


def _flex_text(value: object | None, limit: int) -> str:
    text = " ".join(str(value or "").replace("\xa0", " ").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _detail_components(item: dict[str, Any]) -> list[str]:
    values = item.get("components", [])
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(_flex_text(value, 180) for value in values if _flex_text(value, 180)))


def _detail_choices(item: dict[str, Any]) -> list[tuple[str, list[str]]]:
    values = item.get("choices", [])
    if not isinstance(values, list):
        return []
    result: list[tuple[str, list[str]]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        group = _flex_text(value.get("group"), 180)
        options = value.get("options", [])
        if not isinstance(options, list):
            continue
        cleaned = list(
            dict.fromkeys(
                option_text
                for option in options
                if (option_text := _flex_text(option, 180))
            )
        )
        if group and cleaned:
            result.append((group, cleaned))
    return result


def _detail_menu_items(item: dict[str, Any]) -> list[str]:
    """Read the worksheet's optional nested ``menu_items`` shape too."""

    values = item.get("menu_items", [])
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        if isinstance(value, dict):
            label = next(
                (
                    _flex_text(value.get(key), 180)
                    for key in ("name", "item", "title", "label")
                    if _flex_text(value.get(key), 180)
                ),
                "",
            )
            quantity = _flex_text(
                value.get("quantity") or value.get("qty") or value.get("amount"),
                40,
            )
            if label and quantity:
                label = f"{label} x{quantity}"
        else:
            label = _flex_text(value, 180)
        if label:
            result.append(label)
    return list(dict.fromkeys(result))


def format_menu_detail(item: dict[str, Any], menu_url: str | None = None) -> str:
    """Format one menu card as the detailed Thai reply requested by the lab."""

    page_url = _https_url(menu_url or os.getenv("KFC_MENU_URL"), DEFAULT_MENU_URL)
    lines = [_flex_text(item.get("name"), 180) or "เมนู KFC"]
    if item.get("price"):
        lines.append(f"💰 ราคา: {_flex_text(item['price'], 40)}")
    if item.get("description"):
        lines.append(f"📝 {_flex_text(item['description'], 900)}")

    components = _detail_components(item)
    nested_items = _detail_menu_items(item)
    choices = _detail_choices(item)
    if nested_items or components or choices:
        lines.append("📋 รายการย่อย:")
    if nested_items:
        lines.extend(f"• {subitem}" for subitem in nested_items)
    if components:
        lines.append("📦 ประกอบด้วย:")
        lines.extend(f"• {component}" for component in components)

    if choices:
        lines.append("🎛️ เลือกได้:")
        for group, options in choices:
            lines.append(f"• {group}")
            lines.extend(f"  - {option}" for option in options)

    lines.append(f"🔗 {_https_url(item.get('url'), page_url)}")
    text = "\n".join(lines)
    if len(text) <= 4_500:
        return text
    return text[:4_499].rstrip() + "…"


class KfcMenuImageProvider:
    """Load KFC meal image URLs and cache them for fast LINE replies."""

    def __init__(
        self,
        scraper: KfcScraper | None = None,
        max_results: int | None = None,
        cache_seconds: float | None = None,
    ) -> None:
        self.scraper = scraper or KfcScraper()
        self.max_results = max(
            1,
            int(
                max_results
                or os.getenv("MENU_CAROUSEL_RESULTS")
                or os.getenv("MAX_RESULTS", "5")
            ),
        )
        self.cache_seconds = max(
            1.0,
            float(
                cache_seconds
                if cache_seconds is not None
                else os.getenv("MENU_IMAGE_CACHE_SECONDS", "600")
            ),
        )
        self._items: list[dict[str, Any]] = []
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def get_items(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            if self._items and now < self._expires_at:
                return [dict(item) for item in self._items]
            items = self.scraper.scrape_menu_images(self.max_results)
            if not items:
                raise KfcScraperError("KFC returned no meal images")
            self._items = [dict(item) for item in items[: self.max_results]]
            self._expires_at = time.monotonic() + self.cache_seconds
            return [dict(item) for item in self._items]


def search_kfc_menu_items(
    question: str,
    answerer: KfcQuestionAnswerer,
    image_items: list[dict[str, Any]],
    *,
    limit: int = MENU_SEARCH_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    """Rank menu records and merge live image-card fields into the results."""

    result_limit = max(1, int(limit))
    provider_items = [dict(item) for item in image_items if isinstance(item, dict)]
    by_id = {
        str(item.get("id")).strip(): item
        for item in provider_items
        if str(item.get("id") or "").strip()
    }
    by_name = {
        _compact_menu_key(item.get("name")): item
        for item in provider_items
        if _compact_menu_key(item.get("name"))
    }

    search_method = getattr(answerer, "search_items", None)
    ranked: list[tuple[float, dict[str, Any]]] = []
    if callable(search_method):
        try:
            ranked = list(
                search_method(
                    question,
                    kind="menu",
                    limit=result_limit,
                )
            )
        except TypeError:
            # Keep the helper compatible with small test doubles and older
            # answerer implementations that only accept question and limit.
            ranked = list(search_method(question, limit=result_limit))
        except Exception:
            LOGGER.exception("KFC menu ranking failed")
            ranked = []
    else:
        # The injected provider is already the source of truth for the old
        # worksheet implementation, so it remains a safe fallback in tests or
        # during a partial upgrade.
        ranked = [(0.0, item) for item in provider_items[:result_limit]]

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, raw_item in ranked:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        provider_item = by_id.get(str(item.get("id") or "").strip())
        if provider_item is None:
            provider_item = by_name.get(_compact_menu_key(item.get("name")))
        if provider_item is not None:
            for field in (
                "id",
                "image_url",
                "price",
                "description",
                "components",
                "choices",
                "url",
            ):
                if not item.get(field) and provider_item.get(field):
                    item[field] = provider_item[field]
        item_key = str(item.get("id") or _compact_menu_key(item.get("name"))).strip()
        if not item_key or item_key in seen or not _https_url(item.get("image_url")):
            continue
        if not _flex_text(item.get("name"), 80):
            continue
        item["_similarity"] = float(score)
        seen.add(item_key)
        results.append(item)
        if len(results) >= result_limit:
            break
    return results


def build_kfc_menu_overview_reply(
    answerer: KfcQuestionAnswerer,
    image_items: list[dict[str, Any]],
    menu_url: str | None = None,
) -> list[object] | str:
    """Fill one Carousel with as many catalog menus as LINE permits."""

    all_items = search_kfc_menu_items(
        "เมนู",
        answerer,
        image_items,
        limit=MENU_OVERVIEW_MAX_RESULTS,
    )
    if not all_items:
        return "ยังไม่พบเมนู KFC ในข้อมูลล่าสุด กรุณาลองใหม่อีกครั้ง"

    visible_items = all_items[:MENU_OVERVIEW_CAROUSEL_LIMIT]
    summary = (
        f"🍗 เมนู KFC ทั้งหมด {len(all_items)} รายการ\n"
        f"แสดง {len(visible_items)} รายการแรกใน Carousel\n"
        "กด “ดูรายละเอียด” เพื่อดูรายการย่อยและตัวเลือก"
    )
    carousel = build_kfc_menu_flex(
        visible_items,
        menu_url=menu_url,
        result_limit=MENU_OVERVIEW_CAROUSEL_LIMIT,
        alt_text="เมนู KFC",
    )
    return [summary, carousel]


def _matches_menu_label(query: str, item: dict[str, Any]) -> bool:
    """Whether a partial query appears in a menu name or one of its aliases."""

    compact_query = _compact_menu_key(query)
    if len(compact_query) < 2:
        return False
    aliases = item.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []
    labels = [item.get("name", ""), *aliases]
    return any(compact_query in _compact_menu_key(label) for label in labels)


def _deduplicate_menu_display_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Avoid duplicate cards when the source catalog repeats the same SKU."""

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (
            _compact_menu_key(item.get("name")),
            _flex_text(item.get("price"), 40),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        results.append(item)
    return results


def _filter_similar_menu_items(
    query: str, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep every direct partial-name match, otherwise keep close fuzzy hits."""

    direct_matches = [item for item in candidates if _matches_menu_label(query, item)]
    if direct_matches:
        return _deduplicate_menu_display_items(direct_matches)
    if not candidates:
        return []

    best_score = float(candidates[0].get("_similarity", 0.0))
    # A typo does not share literal characters with the catalog name.  Keep
    # scores close to the best hit, rather than padding the Carousel with
    # unrelated products merely to reach its card limit.
    minimum_score = max(0.60, best_score - 0.18)
    return _deduplicate_menu_display_items(
        [
            item
            for item in candidates
            if float(item.get("_similarity", 0.0)) >= minimum_score
        ]
    )


def build_kfc_menu_search_reply(
    question: str,
    answerer: KfcQuestionAnswerer,
    image_items: list[dict[str, Any]],
    menu_url: str | None = None,
) -> list[object] | str:
    """Return every relevant search hit that fits in one LINE Carousel."""

    candidates = search_kfc_menu_items(
        question,
        answerer,
        image_items,
        limit=MENU_SEARCH_CANDIDATE_LIMIT,
    )
    matches = _filter_similar_menu_items(question, candidates)
    if not matches:
        return (
            f"ไม่พบเมนูที่ใกล้เคียงกับ “{_flex_text(question, 120)}”\n"
            "ลองพิมพ์ชื่อเมนูให้ยาวขึ้น หรือพิมพ์ menu เพื่อเปิดรายการเมนูครับ"
        )
    visible_matches = matches[:MENU_SEARCH_CAROUSEL_LIMIT]
    display_notice = (
        f"แสดง {len(visible_matches)} รายการใน Carousel"
        if len(matches) <= MENU_SEARCH_CAROUSEL_LIMIT
        else (
            f"พบทั้งหมด {len(matches)} รายการ แต่ Carousel แสดงได้สูงสุด "
            f"{MENU_SEARCH_CAROUSEL_LIMIT} รายการ"
        )
    )
    summary = (
        f"🔎 ผลการค้นหา: {_flex_text(question, 120)}\n"
        f"พบเมนูใกล้เคียง {len(matches)} รายการ\n{display_notice}\n"
        "แตะปุ่ม “ดูรายละเอียด” เพื่อดูรายการย่อยและตัวเลือก"
    )
    carousel = build_kfc_menu_flex(
        visible_matches,
        menu_url=menu_url,
        result_limit=MENU_SEARCH_CAROUSEL_LIMIT,
        alt_text="ผลการค้นหาเมนู KFC",
    )
    return [summary, carousel]


def _build_kfc_menu_bubble(item: dict[str, Any]) -> dict[str, Any] | None:
    """Build one product card shared by search results and overview pages."""

    image_url = _https_url(item.get("image_url"))
    name = _flex_text(item.get("name"), 80)
    if not image_url or not name:
        return None
    item_key = str(item.get("id") or item.get("name") or "").strip()
    if not item_key:
        return None
    postback_data = "menu_detail=" + quote(item_key, safe="")
    body_contents: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": name,
            "weight": "bold",
            "size": "sm",
            "wrap": True,
            "maxLines": 2,
        }
    ]
    price = _flex_text(item.get("price"), 40)
    if price:
        body_contents.append(
            {
                "type": "text",
                "text": price,
                "size": "sm",
                "color": "#D71920",
                "margin": "sm",
            }
        )
    description = _flex_text(item.get("description"), 110)
    if description:
        body_contents.append(
            {
                "type": "text",
                "text": description,
                "size": "xs",
                "color": "#666666",
                "wrap": True,
                "margin": "sm",
                "maxLines": 3,
            }
        )
    return {
        "type": "bubble",
        "size": "mega",
        "hero": {
            "type": "image",
            "url": image_url,
            "size": "full",
            "aspectRatio": "1:1",
            "aspectMode": "cover",
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "none",
            "contents": body_contents,
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#D71920",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "ดูรายละเอียด",
                        "data": postback_data,
                    },
                }
            ],
        },
    }


def build_kfc_menu_flex(
    items: list[dict[str, Any]],
    menu_url: str | None = None,
    *,
    result_limit: int | None = None,
    alt_text: str | None = None,
) -> FlexMessage:
    """Build the shopping-style Flex carousel shown in the lab worksheet."""

    configured_limit = max(
        1, min(12, int(os.getenv("MENU_CAROUSEL_RESULTS", "5")))
    )
    carousel_limit = (
        configured_limit
        if result_limit is None
        else max(1, min(12, int(result_limit)))
    )

    bubbles = [
        bubble
        for item in items[:carousel_limit]
        if (bubble := _build_kfc_menu_bubble(item)) is not None
    ]
    if not bubbles:
        raise ValueError("No valid KFC image cards were available")
    container = FlexContainer.from_dict({"type": "carousel", "contents": bubbles})
    return FlexMessage(
        altText=_flex_text(alt_text or "เมนู KFC", 400),
        contents=container,
    )


def _has_high_confidence_menu_name(text: str, answerer: object) -> bool:
    """Route a misspelled/partial product name to the menu carousel."""

    search_method = getattr(answerer, "search_items", None)
    if not callable(search_method):
        return False
    try:
        matches = search_method(
            text,
            kind="menu",
            limit=1,
            minimum_score=0.65,
        )
    except TypeError:
        try:
            matches = search_method(text, limit=1)
        except Exception:
            return False
    except Exception:
        LOGGER.exception("KFC menu name detection failed")
        return False
    if not matches:
        return False
    try:
        return float(matches[0][0]) >= 0.65
    except (IndexError, TypeError, ValueError):
        return False


def _should_show_menu_carousel(text: str, prediction: object, answerer: object) -> bool:
    return (
        getattr(prediction, "intent", "") == "menu"
        or _looks_like_menu_request(text)
        or _has_high_confidence_menu_name(text, answerer)
    )


@dataclass(frozen=True)
class Settings:
    """Only the non-secret shape of the runtime configuration."""

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
    """Prevent duplicate replies when LINE redelivers an event."""

    def __init__(self, ttl_seconds: float = 600, max_entries: int = 1_000) -> None:
        self.ttl_seconds = max(1, float(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._events: dict[str, float] = {}
        self._lock = threading.Lock()

    def claim(self, event_id: str) -> bool:
        """Return ``True`` only for the first delivery of an event ID."""

        now = time.monotonic()
        with self._lock:
            for known_event_id, expiry in list(self._events.items()):
                if expiry <= now:
                    del self._events[known_event_id]
            if event_id in self._events:
                return False
            while len(self._events) >= self.max_entries:
                del self._events[next(iter(self._events))]
            self._events[event_id] = now + self.ttl_seconds
            return True


def create_app(
    settings: Settings | None = None,
    answerer: KfcQuestionAnswerer | None = None,
    intent_classifier: IntentClassifier | None = None,
    reply_sender: Callable[[str, object], None] | None = None,
    menu_image_provider: Callable[[], list[dict[str, Any]]] | None = None,
) -> Flask:
    """Create an injectable Flask application suitable for real use and tests."""

    settings = settings or Settings.from_env()
    app = Flask(__name__)
    app.config["LINE_CONFIGURED"] = settings.configured
    app.config["KFC_ANSWERER"] = answerer or KfcQuestionAnswerer()
    app.config["KFC_INTENT_CLASSIFIER"] = intent_classifier or IntentClassifier()
    app.config["KFC_MENU_IMAGE_PROVIDER"] = (
        menu_image_provider or KfcMenuImageProvider().get_items
    )
    app.config["WEBHOOK_EVENT_DEDUPLICATOR"] = RecentWebhookEvents(
        ttl_seconds=float(os.getenv("WEBHOOK_EVENT_TTL_SECONDS", "600")),
        max_entries=int(os.getenv("WEBHOOK_EVENT_MAX_ENTRIES", "1000")),
    )

    # Constructing a handler with a harmless placeholder lets /healthz keep
    # working before .env is configured, while /callback still rejects traffic.
    handler = WebhookHandler(settings.channel_secret or "missing-channel-secret")

    def send_reply(reply_token: str, message: object | list[object]) -> bool:
        raw_messages = message if isinstance(message, list) else [message]
        outgoing_messages = [
            TextMessage(text=str(value)[:4500])
            if isinstance(value, str)
            else value
            for value in raw_messages
        ]
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
                        messages=outgoing_messages,
                    )
                )
        except ApiException as exc:
            # A reply token is single-use and expires quickly.  Returning 200
            # avoids a redelivery loop after an irrecoverable reply failure.
            LOGGER.warning(
                "LINE reply failed (status=%s, reason=%s)",
                getattr(exc, "status", None),
                getattr(exc, "reason", None),
            )
            return False
        return True

    @handler.add(PostbackEvent)
    def handle_menu_postback(event: PostbackEvent) -> None:
        event_id = getattr(event, "webhook_event_id", "") or ""
        deduplicator: RecentWebhookEvents = app.config["WEBHOOK_EVENT_DEDUPLICATOR"]
        if event_id and not deduplicator.claim(event_id):
            LOGGER.info("Ignoring a duplicate LINE postback event")
            return

        data = (event.postback.data or "").strip()
        if not data.startswith("menu_detail="):
            send_reply(event.reply_token, "ไม่พบรายละเอียดเมนูที่เลือก กรุณาพิมพ์ menu อีกครั้ง")
            return
        item_key = unquote(data.split("=", 1)[1])
        try:
            try:
                items = app.config["KFC_MENU_IMAGE_PROVIDER"]()
            except Exception as exc:
                # A search result can still be resolved from the local scraped
                # catalog even when refreshing the small live image cache fails.
                LOGGER.warning("KFC image cache unavailable for postback: %s", exc)
                items = []
            item = next(
                (
                    candidate
                    for candidate in items
                    if str(candidate.get("id") or candidate.get("name") or "")
                    == item_key
                ),
                None,
            )
            if item is None:
                finder = getattr(app.config["KFC_ANSWERER"], "find_item", None)
                if callable(finder):
                    item = finder(item_key, kind="menu")
            if item is None:
                send_reply(event.reply_token, "ไม่พบเมนูที่เลือกในข้อมูลล่าสุด กรุณาพิมพ์ menu อีกครั้ง")
                return
            image_url = _https_url(item.get("image_url"))
            messages: list[object] = []
            if image_url:
                messages.append(
                    ImageMessage(
                        originalContentUrl=image_url,
                        previewImageUrl=image_url,
                    )
                )
            messages.append(
                TextMessage(
                    text=format_menu_detail(
                        item,
                        menu_url=os.getenv("KFC_MENU_URL", DEFAULT_MENU_URL),
                    )
                )
            )
            send_reply(event.reply_token, messages)
        except Exception:
            LOGGER.exception("KFC menu detail postback failed")
            send_reply(
                event.reply_token,
                "ขออภัย ไม่สามารถโหลดรายละเอียดเมนูนี้ได้ กรุณาลองใหม่อีกครั้ง",
            )

    @handler.add(MessageEvent, message=TextMessageContent)
    def handle_text_message(event: MessageEvent) -> None:
        event_id = getattr(event, "webhook_event_id", "") or ""
        deduplicator: RecentWebhookEvents = app.config["WEBHOOK_EVENT_DEDUPLICATOR"]
        if event_id and not deduplicator.claim(event_id):
            LOGGER.info("Ignoring a duplicate LINE webhook event")
            return
        text = (event.message.text or "").strip()
        if text.casefold() in {"menu", "เมนู"}:
            try:
                try:
                    image_items = app.config["KFC_MENU_IMAGE_PROVIDER"]()
                except Exception as exc:
                    LOGGER.warning("KFC image cache unavailable for menu overview: %s", exc)
                    image_items = []
                response = build_kfc_menu_overview_reply(
                    app.config["KFC_ANSWERER"],
                    image_items,
                    menu_url=os.getenv("KFC_MENU_URL", DEFAULT_MENU_URL),
                )
            except Exception:
                LOGGER.exception("KFC menu image reply failed")
                response = (
                    "ขออภัย ไม่สามารถโหลดรูปเมนู KFC ได้ในขณะนี้ "
                    "กรุณาลองใหม่อีกครั้ง"
                )
        else:
            try:
                prediction = app.config["KFC_INTENT_CLASSIFIER"].detect(text)
                answerer = app.config["KFC_ANSWERER"]
                if _should_show_menu_carousel(text, prediction, answerer):
                    try:
                        image_items = app.config["KFC_MENU_IMAGE_PROVIDER"]()
                    except Exception as exc:
                        # Search results include image URLs from the stored
                        # scraper snapshot, so a live cache refresh is optional.
                        LOGGER.warning("KFC image cache unavailable for search: %s", exc)
                        image_items = []
                    if _is_menu_overview_request(text, answerer):
                        response = build_kfc_menu_overview_reply(
                            answerer,
                            image_items,
                            menu_url=os.getenv("KFC_MENU_URL", DEFAULT_MENU_URL),
                        )
                    else:
                        response = build_kfc_menu_search_reply(
                            text,
                            answerer,
                            image_items,
                            menu_url=os.getenv("KFC_MENU_URL", DEFAULT_MENU_URL),
                        )
                else:
                    response = _reply_for_intent(prediction.intent)
                    if response is None:
                        response = answerer.answer(text)
            except Exception:
                LOGGER.exception("KFC question answering failed")
                response = "ขออภัย ระบบค้นหาข้อมูล KFC ขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง"
        send_reply(event.reply_token, response)

    @app.get("/")
    def index():
        return "LINE KFC Menu & Promotion QA chatbot is running"

    @app.get("/healthz")
    def healthz():
        return jsonify(
            {
                "status": "ok",
                "line_configured": settings.configured,
            }
        )

    @app.post("/")
    @app.post("/callback")
    def callback():
        if not settings.configured:
            LOGGER.error("Webhook credentials are not configured")
            abort(500, description="LINE credentials are not configured")
        signature = request.headers.get("X-Line-Signature")
        if not signature:
            abort(400, description="Missing X-Line-Signature header")
        # LINE signs these original bytes, not a reparsed/reformatted JSON body.
        body = request.get_data(as_text=True)
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
