"""LINE webhook adapter with fast acknowledgement and delayed push response."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from assistant import SmartDocAssistant, _is_greeting
from chat_history import ChatHistory, conversation_key
from config import Settings
from line_ux import LineUX, quick_reply

LOGGER = logging.getLogger(__name__)
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"


def verify_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    """Verify X-Line-Signature using constant-time comparison."""

    if not signature or not channel_secret:
        return False
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature)


def _split_text(text: str, limit: int = 4900) -> list[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    while text:
        cut = min(limit, len(text))
        if cut < len(text):
            boundary = text.rfind("\n", 0, cut)
            if boundary >= int(limit * 0.6):
                cut = boundary
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    return parts[:5]


async def _line_post(url: str, token: str, payload: dict[str, Any]) -> None:
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()


def create_app(assistant: SmartDocAssistant, config: Settings):
    """Create a FastAPI app. Real credentials are read from config, never logged."""

    app = FastAPI(title="SmartDoc Assistant LINE Webhook", version="1.0.0")
    seen: dict[str, float] = {}
    history = ChatHistory(config.artifact_dir / "chat_history.sqlite3", config.history_max_turns, config.history_ttl_seconds)
    ux = LineUX(config.index_dir)
    processing = asyncio.Lock()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "index_exists": (config.index_dir / "manifest.json").is_file(),
            "line_configured": bool(config.line_channel_secret and config.line_access_token),
            "model": config.ollama_model,
            "history_max_turns": config.history_max_turns,
            "history_ttl_seconds": config.history_ttl_seconds,
            "line_ux": True,
        }

    async def process_event(event: dict[str, Any], destination: str, key: str | None) -> None:
        # One inference queue protects the shared models and preserves turn order.
        async with processing:
            await process_answer(event, destination, key)

    async def process_answer(event: dict[str, Any], destination: str, key: str | None) -> None:
        try:
            question = event["message"]["text"].strip()
            if question in ("ล้างประวัติ", "เริ่มใหม่"):
                if key:
                    history.clear(key)
                await _line_post(LINE_PUSH_URL, config.line_access_token,
                                 {"to": destination, "messages": [{"type": "text", "text": "ล้างประวัติแล้วครับ เริ่มถามเรื่องใหม่ได้เลย", "quickReply": quick_reply()}]})
                return
            turns = history.get(key) if key else []
            result = await asyncio.to_thread(assistant.answer, question, True, chat_history=turns)
            text = result.answer
            topic = ""
            if result.grounded and result.sources:
                topic = result.sources[0].get("section_title", "")
            messages = [{"type": "text", "text": part} for part in _split_text(text)]
            messages[-1]["quickReply"] = quick_reply(topic)
            await _line_post(
                LINE_PUSH_URL,
                config.line_access_token,
                {"to": destination, "messages": messages},
            )
            if key and result.grounded:
                history.append(key, question, text, topic)
        except Exception:
            LOGGER.exception("LINE background answer failed")
            try:
                await _line_post(
                    LINE_PUSH_URL,
                    config.line_access_token,
                    {
                        "to": destination,
                        "messages": [
                            {
                                "type": "text",
                                "text": "ระบบขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง หรือตรวจสอบเอกสารฉบับทางการ",
                            }
                        ],
                    },
                )
            except Exception:
                LOGGER.exception("LINE error notification failed")

    @app.post("/callback")
    async def callback(request: Request, background_tasks: BackgroundTasks) -> dict[str, str]:
        if not config.line_channel_secret or not config.line_access_token:
            raise HTTPException(status_code=503, detail="LINE credentials are not configured")
        body = await request.body()
        if len(body) > 1_000_000:
            raise HTTPException(status_code=413, detail="Payload too large")
        if not verify_signature(body, request.headers.get("x-line-signature", ""), config.line_channel_secret):
            raise HTTPException(status_code=400, detail="Invalid LINE signature")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("events", []), list):
            raise HTTPException(status_code=400, detail="Invalid events")

        now = time.time()
        for key, timestamp in list(seen.items()):
            if now - timestamp > 3600:
                del seen[key]
        for event in payload.get("events", []):
            if not isinstance(event, dict):
                continue
            if event.get("type") == "follow":
                text = "ถาม Assistant"
            elif event.get("type") == "message" and event.get("message", {}).get("type") == "text":
                text = str(event["message"].get("text", "")).strip()
            else:
                continue
            if not 2 <= len(text) <= 1500:
                continue
            event_id = event.get("webhookEventId") or event.get("message", {}).get("id") or event.get("replyToken")
            if not event_id or event_id in seen:
                continue
            seen[event_id] = now
            source = event.get("source", {})
            destination = source.get("groupId") or source.get("roomId") or source.get("userId")
            if not destination:
                continue
            # Navigation replies immediately; RAG acknowledges before background inference.
            reply_token = event.get("replyToken")
            navigation = ux.navigation("ถาม Assistant" if _is_greeting(text) else text)
            if navigation:
                if reply_token:
                    try:
                        await _line_post(LINE_REPLY_URL, config.line_access_token,
                                         {"replyToken": reply_token, "messages": navigation})
                    except Exception as exc:
                        seen.pop(event_id, None)
                        LOGGER.exception("LINE navigation failed")
                        raise HTTPException(status_code=502, detail="LINE navigation delivery failed") from exc
                continue
            key = conversation_key(source)
            clearing = text in ("ล้างประวัติ", "เริ่มใหม่")
            if reply_token and not clearing:
                try:
                    await _line_post(
                        LINE_REPLY_URL,
                        config.line_access_token,
                        {"replyToken": reply_token, "messages": [{"type": "text", "text": config.line_ack_text}]},
                    )
                except Exception:
                    LOGGER.exception("LINE acknowledgement failed")
            background_tasks.add_task(process_event, event, destination, key)
        return {"status": "ok"}

    return app
