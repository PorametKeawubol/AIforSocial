"""LINE webhook with immediate acknowledgement and in-process deduplication."""
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from fastapi import BackgroundTasks, HTTPException, Request
from core import ROOT


def verify_signature(body: bytes, signature: str, secret: str) -> bool:
    expected = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    return hmac.compare_digest(expected, signature)


def install(app, answer_function):
    seen = {}
    processing = asyncio.Lock()

    async def reply(event, token):
        from starlette.concurrency import run_in_threadpool
        import httpx
        async with processing:
            try:
                out = await run_in_threadpool(answer_function, event['message']['text'])
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        'https://api.line.me/v2/bot/message/reply',
                        headers={'Authorization': 'Bearer ' + token},
                        json={'replyToken': event['replyToken'], 'messages': [
                            {'type': 'text', 'text': out['answer'][:4900]}]})
                    response.raise_for_status()
                with (ROOT / 'artifacts' / 'line_replies.jsonl').open('a') as f:
                    f.write(json.dumps({'query': event['message']['text'], 'answer': out['answer'],
                                        'sent_at': time.time()}, ensure_ascii=False) + '\n')
            except Exception:
                logging.exception('LINE reply failed')

    @app.post('/callback')
    async def callback(request: Request, background_tasks: BackgroundTasks):
        secret = os.getenv('CHANNEL_SECRET')
        token = os.getenv('CHANNEL_ACCESS_TOKEN')
        if not secret or not token:
            raise HTTPException(503, 'LINE credentials are not configured')
        body = await request.body()
        if len(body) > 1_000_000:
            raise HTTPException(413, 'Payload too large')
        if not verify_signature(body, request.headers.get('x-line-signature', ''), secret):
            raise HTTPException(400, 'Invalid signature')
        try:
            events = json.loads(body).get('events', [])
        except (ValueError, AttributeError):
            raise HTTPException(400, 'Invalid JSON')
        now = time.time()
        for key, timestamp in list(seen.items()):
            if now - timestamp > 3600:
                del seen[key]
        for event in events:
            if event.get('type') != 'message' or event.get('message', {}).get('type') != 'text':
                continue
            text = event['message']['text']
            if not 2 <= len(text) <= 1500:
                continue
            key = event.get('webhookEventId') or event['message'].get('id') or event['replyToken']
            if key in seen:
                continue
            seen[key] = now
            background_tasks.add_task(reply, event, token)
        return {'status': 'ok'}
