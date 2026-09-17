"""LINE history lab: per-user memory, contextual retrieval, authentic reply log."""
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
RAG = LAB.parent / 'Hybrid Graph RAG Chatbot (FAISS + Neo4j + Ollama)' / 'rag-hybrid-bot'
sys.path.insert(0, str(RAG))
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
import httpx

load_dotenv(RAG / '.env')
ARTIFACTS = LAB / 'LINE_history_5'
ARTIFACTS.mkdir(exist_ok=True)
MAX_HISTORY_MESSAGES = 6
user_chat_history = {}
seen = {}
processing = asyncio.Lock()
retriever = None
app = FastAPI(title='LINE History Chat Lab')


def answer_question(question, history):
    global retriever
    from rag_search import Retriever
    if retriever is None:
        retriever = Retriever()
    return retriever.answer(question, chat_history=history)


async def handle_event(event):
    async with processing:
        user_id = event['source']['userId']
        question = event['message']['text'].strip()
        history = list(user_chat_history.get(user_id, []))
        try:
            if question == 'ล้างประวัติ':
                result = {'answer': 'ล้างประวัติการสนทนาเรียบร้อยแล้วครับ'}
            else:
                result = await run_in_threadpool(answer_question, question, history)
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post('https://api.line.me/v2/bot/message/reply',
                    headers={'Authorization': 'Bearer ' + os.environ['CHANNEL_ACCESS_TOKEN']},
                    json={'replyToken': event['replyToken'],
                          'messages': [{'type': 'text', 'text': result['answer'][:4900]}]})
                response.raise_for_status()
            if question == 'ล้างประวัติ':
                user_chat_history.pop(user_id, None)
            else:
                user_chat_history[user_id] = (history + [
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': result['answer'][:4900]}
                ])[-MAX_HISTORY_MESSAGES:]
            # No user IDs, reply tokens, or credentials in the submission log.
            with (ARTIFACTS / 'replies.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps({'question': question, 'answer': result['answer'][:4900],
                    'search_query': result.get('search_query', question),
                    'history_messages': len(history), 'citations': result.get('citations', []),
                    'warnings': result.get('warnings', []), 'sent_at': time.time()},
                    ensure_ascii=False) + '\n')
        except Exception:
            logging.exception('LINE history reply failed')


@app.get('/health')
def health():
    return {'status': 'ok', 'max_history_messages': MAX_HISTORY_MESSAGES}


@app.post('/callback')
async def callback(request: Request, background_tasks: BackgroundTasks):
    secret = os.getenv('CHANNEL_SECRET')
    if not secret or not os.getenv('CHANNEL_ACCESS_TOKEN'):
        raise HTTPException(503, 'LINE credentials missing')
    body = await request.body()
    if len(body) > 1_000_000:
        raise HTTPException(413, 'Payload too large')
    signature = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    if not hmac.compare_digest(signature, request.headers.get('x-line-signature', '')):
        raise HTTPException(400, 'Invalid signature')
    try:
        events = json.loads(body)['events']
        if not isinstance(events, list):
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, 'Invalid payload')
    now = time.time()
    for key, timestamp in list(seen.items()):
        if now - timestamp > 3600:
            del seen[key]
    for event in events:
        if event.get('type') != 'message' or event.get('message', {}).get('type') != 'text':
            continue
        # This lab uses direct user chats; group context is deliberately excluded.
        if event.get('source', {}).get('type') != 'user' or not event['source'].get('userId'):
            continue
        question = event['message'].get('text', '').strip()
        if not 2 <= len(question) <= 1500:
            continue
        key = event.get('webhookEventId') or event['message'].get('id')
        if not key or key in seen:
            continue
        seen[key] = now
        background_tasks.add_task(handle_event, event)
    return {'status': 'ok'}
