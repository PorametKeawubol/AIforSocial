import base64
import hashlib
import hmac
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
import line_adapter


def test_duplicate_event_replies_once(monkeypatch, tmp_path):
    monkeypatch.setenv('CHANNEL_SECRET', 'test-secret')
    monkeypatch.setenv('CHANNEL_ACCESS_TOKEN', 'test-token')
    monkeypatch.setattr(line_adapter, 'ROOT', tmp_path)
    (tmp_path / 'artifacts').mkdir()
    sent = []
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            sent.append(kwargs['json'])
            class Response:
                def raise_for_status(self): pass
            return Response()
    import httpx
    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    app = FastAPI()
    line_adapter.install(app, lambda q: {'answer': 'test answer'})
    body = json.dumps({'events': [{'type': 'message', 'webhookEventId': 'event-1',
        'message': {'id': 'm1', 'type': 'text', 'text': 'test question'},
        'replyToken': 'test-reply'}]}).encode()
    signature = base64.b64encode(hmac.new(b'test-secret', body, hashlib.sha256).digest()).decode()
    with TestClient(app) as client:
        for _ in range(2):
            assert client.post('/callback', content=body, headers={'x-line-signature': signature}).status_code == 200
        assert client.post('/callback', content=body, headers={'x-line-signature': 'invalid'}).status_code == 400
    assert len(sent) == 1
    assert len((tmp_path / 'artifacts' / 'line_replies.jsonl').read_text().splitlines()) == 1
