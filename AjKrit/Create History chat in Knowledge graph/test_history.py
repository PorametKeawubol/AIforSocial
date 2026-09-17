import base64
import hashlib
import hmac
import json
from fastapi.testclient import TestClient
import webhook


def test_history_isolation_limit_reset_signature_and_dedup(monkeypatch, tmp_path):
    monkeypatch.setenv('CHANNEL_SECRET', 'test-secret')
    monkeypatch.setenv('CHANNEL_ACCESS_TOKEN', 'test-token')
    monkeypatch.setattr(webhook, 'ARTIFACTS', tmp_path)
    webhook.user_chat_history.clear()
    webhook.seen.clear()
    calls, sent = [], []

    def answer(question, history):
        calls.append((question, list(history)))
        return {'answer': 'answer:' + question}

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            sent.append(kwargs['json'])
            class Response:
                def raise_for_status(self): pass
            return Response()

    monkeypatch.setattr(webhook, 'answer_question', answer)
    monkeypatch.setattr(webhook.httpx, 'AsyncClient', Client)
    with TestClient(webhook.app) as client:
        def post(n, user='user-a', question=None, valid=True):
            payload = {'events': [{'type': 'message', 'webhookEventId': str(n),
                'source': {'type': 'user', 'userId': user}, 'replyToken': 'test-reply',
                'message': {'type': 'text', 'id': str(n), 'text': question or f'question {n}'}}]}
            body = json.dumps(payload).encode()
            sig = base64.b64encode(hmac.new(b'test-secret', body, hashlib.sha256).digest()).decode()
            return client.post('/callback', content=body,
                headers={'x-line-signature': sig if valid else 'invalid'})

        assert post(0, valid=False).status_code == 400
        assert not calls
        for n in range(1, 6):
            assert post(n).status_code == 200
        assert [len(history) for _, history in calls] == [0, 2, 4, 6, 6]
        assert calls[1][1][0]['content'] == 'question 1'
        assert calls[4][1][0]['content'] == 'question 2'
        assert post(5).status_code == 200
        assert len(sent) == 5
        post(6, user='user-b')
        assert calls[-1][1] == []
        post(7, question='ล้างประวัติ')
        assert 'user-a' not in webhook.user_chat_history
        assert 'user-b' in webhook.user_chat_history
        post(8)
        assert calls[-1][1] == []
    assert len((tmp_path / 'replies.jsonl').read_text().splitlines()) == 8


def test_retrieval_expands_followup_and_passes_history(monkeypatch):
    import rag_search
    history = [{'role': 'user', 'content': 'โรคใบด่าง CVMV คืออะไร'},
               {'role': 'assistant', 'content': 'เป็นโรคในพริก [หน้า 30]'}]
    prompts = []
    def chat(messages, **kwargs):
        prompts.append(messages)
        return 'แพร่โดยการสัมผัสและเพลี้ยอ่อน [หน้า 30]', {}
    monkeypatch.setattr(rag_search, 'chat', chat)
    r = rag_search.Retriever.__new__(rag_search.Retriever)
    queries = []
    def search(query):
        queries.append(query)
        return {'chunks': [{'page': 30, 'text': 'แพร่เชื้อด้วยการสัมผัส มีเพลี้ยอ่อนเป็นพาหะ'}],
                'facts': [], 'warnings': []}
    monkeypatch.setattr(r, 'search', search)
    result = r.answer('แล้วโรคนี้แพร่อย่างไร?', chat_history=history)
    assert len(queries) == 1
    assert queries[0].startswith('แล้วโรคนี้แพร่อย่างไร?')
    assert 'โรคใบด่าง CVMV คืออะไร' in queries[0]
    assert len(prompts) == 1
    assert prompts[0][1:3] == history
    assert result['query'] == 'แล้วโรคนี้แพร่อย่างไร?'
    assert result['history_messages'] == 2
    assert result['citations'] == [30]
