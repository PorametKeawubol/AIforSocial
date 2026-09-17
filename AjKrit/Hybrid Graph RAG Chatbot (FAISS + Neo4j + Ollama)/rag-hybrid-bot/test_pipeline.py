import hashlib, json
import pytest
from core import clean, Facts, DATA, read
from build_index import RELATIONS,TYPES,valid_fact_types,grounded_names

def test_pdf_and_chunks_provenance():
    pages={p['page']:p['text'] for p in read('pages.json')}
    chunks=read('chunks.json')
    assert len(pages)==48
    assert len({c['id'] for c in chunks})==len(chunks)
    assert all(c['text'] in pages[c['page']] for c in chunks)

def test_graph_quotes_and_categories():
    pages={p['page']:p['text'] for p in read('pages.json')}
    facts=read('facts.json')
    assert facts
    assert all(f['evidence'] in pages[f['page']] and f['relation'] in RELATIONS and f['subject_type'] in TYPES and f['object_type'] in TYPES for f in facts)
    assert all(valid_fact_types(f) and grounded_names(f,pages[f["page"]]) for f in facts)
    assert not read('graph_failures.json')

def test_index_manifest():
    import faiss
    m=read('index_manifest.json');i=faiss.read_index(str(DATA/'index.faiss'))
    assert m['chunks_sha256']==hashlib.sha256((DATA/'chunks.json').read_bytes()).hexdigest()
    assert i.ntotal==len(read('chunks.json')) and i.d==m['dimension']

def test_database_export_matches_facts():
    assert len(read('facts.json'))==len(read('neo4j_export.json'))

def test_answer_citation_membership():
    answers=read('answers.json')
    assert answers
    for r in answers:
        assert set(r['citations']) <= {c['page'] for c in r['chunks']}
        assert not r['warnings']

def test_api_rejects_invalid_request():
    from fastapi.testclient import TestClient
    from app import app
    c=TestClient(app)
    assert c.post('/api/ask',json={'query':''}).status_code==422
    assert c.post('/api/ask',json={'query':'a'*1501}).status_code==422
    assert c.get('/api/status').status_code==200

def test_line_signature_and_missing_credentials(monkeypatch):
    import base64,hmac,hashlib
    from line_adapter import verify_signature
    body=b'{"events":[]}'
    signature=base64.b64encode(hmac.new(b'test-secret',body,hashlib.sha256).digest()).decode()
    assert verify_signature(body,signature,'test-secret')
    assert not verify_signature(body+b' ',signature,'test-secret')
    from fastapi.testclient import TestClient
    from app import app
    monkeypatch.delenv('CHANNEL_SECRET',raising=False)
    monkeypatch.delenv('CHANNEL_ACCESS_TOKEN',raising=False)
    assert TestClient(app).post('/callback',content=body).status_code==503

def test_quote_recovery_never_invents_text():
    from build_index import resolve_quote
    assert resolve_quote('โรคแคงเกอร์ เกิดจากเชื้อแบคทีเรีย','โรคแคงเกอร์\nเกิดจากเชื้อแบคทีเรีย')=='โรคแคงเกอร์\nเกิดจากเชื้อแบคทีเรีย'
    assert resolve_quote('โรคแคงเกอร์เกิดจากเชื้อรา','โรคแคงเกอร์เกิดจากเชื้อแบคทีเรีย') is None

def test_out_of_document_abstention():
    r=next(r for r in read('answers.json') if 'Windows' in r['query'])
    assert 'ไม่พบข้อมูล' in r['answer']
    assert not r['citations']

def test_graph_outage_keeps_vector_lexical_retrieval(monkeypatch):
    import numpy as np
    import rag_search
    r=rag_search.Retriever()
    monkeypatch.setattr(rag_search,'embed',lambda texts,model:np.array([r.index.reconstruct(0)],dtype='float32'))
    def offline():raise ConnectionError('test outage')
    monkeypatch.setattr(rag_search,'driver',offline)
    result=r.search('โรคในพืช')
    assert result['chunks']
    assert result['warnings']==['Neo4j unavailable: ConnectionError']
    assert 'dense' in result['rankings'] and 'bm25' in result['rankings']


def test_reingestion_idempotence():
    assert read("ingest_idempotence.json")["passed"]
