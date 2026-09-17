"""Local, evidence-linked Hybrid Graph RAG. No generated Cypher or pickle loading."""
import os, json, hashlib, re, time
from pathlib import Path
from functools import lru_cache
from typing import Literal
import httpx
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
DATA = ROOT / 'artifacts'
DATA.mkdir(exist_ok=True)
PDF = Path(os.getenv('PDF_PATH', str(ROOT.parent / '9-file-4786.pdf')))
MODEL = os.getenv('OLLAMA_MODEL', 'qwen3.5:9b')
EMBED = os.getenv('EMBED_MODEL', 'BAAI/bge-m3')
HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8'); tmp.replace(path)

def read(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))

def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()[:24]

def clean(text):
    return re.sub(r'[ĻÎÊĕ\ufffd]', '', text).replace('กdำ', 'กำ').replace('ํา','ำ').replace('\u0000','').strip()

def driver():
    return GraphDatabase.driver(os.getenv('NEO4J_URI', 'bolt://localhost:7688'), auth=(os.getenv('NEO4J_USERNAME','neo4j'),os.environ['NEO4J_PASSWORD']))

def chat(messages, schema=None, model=None, limit=1800):
    payload = dict(model=model or MODEL, messages=messages, stream=False, think=False,
                   options=dict(temperature=0, num_ctx=8192, num_predict=limit), keep_alive='15m')
    if schema:
        payload['format'] = schema
        payload['messages'] = [dict(role='system',content='\n'.join(m['content'] for m in messages if m['role']=='system')+'\nReturn ONLY valid JSON matching this schema: '+json.dumps(schema,ensure_ascii=False)), *[m for m in messages if m['role']!='system']]
    started = time.perf_counter()
    with httpx.Client(timeout=600) as client:
        response = client.post(HOST+'/api/chat', json=payload); response.raise_for_status()
    result = response.json()
    if result.get('done_reason') == 'length':
        raise RuntimeError('LLM output truncated; raise token limit or reduce input')
    return result['message']['content'], {'seconds':round(time.perf_counter()-started,3), 'model':payload['model'],
            'eval_count':result.get('eval_count'), 'eval_duration':result.get('eval_duration'), 'prompt_eval_count':result.get('prompt_eval_count')}

class Fact(BaseModel):
    subject: str = Field(min_length=1)
    subject_type: Literal['Crop','Disease','Pathogen','Symptom','Treatment','Topic']
    relation: Literal['HAS_DISEASE','CAUSED_BY','EXHIBITS','TREATED_BY','PREVENTED_BY','RELATED_TO']
    object: str = Field(min_length=1)
    object_type: Literal['Crop','Disease','Pathogen','Symptom','Treatment','Topic']
    evidence: str = Field(description='Exact continuous quote from the source text proving this relation', min_length=4)

class Facts(BaseModel):
    facts: list[Fact] = Field(default_factory=list)

@lru_cache(maxsize=2)
def encoder(model=EMBED):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(6)
    m = SentenceTransformer(model, device='cpu', local_files_only=os.getenv('HF_HUB_OFFLINE')=='1')
    m.max_seq_length = 1024
    return m

def embed(texts, model=EMBED):
    return encoder(model).encode(texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype('float32')

def tokenize(text):
    from pythainlp.tokenize import word_tokenize
    return [t.lower() for t in word_tokenize(text, engine='newmm') if t.strip()]

STYLE = '''body{font-family:"Noto Sans Thai",Tahoma,sans-serif;background:#eff5f2;color:#17352d;margin:0;padding:40px;line-height:1.7}main{max-width:1200px;margin:auto}h1{font-size:32px}h2{color:#207359}article{background:white;padding:28px;border-radius:18px;margin:22px 0;border:1px solid #d9e6df}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #d9e6df;padding:10px;text-align:left}.tag{color:#267454;font-weight:bold}a{color:#207359}'''

def checkpoint(number, title, summary):
    from html import escape
    p = DATA / 'checkpoints' / f'{number:02d}'
    p.mkdir(parents=True, exist_ok=True)
    dump(p/'results.json', summary)
    (p/'report.html').write_text(f'<!doctype html><html lang="th"><meta charset="utf-8"><style>{STYLE}</style><main><span class="tag">HYBRID GRAPH RAG · CHECKPOINT {number:02d}</span><h1>{escape(title)}</h1><article><pre>{escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre></article></main></html>',encoding='utf-8')
    print(f'Checkpoint {number:02d}: {title}', flush=True)
