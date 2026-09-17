from threading import Lock
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from core import ROOT, DATA, PDF, read, dump
from rag_search import Retriever
app=FastAPI(title='Plant Knowledge · Hybrid Graph RAG')
lock=Lock()
retriever=None
class Query(BaseModel):
    query:str=Field(min_length=2,max_length=1500)
@app.get('/')
def home():return FileResponse(ROOT/'web/index.html')
@app.get('/api/status')
def status():
    return {'source':read('source.json'),'index':read('index_manifest.json'),'graph_facts':len(read('neo4j_export.json'))}
@app.get('/api/graph')
def graph():return read('neo4j_export.json')
@app.get('/api/examples')
def examples():return read('answers.json') if (DATA/'answers.json').exists() else []
@app.get('/document.pdf')
def document():return FileResponse(PDF,media_type='application/pdf')
@app.post('/api/ask')
def ask(body:Query):
    global retriever
    if not lock.acquire(blocking=False):raise HTTPException(429,'กำลังประมวลผลคำถามก่อนหน้า')
    try:
        if retriever is None:retriever=Retriever()
        result=retriever.answer(body.query)
        dump(DATA/'last_answer.json',result)
        return result
    except Exception as e:raise HTTPException(503,'ประมวลผลไม่สำเร็จ: '+type(e).__name__)
    finally:lock.release()
app.mount('/checkpoints',StaticFiles(directory=DATA/'checkpoints'),name='checkpoints')

from line_adapter import install
install(app,lambda query:ask(Query(query=query)))
