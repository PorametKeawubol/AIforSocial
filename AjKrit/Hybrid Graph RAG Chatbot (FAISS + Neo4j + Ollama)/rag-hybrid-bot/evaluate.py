"""Small, manually page-labelled diagnostic set; not a held-out quality benchmark."""
import csv, time
import numpy as np
from core import *
from rag_search import Retriever
GOLD=[
('การห่มดินช่วยรักษาความชื้นและจุลินทรีย์อย่างไร',[6,7,8]),
('ใช้วัสดุอะไรคลุมหน้าดินเพื่อแก้ดินเค็ม',[7,8]),
('สังเกตดินเปรี้ยวในนาข้าวจากใบได้อย่างไร',[11]),
('น้ำส้มควันไม้ต้องพักกี่เดือนก่อนนำไปใช้',[13]),
('น้ำส้มควันไม้สำหรับพ่นดินก่อนปลูกต้องเจือจางเท่าไร',[14,15]),
('โรคใบไหม้แผลใหญ่ในข้าวโพดมีสาเหตุและอาการอะไร',[17]),
('ข้าวโพดมีใบลายและผงสีขาวตอนเช้าเกี่ยวข้องกับโรคอะไร',[18]),
('โรคแคงเกอร์เกิดจากเชื้อชนิดใดและพบในพืชอะไร',[20]),
('การใช้ปูนขาวป้องกันแคงเกอร์มีข้อควรระวังอะไร',[22]),
('พริกเป็นจุดฉ่ำน้ำรูปวงรีและแผลขยายเกี่ยวกับโรคอะไร',[24]),
('โรคเหี่ยวจากเชื้อแบคทีเรียในพริกมีอาการอย่างไร',[28]),
('โรคใบด่างจากไวรัส PVY มีอาการอะไร',[31]),
('โรครากปมในพริกเกิดจากอะไร',[36]),
('วิธีล้างใบป้องกันราน้ำค้างทำช่วงเวลาใด',[40]),
('โรครากเน่าโคนเน่าในทุเรียนมีอาการเริ่มต้นอย่างไร',[46]),
('สูตรกำจัดวัชพืชด้วยน้ำส้มสายชูใช้วัตถุดิบอะไร',[44]),
]

def run():
    dump(DATA/'evaluation_questions.json',[dict(query=q,relevant_pages=p) for q,p in GOLD])
    r=Retriever();rows=[]
    baseline='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    started=time.perf_counter();v=embed([c['text'] for c in r.chunks],baseline);qv=embed([q for q,_ in GOLD],baseline)
    base_seconds=time.perf_counter()-started
    for mode in ['minilm_dense','dense','bm25','hybrid']:
        for idx,(q,gold) in enumerate(GOLD):
            if mode=='minilm_dense':
                order=np.argsort(-(v@qv[idx]))[:5];pages=[r.chunks[i]['page'] for i in order];seconds=None;warnings=[]
            else:
                res=r.search(q,k=5,mode=mode);pages=[c['page'] for c in res['chunks']];seconds=res['seconds'];warnings=res['warnings']
            rank=next((i for i,p in enumerate(pages,1) if p in gold),None)
            rows.append(dict(mode=mode,query=q,gold_pages=gold,retrieved_pages=pages,hit_at_1=int(pages[0] in gold),hit_at_5=int(rank is not None),mrr_at_5=1/rank if rank else 0,seconds=seconds,warnings=warnings))
    summary={mode:{key:round(float(np.mean([r[key] for r in rows if r['mode']==mode])),4) for key in ['hit_at_1','hit_at_5','mrr_at_5']} for mode in ['minilm_dense','dense','bm25','hybrid']}
    dump(DATA/'retrieval_evaluation.json',dict(summary=summary,results=rows,baseline_encoding_seconds=base_seconds))
    with (DATA/'retrieval_evaluation.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    checkpoint(5,'เปรียบเทียบการค้นคืน 4 วิธี · 16 คำถาม',dict(metrics=summary,questions=len(GOLD),gold='Manually labelled relevant PDF pages; Hit@k means at least one relevant page.',limitations='Small diagnostic set, not held-out; retrieval metrics do not measure answer correctness. Hybrid preserves dense top-3 after inspecting this diagnostic set; post-tuning results are not independent validation. No reranker.',failures=[row for row in rows if not row['hit_at_5']]))

if __name__=='__main__':run()
