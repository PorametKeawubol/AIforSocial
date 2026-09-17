import time
from core import *
from rag_search import Retriever
QUERIES=[
 'โรคแคงเกอร์เกิดจากเชื้อชนิดใดและพบในพืชอะไร',
 'โรครากเน่าโคนเน่าในทุเรียนมีอาการเริ่มต้นอย่างไร',
 'โรครากปมในพริกเกิดจากอะไร',
 'วิธีล้างใบป้องกันราน้ำค้างทำช่วงเวลาใด',
 'เอกสารนี้บอกวิธีติดตั้ง Windows 11 อย่างไร',
]
if __name__=='__main__':
    r=Retriever();answers=[]
    for q in QUERIES:
        out=r.answer(q);answers.append(out);dump(DATA/'answers.json',answers)
        print(q,out['generation'],flush=True)
    checkpoint(6,'ทดสอบ Chatbot · คำตอบ อ้างอิง และคำถามนอกเอกสาร',dict(model=MODEL,answers=[{k:a[k] for k in ['query','answer','citations','generation','warnings']} for a in answers],note='Page citation membership is validated automatically. It does not prove all claims are entailed by the cited page.'))
