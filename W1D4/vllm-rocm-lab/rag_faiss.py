import os

import httpx
import faiss
import numpy as np
from openai import APIConnectionError, OpenAI
from sentence_transformers import SentenceTransformer

MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-0.6B")
BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")

documents = [
    "Data Engineering คือการจัดการข้อมูลตั้งแต่การเก็บรวบรวมจนถึงการใช้งาน",
    "Machine Learning คือการให้คอมพิวเตอร์เรียนรู้จากข้อมูล",
    "Deep Learning คือส่วนหนึ่งของ Machine Learning ที่ใช้ Neural Network",
    "FAISS ใช้สำหรับค้นหาความคล้ายของเวกเตอร์อย่างรวดเร็ว",
    "RAG คือระบบที่นำข้อมูลจากเอกสารมาใช้ประกอบการตอบของ AI",
]

embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
embeddings = np.array(embedder.encode(documents)).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def search(query: str, k: int = 2) -> list[str]:
    query_vec = np.array(embedder.encode([query])).astype("float32")
    _, indices = index.search(query_vec, k)
    return [documents[i] for i in indices[0]]


question = "FAISS คืออะไร"
results = search(question)
context = "\n".join(results)

prompt = f"""ใช้ข้อมูลต่อไปนี้ตอบคำถาม หากข้อมูลไม่พอให้ตอบว่าไม่พบข้อมูลในเอกสาร

{context}

คำถาม: {question} /no_think
ตอบเป็นภาษาไทยแบบสั้น:
"""

print("จำนวนเอกสาร:", index.ntotal)
print("ข้อมูลที่ค้นพบ:")
for result in results:
    print("-", result)

try:
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256,
    )
except (httpx.HTTPError, APIConnectionError) as exc:
    print("\nคำตอบจาก RAG:")
    print(f"ยังเรียก vLLM API ไม่ได้: {exc}")
    print(f"เปิด OpenAI-compatible server ที่ {BASE_URL} แล้วรันไฟล์นี้อีกครั้ง")
else:
    print("\nคำตอบจาก RAG:")
    print(response.choices[0].message.content)
