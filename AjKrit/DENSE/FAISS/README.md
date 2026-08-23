# เปรียบเทียบ FAISS และ ChromaDB สำหรับ Hybrid RAG

โปรเจกต์นี้ทำตามใบงานโดยเปรียบเทียบการค้นคืนเอกสารภาษาไทยด้วย
FAISS และ ChromaDB บนเอกสารชุดเดียวกัน และใช้ embedding ชุดเดียวกันจาก
`BAAI/bge-m3`

ในใบงานนี้ Hybrid Retrieval หมายถึง **Semantic Search + Metadata Filter**
ไม่ใช่การรวม BM25 กับ dense retrieval และโปรเจกต์นี้ยังไม่มี LLM สำหรับ
สร้างคำตอบ

## สิ่งที่มีในโฟลเดอร์

- `hybrid_rag_comparison.py` — โปรแกรมหลักสำหรับรันทุก Exercise
- `test_hybrid_rag_comparison.py` — unit tests ที่ไม่ดาวน์โหลดโมเดล
- `requirements.txt` — dependencies ที่ต้องใช้
- `🧪 ใบงานฝึกปฏิบัติ_ เปรียบเทียบ FAISS และ ChromaDB สำหรับ Hybrid RAG.md`
  — โจทย์ต้นฉบับ

## ติดตั้ง

เปิด PowerShell ที่โฟลเดอร์นี้ แล้วสร้าง virtual environment ของโครงการ
หรือใช้ environment ที่มีอยู่แล้ว:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

การรันครั้งแรกจะดาวน์โหลด `BAAI/bge-m3` จาก Hugging Face หากยังไม่มีอยู่ใน
cache จึงต้องเชื่อมต่ออินเทอร์เน็ต

## รันชุดข้อมูลหลัก 8 เอกสาร

```powershell
.\.venv\Scripts\python.exe hybrid_rag_comparison.py
```

คำสั่งนี้ทำ Exercise 1, Exercise 2, การค้นหาแบบ category filter และวัด
Recall@1/@3/@5 กับ latency พร้อมพิมพ์ตารางสรุปผลของแต่ละ Exercise ใน console

## รัน Exercise 3 พร้อมเอกสารเพิ่ม 5 รายการ

```powershell
.\.venv\Scripts\python.exe hybrid_rag_comparison.py --include-extensions --output-dir results/extensions
```

ตัวเลือกสำคัญเพิ่มเติม:

```powershell
.\.venv\Scripts\python.exe hybrid_rag_comparison.py --benchmark-runs 10
.\.venv\Scripts\python.exe hybrid_rag_comparison.py --chroma-dir chroma_db_comparison
```

## ทดสอบ

```powershell
.\.venv\Scripts\python.exe -m pytest test_hybrid_rag_comparison.py -v
```

## รายงานที่ได้

โปรแกรมสร้างไฟล์ใน `results/` หรือโฟลเดอร์ที่กำหนดผ่าน `--output-dir`

- `semantic_top3.csv` — Top-3 ของคำถามเชิงความหมาย 5 ข้อจาก FAISS และ
  ChromaDB
- `exact_code_comparison.csv` — ผลของ FAISS pure semantic, FAISS พร้อม
  application-side metadata filter และ ChromaDB `where` filter สำหรับรหัส
  `REG-2045`, `IT-7788`, `FIN-4020`, `LIB-7033`
- `metrics.json` — จำนวนเอกสาร, Recall@1/@3/@5, mean/median retrieval
  latency และผล ChromaDB category filter (`category = REG`)

FAISS แสดงค่า **similarity** เพราะใช้ normalized vector กับ `IndexFlatIP`
ส่วน ChromaDB แสดงค่า **cosine distance** จึงไม่ควรเทียบค่าตัวเลขทั้งสอง
ระบบกันโดยตรง แม้อันดับผลลัพธ์จะเปรียบเทียบกันได้

ตัวเลข latency วัดเฉพาะการค้นคืนหลังจากสร้าง query embedding แล้ว เพื่อไม่ให้
เวลา encode ซึ่งสูงกว่ามากใน corpus ขนาดเล็ก กลบความต่างของ vector store

## คำตอบคำถามอภิปราย

1. เมื่อใช้ embedding เดียวกัน FAISS และ ChromaDB ควรให้ผล semantic search
   ใกล้เคียงกัน เพราะใช้ความสัมพันธ์ของ vector ชุดเดียวกัน แต่ลำดับหรือค่าคะแนน
   อาจต่างกันได้จากวิธี index และ convention ของคะแนน (similarity เทียบกับ
   distance)
2. FAISS เป็น vector library ที่เก็บและค้นหา vector เป็นหลัก จึงไม่ได้จัดการ
   metadata filter ให้โดยตรง โปรแกรมต้องเก็บ metadata และเขียน logic สำหรับ
   กรอง `doc_code` เอง
3. ChromaDB เก็บ metadata ไปพร้อมเอกสารและ embedding จึงใช้
   `where={"doc_code": "REG-2045"}` เพื่อจำกัด candidate ให้ตรงรหัสก่อน/ร่วมกับ
   semantic retrieval ได้ ทำให้ไม่คืนเอกสารที่เพียงคล้ายเชิงความหมายแต่เป็นคนละรหัส

## ความเป็นธรรมของการเปรียบเทียบ

โปรแกรมโหลด Sentence Transformer เพียงครั้งเดียว แล้วส่ง document embedding
และ query embedding ที่ normalize แล้วชุดเดียวกันให้ทั้ง FAISS และ ChromaDB
ChromaDB จึงไม่สร้าง embedding ซ้ำด้วย embedding function ภายใน ซึ่งช่วยให้ผลต่าง
ที่พบอธิบายได้ว่าเกิดจาก vector store และ metadata capability มากกว่าเกิดจากโมเดล
