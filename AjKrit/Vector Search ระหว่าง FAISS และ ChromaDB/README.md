# FAISS vs ChromaDB: Hybrid RAG Vector Search

โค้ดนี้ทำใบงานเปรียบเทียบ Vector Search จากข้อมูลในโฟลเดอร์เดียวกัน โดยใช้ embedding ชุดเดียวกันจาก `BAAI/bge-m3` แล้วส่ง vector ที่ normalize แล้วให้ทั้ง FAISS และ ChromaDB

## ไฟล์สำคัญ

- `faiss_chromadb_benchmark.py` — โหลดข้อมูล สร้าง index วัด build time/query latency และเขียนรายงาน
- `test_faiss_chromadb_benchmark.py` — unit tests และ integration tests แบบขนาดเล็ก
- `thai_qa_utf8.json` — corpus ที่ให้มา
- `thai_qa_paraphrase_15.csv` — query 15 รายการที่ให้มา
- `ใบงานปฏิบัติการ_ การเปรียบเทียบประสิทธิภาพ Vector Search ระหว่าง FAISS และ ChromaDB สำหรับระบบ Hybrid RAG.md` — ใบงานและคำอธิบายผล

## ติดตั้งและทดสอบ

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

หากใช้ Windows ให้เปลี่ยน `.venv/bin/python` เป็น `.venv\Scripts\python.exe`

## รัน benchmark

คำสั่งหลักใช้ context ที่มีค่า `input` จำนวน 10,000 รายการ, query ทั้ง 15 รายการ, Top-K=5 และวัดซ้ำ query ละ 10 รอบ

```bash
.venv/bin/python faiss_chromadb_benchmark.py --no-progress
```

ตัวเลือกที่ใช้บ่อย:

```bash
.venv/bin/python faiss_chromadb_benchmark.py --limit 10000 --benchmark-repeats 30
.venv/bin/python faiss_chromadb_benchmark.py --limit 5000 --query-limit 5 --benchmark-repeats 5
```

ผลลัพธ์จะอยู่ใน `results/` และฐานข้อมูล persistent อยู่ใน `chroma_db/`:

- `build_metrics.csv` — build time, ขนาด vector และ storage
- `latency_samples.csv` — latency ของทุก query/repeat
- `query_results_topk.csv` — ผล Top-K ของทั้งสองระบบ
- `metrics.json` — สรุปตัวเลขทั้งหมดสำหรับกรอกใบงาน

## หมายเหตุเรื่องข้อมูล

ไฟล์ JSON มี 41,740 แถว แต่มีเพียง 14,740 แถวที่มี context ในฟิลด์ `input`; แถวจากบาง source เช่น `wiki_qa` ไม่มี context จึงนำไป embed ไม่ได้ โปรแกรมจะข้ามแถวดังกล่าวและเลือก 10,000 แถวแรกที่มี `input` แบบ deterministic โดยสร้าง document ID จากตำแหน่งแถวใหม่ เพราะ `__index_level_0__` ซ้ำกันในข้อมูลจริง

## ตีความผล

- FAISS ใช้ `IndexFlatIP` เป็น exact search ใน memory; vector ที่ normalize แล้วทำให้ inner product เทียบเท่า cosine similarity และค่ามากกว่าดีกว่า
- ChromaDB ใช้ collection ที่กำหนด `hnsw:space=cosine`; ผลลัพธ์เป็น cosine distance และค่าน้อยกว่าดีกว่า จึงไม่ควรเทียบค่าคะแนนดิบข้ามระบบโดยตรง
- latency ที่รายงานเริ่มจับเวลาหลัง query embedding เสร็จแล้ว เพื่อไม่ให้เวลาโหลด/encode โมเดลกลบความต่างของ vector engine
- ตัวอย่าง metadata filter ใช้ `source`: FAISS ต้องค้นเต็มแล้วกรองใน application ส่วน ChromaDB ใช้ `where` ใน collection ได้โดยตรง ซึ่งเป็นจุดสำคัญของ Hybrid RAG
