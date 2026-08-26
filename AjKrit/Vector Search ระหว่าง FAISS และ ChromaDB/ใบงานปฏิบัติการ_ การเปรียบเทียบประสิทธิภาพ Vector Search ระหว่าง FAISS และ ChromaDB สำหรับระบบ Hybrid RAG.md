# ใบงานปฏิบัติการ: การเปรียบเทียบประสิทธิภาพ Vector Search ระหว่าง FAISS และ ChromaDB สำหรับระบบ Hybrid RAG

## 1. วัตถุประสงค์

1. ศึกษาการสร้าง vector index และการค้นหาความสัมพันธ์ของข้อความด้วย FAISS และ ChromaDB
2. วัดเวลา build index และ query latency บน corpus ภาษาไทย/หลายภาษาขนาด 10,000 รายการ
3. วิเคราะห์ความเหมาะสมของ vector engine สำหรับสถาปัตยกรรม Hybrid RAG ที่ต้องใช้ semantic retrieval ร่วมกับ metadata filter

## 2. ข้อมูลและข้อสังเกตจากไฟล์ที่ให้มา

- `thai_qa_utf8.json` มี 41,740 แถวจากหลาย source
- มี 14,740 แถวที่มีข้อความ context ในฟิลด์ `input`; แถวที่ไม่มี `input` ไม่สามารถนำไปสร้าง document embedding ได้
- โปรแกรมจึงเลือก 10,000 แถวแรกที่มี `input` แบบ deterministic เมื่อใช้ค่าเริ่มต้น `--limit 10000`
- ฟิลด์ `__index_level_0__` ซ้ำกันในข้อมูลจริง จึงใช้ ID `doc-00000`, `doc-00001`, ... จากตำแหน่ง document ที่เลือก เพื่อป้องกัน ChromaDB duplicate ID
- `thai_qa_paraphrase_15.csv` มี query 15 รายการ ใช้คอลัมน์ `instruction`

## 3. ทฤษฎีที่เกี่ยวข้อง

### FAISS

FAISS เป็น library ที่เน้น similarity search ใน memory โปรแกรมนี้ใช้ `IndexFlatIP` ซึ่งเป็น exact search โดยคำนวณ inner product กับ vector ทั้งหมด เมื่อ normalize vector แล้ว inner product จะเทียบเท่า cosine similarity ข้อดีคือโครงสร้างเรียบง่ายและ latency ต่ำ ข้อจำกัดคือ FAISS ไม่ได้จัดการ document metadata หรือ persistence ให้โดยอัตโนมัติ

### ChromaDB

ChromaDB เป็น vector database ที่เก็บ document, embedding และ metadata ไว้ด้วยกัน โปรแกรมกำหนด collection เป็น `hnsw:space=cosine` และส่ง embedding ที่คำนวณไว้แล้วเข้าไปโดยตรง ChromaDB จึงไม่ encode ซ้ำ และสามารถใช้ `where` กรอง metadata เช่น `source` ระหว่าง query ได้

### ความเป็นธรรมของการทดลอง

ทั้งสองระบบใช้ document/query embedding ชุดเดียวกันจาก `BAAI/bge-m3` และ normalize เป็น `float32` เหมือนกัน เวลา encode โมเดลถูกวัดแยกจาก retrieval latency การวัด latency จึงสะท้อน overhead ของ vector engine มากกว่าเวลา inference ของโมเดล

## 4. การเตรียมสภาพแวดล้อม

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

บน Windows ใช้ `.venv\Scripts\python.exe` แทน `.venv/bin/python`

## 5. วิธีรันการทดลอง

รัน benchmark ตามค่ามาตรฐานของใบงาน: corpus 10,000 รายการ, Top-K=5, query 15 รายการ และวัดซ้ำ query ละ 10 รอบ

```bash
.venv/bin/python faiss_chromadb_benchmark.py --no-progress
```

ถ้าต้องการทดสอบเร็วระหว่างพัฒนา:

```bash
.venv/bin/python faiss_chromadb_benchmark.py \
  --limit 1000 \
  --query-limit 5 \
  --benchmark-repeats 3 \
  --no-progress
```

โปรแกรมจะสร้างไฟล์ดังนี้

| ไฟล์ | รายละเอียด |
|---|---|
| `results/build_metrics.csv` | build time, dimension, vector memory estimate และ Chroma storage |
| `results/latency_samples.csv` | latency ราย query และรายรอบ |
| `results/query_results_topk.csv` | ผล Top-K ของ FAISS และ ChromaDB |
| `results/metrics.json` | สรุป metrics สำหรับกรอกใบงาน |

## 6. ขั้นตอนการทดลองที่โปรแกรมทำ

1. โหลด JSON และเลือก document ที่มี `input` ครบ 10,000 รายการ
2. โหลด `instruction` จาก CSV เป็น query จำนวน 15 รายการ
3. โหลด `BAAI/bge-m3` เพียงครั้งเดียว และ encode document/query ด้วย `normalize_embeddings=True`
4. สร้าง `faiss.IndexFlatIP` และจับเวลาเฉพาะการสร้าง index กับการ add vector
5. สร้าง persistent ChromaDB collection แบบ cosine และ insert เป็น batch โดยใช้ vector ชุดเดียวกับ FAISS
6. ค้นหา Top-5 จากทั้งสองระบบ
7. warm up ก่อน แล้ววัด retrieval latency โดยไม่รวม query embedding
8. สาธิต metadata filter ด้วย `source`: FAISS กรองใน application หลัง full ranking ส่วน ChromaDB ใช้ `where` ใน query

## 7. ตารางบันทึกผล

คัดลอกจาก `results/build_metrics.csv` และ `results/metrics.json` หลังรันจริง

| Metric | FAISS (`IndexFlatIP`) | ChromaDB (HNSW cosine) |
|---|---:|---:|
| จำนวน document | ดู `documents` | ดู `documents` |
| Dimension | ดู `dimension` | ดู `dimension` |
| Index build time (วินาที) | ดู `build_time_seconds` | ดู `build_time_seconds` |
| Vector memory estimate (MiB) | ดู `vector_memory_mb` | ดู `vector_memory_mb` |
| Persisted storage (MiB) | 0 (index อยู่ใน memory) | ดู `persisted_storage_mb` |
| Average query latency (ms/query) | ดู `metrics.json` → `latency` → `FAISS` → `mean_ms` | ดู `metrics.json` → `latency` → `ChromaDB` → `mean_ms` |
| Min / Max latency (ms) | `min_ms` / `max_ms` | `min_ms` / `max_ms` |
| P95 latency (ms) | `p95_ms` | `p95_ms` |
| Throughput (QPS) | `qps` | `qps` |

> ค่า score ใน `query_results_topk.csv` ไม่ควรนำมาเทียบตรง ๆ: FAISS รายงาน similarity (มากดีกว่า) แต่ ChromaDB รายงาน cosine distance (น้อยดีกว่า)

## 8. คำถามวิเคราะห์และแนวคำตอบ

### 8.1 ทำไม query latency จึงต่างกัน

`IndexFlatIP` ของ FAISS อยู่ใน memory และเรียก native C++ เพื่อคำนวณกับ array ที่ต่อเนื่องเป็นหลัก จึงมีชั้น abstraction และ I/O น้อย โดยแลกกับการ scan vector ทุกตัวแบบ exact ส่วน ChromaDB ต้องผ่าน database API, จัดการ collection/metadata และอ่านโครงสร้าง HNSW ที่ persistent; จึงมี overhead ต่อ query มากกว่า แต่ได้ความสามารถด้าน document lifecycle, persistence และ filter ในตัว ผลจริงขึ้นกับ version, CPU, จำนวน threads, storage และ warm-up จึงควรใช้ตัวเลขจาก `metrics.json` ของเครื่องที่รันจริง

### 8.2 IndexFlatIP ใช้ memory เท่าไร และควรปรับอย่างไรเมื่อข้อมูลโต

ถ้าใช้ BGE-M3 ที่ dimension 1,024 และ `float32`, vector memory ขั้นต่ำของ 10,000 รายการคือ

```text
10,000 × 1,024 × 4 bytes = 40,960,000 bytes ≈ 39.06 MiB
```

ยังไม่รวม metadata, allocator และ process overhead `IndexFlatIP` เป็น exact search จึงใช้ memory เพิ่มตาม `N × dimension` และเวลาค้นหาเพิ่มตามจำนวน vector หาก corpus โตมากควรพิจารณา `IndexHNSWFlat` เมื่อต้องการ recall สูงและ latency ต่ำ, `IndexIVFFlat` เมื่อต้องการลดจำนวน candidate ที่ scan โดยต้อง train quantizer ก่อน หรือ `IndexIVFPQ` เมื่อต้องการลด memory มากขึ้นโดยยอมแลก recall

### 8.3 ระบบใดเหมาะกับ Hybrid RAG

- เลือก FAISS เมื่อระบบมี retrieval service ที่ควบคุม lifecycle เอง, corpus อยู่ใน memory, ต้องการ latency ต่ำ และ metadata filter ทำใน application ได้
- เลือก ChromaDB เมื่อทีมต้องการเก็บ document/metadata/persistence ในระบบเดียว และใช้ filter เช่น source, tenant หรือ category ระหว่าง retrieval
- ใน production อาจใช้ FAISS เป็น fast path สำหรับ semantic retrieval และใช้ metadata/keyword layer แยก หรือใช้ ChromaDB เป็นระบบที่พัฒนาเร็วกว่า ทั้งนี้ควรวัด end-to-end latency และ recall บน workload จริงก่อนตัดสินใจ

## 9. สรุปผลการทดลอง

หลังรันให้เขียนสรุปโดยอ้างอิงค่า mean, p95 และ QPS จาก `metrics.json` พร้อมระบุว่า latency ที่วัดเป็น retrieval-only ไม่รวมเวลาสร้าง embedding และอธิบาย trade-off ระหว่างความเร็ว, ความสามารถด้าน metadata/persistence และความแม่นยำของ index ที่เลือก
