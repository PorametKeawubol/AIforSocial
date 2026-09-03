# Basic RAG: Sentence-Transformers + FAISS + Ollama

เวิร์กโฟลว์นี้แยกงานเป็นสองขั้นชัดเจน เพื่อไม่ต้องสร้าง embedding และ FAISS index ทุกครั้งที่ถามคำถาม:

```text
thai_qa_utf8.json (10,000 passages) ──> build_faiss_index.py ──> artifacts/thai_qa_10000/
                                                                         │
thai_qa_paraphrase_15.csv ───────────────────────> rag_qa.py ──> FAISS retrieval + Ollama Q&A reports
```

`DENSE/thai_qa_utf8.json` และ `DENSE/thai_qa_paraphrase_15.csv` มีอยู่ใน workspace แล้ว จึงใช้เป็นค่าเริ่มต้นอัตโนมัติ โดย indexer จะนำ **10,000 รายการแรก** จาก JSON มาใช้และบันทึกที่ `artifacts/thai_qa_10000/`.

## 1. ติดตั้งและเปิด Ollama

```bash
cd "/home/porametk/Desktop/AIforSocial/AIforSocial/AjKrit/Basic RAG with Sentence-Transformers (PhayaThaiBERT) + FAISS + Ollama"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
./run_ollama_rocm.sh
```

เปิด terminal ใหม่ในโฟลเดอร์เดิมก่อนทำขั้นถัดไป

## 2. สร้าง FAISS index เพียงครั้งเดียว

```bash
python build_faiss_index.py
```

สคริปต์สร้าง cosine-similarity index ด้วย normalized embeddings และบันทึก:

- `index.faiss` — FAISS `IndexFlatIP`
- `documents.jsonl` — passage และ metadata ที่เชื่อมกับ vector แต่ละตัว
- `manifest.json` — รุ่น embedding model, จำนวน records และการตั้งค่า index

ค่า embedding เริ่มต้นคือ `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` ตาม Full Code ในใบงาน หากต้องการใช้ WangchanBERT ให้สร้าง index ใหม่ทั้งชุด:

```bash
python build_faiss_index.py \
  --embedding-model airesearch/wangchanberta-base-att-spm-uncased \
  --replace
```

## 3. ตรวจสอบ retrieval บน 15 prompts โดยไม่เรียก LLM

คำสั่งนี้เหมาะสำหรับตรวจว่า index และชุดทดสอบเชื่อมกันถูกต้อง ผลลัพธ์อยู่ที่ `results/thai_qa_15/retrieval_only.{json,csv}`.

```bash
python rag_qa.py --skip-generation
```

รายงานบอก `retrieval_recall_at_3` โดยคำนวณเฉพาะคำถามที่ gold passage อยู่ใน 10,000 records ที่ index ไว้ และแยกจำนวน gold passage ที่อยู่นอก index ออกมาอย่างโปร่งใส

## 4. ทดสอบ RAG + Ollama สามโมเดล

```bash
python rag_qa.py --pull-missing
```

โมเดลเริ่มต้นที่ทดสอบเรียงทีละตัว (คืน VRAM หลังแต่ละคำตอบด้วย `keep_alive: 0`) คือ:

| โมเดล | จุดประสงค์ |
| --- | --- |
| `qwen2.5:1.5b` | baseline ขนาดเล็ก |
| `llama3.2:3b` | baseline ขนาดกลาง |
| `qwen2.5:7b` | เปรียบเทียบคุณภาพกับโมเดลใหญ่ขึ้น |

ผลที่สร้างได้คือ `<model>.json`, `<model>.csv` และ `model_comparison.json` ใน `results/thai_qa_15/` โดยมีคำตอบ, context ที่ค้นได้, เวลา generation, `Hit@3` ของ retrieval และการตรวจว่า answer มี expected answer หรือไม่

เพื่อไม่ให้ passage ยาวเกิน context window ของ Ollama ตัว runner จะส่ง Context รวมไม่เกิน 8,000 ตัวอักษรและแบ่งพื้นที่ให้แต่ละผลลัพธ์ที่ค้นได้อย่างสมดุล ปรับได้ เช่น `--max-context-chars 6000`.

หากต้องการรันเฉพาะโมเดลที่ติดตั้งแล้ว:

```bash
python rag_qa.py --models qwen2.5:1.5b llama3.2:3b
```

## ตัวเลือกสำคัญ

ใช้ path ชุดข้อมูลของตนเอง:

```bash
python build_faiss_index.py --dataset /path/to/thai_qa_utf8.json --limit 10000
python rag_qa.py --questions /path/to/thai_qa_paraphrase_15.csv
```

`basic_rag_ollama.py` ยังคงอยู่สำหรับทดลองตามใบงานฉบับสั้น (8 เอกสารในหน่วยความจำ) แต่การส่งงานตามข้อกำหนดให้ใช้ `build_faiss_index.py` และ `rag_qa.py`.
