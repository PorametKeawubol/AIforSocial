# รายงานส่งงาน: Basic RAG with Sentence-Transformers + FAISS + Ollama

## สิ่งที่ดำเนินการ

1. แยกขั้นตอนสร้าง index (`build_faiss_index.py`) ออกจากขั้นตอน RAG Q&A (`rag_qa.py`)
2. สร้าง FAISS `IndexFlatIP` ด้วย cosine similarity จาก normalized embeddings ของ `thai_qa_utf8.json` จำนวน 10,000 passages
3. ทดสอบคำถาม paraphrase จำนวน 15 ข้อจาก `thai_qa_paraphrase_15.csv`
4. เปรียบเทียบ Ollama 3 โมเดล โดยรันทีละโมเดล: `qwen2.5:1.5b`, `llama3.2:3b`, `qwen2.5:7b`

## การตั้งค่า index

| รายการ | ค่า |
| --- | --- |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| จำนวน passages | 10,000 |
| Vector dimension | 384 |
| FAISS index | `IndexFlatIP` |
| Similarity | Cosine similarity (L2-normalized embeddings) |
| อุปกรณ์ | CUDA / AMD Radeon RX 7600S |

รายละเอียดที่สร้างจริงอยู่ใน `artifacts/thai_qa_10000/manifest.json`.

## ผล Retrieval

| รายการ | ผล |
| --- | ---: |
| จำนวนคำถาม | 15 |
| Gold passage ที่อยู่ใน index | 13 |
| Gold passage ที่อยู่นอก 10,000 records | 2 |
| Retrieval Hit@3 | 1 |
| Retrieval Recall@3 | 0.0769 |

ไฟล์รายละเอียดรายข้อ: `results/thai_qa_15/retrieval_only.csv` และ `retrieval_only.json`.

## ผล RAG + Ollama

| โมเดล | Answer contains expected answer | Rate | เวลาเฉลี่ย/ข้อ |
| --- | ---: | ---: | ---: |
| `qwen2.5:1.5b` | 3/15 | 20.0% | 2.6556 วินาที |
| `llama3.2:3b` | 1/15 | 6.67% | 3.6027 วินาที |
| `qwen2.5:7b` | 2/15 | 13.33% | 5.7363 วินาที |

ผลรายข้อและคำตอบของแต่ละโมเดลอยู่ใน `results/thai_qa_15/` และมีสรุปใน `model_comparison.json`.

## วิธีสร้างผลซ้ำ

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# สร้าง index จาก thai_qa_utf8.json 10,000 records
python build_faiss_index.py

# Terminal อีกหน้าต่าง: เปิด Ollama
./run_ollama_rocm.sh

# Terminal หลัก: ทดสอบ retrieval และสามโมเดล
python rag_qa.py --skip-generation
python rag_qa.py
```

## หมายเหตุเรื่องไฟล์ส่ง

ไฟล์ ZIP นี้ไม่รวม `thai_qa_utf8.json`, `index.faiss` และ `documents.jsonl` เพราะเกินขนาดจำกัดการส่ง 20 MB แต่แนบ source code, manifest และผลลัพธ์ที่รันจริงครบถ้วนแล้ว สามารถสร้าง index ซ้ำได้จากคำสั่งข้างต้นเมื่อมีชุดข้อมูลต้นฉบับ.
