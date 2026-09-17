# Project 2 — SmartDoc Assistant report

## Objective

สร้างผู้ช่วยตอบคำถามภาษาไทยจาก `คู่มือประชาชน สำนักงานตำรวจแห่งชาติ.pdf`
โดยให้คำตอบตรวจสอบย้อนกลับได้ถึงหัวข้อและเลขหน้า และลด hallucination ด้วย
retrieval guardrail และ strict fallback

## Design decisions

### Ingestion and chunking

`ingest.py` เก็บเลขหน้า PDF และเลขหน้าที่พิมพ์ในหนังสือไว้กับทุก chunk ใช้
PyMuPDF แบบ layout-aware จาก text blocks และตรวจตารางด้วย `find_tables()` เมื่อ
พบ พร้อม `pdftotext -layout` เป็น fallback ทำ Unicode normalization, ลบ zero-width
space/joiner, BOM และ soft hyphen, แก้ artifact ที่พบใน PDF, ลบเฉพาะบรรทัด
header/footer ที่ซ้ำใน margin และไม่ลบ body text แบบ blanket rule

การแบ่งข้อความเป็น parent-document chunking สองชั้น: รวม paragraph/bullet ที่อยู่
ใน section เดียวกันเป็น parent passage 2,400 ตัวอักษร แล้วใช้ Thai-friendly
recursive splitting ที่พยายามตัด paragraph, บรรทัด, เครื่องหมายจบประโยค,
เว้นวรรค และ character boundary ตามลำดับเป็น child chunk ขนาด 900 ตัวอักษร
และ overlap 120 ตัวอักษร แต่ละ child เก็บ `parent_text`, `parent_id`,
`section_id`, `section_title`, `pdf_pages` และ `printed_pages` ทำให้ retrieval
แม่นระดับ child และ LLM ได้บริบทต่อเนื่องระดับ parent โดยไม่ส่ง chunk ซ้ำ

### Retrieval and fusion

Dense channel ใช้ normalized `BAAI/bge-m3` บน FAISS `IndexFlatIP` ซึ่งเทียบเท่า
cosine similarity เมื่อ vector ถูก normalize แล้ว Sparse channel ใช้ BM25 กับ
Thai `newmm` tokens ผลลัพธ์ over-fetch แล้วรวมด้วย weighted Reciprocal Rank
Fusion (`dense=0.55`, `sparse=0.45`) พร้อมเก็บ component score ไว้ตรวจสอบ
เปิด Cross-Encoder ได้ด้วย `RERANK_MODEL`

### Answer guardrail

Prompt ระบุให้ใช้ context เท่านั้น, ไม่สร้าง citation, ระบุข้อจำกัดของเอกสารปี
2558 และตอบ JSON มี `answer`, `citations`, `confidence` หลังรับคำตอบระบบจะ
ตรวจ citation ให้ตรงกับ source allow-list; ถ้าไม่มี citation ที่ถูกต้องจะไม่
ปล่อยคำตอบเชิงข้อเท็จจริงออกไป

### LINE latency

Qwen3.8-27B Q4 ใช้ประมาณ 3.18–3.39 tokens/s และ 44–69 วินาทีต่อ probe บน
Ryzen 7 7735HS/RX 7600S 8 GiB ดังนั้น webhook ตอบ acknowledgement ก่อน แล้ว
ทำงานใน background และส่งคำตอบผ่าน Push Message เพื่อไม่ให้ request หมดเวลา

## Feature checklist

- Thai semantic + exact retrieval
- Dynamic top-k and context budget
- Source title/topic/link/page citation
- Invalid citation and low-signal fallback
- Query/retrieval/LLM latency telemetry
- JSON retrieval audit record
- LINE signature verification and event deduplication
- Local-only model and embedding path

## Reproducible commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 -m unittest discover -s tests -p 'test*.py' -v
python3 build_index.py
python3 evaluate.py
OLLAMA_MODEL=qwen3.5:9b OLLAMA_TIMEOUT_SECONDS=180 MAX_ANSWER_TOKENS=240 \
  python3 evaluate.py --run-llm --bert-score
```

## Evaluation interpretation

`evaluate.py` writes `artifacts/evaluation.json`, `.csv` and a Markdown metric
table. With `--run-llm --bert-score`, it generates real answers through Ollama
and reports mean SBERT cosine to the reference answer, answer relevance,
semantic evidence faithfulness, BERTScore Precision/Recall/F1 and a clearly
labelled lexical faithfulness proxy. `--llm-judge` additionally asks the local
LLM to score faithfulness and relevance. Since this is one source PDF and a
small authored test set, results are a project validation signal rather than a
general benchmark; none of these metrics proves current legal correctness.

### Full evaluation result

The rebuilt index contains 130 pages, 3 detected tables, 135 child chunks and
80 parent passages. Retrieval on all 18 authored Thai questions achieved:

| Metric | Result |
|---|---:|
| Hit@5 | 1.000 |
| MRR | 1.000 |
| Mean target rank | 1.000 |
| SBERT cosine to reference | 0.833 |
| Answer relevance cosine | 0.756 |
| Semantic faithfulness proxy | 1.000 |
| Lexical faithfulness proxy | 0.812 |
| BERTScore Precision / Recall / F1 | 0.766 / 0.850 / 0.805 |
| LLM-judge faithfulness / relevance | 0.981 / 1.000 |
| Grounded answers / total | 18 / 18 |
| Citation or LLM error | 0 / 18 |

The full generated-answer benchmark used local Ollama `qwen3.5:9b` so all 18
questions could finish on this CPU workstation. The production `.env` remains
configured for Qwen3.8-27B; a direct Qwen3.8-27B probe produced a grounded
answer with a valid citation, but its approximately 200-second single-question
latency makes a full 18-question benchmark impractical on this hardware. The
JSON/CSV rows record the generation model and the LLM-judge model used for each
answer. The judge is an evidence-support signal, not proof of current legal
correctness.
