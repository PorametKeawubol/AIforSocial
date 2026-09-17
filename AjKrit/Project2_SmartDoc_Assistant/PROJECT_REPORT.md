# SmartDoc Assistant — ผู้ช่วยค้นคู่มือประชาชนผ่าน LINE

สถานะ ณ 18 กันยายน 2026 อ้างอิงโค้ด artifacts และผลทดลองในบทสนทนา
เป็นรายงานโครงการ ไม่ใช่การรับรองคะแนนเต็มหรือความถูกต้องของกฎหมายปัจจุบัน

## 1. ปัญหา วัตถุประสงค์ และขอบเขต

คู่มือประชาชน สำนักงานตำรวจแห่งชาติมี 130 หน้า ผู้ใช้มักไม่รู้ว่าจะเปิดหัวข้อใด
และถามด้วยภาษาพูด เช่น “เพื่อนหายทำไง” ระบบจึงค้นหลักฐานจาก PDF ก่อนให้ LLM
สรุปภาษาไทยพร้อมหัวข้อและเลขหน้า ผ่าน LINE ที่ผู้ใช้เข้าถึงง่าย
เป้าหมายคือค้นได้ทั้งคำตรงและความหมาย ตรวจสอบคำตอบย้อนหลังได้ ถามต่อได้
และสมดุลคุณภาพกับความเร็วบนเครื่อง local

ใช้ PDF เดียว คู่มือฉบับ พ.ศ. 2558 ครอบคลุมการติดต่อสถานีตำรวจ ขั้นตอนและเอกสาร
การขออนุญาต ความปลอดภัย ภัยพิบัติ รถและจราจร และกฎหมายทั่วไป
เป็นผู้ช่วยค้นเอกสาร ไม่ใช่บริการรับแจ้งเหตุฉุกเฉินหรือคำแนะนำกฎหมายปัจจุบันที่รับรองแล้ว
ลิงก์ Facebook เก่าถูกนำออก ใช้ชื่อหัวข้อ หน้า PDF และหน้าหนังสือแทน

## 2. ข้อมูลและค่าปัจจุบัน

| รายการ | ค่า |
|---|---|
| หน้า PDF / หน้าที่มีข้อความ | 130 / 130 |
| อักขระที่สกัด / หลัง cleanup | 88,509 / 85,627 |
| ตารางที่ตรวจพบ | 3 ตารางใน 3 หน้า |
| Child chunks / Parent passages | 135 / 80 |
| Child size / Overlap / Parent size | 900 / 120 / 2,400 ตัวอักษร |
| Embedding | BAAI/bge-m3, 1,024 dimensions |
| Dense / Sparse | FAISS IndexFlatIP / Thai BM25 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Rerank pool | เริ่มต้น 6; อย่างน้อยเท่า requested top-k |
| LLM production | Ollama qwen3.5:9b |
| LLM options | temperature=0, seed=42, think=false, context=8192 tokens |
| Output / Context budget | 420 tokens / 10,000 ตัวอักษร |
| Keep alive | 15 นาที |
| เวลา build ที่บันทึกไว้ | 49.13 วินาที |

135 chunks คือหน่วยข้อความสำหรับค้นหา ไม่ใช่จำนวนเอกสารหรือจำนวนคำตอบ
จำนวนขึ้นกับขนาดและโครงสร้างเอกสาร ไม่ใช่ยิ่งมากยิ่งดี หากเล็กเกินไปอาจขาดบริบท
หน้า PDF อาจต่างจากหน้าหนังสือเพราะมีปกและคำนำ จึงเก็บเลขหน้าทั้งสองแบบ

## 3. Architecture และ workflow

Offline:

PDF → สกัด layout/table → cleanup → parent/child split → embedding → FAISS + BM25

Online:

LINE → signature/dedup → navigation หรือ RAG → resolve history/query expansion
→ Dense + Sparse → weighted RRF → rerank → parent context → Qwen 9B
→ citation validation/format → Push + Quick Reply → SQLite history

Cloudflare Tunnel เปิด HTTPS ให้ LINE เข้าถึง local webhook port 8000
รับ event ที่ /callback และตรวจสถานะที่ /health
Quick Tunnel เปลี่ยน URL เมื่อเปิดใหม่ ต้องอัปเดต Webhook URL ใน LINE Developers
บริการ webhook ใช้ transient systemd user-service smartdoc-webhook
ไม่ auto-start หลัง reboot; วิธีเปิดใหม่อยู่ใน command.txt

## 4. PDF ingestion และ advanced chunking

ingest.py ใช้ PyMuPDF text blocks แบบ layout-aware และ find_tables()
มี pdftotext -layout เป็น fallback ตามเงื่อนไขในโค้ด
ทำ Unicode normalization ลบ zero-width characters, BOM, soft hyphen
แก้ artifacts เฉพาะรูปแบบที่ตรวจพบ และลบ header/footer ที่ซ้ำใน margin
ไม่ลบ body text แบบ blanket rule

พบ layout noise ถูกลบ 2,882 อักขระ ประมาณ 3.26% ของข้อความที่สกัด
ตัวเลขนี้คือสัดส่วนที่ลบ ไม่ใช่หลักฐานว่าขยะที่เหลือต่ำกว่า 5%
ยังต้องตรวจด้วยคน โดยเฉพาะหัวข้อหลายบรรทัดและตาราง

ใช้ section-aware parent-document chunking: รวม paragraph/bullet ใน section
เป็น parent passage จากนั้น recursive splitter แบ่ง child ตาม paragraph,
newline, จุดจบประโยค, space และ character boundary ตามลำดับ
ค้นระดับ child แล้วส่ง parent_text ให้ LLM รักษาข้อความต่อเนื่อง
deduplicate parent ก่อนสร้าง prompt
metadata มี parent_id, section_id/title, pdf_pages, printed_pages และชื่อเอกสาร
เป็น parent-document + recursive chunking ไม่ใช่ semantic breakpoint chunking
ยังไม่พิสูจน์ OCR สำหรับ PDF สแกนทุกประเภท

## 5. Hybrid retrieval และ dynamic optimization

Dense ใช้ normalized BGE-M3 vectors กับ FAISS IndexFlatIP ทำให้ dot product
เทียบเท่า cosine บนเวกเตอร์เหล่านี้ ข้อมูลเล็กจึงใช้ exact search ได้
Sparse ใช้ BM25 และ Thai newmm tokenizer ช่วยคำตรง/ศัพท์เฉพาะ/หมายเลข
รวมอันดับด้วย weighted RRF: dense=0.55, sparse=0.45
มี weighted score อีกส่วนเป็นเกณฑ์ประกอบ ไม่ใช่น้ำหนัก RRF เดียวกัน

candidate_k เริ่มต้น 24 และ over-fetch อย่างน้อย requested_k×4
Dynamic top-k ราว 3–8 ตามความยาวและจำนวน clause
Reranker ประเมิน child text ของ scored pool 6 รายการ หรืออย่างน้อยเท่า requested_k
หลังสำเร็จจัดอันดับเฉพาะ scored pool ไม่ปน Cross-Encoder scores กับ unscored RRF
ถ้า reranker ล้มเหลว/คะแนนผิดรูปแบบหรือไม่ finite จะรักษา RRF เป็น fallback
Context budget เป็น character budget ไม่ใช่ token counting เต็มรูปแบบ
จึงยังต้องตรวจ prompt token telemetry โดยเฉพาะภาษาไทย

## 6. Prompt และ guardrails

System prompt จำกัดให้ใช้ context มีตัวอย่าง JSON และแจ้งว่าเอกสารปี 2558
คำตอบมี answer, citations, confidence ตรวจ citation กับ source allow-list
S1/S2 คือรหัสหลักฐานของคำตอบนั้น ไม่ใช่คะแนนความถูกต้อง
ตัด citation ซ้ำในเนื้อหาและแสดงที่มาเป็นบล็อกท้ายคำตอบ
หลักฐานสัญญาณต่ำหรือคำตอบไม่ผ่านเงื่อนไขใช้ strict fallback
หากต่อ Ollama ไม่ได้ แสดงหลักฐานที่ค้นพบพร้อมแจ้งข้อขัดข้อง
history เป็นบริบท ไม่ใช่หลักฐานอ้างอิง

Guardrails ลดความเสี่ยงแต่ไม่รับรอง zero hallucination
citation ที่มีอยู่จริงไม่พิสูจน์ว่าทุกข้ออ้างถูกต้องหรือใช้กับกรณีผู้ใช้ได้
กฎหมาย/ขั้นตอนปัจจุบันต้องตรวจหน่วยงานทางการ ไม่สรุปจากคู่มือเก่าอย่างเดียว

## 7. LINE UX และ chat history

Rich Menu 3×2 มี ถาม Assistant, ขั้นตอนและเอกสาร, สิทธิและกฎหมาย,
ความปลอดภัย, รถและจราจร, หัวข้อทั้งหมด
Flex Carousel แสดงหัวข้อย่อย Quick Reply ส่งคำถามที่ระบุหัวข้อครบเข้า RAG
คำทักทายและเมนูไม่เรียก retrieval/LLM

SQLite history เก็บสูงสุด 4 รอบ อายุ 24 ชั่วโมง แยก user และ scope ของห้อง/กลุ่ม
บันทึกเฉพาะคำตอบ grounded ที่ส่งสำเร็จ resolve_followup ช่วยคำถาม “ไปที่ไหน”
หรือ “ต้องเตรียมอะไร” ล้างได้ด้วย “ล้างประวัติ”/“เริ่มใหม่”
ข้อมูลหมดอายุลบเมื่อเข้าถึง ไม่ใช่ timer; ไม่มี Neo4j/Knowledge Graph ในระบบนี้

RAG ในแชตส่วนตัวใช้ loading 60 วินาที แทนข้อความรับคำถามทุกครั้ง
รอเกิน 30 วินาทีส่ง notice ครั้งเดียว และ refresh loading ระหว่างรอ
loading รองรับเฉพาะผู้ใช้ที่เปิดหน้าจอแชตส่วนตัวอยู่ กลุ่ม/ห้องและ API ล้มเหลว
ใช้ข้อความ acknowledgement สำรอง งานตอบผ่าน background task และ Push
คิว inference เดียวรักษาลำดับแต่ทำให้หลายผู้ใช้ต้องรอ และไม่ durable ข้าม restart

## 8. Evaluation: แยกรอบและความหมายให้ชัด

### Full evaluation เดิมก่อนลด rerank

ชุด authored Q&A 18 ข้อ มี reference และหนึ่ง target_section ต่อข้อ
artifacts/evaluation_full บันทึก SBERT reference=0.833, relevance=0.756,
semantic faithfulness proxy=1.000, BERTScore P/R/F1=0.766/0.850/0.805,
LLM-judge faithfulness/relevance=0.981/1.000 และ grounded/citation 18/18
เป็นผลย้อนหลัง ไม่ใช่ full evaluation ล่าสุดหลัง rerank 6

### Embedding benchmark เดิม: ไม่มี reranker

| Dense model | Hit@1 | MRR@5 | Recall@5 | Query p95 |
|---|---:|---:|---:|---:|
| BGE-M3 | 17/18 | 0.9722 | 1.000 | 0.126 วินาที |
| E5-large | 18/18 | 1.0000 | 1.000 | 0.128 วินาที |
| Qwen3-Embedding-0.6B | 16/18 | 0.9444 | 1.000 | 0.306 วินาที |

CPU FP32, 512-token cap, 3 repeats/query, chunks เดียวกัน
E5 ใช้ query:/passage: ส่วน Qwen embedding ใช้ query instruction
Hybrid RRF ทั้งสามได้ Hit@1 17/18 ผลใกล้กัน
เมื่อมี target เดียว Recall@k=Hit@k; คืน 5 หัวข้อแล้วเจอ target จะได้ precision=.2
และ F1≈.333 ไม่ใช่ BERTScore F1 และไม่ได้แปลว่าคำตอบผิด 67%
Raw cosine ข้าม embedding เทียบอันดับคุณภาพโดยตรงไม่ได้

### LLM benchmark เดิม: frozen contexts เดียวกัน

| Model | BERTScore F1 | Reference cosine | LLM-only p95 |
|---|---:|---:|---:|
| Qwen 9B | 0.7950 | 0.8312 | 8.92 วินาที |
| Qwen 27B GGUF Q4 | 0.7916 | 0.8301 | 95.70 วินาที |
| Ministral 3 8B | 0.7994 | 0.8141 | 18.87 วินาที |

18 คำถาม/โมเดล ควบคุม prompt, temp=0, think=false, context8192 และ budget420
ใช้ scorer เดียว ไม่มี API errors แต่ Ministral ปนขั้นตอนในบางข้อและ 2 คำตอบชน token limit
Gemma ถูกย้ายไป archived results เพราะพบผสมขั้นตอนและกลับความหมาย
citation contract และ semantic proxy อาจผ่านทั้งที่เนื้อหาผิด จึงตรวจด้วยคนร่วมด้วย
เลือก Qwen 9B เพราะเหมาะกับเวลารอ LINE ไม่ใช่เพราะชนะทุก metric

### E5 ทั้งระบบ: ก่อนลด rerank ทดลองในหน่วยความจำ

| Metric | BGE-M3 | E5-large |
|---|---:|---:|
| Hybrid + reranker Hit@1 / Recall | 18/18 | 18/18 |
| BERTScore F1 | 0.8047 | 0.8061 |
| Reference cosine | 0.8327 | 0.8322 |
| End-to-end p95 ไม่รวม LINE delivery | 15.46 วินาที | 14.90 วินาที |

ไม่เขียน artifacts ตามคำขอเดิม ตัวเลขมาจาก console/session
F1 เพิ่มเพียง .0014 และยังตกหล่นบางข้อ จึงไม่ย้าย production ไป E5
ไม่อ้างนัยสำคัญหรือความเร็วที่แน่นอนจากรอบเดียว

### ลด rerank pool: ทดลองล่าสุด

| Candidates | Mean rerank | p95 rerank | Hit@1 / Recall |
|---|---:|---:|---:|
| 10 | 5.45 วินาที | 6.04 วินาที | 18/18 |
| 6 | 2.88 วินาที | 3.49 วินาที | 18/18 |
| 5 | 2.32 วินาที | 2.67 วินาที | 18/18 |

CPU 4 threads สลับลำดับ pool sizes จัดอันดับเฉพาะ scored pool ทุก variant
จึงควบคุม semantics การจัดอันดับ ไม่ใช่เทียบโค้ดเก่าที่ปน RRF โดยตรง
เลือก 6 เผื่อหลักฐานรอง ลดเวลา reranker เฉลี่ยราว 47% ไม่ใช่เวลารวม
ตรวจคำตอบจริง 3 ข้อ: สิทธิผู้ถูกจับ ขโมยขึ้นบ้าน ทวงหนี้ ไม่พบถอยลงในข้อที่ตรวจ
บางข้อกล่าวถึงข้อมูลที่แบบ 10 ตกหล่น แต่ยังไม่ทำ full answer evaluation 18 ข้อใหม่
ตัวเลขรอบนี้มาจาก console/session ไม่ใช่ artifacts/model_benchmark เดิม

## 9. Performance, security และ reproducibility

เครื่อง Ryzen 7 7735HS, RAM ราว30GiB, RX7600S VRAM8GiB
PyTorch เป็น CPU-only ส่วน Ollama /api/ps รายงาน Qwen9B อยู่ใน VRAM ทั้งโมเดล ณ การตรวจ
LLM benchmark เดิมเฉลี่ย6.06วินาที: อ่าน prompt2.71, generation2.84, ส่วนอื่น.51
cold start, context/output length, queue, concurrent traffic และ network มีผลต่อเวลา
p95 ของแต่ละช่วงไม่สามารถบวกเป็น p95 รวม และค่าทดสอบไม่ใช่ SLA

Webhook ตรวจ HMAC signature, จำกัด payload, dedup events ใน memory
secrets อ่านจาก .env ไม่ใส่รายงานหรือสไลด์ secrets ที่เคยเผยแพร่ควร rotate
local inference ไม่ได้หมายความว่าข้อมูลไม่ออกจากเครื่องทั้งหมด: ข้อความผ่าน LINE API/tunnel
history มีข้อมูลส่วนตัว ต้องกำหนด retention/backup/privacy policy และไม่เผยแพร่

Modular architecture, config, requirements ที่ pin versions ด้วย == และ tests ล่าสุด32/32ผ่าน
ไม่ใช่หลักฐาน load testing/security audit ครบถ้วน
command.txt รวมการรัน build/CLI/evaluation/benchmark/service/tunnel
frozen benchmark เดิมไม่เปลี่ยนตามโค้ดใหม่ ต้องแยก output เมื่อประเมินใหม่

## 10. หน้าที่ไฟล์

| กลุ่ม | ไฟล์ |
|---|---|
| Runtime | app.py, line_webhook.py, assistant.py, retriever.py, llm.py, models.py, config.py |
| UX/history | line_ux.py, chat_history.py, artifacts/chat_history.sqlite3 |
| Production index | artifacts/index/index.faiss, chunks.json, manifest.json |
| Preparation | ingest.py, build_index.py, setup_richmenu.py, assets/richmenu_thaipol.png |
| Evaluation | evaluate.py, benchmark_models.py, eval/questions.jsonl, tests/ |
| Docs/environment | requirements.txt, .env.example, command.txt, README.md |

setup_richmenu.py ไม่รันประจำ แต่ยังใช้เปลี่ยน/ตรวจเมนู PDF เก็บไว้ rebuild/ตรวจแหล่งข้อมูล
ผล benchmark/evaluation เป็นหลักฐานทดลอง ไม่ใช่ runtime dependency

## 11. Rubric mapping และแผนต่อยอด

| Criterion | น้ำหนัก | หลักฐาน / ช่องว่าง |
|---|---:|---|
| PDF ingestion/chunking | 15% | layout/table, cleanup, recursive+parent; OCR/residual noise ยังไม่พิสูจน์ |
| Hybrid/fusion | 25% | FAISS+BM25+weighted RRF+Cross-Encoder |
| Dynamic/embedding | 15% | multilingual models, dynamic k, character budget; ยังไม่ token counting เต็มรูปแบบ |
| Prompt/guardrail | 20% | JSON example, context-only, citation allow-list, fallback; ไม่รับรอง zero hallucination |
| Quantitative evaluation | 15% | Q&A, SBERT/BERTScore/proxies/judge/ตาราง; latest full eval ต้องรัน |
| Architecture/MLOps | 10% | modular config, pins, tests/commands; durable queue/load/security ยังไม่ครบ |

ครอบคลุมแกนหลัก rubric แต่คะแนนจริงขึ้นกับผู้ประเมิน ไม่ประกาศว่า100/production-gradeทุกเงื่อนไข
ชุด18ข้อเป็น authored in-domain targetเดียว ยังไม่วัด refusal/OOD, history correctness,
ภาษาพูดยาก, multi-topic และ concurrency
ต่อไปเพิ่ม hard cases/multi-relevant labels/human annotation, nDCG/Recall@1/3,
paired uncertainty estimates และ full evaluation rerank6
อัปเดตคู่มือ เพิ่ม emergency routing ที่ตรวจข้อมูลปัจจุบัน
ทดสอบ reranker เล็กลง/context selection/history-aware cache และ durable queue
วัดข้อมูลตกหล่นด้วย ไม่ลด token limit อย่างเดียว

## 12. หลักฐานอ้างอิง

Build: artifacts/build_metrics.json และ artifacts/index/manifest.json
Full evaluation เดิม: artifacts/evaluation_full.json/.csv/.md
Model benchmark เดิม: artifacts/model_benchmark/comparison.json/.md,
*_answers.json และ review_notes.md
Latest code: retriever.py, line_webhook.py, tests/
LINE loading: https://developers.line.biz/en/docs/messaging-api/use-loading-indicator/
E5 end-to-end และ pool-size experiment: console output ในบทสนทนา ไม่ได้บันทึกผลแยกใน artifacts
