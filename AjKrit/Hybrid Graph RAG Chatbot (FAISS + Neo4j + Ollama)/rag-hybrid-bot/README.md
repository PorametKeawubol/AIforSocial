# Hybrid Graph RAG — คู่มือความรู้ในการกำจัดโรคพืช

ระบบอ่าน PDF ภาษาไทย 48 หน้า → FAISS + Thai BM25 + Neo4j → ตอบด้วย Ollama พร้อมอ้างอิงหน้า PDF มีเว็บ Chatbot และ export หลักฐานแต่ละ checkpoint

## เปิดใช้งานบนเครื่องนี้

```bash
cd '/home/porametk/Desktop/AIforSocial/AIforSocial/AjKrit/Hybrid Graph RAG Chatbot (FAISS + Neo4j + Ollama)/rag-hybrid-bot'
# ถ้า Ollama ยังไม่ทำงาน ให้เปิดในอีก terminal:
ollama serve
# ใช้ Docker Desktop context บนเครื่องนี้
systemctl --user start docker-desktop
docker compose -p hybrid-plant-rag up -d
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8090
```

- Chatbot: http://127.0.0.1:8090
- Neo4j Browser: http://localhost:7475 — ใช้บัญชีใน `.env`
- API: http://127.0.0.1:8090/docs
- รายงานและภาพ: `artifacts/checkpoints/01` ถึง `07`
- ผลคำตอบจริง: `artifacts/answers.json`
- กราฟที่อ่านกลับจากฐานข้อมูลจริง: `artifacts/neo4j_export.csv` และ `.json`
- GraphML แบบพกพา: `artifacts/knowledge_graph.graphml`
- รายงานรวม: `artifacts/REPORT.html`
- คำสั่งสำรวจกราฟ: `graph_queries.cypher`

โปรเจกต์ใช้ Neo4j แยก container, volume และพอร์ตจากงานเก่า ใช้ labels `RagPage`, `RagChunk`, `RagEntity`, `RagFact`; ไม่ลบกราฟงานอื่น

## ติดตั้งใหม่

ต้องมี Python 3.12, Docker Compose, Ollama และพื้นที่สำหรับโมเดล

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
cp .env.example .env
# แก้ NEO4J_PASSWORD ก่อนเริ่ม container
ollama pull qwen3.5:9b
docker compose -p hybrid-plant-rag up -d
./run_pipeline.sh
```

PDF เริ่มต้นคือ `../9-file-4786.pdf`; เปลี่ยนได้ด้วย `PDF_PATH` ใน `.env`

เครื่องที่ทำงานจริงใช้ `.venv/lib/python3.12/site-packages/shared-ml.pth` อ้างอิง ML packages จาก virtualenv เดิมในโฟลเดอร์ `Vector Search ระหว่าง FAISS และ ChromaDB` เพื่อประหยัดพื้นที่ ไม่ได้แก้ package ใน environment เดิม การติดตั้งใหม่ตามคำสั่งข้างบนจะไม่ต้องพึ่ง environment นี้ `requirements.lock.txt` เป็น inventory รุ่นที่ใช้จริง รวม dependencies จาก environment ร่วม ไม่ใช่ lockfile แบบข้ามแพลตฟอร์ม

## ประมวลผลและเก็บผล

```bash
.venv/bin/python build_index.py --stage extract
.venv/bin/python build_index.py --stage vectors
.venv/bin/python build_index.py --stage graph
.venv/bin/python build_index.py --stage ingest
.venv/bin/python evaluate.py
.venv/bin/python run_answers.py
.venv/bin/python finalize.py
.venv/bin/python write_report.py
# เปิดเว็บก่อน แล้วบันทึกภาพ/PDF:
.venv/bin/python capture.py
.venv/bin/python export_bundle.py
```

`build_index.py` ไม่ใส่ stage จะทำครบทั้ง 4 ขั้น การสกัดกราฟ cache ตาม hash ของ PDF, ข้อความ, ชื่อโมเดล และรุ่น prompt/schema หากโมเดลเดิมถูกเปลี่ยน weights ภายใต้ tag เดิม ให้ล้างเฉพาะ `artifacts/graph_cache` หรือเปลี่ยน version key ก่อนรันใหม่

Checkpoint:

1. อ่าน PDF, เก็บข้อความดิบ/ข้อความแก้ encoding, แบ่ง chunks พร้อมเลขหน้าและ hash
2. สร้าง BGE-M3 dense embeddings และ FAISS cosine ผ่าน normalized inner product
3. สกัด facts ด้วย JSON schema และตรวจ evidence กับข้อความจริง
4. เขียนกราฟใน transaction เดียวและ export กลับจาก Neo4j
5. เปรียบเทียบ MiniLM dense / BGE-M3 dense / BM25 / Hybrid ด้วย 16 คำถาม
6. ตอบคำถามจริง 5 ข้อ รวมคำถามนอกเอกสาร
7. ผลทดสอบระบบ, รุ่นแพ็กเกจ และสรุปสถานะ

ทุก checkpoint มี `results.json`, `report.html`, `screenshot.png`, `report.pdf` หลังรัน capture สำเร็จ รายงาน HTML เป็นผลที่สร้างจากข้อมูลจริง และภาพแคปเป็นภาพจาก browser ที่เปิดรายงานนั้น ไม่ใช่ภาพจำลองการทำงานของ Neo4j Browser ส่วน `chatbot_desktop.png` และ `chatbot_mobile.png` แคปหน้าเว็บจริง

## การออกแบบและการเปลี่ยนจากใบงาน

- **Qwen3.5 9B** แทน Llama 3.2 1B; ใช้ GPU ผ่าน Ollama และ context 8,192 เพื่อเหมาะกับ VRAM 8 GB เปลี่ยนด้วย `OLLAMA_MODEL` ไม่มีการรับประกันว่าเป็นโมเดลที่ดีที่สุดในทุกงาน
- **BAAI/bge-m3** แทน multilingual MiniLM; 1,024 dimensions รัน CPU เพื่อเว้น VRAM ให้ LLM เปลี่ยน `EMBED_MODEL` แล้ว rebuild vectors
- **PyMuPDF** รักษาลำดับข้อความ PDF; normalize เฉพาะอักขระฟอนต์ที่ตรวจพบ เก็บ raw text ให้ตรวจกลับได้ ไม่มี OCR อัตโนมัติ
- **FAISS IndexFlatIP** exact search เหมาะกับเอกสารขนาดเล็ก ไม่ใช้ ANN ที่เสีย recall โดยไม่จำเป็น และไม่โหลด pickle
- **Thai BM25 + Reciprocal Rank Fusion** รวม dense/lexical/graph เป็นอันดับ รักษา dense top-3 เป็นแกนแล้วเติมผล fusion; ค่าน้ำหนัก graph 0.35 ลดผลกระทบจาก entity กว้าง เช่น “เชื้อรา”
- **Native Neo4j driver** ใช้ parameterized Cypher; ไม่ให้ LLM สร้าง query มารัน และไม่ต้องพึ่ง APOC หรือ LangChain experimental
- **กราฟแบบ fact node** เก็บชื่อ relation, เลขหน้า, evidence และ source hash ต่อ fact; ติดตามความสัมพันธ์ที่ชื่อซ้ำกันข้ามบริบทได้
- **Page-preserving chunking** 1,300 ตัวอักษร overlap 250; เป็นค่าประนีประนอม ไม่ใช่การแบ่งหัวข้อสมบูรณ์
- **FastAPI + เว็บในเครื่อง** สำหรับถาม, ตรวจหลักฐาน, ดาวน์โหลด JSON และดูกราฟ

## ข้อจำกัดที่สำคัญต่อการอ่านผล

ผล Hit@1/Hit@5/MRR@5 วัดว่าค้นเจอหน้าอ้างอิงหรือไม่ ไม่ใช่คะแนนความถูกต้องของคำตอบ ชุด 16 คำถามจัดทำจาก PDF เดียวกัน ใช้ตรวจระบบเบื้องต้น ไม่ใช่ชุดทดสอบอิสระสำหรับกล่าวอ้างคุณภาพทั่วไป มีการปรับวิธี fusion หลังเห็นผลชุดนี้ จึงมีความเสี่ยง overfit และต้องทดสอบกับคำถามใหม่ก่อนสรุปความสามารถทั่วไป

การตรวจ evidence ยืนยันว่าข้อความอ้างมีอยู่จริง แต่ยังไม่ได้พิสูจน์ว่าความสัมพันธ์ตีความถูกทั้งหมด โมเดลอาจสกัดไม่ครบ โดยเฉพาะหน้าที่ข้อมูลต่อเนื่องจากหน้าก่อน ต้องตรวจ `graph_rejected.json` และเทียบกับ PDF ไฟล์ `graph_review.json` ระบุรายการคัดออกและ facts ที่ Codex ทบทวนจากต้นฉบับเพิ่มเติม มี source hash จำกัดผลให้ใช้กับ PDF นี้เท่านั้น `origin` ในผล export แยก `model` กับ `codex_source_review` ชัดเจน การทบทวนนี้ยังเป็น AI review ไม่ใช่ผู้เชี่ยวชาญอิสระ ไม่มีการสมมติความสัมพันธ์จากการวางชื่อใกล้กัน

เลขหน้าอ้างอิงคือหน้า PDF (เริ่ม 1) ส่วน generation ใช้ข้อมูลจากหนังสือปี 2562 ไม่ได้ตรวจปรับเนื้อหาเกษตรให้เป็นข้อมูลปัจจุบัน ระบบไม่ควรใช้ผล benchmark นี้รับรองคำแนะนำทางการเกษตร

หาก Neo4j ใช้งานไม่ได้ search จะแจ้ง warning และใช้ dense + BM25; ข้อผิดพลาดขณะเขียน graph จะไม่ถูกกลืน ไม่มีการรับประกัน latency จากการทดสอบไม่กี่ครั้ง

## LINE / Cloudflare

มี `/callback` และตรวจ HMAC signature ใน `line_adapter.py` หากไม่มี `CHANNEL_ACCESS_TOKEN` / `CHANNEL_SECRET` จะตอบ 503 เว็บ Chatbot ใช้งานแยกได้ตามปกติ ทดสอบรับคำถามและตอบกลับผ่าน LINE จริงแล้ว เมื่อ 11 กันยายน 2569 ผ่าน Cloudflare Quick Tunnel

เมื่อมี credentials และต้องการเปิดเชื่อมต่อจริง ให้ใส่ใน `.env`, restart เว็บ แล้วใช้:

```bash
cloudflared tunnel --url http://127.0.0.1:8090
```

ตั้ง Webhook URL เป็น `https://<tunnel-host>/callback` ใน LINE Developers Webhook ตอบรับทันที แล้วประมวลผลเบื้องหลังทีละข้อความ พร้อมข้าม event ID ซ้ำภายใน 1 ชั่วโมง และบันทึกคำตอบที่ส่งสำเร็จใน `artifacts/line_replies.jsonl` สถานะคิวและการตรวจซ้ำเก็บในหน่วยความจำ จึงหายเมื่อรีสตาร์ต เหมาะสำหรับ lab; การใช้งานจริงควรเพิ่ม durable queue และ persistent deduplication และจัดการ reply token ที่หมดอายุหากคิวรอนาน

## แหล่งอ้างอิงการเลือกเทคโนโลยี

- ใบงาน `../การสร้างระบบ Hybrid Graph RAG Chatbot (FAISS + Neo4j + Ollama) .md`
- https://ollama.com/library/qwen3.5:9b
- https://huggingface.co/BAAI/bge-m3
- https://docs.ollama.com/capabilities/structured-outputs
- https://neo4j.com/docs/python-manual/current/query-simple/

- LINE signature: https://developers.line.biz/en/docs/messaging-api/verify-webhook-signature/
