# ผลการทำ Lab: PDF RAG ด้วย FAISS, Sentence-BERT และ LLaMA

วันที่ทดสอบ: 3 กันยายน 2026  
เอกสาร: `solarcell-basic-knowledge-SolarHub.pdf`  
Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`  
LLM: `llama3.2:1b` ผ่าน Ollama

## Lab 1 — FAISS Indexing

| รายการวัด | ผลลัพธ์ |
| --- | ---: |
| จำนวนหน้า PDF | 22 |
| จำนวน lines ที่ดึงได้ | 309 |
| จำนวน chunks | 103 |
| ขนาด chunk / overlap | 5 / 2 lines |
| Embedding dimension | 384 |
| เวลาสร้าง index | 21.48 วินาที |
| ขนาด FAISS index (`index.faiss` + `index.pkl`) | 263,576 bytes (~257.4 KiB) |

ไฟล์ที่สร้างอยู่ใน `faiss_index/` ได้แก่ `index.faiss`, `index.pkl` และ `index_metrics.json`.

## Lab 2 — Top-K Retrieval + RAG

คำถามที่ใช้เหมือนกันทุกครั้ง: **“ระบบออฟกริดคืออะไร?”**

| K | คำตอบที่ LLaMA ส่งกลับ | ข้อสังเกต |
| ---: | --- | --- |
| 1 | “ระบบออฟกริด … คือ ระบบที่ไม่ต้องการเชื่อมต่อกับระบบไฟฟ้าหลัก” | retrieval อันดับแรกเป็น chunk เรื่อง Hybrid (หน้า 8) แต่โมเดลตอบเป็นนิยาม Off Grid จึงไม่สอดคล้องกับ context เพียงพอ |
| 3 | “ระบบออฟกริด … ผลิตไฟฟ้าจากโซล่าเซลล์ แล้วไม่ได้เชื่อมต่อเข้ากับระบบจำหน่ายไฟฟ้า…” | มี chunk หน้า 2 ที่นิยาม Off Grid โดยตรง จึงได้คำตอบถูกต้องและกระชับ |
| 5 | Off Grid เป็นระบบไม่เชื่อมต่อ Grid แต่เพิ่มข้อความว่า “ผลิตไฟฟ้าได้ตลอดเวลา ไม่ว่าจะมีแสงอาทิตย์หรือไม่” | context มากขึ้น แต่ประโยคท้ายเป็นการสรุปเกินข้อมูล เพราะ Off Grid แบบไม่ใช้แบตเตอรี่ใช้ได้เฉพาะเมื่อมีแสงอาทิตย์ |
| 10 | Off Grid หรือ Stand Alone เป็นระบบโซลาร์เซลล์ที่ไม่เชื่อมต่อระบบจำหน่ายของการไฟฟ้านครหลวง/ส่วนภูมิภาค | ได้ความหมายถูกต้อง แต่ context มีหน้า 15 เพิ่มเข้ามาโดยไม่จำเป็น |

**สรุป:** ในการทดสอบนี้ `K=3` เป็นจุดสมดุลที่ดีที่สุด: มีหลักฐานนิยามโดยตรงโดยไม่เพิ่ม context ที่ไม่เกี่ยวข้องมากเกินไป. `K=1` เสี่ยงพลาดเพราะผลลัพธ์อันดับแรกไม่ตรงเจตนาคำถาม ส่วน `K=5` และ `K=10` เพิ่มข้อมูลมากขึ้นไม่ได้ทำให้คำตอบน่าเชื่อถือขึ้นเสมอไป และยังเสี่ยงให้ LLM สรุปเกิน context. ดังนั้น prompt ที่บังคับให้ยึด context ลดความเสี่ยงได้ แต่ไม่รับประกันว่า LLM จะไม่ hallucinate; งานจริงควรเพิ่ม reranker หรือการตรวจ grounding.

## Lab 3 — LINE RAG Bot

`webhook.py` ใช้ LINE Messaging API SDK v3 และเรียก `llm_response()` โดยตรง จึงโหลด FAISS ที่สร้างไว้แล้ว ไม่สร้าง index ใหม่ต่อข้อความ. ทดสอบ endpoint ภายในด้วย Flask test client แล้ว `GET /health` ตอบ `200 {"status": "ok"}`.

การทดสอบตอบข้อความจริงใน LINE ต้องใส่ `CHANNEL_ACCESS_TOKEN` และ `CHANNEL_SECRET` ใน `.env` ของบัญชี LINE Messaging API แล้วเปิด HTTPS tunnel ให้ URL `/callback` เข้าถึงได้. ไม่มีการใส่หรือสร้าง credentials แทนผู้ใช้ในการทดสอบนี้.
