# PDF RAG: FAISS + Sentence-BERT + LLaMA

โปรเจกต์นี้ทำตาม Lab โดยแยกขั้นตอนสร้างดัชนีออกจากการตอบคำถามและ LINE webhook จึงไม่ต้องอ่าน PDF หรือสร้าง embeddings ทุกครั้งที่ได้รับคำถาม

เอกสารตั้งต้นคือ `solarcell-basic-knowledge-SolarHub.pdf` (22 หน้า) และเปลี่ยน PDF ได้ด้วย `--pdf`.

## ติดตั้ง

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama serve
ollama pull llama3.2:1b
```

เปิด terminal อีกหน้าต่างหนึ่งสำหรับคำสั่งที่เหลือ หาก `ollama serve` กำลังทำงานอยู่

## Lab 1 — สร้าง FAISS index

```bash
python build_faiss.py --pdf solarcell-basic-knowledge-SolarHub.pdf
```

จะสร้าง `faiss_index/index.faiss`, `faiss_index/index.pkl` และ `faiss_index/index_metrics.json` ซึ่งบันทึกจำนวนหน้า, lines, chunks, embedding dimension, เวลา และขนาดดัชนีโดยอัตโนมัติ

## Lab 2 — ทดลอง Top-K และ RAG

```bash
python rag_search.py "ระบบออฟกริดคืออะไร?" --k 1
python rag_search.py "ระบบออฟกริดคืออะไร?" --k 3
python rag_search.py "ระบบออฟกริดคืออะไร?" --k 5
python rag_search.py "ระบบออฟกริดคืออะไร?" --k 10
```

หากต้องการประเมินเฉพาะการค้นคืนโดยยังไม่ได้เปิด Ollama ให้เพิ่ม `--retrieve-only`.

## Lab 3 — เชื่อม LINE Bot

1. คัดลอก `.env.example` เป็น `.env` แล้วใส่ Channel access token และ Channel secret จาก LINE Messaging API
2. สร้าง index ตาม Lab 1 และเปิด Ollama ให้พร้อม
3. เริ่ม webhook ด้วย `python webhook.py`
4. ใช้ HTTPS tunnel (เช่น ngrok) ให้ LINE เรียก `https://<your-domain>/callback` ได้ แล้วตั้งค่า URL นี้ใน LINE Developers Console

`/health` ใช้ตรวจว่า Flask ทำงานอยู่ได้ แต่ไม่ได้ตรวจสอบสถานะ Ollama หรือ FAISS.
