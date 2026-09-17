# History Chat บน LINE — คู่มือกำจัดศัตรูพืช

แล็บนี้ใช้บอต LINE `6610110554` ผ่านส่วนขยาย LINE ใน Brave จริง โดยเชื่อมกับ FAISS, Neo4j และ Ollama ของโปรเจกต์เดิม

## หลักฐานส่งงาน

- `LINE_history_5/LINE_history_5.png`: ภาพประวัติคำถามต่อเนื่อง 5 ข้อในหน้าต่าง LINE จริง
- `LINE_history_5/LINE_history_5.pdf`: ภาพสำหรับส่งงานเป็น PDF
- `LINE_history_5/questions_and_answers.txt`: คำถามและคำตอบที่ส่งสำเร็จจริง
- `LINE_history_5/replies.jsonl`: ผลตอบกลับพร้อมจำนวนข้อความประวัติที่ใช้ ไม่มี user ID หรือ token
- `LINE_history_5/source_page_30.txt`: ข้อความต้นฉบับหน้า 30 สำหรับเทียบคำตอบ
- `LINE_history_5/initial_trial/`: หลักฐานช่วงทดลองก่อนปรับให้ตอบเฉพาะประเด็น ไม่ใช่ชุดส่งงานสุดท้าย

## การทำงาน

`webhook.py` ตรวจลายเซ็น LINE แล้วเก็บประวัติแยกตาม `source.userId` สำหรับแชตส่วนตัว จำกัด 6 ข้อความล่าสุด (3 คู่ถาม–ตอบ) เรียก Retriever โดยส่งประวัติไปด้วย และบันทึกประวัติหลังส่งคำตอบสำเร็จ รองรับคำสั่ง `ล้างประวัติ` และป้องกัน event ซ้ำ

`rag-hybrid-bot/rag_search.py` รองรับ `chat_history` โดยรวมคำถามใหม่กับข้อความสนทนาจริงในการค้นคืน แล้วส่งประวัติและหลักฐานเอกสารเข้าโมเดล คำตอบต้องอ้างหน้าเอกสารและตอบเฉพาะโรคที่ถาม ไม่ใช้โมเดลเขียนคำค้นใหม่เพื่อหลีกเลี่ยงการเติมข้อเท็จจริงระหว่างขั้นค้นคืน

## เริ่มบริการอีกครั้ง

จากโฟลเดอร์นี้ เมื่อ Ollama และ Neo4j ของโปรเจกต์เดิมเปิดแล้ว:

```bash
HF_HUB_OFFLINE=1 '../Hybrid Graph RAG Chatbot (FAISS + Neo4j + Ollama)/rag-hybrid-bot/.venv/bin/python' -m uvicorn webhook:app --host 127.0.0.1 --port 8090
cloudflared tunnel --url http://127.0.0.1:8090
```

ตั้ง Webhook URL ของบอตให้ตรงกับ URL tunnel ตามด้วย `/callback` ใช้ credentials จาก `.env` ของโปรเจกต์เดิม ไม่รวม credentials ในไฟล์ส่งงาน Quick Tunnel เปลี่ยน URL เมื่อเปิดใหม่ ประวัติอยู่ในหน่วยความจำและหายเมื่อรีสตาร์ต

## ตรวจสอบ

```bash
'../Hybrid Graph RAG Chatbot (FAISS + Neo4j + Ollama)/rag-hybrid-bot/.venv/bin/python' -m pytest -q test_history.py
```

ทดสอบการใช้บริบทในการค้นคืนและสร้างคำตอบ การแยกผู้ใช้ จำกัดประวัติ ล้างประวัติ ตรวจลายเซ็น และป้องกัน event ซ้ำ ผลล่าสุด: 2 tests passed; regression ของ LINE adapter เดิม: 1 test passed

ผลทดสอบชุดนี้แสดงการใช้ประวัติในบทสนทนาที่เลือก ไม่ใช่การรับรองว่าคำตอบของโมเดลถูกต้องทุกกรณี ภาพและข้อความคำตอบเก็บตามจริง รวมถ้อยคำที่โมเดลสะกดคลาดเคลื่อน ควรเทียบกับต้นฉบับหน้า 30
