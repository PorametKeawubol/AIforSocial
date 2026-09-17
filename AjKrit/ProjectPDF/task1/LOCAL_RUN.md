# รันงาน 1 ในเครื่อง

ผลสำหรับส่งคือกราฟ A และ B ใน Neo4j active database รายละเอียดและ query พร้อมใช้ดูที่ [SUBMISSION.md](SUBMISSION.md)

เครื่องที่ทดสอบ: Ryzen 7 7735HS, RAM 30 GiB, Radeon RX 7600S 8 GiB โมเดล `Qwen3.8-27B GGUF UD-Q4_K_M` รันผ่าน Ollama แบบแบ่ง GPU/RAM โดยไม่ใช้ Cloud API

## โมเดล

- A: `BAAI/bge-m3` + `Qwen3.8-27B GGUF UD-Q4_K_M`
- B: `intfloat/multilingual-e5-large` + `Qwen3.8-27B GGUF UD-Q4_K_M`

ผล candidate รอบสกัดอยู่ใน `artifacts/sections_v2` ส่วนกราฟ A/B หลัง source review อยู่ใน `artifacts/final_ab`

## สร้าง นำเข้า และตรวจ

รันจากโฟลเดอร์ `Project2/task1`:

```bash
.venv/bin/python build_ab_reviewed.py
.venv/bin/python switch_to_ab_database.py
.venv/bin/python verify_ab.py
.venv/bin/python -m unittest discover -p 'test*.py'
```

`switch_to_ab_database.py` ใช้ clean Docker volume และนำเข้าเฉพาะ A/B ฐานข้อมูลเดิมอยู่ใน container `neo4j-GraphRAG` ที่หยุดไว้เป็น recovery backup ส่วน container ที่เปิดใช้งานชื่อ `neo4j-Project2-AB`

## เปิดดู

เปิด <http://localhost:7474/browser/> แล้วใช้ query A หรือ B จาก `ab_views.cypher` ถ้าเปิดหน้า Browser ค้างไว้ตั้งแต่ก่อนสลับฐานข้อมูล ให้รีเฟรชหน้าเว็บหนึ่งครั้ง
