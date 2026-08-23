# LINE Advice Branch Chatbot

โปรเจกต์นี้ทำตามเอกสารฝึกปฏิบัติการ: Flask รับ LINE webhook, Selenium เปิดหน้า Advice เพื่อค้นหาสาขา และตอบชื่อสาขาพร้อมลิงก์กลับไปที่ LINE โดยไม่มีการเรียก Advice API โดยตรง

## ติดตั้ง

ต้องมี Python 3.10 ขึ้นไป และติดตั้ง Chrome/Chromium ในเครื่อง

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

แก้ไฟล์ `.env` แล้วใส่ `LINE_CHANNEL_SECRET` และ `LINE_CHANNEL_ACCESS_TOKEN` จาก LINE Developers Console โดยไม่ต้องใส่ credential ลงใน source code

ถ้า Chrome/Chromium ไม่ได้อยู่ใน PATH ให้กำหนด `CHROME_BINARY` ใน `.env` เป็น path ของ executable เช่น `/usr/bin/google-chrome`

## รัน Flask

```bash
python webhook_server.py
```

ตรวจสอบได้ที่ `http://127.0.0.1:5000/healthz` ควรได้สถานะ `ok` และ `line_configured: true`

## เปิดให้ LINE เรียกด้วย Cloudflare Tunnel

เปิด terminal อีกหน้าต่างแล้วรัน:

```bash
cloudflared tunnel --url http://localhost:5000
```

นำ URL HTTPS ที่ได้ไปตั้งใน LINE Developers Console เป็น:

```text
https://<ชื่อ-tunnel>.trycloudflare.com/callback
```

จากนั้นกด Verify และเปิด Use Webhook ตามเอกสาร

## การใช้งาน

ส่งข้อความชื่อจังหวัด อำเภอ หรือพื้นที่ เช่น `หาดใหญ่` ให้ LINE Official Account ระบบจะค้นหาผ่านหน้า `advice.co.th/wheretobuy` แล้วตอบผลลัพธ์กลับมา หากส่ง `help` จะได้รับคำแนะนำการใช้งาน

การค้นหาแต่ละครั้งจะเปิดและควบคุมหน้า Advice ด้วย Selenium จึงใช้เวลามากกว่าการเรียก API โดยตรง แต่โค้ดยังกัน `webhookEventId` ซ้ำ 10 นาที เพื่อไม่ให้ตอบผลเดิมซ้ำเมื่อ LINE ส่ง event ซ้ำอีกครั้ง

ใน LINE Official Account Manager ให้ปิด **Auto-response messages** และลบ/ปิด reply rule เก่าที่ไม่ต้องการด้วย มิฉะนั้นข้อความจากระบบเก่าจะยังถูกส่งคู่กับข้อความจาก webhook ได้

หน้า Advice มีการเปลี่ยน selector จากตัวอย่างเดิมในเอกสารแล้ว โค้ดจึงรองรับทั้ง `#shop_find` / `.list-items-branch h3 > a` และ selector ปัจจุบัน `input.form-control-adv` / `.t-branch-name`

เมื่อคำค้นตรงกับชื่อจังหวัด เช่น `สงขลา` โค้ดจะอ่านหัว accordion จังหวัดของหน้า Advice แล้วดึงเฉพาะ card ในกลุ่มนั้น จึงไม่ปนสาขา `ปัตตานี` ที่มีคำว่า `สงขลา` อยู่ในชื่อสาขา

ในหน้าเว็บรุ่นปัจจุบัน card ของสาขาใช้ click handler และไม่มี `href` แยกรายสาขา โค้ดจึงสร้างลิงก์ค้นหาของสาขานั้นให้แทน ส่วน selector รุ่นเก่าจะดึง `href` จริงตามเอกสาร

## ทดสอบ

```bash
pytest -q
```

การทดสอบใช้ fake Selenium searcher และไม่เปิด Chrome หรือเรียก LINE API จริง
