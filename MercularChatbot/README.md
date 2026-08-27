# MercuMate — Mercular LINE Shopping Assistant (Assignment I)

**MercuMate** คือแชตบอต LINE ภาษาไทยสำหรับค้นหาสินค้า Mercular
ด้วยภาษาธรรมชาติ เช่น
`หาเมาส์ Logitech ไม่เกิน 3,000 บาท เอาเฉพาะของพร้อมส่ง` แล้วแสดงสินค้า
ที่ตรงเงื่อนไขเป็น **Random Top 5 Flex Carousel**

โปรเจกต์นี้ออกแบบตามเกณฑ์ใน `Assignment_I/Scoring_criteria.md` โดยตรง:

- ดึงและทำความสะอาด ชื่อ ราคา รูป แบรนด์ หมวดหมู่ สต็อก และ URL จากหน้าเว็บสาธารณะ
- แยก Intent และ Entity หลายเงื่อนไข รองรับภาษาพูด ตัวเลขไทย และคำพิมพ์ผิดที่พบบ่อย
- สุ่มจากกลุ่มสินค้าที่เกี่ยวข้องโดยไม่ซ้ำในชุดเดียว และจำผลล่าสุดแยกตามผู้ใช้
- ส่ง Flex Carousel พร้อมปุ่ม `ดูรายละเอียด` / `ซื้อที่ Mercular` และ Quick Reply
- มี Rich Menu ธีม MercuMate 6 ช่อง และ Message Lab สำหรับทดลองข้อความ LINE
  11 ชนิดโดยไม่รบกวนเส้นทางค้นหาสินค้าหลัก
- ตอบจาก snapshot ในเครื่อง จึงไม่รอ scraping ระหว่าง webhook และรักษา latency ต่ำ

> นี่คือโปรเจกต์เพื่อการศึกษาและไม่ได้เป็นบอตทางการของ Mercular ราคาและสต็อก
> อาจเปลี่ยนแปลงได้ ให้ยืนยันอีกครั้งบนหน้าสินค้าก่อนซื้อ

## โครงสร้าง

```text
LINE webhook (app.py)
  ├─ PhayaThaiBERT intent + exact Thai entities (bert_nlp.py + nlp.py)
  ├─ hard filters → relevance → fair random Top 5 (recommender.py)
  ├─ local last-known-good catalog (repository.py)
  └─ Flex / Quick Reply views (line_views.py)

Scheduled/manual refresh
  └─ public category HTML → clean/dedupe → atomic JSON (scraper.py)
```

## ติดตั้ง

ต้องใช้ Python 3.10 ขึ้นไป

```bash
cd MercularChatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

ใส่ `LINE_CHANNEL_SECRET` และ `LINE_CHANNEL_ACCESS_TOKEN` ใน `.env` เท่านั้น
ห้ามใส่ credential ลงใน source code หรือ commit `.env`

### PhayaThaiBERT NLP

MercuMate ใช้ `clicknext/phayathaibert` เพื่อช่วยจำแนก intent ภาษาไทยจากความหมาย
ของประโยค และใช้ rule parser ควบคู่กันเพื่อดึงเงื่อนไขสินค้าที่ต้องแม่นยำ เช่น ราคา
แบรนด์ สต็อก และการเรียงผลลัพธ์ หากโหลดโมเดลไม่ได้ บอตจะตอบด้วย rule parser เดิม
โดยอัตโนมัติและไม่ทำให้ webhook ล่ม

ค่าเริ่มต้นใน `.env` คือ `NLP_BACKEND=phayathaibert` โมเดลจะดาวน์โหลดจาก Hugging
Face เมื่อได้รับข้อความครั้งแรก (ไฟล์ weights ราว 1.1 GB); สำหรับ production ควร
pre-cache น้ำหนักโมเดลแล้วตั้ง `PHAYATHAIBERT_LOCAL_FILES_ONLY=true` หากต้องการปิด
โมเดลชั่วคราว ให้ตั้ง `NLP_BACKEND=rules`.

ค่า `PHAYATHAIBERT_MIN_CONFIDENCE` เริ่มต้นเป็น `0.30` ซึ่งเหมาะกับคะแนน semantic
similarity ของ base model; เพิ่มค่านี้ได้หากต้องการให้ rule parser เป็นตัวตัดสินมากขึ้น.

หากต้องการใช้ Image, Video, Audio, Imagemap หรือ Template demo ให้ใส่ origin
HTTPS ที่ LINE เข้าถึงได้ด้วย เช่น:

```dotenv
PUBLIC_BASE_URL=https://your-current-tunnel.ngrok-free.app
```

## เตรียม catalog

ตัว scraper อ่านเฉพาะหน้า category/product สาธารณะที่ robots.txt อนุญาต ใช้
timeout + retry/backoff เว้นช่วง request และบันทึก source/fetched time ทุกครั้ง
พร้อม quality gate ที่ไม่ยอมแทนที่ข้อมูลเดิมเมื่อหน้า category ล้มเหลวเกิน 40%
หรือจำนวนสินค้าลดฮวบผิดปกติ

```bash
python scraper.py --refresh
python cli.py "หาเมาส์เกมมิ่งไม่เกิน 3000 พร้อมส่ง"
```

ก่อนใช้งานกับข้อมูลจำนวนมากหรือเผยแพร่จริง ควรขออนุญาต/official feed จาก
Mercular ก่อน เนื่องจากเงื่อนไขเว็บไซต์จำกัดการคัดลอกเนื้อหา โปรเจกต์นี้ตั้งใจ
เก็บเพียง snapshot ขนาดเล็กที่จำเป็นสำหรับการสาธิต และไม่แตะ `/cart`, `/cms`,
`/browse` หรือ `/my-account`

แนะนำให้ refresh ด้วย cron หรืองานภายนอกวันละหนึ่งครั้ง ไม่ควรเรียก scraper
จาก webhook:

```cron
15 3 * * * cd /path/to/MercularChatbot && .venv/bin/python scraper.py --refresh
```

## รัน LINE webhook

```bash
python app.py
curl http://127.0.0.1:5000/healthz
curl -i http://127.0.0.1:5000/readyz
```

เปิด tunnel อีก terminal:

```bash
./scripts/start_tunnel.sh
```

นำ URL ที่ได้ไปตั้งใน LINE Developers Console เป็น
`https://<tunnel>.trycloudflare.com/callback` จากนั้นกด Verify, เปิด Use webhook,
และปิด Auto-response ใน LINE Official Account Manager เพื่อไม่ให้ตอบซ้ำ

## Rich Menu และ Message Lab

Rich Menu ขนาด `2500×1686` ใช้ภาพ mascot/cover ของ MercuMate และมี 6 action:
ค้นหาสินค้า, เกมมิ่ง, ออดิโอ, แก็ดเจ็ต, Message Lab และช่วยเหลือ โดย action
ค้นหาทั้งหมดยังส่งข้อความเข้า NLP ตัวเดิม จึงช่วยผู้ใช้เริ่มคำสั่งได้โดยไม่ทำทางลัด
ข้ามระบบค้นหา

ตรวจและติดตั้งเมนูแบบทำซ้ำได้:

```bash
./scripts/render_rich_menu.sh
python scripts/validate_line_messages.py
python scripts/setup_rich_menu.py --dry-run
python scripts/setup_rich_menu.py
```

Message Lab รองรับ Text, Text v2, Sticker, Image, Video, Audio, Location,
Coupon, Imagemap, Template และ Flex โดยส่งทีละชนิดเพื่อไม่เกินข้อจำกัด 5 message
objects ต่อหนึ่ง reply หากต้องการส่ง `CouponMessage` จริง ต้องสร้าง Coupon ที่ได้รับ
อนุญาตใน LINE Official Account Manager แล้วใส่ `LINE_COUPON_ID`; หากยังไม่มี
ระบบจะแสดง Flex demo ที่ระบุชัดว่าไม่มีส่วนลดจริง

ไฟล์สื่อสำเร็จรูปอยู่ใน `static/` ส่วน source/renderable assets อยู่ใน `assets/`.
Rich Menu แสดงบน LINE มือถือ แต่ไม่แสดงบน LINE PC และ URL tunnel ชั่วคราวต้อง
อัปเดต `PUBLIC_BASE_URL` เมื่อเปลี่ยน

## ตัวอย่างข้อความ

- `มีสินค้าอะไรแนะนำบ้าง`
- `หาหูฟัง Xiaomi ไม่เกิน 3000`
- `อยากได้เม้าเกมมิ่ง logitec งบ 3k พร้อมส่ง` (ภาษาพูด + typo)
- `คีย์บอร์ดราคา 1000 ถึง 3500 เรียงถูกสุด`
- `ลำโพง Marshall มากกว่า 9000 แต่ไม่เกิน 20000`
- `สุ่มใหม่` เพื่อรับชุดใหม่ที่ยังตรงเงื่อนไขเดิม
- `ช่วยเหลือ`

หากสินค้าตรงเงื่อนไขน้อยกว่า 5 รายการ บอตจะแสดงเท่าที่มีโดยไม่เติมสินค้า
ผิดเงื่อนไข หากไม่พบเลย บอตจะแนะนำให้ลดเงื่อนไขผ่าน Quick Reply

การเรียงตามราคาและส่วนลดใช้ค่าที่อยู่ใน snapshot โดยตรง ส่วนคำขอ “ขายดี”
หรือ “ใหม่ล่าสุด” จะจัดตามความเกี่ยวข้องพร้อมแจ้งข้อจำกัด เพราะหน้า category
ปัจจุบันไม่มีตัวเลขยอดขายหรือวันวางจำหน่าย จึงไม่สร้างสัญญาณเหล่านั้นขึ้นเอง

## ทดสอบและหลักฐานคะแนน

ชุดทดสอบทั้งหมดเป็น offline ไม่เรียก Mercular หรือ LINE API จริง:

```bash
pytest -q
python scripts/evaluate_nlp.py
python scripts/benchmark.py
```

รายละเอียดการจับคู่เกณฑ์และ TC-01 ถึง TC-12 อยู่ที่
[`docs/SCORING_EVIDENCE.md`](docs/SCORING_EVIDENCE.md) พร้อมรายงาน
[`NLP`](docs/NLP_EVALUATION.md) และ [`performance`](docs/PERFORMANCE_REPORT.md)
ที่รันซ้ำได้

`scripts/validate_line_messages.py` และ `scripts/setup_rich_menu.py --dry-run`
เป็น optional live validation จึงไม่ถูกรวมใน offline test suite

## การดูแลข้อมูลและความปลอดภัย

- ตรวจ LINE signature ก่อนอ่าน event และกัน `webhookEventId` ที่ LINE ส่งซ้ำ
- ไม่รับที่อยู่ ข้อมูลบัตร ข้อมูลบัญชี หรือดำเนินการชำระเงินในแชต
- ปุ่มซื้อพาไปยัง HTTPS product URL ของ Mercular เท่านั้น
- snapshot เสียจะไม่แทนที่ last-known-good catalog
- price/stock ที่ไม่มีหรือเก่าแสดงอย่างตรงไปตรงมา ไม่เดาค่า
- log ไม่บันทึก channel token, secret หรือข้อความส่วนบุคคลเต็มรูปแบบ
