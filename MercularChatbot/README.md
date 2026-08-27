# MercuMate — Mercular LINE Shopping Assistant (Assignment I)

**MercuMate** คือแชตบอต LINE ภาษาไทยสำหรับค้นหาสินค้า Mercular
ด้วยภาษาธรรมชาติ เช่น
`หาเมาส์ Logitech ไม่เกิน 3,000 บาท เอาเฉพาะของพร้อมส่ง` แล้วแสดงสินค้า
ที่ตรงเงื่อนไขเป็น **Random Top 5 Flex Carousel**

โปรเจกต์นี้ออกแบบตามเกณฑ์ใน `Assignment_I/Scoring criteria.txt` โดยตรง:

- ดึงและทำความสะอาด ชื่อ ราคา รูป แบรนด์ หมวดหมู่ สต็อก และ URL จากหน้าเว็บสาธารณะ
- แยก Intent และ Entity หลายเงื่อนไข รองรับภาษาพูด ตัวเลขไทย และคำพิมพ์ผิดที่พบบ่อย
- สุ่มจากกลุ่มสินค้าที่เกี่ยวข้องโดยไม่ซ้ำในชุดเดียว และจำผลล่าสุดแยกตามผู้ใช้
- ส่ง Flex Carousel พร้อมปุ่ม `ดูรายละเอียด` / `ซื้อที่ Mercular` และ Quick Reply
- มี Rich Menu ธีม MercuMate 6 ช่องสำหรับค้นหา หมวดสินค้า โปรโมชัน และวิธีใช้
- มี Message Lab แยกสำหรับทดลองข้อความ LINE 11 ชนิด โดยไม่ปะปนกับเมนูผู้ใช้จริง
- ตอบจาก snapshot ในเครื่อง จึงไม่รอ scraping ระหว่าง webhook และรักษา latency ต่ำ

> นี่คือโปรเจกต์เพื่อการศึกษาและไม่ได้เป็นบอตทางการของ Mercular ราคาและสต็อก
> อาจเปลี่ยนแปลงได้ ให้ยืนยันอีกครั้งบนหน้าสินค้าก่อนซื้อ

## โครงสร้าง

```text
LINE webhook (app.py)
  ├─ PhayaThaiBERT intent + exact Thai entities (bert_nlp.py + nlp.py)
  ├─ hierarchical category picker from breadcrumbs (catalog_navigation.py)
  ├─ hard filters → relevance → fair random Top 5 (recommender.py)
  ├─ local last-known-good catalog (repository.py)
  ├─ bounded TTL context + webhook deduplication (session_state.py)
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

### Daily catalog and price history

`scripts/sync_catalog.py` ทำงานนอก webhook เสมอ และเขียน price, discount และ
stock observation หนึ่งรายการต่อสินค้า/วันลง SQLite. ค่า `--scope seed` คือ 13
หน้า demo เดิม; `--scope sitemap-leaves` อ่าน category sitemap สาธารณะเพื่อเลือก
leaf categories เช่น `audio/dap-dac-amp/dac-amplifiers` โดยไม่เรียก `/browse`.
ทั้งสอง scope จะเพิ่มหน้า collection พิเศษ `flash-sale` และ `new-arrival` ต่อท้าย
หมวดปกติด้วย สินค้าที่ซ้ำจะถูกเก็บเพียงรายการเดียว แต่ติด tag
`collection:flash-sale` หรือ `collection:new-arrival` ไว้ใน snapshot. หน้า
`new-arrival` อาจไม่มีสินค้าในบางช่วง จึงบันทึกเป็น collection ว่างโดยไม่ทำให้
รอบ sync ล้มเหลว.
โหมด sitemap ต้องใช้เฉพาะเมื่อได้รับอนุญาตให้สร้าง product index ในวงกว้างแล้ว:

```bash
# Demo scope: refresh snapshot พร้อมบันทึกประวัติรายวัน
python scripts/sync_catalog.py --scope seed

# Authorized broad category sync; page-level cap ปรับได้ตามข้อตกลงของ source
python scripts/sync_catalog.py --scope sitemap-leaves --max-products-per-category 100

# Retry only paths that failed in the current snapshot; successful paths are untouched
python scripts/sync_catalog.py --retry-failed
```

ตั้ง cron ให้ทำวันละหนึ่งครั้ง ไม่ควรเรียกงานนี้จาก webhook:

```cron
15 3 * * * cd /path/to/MercularChatbot && .venv/bin/python scripts/sync_catalog.py --scope sitemap-leaves >> catalog-sync.log 2>&1
45 4 * * * cd /path/to/MercularChatbot && .venv/bin/python scripts/sync_promotions.py >> promotion-sync.log 2>&1
```

### Promotion articles

`scripts/sync_promotions.py` อ่านการ์ดบทความจากหน้า
`/category-review-article/promotion` วันละครั้งและเขียน
`data/mercular_promotions.json` แบบ atomic. ถ้าหน้าเว็บตอบ 429, โครงสร้างเปลี่ยน
หรือไม่พบรายการ ระบบจะไม่เขียนทับ last-known-good snapshot. ใน LINE พิมพ์
`มีโปรโมชันอะไรบ้าง` หรือสะกดใกล้เคียงอย่าง `มีโปรโมชั้นอะไรบ้าง` เพื่อดูสูงสุด 5
รายการพร้อมลิงก์บทความทางการ. ควรจัดเวลางานนี้ไม่ให้ชนกับ product detail job.

### Functional-requirement coverage

| Requirement | สถานะ | หมายเหตุ |
| --- | --- | --- |
| FR-01 Search | รองรับ | หมวด แบรนด์ งบ และ typo |
| FR-02 Recommendation | รองรับ | use-case ที่มี alias เช่น FPS |
| FR-03 Comparison | รองรับตามข้อมูล | ต้องพบชื่อทั้งสองรุ่นใน snapshot; ไม่เดารุ่นที่หาย |
| FR-04 Constraint filtering | รองรับ | เงื่อนไขทั้งหมดเป็น AND; ไม่มีสินค้าตรงจริงจะแจ้งไม่พบ |
| FR-05 Product Q&A | รองรับตามข้อมูล | แตะดูรายละเอียดก่อน; ตอบ Bluetooth/น้ำหนัก/สเปกจากข้อมูลที่ scrape เท่านั้น |
| FR-06 Context | รองรับ | `ขอถูกกว่านี้` และ `เอา Logitech อย่างเดียว` ภายใน TTL |
| FR-07 Alternative | รองรับ | หลังแตะสินค้าต้นแบบ หา category เดียวกันที่ราคาต่ำกว่า |
| FR-08 Use case | รองรับบางชุด | FPS, Valorant และ lightweight มี alias; use-case ใหม่ต้องเพิ่ม/evaluate |
| FR-09 Multi-constraint | รองรับ | รวม wireless, สี, งบ และ `ไม่เอา <brand>` |
| FR-10 Carousel | รองรับ | สูงสุด 5 การ์ดพร้อมชื่อ ราคา รูป และลิงก์ |

### Product-page details

ข้อมูลหน้า category ไม่ได้มีสเปกทั้งหมด จึงมี job แยก
`scripts/enrich_product_details.py` อ่าน `__NEXT_DATA__` ที่ server render อยู่ใน
product page ด้วย HTTP ก่อน จึงไม่ต้องเปิด Chromium สำหรับสินค้าส่วนใหญ่; หากหน้าใด
ไม่มี payload นี้จึงค่อย fallback ไป Playwright แบบ headless. ได้ข้อมูลภาพรวม,
คุณสมบัติเด่น, ตารางสเปก, คะแนน/จำนวนรีวิวเมื่อหน้าแสดง, ประกัน และสิทธิ์บริการ
ข้อมูลเหล่านี้ถูกเก็บใน snapshot และแสดงในข้อความ “ดูรายละเอียด” ของ LINE.
ไม่เรียกจาก webhook และไม่เก็บบทความรีวิวเต็มหน้า.

ติดตั้ง browser หนึ่งครั้งสำหรับ fallback แล้วทดลองจำนวนน้อยก่อน:

```bash
.venv/bin/playwright install chromium
.venv/bin/python scripts/enrich_product_details.py --limit 5
```

คำสั่งจะข้ามสินค้าที่มี `detail_updated_at` แล้วโดยปริยาย จึงใช้รันทุกคืนเพื่อเติม
เฉพาะสินค้าใหม่ได้; ใช้ `--refresh-existing` เมื่อต้องการอัปเดตรายละเอียดเดิมทั้งหมด.
หาก source ตอบ `HTTP 429` job จะ checkpoint หน้าที่สำเร็จแล้วและหยุดทันที ไม่วน
request ต่อ; ให้รอตาม `retry_after_seconds` (ถ้ามี) ก่อนรันใหม่.

หากต้องเติมข้อมูลทั้งหมดบนเครื่องที่เปิดทิ้งไว้ได้ ให้ใช้ safe resume mode;
มันดึงครั้งละ 5 หน้าและพัก 5 นาทีระหว่างทุก batch (รวมถึงเมื่อถูก rate-limit):

```bash
.venv/bin/python scripts/enrich_product_details.py --until-complete
```

จำกัดให้เก็บ detail เฉพาะหมวดหลักได้ เช่น คอมพิวเตอร์และ Smartphone / Tablet /
ACC:

```bash
.venv/bin/python scripts/enrich_product_details.py --until-complete \
  --category-scope computer --category-scope smartphone-tablet-acc
```

เมื่อทดสอบแล้วว่า source รับได้ ให้ใช้ HTTP mode ที่ default อยู่และคงช่วงห่างต่อหน้า
อย่างน้อย 3 วินาทีเพื่อลดเวลาโดยไม่เพิ่ม browser concurrency; `429` จะพัก 5 นาทีเอง:

```bash
DETAIL_SCRAPER_DELAY_SECONDS=3 .venv/bin/python scripts/enrich_product_details.py \
  --until-complete --batch-size 25 --between-batches-seconds 0 \
  --category-scope computer --category-scope smartphone-tablet-acc
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

Rich Menu v3 ขนาด `2500×1686` ใช้ภาพ mascot/cover ของ MercuMate และมี 6 action:
สินค้าทั้งหมด, เกมมิ่ง, คอมพิวเตอร์, มือถือ/แท็บเล็ต, โปรโมชัน และช่วยเหลือ
สี่ปุ่มแรกเปิด Flex category picker จาก breadcrumb ใน snapshot; หมวดที่ยังมีระดับย่อย
จะเปิดให้เลือกต่อ และมี “ดูทั้งหมดในหมวดนี้” ก่อนเข้าสู่ Random Top 5 การค้นหาจาก
picker ใช้ exact category-path constraint จึงไม่ปนสินค้าที่มีเพียงคำคล้ายกันในชื่อ
ทุกปุ่มมี contract test ตั้งแต่ payload → navigation/intent → webhook response ใน
`tests/test_rich_menu.py` และ `tests/test_app.py` ส่วน Message Lab ยังเรียกได้ด้วยข้อความ
`เดโมข้อความ` แต่ไม่นำพื้นที่เมนูหลักของผู้ใช้ไปใช้

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

- `มีสินค้าอะไรบ้าง` — เปิดเมนูเลือกหมวดและหมวดย่อย
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
