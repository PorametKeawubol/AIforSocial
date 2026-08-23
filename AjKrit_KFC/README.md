# LINE KFC Menu & Promotion QA

LINE chatbot สำหรับค้นหาข้อมูลเมนูและโปรโมชันของ KFC Thailand ตามใบงาน: Flask รับ webhook, ดึงข้อมูลแบบ dynamic จากหน้า KFC, เก็บเป็น JSON และค้นหาด้วย multilingual BERT embedding

## สิ่งที่มีในโครงงาน

- `webhookserver.py` — LINE Messaging API webhook (`POST /callback`) และ health check
- `intent_classifier.py` — Sentence-BERT Intent Detection สำหรับ 6 intent: `menu`, `promotion`, `location`, `greeting`, `order`, `thanks` พร้อม `other` fallback
- `scraper.py` — ดึงเมนู/โปรโมชันจากข้อมูล public ของ KFC Thailand, บันทึก `data/kfc_menu.json` และดึง URL รูปเมนูสำหรับ Flex
- `qa_engine.py` — ค้นหา semantic ด้วย `sentence-transformers` พร้อม keyword fallback เมื่อโมเดลใช้งานไม่ได้
- `scripts/start_tunnel.sh` — เปิด Cloudflare Quick Tunnel ไปยัง Flask

KFC Thailand ใช้หน้าแบบ dynamic จึงไม่มีรายละเอียดเมนูอยู่ใน HTML เริ่มต้นเสมอไป ตัว scraper จึงอ่าน public content configuration ที่หน้าเว็บใช้เองก่อน แล้วมี BeautifulSoup/Selenium fallback สำหรับหน้า HTML ที่เปลี่ยนในอนาคต โดยไม่เก็บ token ของ KFC ไว้ใน source code

## ติดตั้ง

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

ใส่ `LINE_CHANNEL_SECRET` และ `LINE_CHANNEL_ACCESS_TOKEN` ใน `.env` เท่านั้น ห้าม commit ไฟล์นี้

หาก Chromium ไม่อยู่ใน PATH ให้ตั้ง `CHROME_BINARY` เช่น `/snap/bin/chromium`
ระบบใช้ `chromedriver-autoinstaller` ให้จับคู่ ChromeDriver อัตโนมัติได้ โดยอ้างอิงชุดไบนารีจาก [Chrome for Testing](https://googlechromelabs.github.io/chrome-for-testing/)

## สร้างฐานข้อมูลและ BERT index

```bash
python scraper.py --refresh --build-index
```

คำสั่งนี้บันทึก snapshot ที่ `data/kfc_menu.json` และสร้าง cache embedding ไว้ใน `data/kfc_embeddings.npz` (ไฟล์ cache ไม่ถูก version control) หลังจากนั้นสามารถทดสอบได้ เช่น

```bash
python scraper.py --ask "เดอะบอกซ์ ซิกเนเจอร์ คืออะไร"
python scraper.py --ask "มีโปรโมชั่นอะไรบ้าง"
```

ทดสอบผลการทดลอง Intent Detection จำนวน 42 ข้อความ (intent ละ 7 ข้อความ) ได้ด้วย:

```bash
python scripts/evaluate_intents.py
```

รายงานผลอยู่ที่ [`docs/Lab_6_Intent_Report.md`](docs/Lab_6_Intent_Report.md)

## รัน LINE webhook

```bash
python webhookserver.py
```

ตรวจสถานะได้ที่ <http://127.0.0.1:5000/healthz>

เปิดอีก terminal สำหรับ Cloudflare Tunnel:

```bash
./scripts/start_tunnel.sh
```

นำ HTTPS URL ที่ cloudflared แสดงไปตั้งใน LINE Developers Console เป็น:

```text
https://<your-tunnel>.trycloudflare.com/callback
```

จากนั้นกด **Verify** และเปิด **Use webhook** ใน LINE Developers Console. ควรปิด Auto-response message ใน LINE Official Account Manager เพื่อไม่ให้มีข้อความตอบซ้ำ

เมื่อผู้ใช้ส่งข้อความ `menu` (หรือ `เมนู`) บอตจะอ่าน URL รูปจาก catalog หน้า [KFC meals](https://www.kfc.co.th/menu/meals) แล้วตอบกลับเป็น Flex carousel 5 ใบตามใบงาน ภายในแต่ละใบมีรูป ชื่อ ราคา และปุ่ม `ดูรายละเอียด` เมื่อกดปุ่ม บอตจะส่งรูปเมนูนั้นพร้อมราคา ส่วนประกอบ ตัวเลือก และลิงก์เมนูเป็นข้อความใน reply เดียว ปรับจำนวนการ์ดได้ด้วย `MENU_CAROUSEL_RESULTS` (ไม่เกิน 10) และรูปจะถูก cache ตาม `MENU_IMAGE_CACHE_SECONDS` เพื่อให้การตอบด้วย reply token ทำได้รวดเร็ว

ข้อความธรรมชาติจะผ่าน Intent Classifier ก่อน: `greeting`, `thanks`, `location` และ `order` มีคำตอบสนทนาโดยตรง ส่วน `menu`, `promotion` และ `other` จะส่งต่อให้ QA engine ค้นหาข้อมูลเมนู/โปรโมชันต่อ

## ตัวอย่างคำถาม

- `มีเมนูไก่ทอดอะไรบ้าง`
- `เดอะบอกซ์ ซิกเนเจอร์ คืออะไร`
- `เมนูนี้มีอะไรบ้าง เดอะบอกซ์ ออลสตาร์`
- `มีโปรโมชั่นอะไรบ้าง`
- `help`
- `menu` — แสดงรูปเมนู KFC แบบ Flex carousel

ข้อมูลราคาและโปรโมชันเปลี่ยนได้ตามช่องทาง/ช่วงเวลา ให้รัน `python scraper.py --refresh --build-index` ก่อนการสาธิตหรือเมื่อต้องการข้อมูลใหม่

## ทดสอบ

```bash
python -m pytest -q
```

ชุดทดสอบใช้ข้อมูลจำลองและไม่เรียก KFC, Chrome หรือ LINE API จริง ส่วนการทดสอบดึงรูปจริงทำได้ด้วย:

```bash
python - <<'PY'
from scraper import KfcScraper
for item in KfcScraper().scrape_menu_images(3):
    print(item["name"], item["image_url"])
PY
```
