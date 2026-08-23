from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# กำหนด URL และตั้งค่า Chrome 
# กำหนดเว็บไซต์ที่จะ Scrape และสร้าง Chrome WebDriver
# โดยเปิดใช้งาน Headless Mode เพื่อให้ทำงานเบื้องหลัง
URL = "https://www.ais.th/consumers/promotions"

options = Options()
options.add_argument("--headless=new")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)

# เปิดเว็บไซต์และรอโหลดข้อมูล 
# เข้าเว็บไซต์โปรโมชั่นของ AIS และรอจนกว่าการ์ดโปรโมชั่น
# จะถูกโหลดเสร็จ ก่อนเริ่มดึงข้อมูล
driver.get(URL)

WebDriverWait(driver, 30).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, ".cms-content-card"))
)

# ดึงข้อมูลโปรโมชั่นทั้งหมด
# ค้นหาการ์ดโปรโมชั่นทั้งหมดในหน้าเว็บ
cards = driver.find_elements(By.CSS_SELECTOR, ".cms-content-card")

promotions = []
seen = set()

# วนลูปดึงข้อมูลจากแต่ละการ์ด ได้แก่
# ชื่อโปรโมชั่น รายละเอียด ลิงก์ และรูปภาพ
# พร้อมตรวจสอบไม่ให้เก็บ URL ซ้ำ
for card in cards:
    try:
        link = card.find_element(By.TAG_NAME, "a")
        href = urljoin(URL, link.get_attribute("href"))

        if href in seen:
            continue

        title = ""

        for tag in ["h1", "h2", "h3", "h4", "h5"]:
            elems = card.find_elements(By.TAG_NAME, tag)
            if elems:
                title = elems[0].text.strip()
                break

        if not title:
            continue

        desc = ""
        p = card.find_elements(By.TAG_NAME, "p")
        if p:
            desc = p[0].text.strip()

        image_url = ""
        imgs = card.find_elements(By.TAG_NAME, "img")
        if imgs:
            src = imgs[0].get_attribute("src")
            image_url = urljoin(URL, src)

        promotions.append({
            "title": title,
            "description": desc,
            "url": href,
            "image_url": image_url
        })

        seen.add(href)

    except Exception:
        # หากเกิดข้อผิดพลาดกับการ์ดใด ให้ข้ามไปการ์ดถัดไป
        pass

# ปิด Browser
# ปิด Chrome หลังจากดึงข้อมูลเสร็จเรียบร้อย
driver.quit()

# เตรียมข้อมูลและบันทึกไฟล์ 
# สร้างข้อมูลในรูปแบบ JSON พร้อมบันทึกเวลาที่ Scrape
payload = {
    "source": URL,
    "scraped_at": datetime.now(timezone.utc).isoformat(),
    "count": len(promotions),
    "promotions": promotions,
}

# สร้างโฟลเดอร์ output หากยังไม่มี
Path("output").mkdir(exist_ok=True)

# บันทึกข้อมูลเป็นไฟล์ JSON
with open("output/ais_promotions.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

# บันทึกข้อมูลเป็นไฟล์ CSV
with open("output/ais_promotions.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["title", "description", "url", "image_url"],
    )
    writer.writeheader()
    writer.writerows(promotions)

# แสดงผลลัพธ์
# แสดงจำนวนโปรโมชั่นที่ดึงได้ และชื่อของแต่ละโปรโมชั่น
print(f"เจอ {len(promotions)} โปรโมชั่น")

for i, p in enumerate(promotions, 1):
    print(f"{i}. {p['title']}")