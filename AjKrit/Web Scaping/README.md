# AIS Promotions Scraper

สคริปต์นี้ใช้ Selenium เปิดหน้าเว็บ AIS ที่มี JavaScript และใช้ BeautifulSoup แยกเฉพาะการ์ดโปรโมชั่นจากหน้า:

`https://www.ais.th/consumers/promotions`

ผลลัพธ์มีข้อมูล `title`, `description`, `url` และ `image_url` ในไฟล์ JSON และ CSV

## ติดตั้ง

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

ต้องมี Google Chrome ติดตั้งอยู่ในเครื่อง Selenium 4 จะจัดการ ChromeDriver ให้โดยอัตโนมัติ

## รัน

```powershell
python ais_promotions_scraper.py
```

ไฟล์ผลลัพธ์จะอยู่ที่ `output\ais_promotions.json` และ `output\ais_promotions.csv` หากต้องการเห็นหน้าต่าง Chrome ให้เพิ่ม `--headful`
