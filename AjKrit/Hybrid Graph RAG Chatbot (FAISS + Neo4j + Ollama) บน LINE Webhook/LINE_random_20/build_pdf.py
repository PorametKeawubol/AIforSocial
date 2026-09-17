from pathlib import Path
from collections import Counter
from html import escape
import json
import zipfile
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / 'answers.json').read_text())
counts = Counter(r['category'] for r in rows)
assert len(rows) == 20 and len(counts) == 8
css = '''
@font-face {font-family: Thai; src: url("file:///usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf");}
@font-face {font-family: Thai; font-weight:700; src: url("file:///usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf");}
@page {size: A4 landscape; margin: 10mm;}
* {box-sizing:border-box;} body {margin:0; font-family:Thai,sans-serif; color:#000;}
.page {height:189mm; break-after:page; overflow:hidden; padding-top:8mm; position:relative;}
.author {position:absolute; top:0; right:0; font-size:11pt; color:#000;}
.page:last-child {break-after:auto;}
h1 {font-size:25pt; margin:3mm 0 2mm;}
.subtitle {font-size:12pt; color:#000; margin:0 0 6mm;}
table {width:85%; border-collapse:collapse; font-size:14pt;}
th {background:white; color:#000; text-align:left;}
th,td {padding:2.2mm 5mm; border:1px solid #000;}
th:last-child,td:last-child {width:25mm; text-align:center;}
tbody tr:nth-child(even) {background:white;}
tfoot {font-weight:bold; background:white;}
.note {font-size:11pt; margin-top:5mm; color:#000;}
.heading {height:23mm; padding:0 0 2mm;}
.meta {font-size:11pt; color:#000; font-weight:bold; margin-bottom:1.3mm;}
.question {font-size:13pt; font-weight:bold; line-height:1.5; margin:0;}
.screenshot {display:block; width:277mm; height:157mm; max-height:157mm; object-fit:contain; border:1px solid #dbe6e3;}
'''
summary = '<section class="page"><div class="author">6610110554 ปรเมษ แก้วอุบล</div><h1>คำถามและคำตอบจาก LINE</h1><p class="subtitle">สุ่มคำถาม 20 ข้อ ครอบคลุม 8 หมวดหมู่</p><table><thead><tr><th>หมวด</th><th>จำนวน</th></tr></thead><tbody>'
summary += ''.join(f'<tr><td>{escape(cat)}</td><td>{count}</td></tr>' for cat,count in counts.items())
summary += '</tbody><tfoot><tr><td>รวม</td><td>20</td></tr></tfoot></table><p class="note">หน้าถัดไปแสดงลำดับ ข้อต้นฉบับ คำถาม และภาพคำตอบจาก LINE จริง<br>ข้อต้นฉบับอ้างอิงหมายเลข id ในไฟล์ plant_disease_chat_questions.csv</p></section>'
pages = [summary]
for i,r in enumerate(rows,1):
    pages.append(f'<section class="page"><div class="author">6610110554 ปรเมษ แก้วอุบล</div><div class="heading"><div class="meta">ลำดับ {i:02d} / 20 &nbsp; | &nbsp; ข้อต้นฉบับ {escape(r["id"])} &nbsp; | &nbsp; {escape(r["category"])}</div><p class="question">คำถาม: {escape(r["question"])}</p></div><img class="screenshot" src="{escape(r["screenshot"])}"></section>')
html = '<!doctype html><html lang="th"><meta charset="utf-8"><title>LINE — 20 คำถาม ครบ 8 หมวด</title><style>'+css+'</style><body>'+''.join(pages)+'</body></html>'
(ROOT/'LINE_random_20.html').write_text(html)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto((ROOT/'LINE_random_20.html').as_uri())
    page.evaluate('document.fonts.ready')
    assert page.locator('img').evaluate_all('(imgs) => imgs.every(i => i.complete && i.naturalWidth > 0)')
    page.pdf(path=str(ROOT/'LINE_random_20.pdf'),print_background=True,prefer_css_page_size=True)
    browser.close()
readme=ROOT/'README.md'
s=readme.read_text()
s=s.replace('| หน้า PDF | ข้อต้นฉบับ | คำถาม |','| หน้า PDF (หน้า 1 เป็นสรุปหมวด) | ข้อต้นฉบับ | คำถาม |')
lines=s.splitlines()
for i,r in enumerate(rows,1):
    old=f'| {i} | {r["id"]} | {r["question"]} |'
    lines=[f'| {i+1} | {r["id"]} | {r["question"]} |' if line==old else line for line in lines]
readme.write_text('\n'.join(lines)+'\n')
with zipfile.ZipFile(ROOT.parent/'LINE_random_20.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(ROOT.iterdir()):
        if f.is_file(): z.write(f,f.name)
print('Updated PDF and ZIP.')
