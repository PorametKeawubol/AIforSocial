"""Package an actual LINE window capture; never recreate chat bubbles."""
from pathlib import Path
from html import escape
import json
import zipfile
from playwright.sync_api import sync_playwright

LAB = Path(__file__).resolve().parent
ROOT = LAB / 'LINE_history_5'
rows = [json.loads(line) for line in (ROOT / 'replies.jsonl').read_text().splitlines()]
assert len(rows) == 5
assert [r['history_messages'] for r in rows] == [0, 2, 4, 6, 6]
assert all(not r['warnings'] for r in rows)
assert (ROOT / 'LINE_history_5.png').is_file()

text = ['ประวัติคำถามต่อเนื่อง 5 ข้อใน LINE',
        'หัวข้อ: โรคใบด่างจากไวรัส CVMV ในพริก',
        'เอกสาร: คู่มือเกษตรกร รู้ไว้ใช้จริง ชุดความรู้ในการกำจัดศัตรูพืช หน้า 30',
        'วันที่: 12 กันยายน 2569 | บอต: 6610110554',
        'คำตอบต่อไปนี้ถอดจากข้อความที่บอตส่งเข้า LINE จริง ไม่ได้แก้คำตอบย้อนหลัง', '']
for i, row in enumerate(rows, 1):
    text += [f'{i}. ผู้ใช้: {row["question"]}', f'บอต: {row["answer"]}',
             f'จำนวนข้อความประวัติที่ใช้: {row["history_messages"]}', '']
text += ['หมายเหตุ: คำตอบข้อ 1 สะกด “หยอดหด” คลาดเคลื่อนจากต้นฉบับ “ใบยอดหด”',
         'ภาพและข้อความเก็บตามจริงเพื่อใช้ตรวจสอบการทำงานของระบบ']
(ROOT / 'questions_and_answers.txt').write_text('\n'.join(text), encoding='utf-8')

html = '''<!doctype html><html lang="th"><meta charset="utf-8">
<title>ประวัติคำถามต่อเนื่อง 5 ข้อใน LINE</title><style>
@font-face{font-family:Thai;src:url(file:///usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf)}
@page{size:A4;margin:10mm}*{box-sizing:border-box}body{margin:0;font-family:Thai,sans-serif;color:#111}
h1{font-size:18pt;margin:0 0 2mm}p{font-size:10pt;margin:1mm 0;line-height:1.5}
img{display:block;width:190mm;height:237mm;object-fit:contain;margin:3mm 0 1mm}
.small{font-size:8pt;color:#444}
</style><h1>ประวัติคำถามต่อเนื่อง 5 ข้อใน LINE</h1>
<p>โรคใบด่างจากไวรัส CVMV ในพริก · คู่มือกำจัดศัตรูพืช หน้า 30</p>
<p>บอต 6610110554 · 12 กันยายน 2569 · ภาพจากส่วนขยาย LINE ใน Brave จริง</p>
<img src="LINE_history_5.png" alt="ประวัติถามตอบจริง 5 ข้อใน LINE">
<p class="small">ระบบใช้ประวัติ 0, 2, 4, 6, 6 ข้อความตามลำดับ เพื่อเข้าใจคำถามที่อ้างถึงบริบทเดิม</p>
<p class="small">หมายเหตุ: ข้อ 1 บอตสะกด “หยอดหด” คลาดเคลื่อนจากต้นฉบับ “ใบยอดหด”; ภาพคงคำตอบจริง</p></html>'''
(ROOT / 'LINE_history_5.html').write_text(html, encoding='utf-8')
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto((ROOT / 'LINE_history_5.html').as_uri())
    page.evaluate('document.fonts.ready')
    assert page.locator('img').evaluate('(img) => img.complete && img.naturalWidth > 0')
    page.pdf(path=str(ROOT / 'LINE_history_5.pdf'), print_background=True, prefer_css_page_size=True)
    browser.close()
files = ['LINE_history_5.png', 'LINE_history_5.pdf', 'LINE_history_5.html',
         'questions_and_answers.txt', 'source_page_30.txt', 'replies.jsonl']
with zipfile.ZipFile(LAB / 'LINE_history_5.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for name in files:
        z.write(ROOT / name, name)
print('Created PDF, transcript, and submission ZIP.')
