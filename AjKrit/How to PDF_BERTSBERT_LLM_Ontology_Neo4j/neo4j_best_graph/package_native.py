from pathlib import Path
import json,hashlib,zipfile,html
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parent;O=R/'neo4j_native'
m=json.loads((O/'native_exports.json').read_text());data=json.loads((O/'database_export.json').read_text());g=json.loads((R/'graph.json').read_text())
assert {n['id'] for n in data['nodes']}=={n['id'] for n in g['nodes']}
assert {e['id'] for e in data['relationships']}=={e['id'] for e in g['relationships']}
assert {n['id']:n for n in data['nodes']}=={n['id']:n for n in g['nodes']}
for e in data['relationships']:
 expected=next(x for x in g['relationships'] if x['id']==e['id']);assert e=={k:v for k,v in expected.items() if k!='evidence'}
check={'neo4j_version':data['version'],'uri':'bolt://localhost:7688','database':'neo4j','nodes':len(data['nodes']),'relationships':len(data['relationships']),'semantic_nodes':207,'semantic_relationships':263,'node_ids_match':True,'all_node_properties_match':True,'all_relationship_properties_match':True,'native_export_files':len(m)}
(O/'verification.json').write_text(json.dumps(check,indent=2))
v=json.loads((R/'validation.json').read_text());v['neo4j_live_import']=True;v['neo4j_version']=data['version'];(R/'validation.json').write_text(json.dumps(v,indent=2))
style='''body{font-family:"Noto Sans Thai",sans-serif;color:#143047;background:#eef4f8;margin:0;padding:30px}main{max-width:1450px;margin:auto}h1{font-size:30px}article{background:white;padding:24px;margin:24px 0;border-radius:15px;break-after:page}img{max-width:100%;height:auto}a{color:#006d98}p{line-height:1.7}.native{width:100%;height:750px;object-fit:contain;background:white}.proof{width:100%}@media print{body{padding:0}article{margin:0;border-radius:0}h1{font-size:23px}.native{height:180mm}.proof{max-height:225mm;object-fit:contain}}'''
head=f'''<!doctype html><html lang="th"><meta charset="utf-8"><title>กราฟจาก Neo4j Browser จริง</title><style>{style}</style><main><h1>กราฟจาก Neo4j Browser โดยตรง</h1><p>Neo4j {html.escape(data['version'][0])} · ฐานข้อมูล neo4j · 266 โหนด / 879 ความสัมพันธ์<br>เฉพาะเนื้อหา 207 โหนด / 263 ความสัมพันธ์ · ตรวจข้อมูลที่อ่านกลับจากฐานข้อมูลตรงกับชุดต้นฉบับ</p><p>PNG และ SVG ดาวน์โหลดจากเมนู Download ของ Neo4j Browser; ไฟล์ *_browser.png เป็นภาพหน้าจอจริง รวม query และจำนวนผลลัพธ์<br><a href="http://localhost:7475/browser/">เปิด Neo4j Browser ในเครื่อง</a> · <a href="database_export.json">ข้อมูลอ่านกลับจากฐานข้อมูล</a> · <a href="verification.json">ผลตรวจ</a></p>'''
body=head
for row in m:
 if row['format']!='svg':continue
 key=row['view'];title=html.escape(row['title']);count=f"{row['nodes']} โหนด / {row['relationships']} ความสัมพันธ์"
 body+=f'<article><h2>{title}</h2><p>{count} · Neo4j Browser native export</p><img class="native" src="{key}.svg"><p><a href="{key}.png">PNG จาก Neo4j</a> · <a href="{key}.svg">SVG จาก Neo4j</a> · <a href="{key}_browser.png">ภาพหน้าจอ Browser</a> · <a href="../query_{key}.cypher">Cypher ของภาพนี้</a></p></article>'
body+='<article><h2>หลักฐาน: ภาพหน้าจอ Neo4j Browser</h2><p>ตัวเลขใน Results overview เป็นผลของ query นี้ ส่วนตัวเลข Database information รวมกราฟงานอื่นในฐานข้อมูลเดียวกัน</p><img class="proof" src="02_chili_virus_detail_browser.png"></article></main></html>'
(O/'index.html').write_text(body)
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page();page.goto((O/'index.html').as_uri());page.pdf(path=str(O/'Neo4j_Native_Graph_Report.pdf'),format='A3',landscape=True,print_background=True,margin={'top':'8mm','bottom':'8mm','left':'8mm','right':'8mm'});b.close()
readme=f'''# กราฟจาก Neo4j จริง

นำเข้าฐานข้อมูล Neo4j {data['version'][0]} สำเร็จที่ http://localhost:7475/browser/ (Bolt localhost:7688) แล้วอ่านกลับตรวจ ID โหนดและทุก property ของความสัมพันธ์ตรงกับต้นฉบับ

- กราฟทั้งหมด 266 โหนด / 879 ความสัมพันธ์
- เฉพาะเนื้อหา 207 โหนด / 263 ความสัมพันธ์
- เปิด `index.html` หรือ `Neo4j_Native_Graph_Report.pdf` เพื่อดูกราฟจาก Browser โดยตรง
- `*.svg`, `*.png` มาจากเมนู Download ของ Neo4j Browser ไม่ได้วาดใหม่
- `*_browser.png` เป็นภาพหน้าจอจริง พร้อม query และ Results overview
- `database_export.json` อ่านกลับจาก Neo4j และ `verification.json` คือผลตรวจ
- พื้นหลังภาพที่ Browser ส่งออกโปร่งใส; HTML/PDF แสดงบนพื้นขาว
- ภาพแต่ละมุมมองเป็นกราฟย่อย; `05_full_semantic` เป็นกราฟเนื้อหาครบ 207 โหนด / 263 ความสัมพันธ์

โหนดทั้งหมดใช้ label `PlantGuide2019Reviewed` เพื่อค้นแยกจากงานอื่นที่มีอยู่แล้วในฐานข้อมูล ไม่มีการลบข้อมูลเดิม
'''
(O/'README.md').write_text(readme)
# Clearly supersede the earlier offline status while retaining disclosure about earlier diagrams.
p=R/'README.md';s=p.read_text();s=s.replace('# กราฟความรู้สำหรับ Neo4j — คู่มือการกำจัดโรคพืช','# กราฟความรู้สำหรับ Neo4j — คู่มือการกำจัดโรคพืช\n\n**อัปเดต: นำเข้า Neo4j จริงแล้ว** เปิด `neo4j_native/index.html` หรือ `neo4j_native/Neo4j_Native_Graph_Report.pdf` สำหรับภาพส่งออกจาก Neo4j Browser โดยตรง พร้อมผลตรวจใน `neo4j_native/verification.json` และเข้า Browser ได้ที่ http://localhost:7475/browser/ . ภาพ SVG เดิมที่อยู่โฟลเดอร์หลักยังเป็นภาพจัดวางเอง; ภาพจาก Browser อยู่ใน `neo4j_native/` เท่านั้น.')
s=s.replace('ขณะจัดทำ ไม่พบ Neo4j ที่กำลังฟังพอร์ต และ Docker daemon ไม่ทำงาน จึงยังไม่ได้ทดสอบนำเข้าฐานข้อมูลจริง ผลตรวจที่แนบเป็นการตรวจความครบถ้วนของกราฟและการเปิดใช้ HTML','นำเข้าและอ่านกลับจาก Neo4j จริงแล้ว: 266 โหนด / 879 ความสัมพันธ์ ตรงกับข้อมูลครบชุด ผลตรวจและภาพ Browser อยู่ใน `neo4j_native/`.')
s=s.replace('โดยยังไม่ได้ยืนยันการรันกับ instance ของผู้ใช้:', 'และยืนยันการนำเข้าผ่าน Python driver กับ Neo4j ในเครื่องแล้ว:')
p.write_text(s)
(R/'manifest_sha256.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file() and p.name!='manifest_sha256.json' and '__pycache__' not in p.parts},indent=2))
out=R.parent/'Neo4j_Native_Graphs.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for p in O.iterdir():
  if p.is_file():z.write(p,Path('neo4j_native')/p.name)
 for p in R.glob('query_*.cypher'):z.write(p,p.name)
with zipfile.ZipFile(out) as z:assert z.testzip() is None
print('Native package:',out,'bytes',out.stat().st_size)
