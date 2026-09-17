from pathlib import Path
import json, html, re, math
import networkx as nx
R=Path(__file__).resolve().parent
G=json.loads((R/'graph.json').read_text());N={n['id']:n for n in G['nodes']};E=G['relationships'];SEM=[e for e in E if e['layer']=='semantic']; meta=json.loads((R/'validation.json').read_text());V=json.loads((R/'views.json').read_text())
COLORS={'Crop':'#15956a','Disease':'#df6470','DiseaseGroup':'#bd536b','Pathogen':'#8d72c7','Biocontrol':'#00a6a1','Symptom':'#dfa52d','Treatment':'#397cce','Material':'#a87b4f','Section':'#587287','Document':'#254353','Soil':'#926b47','Pest':'#d67832','Environment':'#688a9b','Season':'#799c50','Chemical':'#9e67a5'}
def nid(name,kind=None):return next(k for k,n in N.items() if n['name']==name and (kind is None or n['kind']==kind))
def pair(a,b):return [e for e in E if e['source']==a and e['target']==b]
def subedges(ids):return [e for e in SEM if e['source'] in ids and e['target'] in ids]
views=[]
def register(key,title,subtitle,pos,edges,w,h):
 views.append(dict(key=key,title=title,subtitle=subtitle,positions=pos,edges=[e['id'] for e in edges],width=w,height=h))
# Overview: actual document/section/semantic paths; representative subgraph, all 10 chapters.
chapters=[('การห่มดินแก้ดินเค็ม','ดินเค็ม','Soil','การห่มดิน','Treatment'),('การจัดการดินเปรี้ยว','ดินเปรี้ยว','Soil','เปลี่ยนมาใช้ปุ๋ยอินทรีย์','Treatment'),('น้ำส้มควันไม้','น้ำส้มควันไม้','Material','พักให้ตกตะกอนและแยกชั้น','Preparation'),('โรคข้าวโพด','ข้าวโพด','Crop','โรคราน้ำค้าง (ข้าวโพด)','Disease'),('แคงเกอร์ในมะนาว','มะนาว','Crop','โรคแคงเกอร์ (มะนาว)','Disease'),('โรคพริก','พริก','Crop','โรคใบด่าง CVMV (พริก)','Disease'),('ราน้ำค้างฤดูหนาว','พืชตระกูลแตง','Crop','กลุ่มโรคราน้ำค้าง','DiseaseGroup'),('เชื้อราที่มากับฝน','การจัดการเชื้อราฤดูฝน','Treatment','จัดระบบระบายน้ำไม่ให้ท่วมขัง','Treatment'),('สูตรกำจัดวัชพืช','วัชพืช','Pest','สูตรน้ำส้มสายชู','Treatment'),('รากเน่าโคนเน่าในทุเรียน','ทุเรียน','Crop',V['durian'],'Disease')]
doc=nid('คู่มือเกษตรกร: ความรู้ในการกำจัดโรคพืช (2562)');pos={doc:[175,850]};es=[]
for i,(c,a,ak,b,bk) in enumerate(chapters):
 y=245+i*132;s=nid(c,'Section');aa=nid(a,ak);bb=nid(b,bk);pos[s]=[620,y];pos[aa]=[1110,y];pos[bb]=[1670,y]
 es+=pair(doc,s)+pair(s,aa)+pair(aa,bb)
register('01_overview','ภาพรวมความรู้ครบ 10 บท','เส้นทางตัวแทนจากกราฟเต็ม • แยกโครงสร้างเอกสารกับความสัมพันธ์เชิงเนื้อหา • เลขหน้าใช้ลำดับหน้า PDF',pos,es,1980,1600)
# Virus detail with common-vector bridge and shared management.
pos={nid('พริก','Crop'):[160,820],nid('เพลี้ยอ่อน','Pest'):[1430,285]}
for i,name in enumerate(V['virus_diseases']):
 d=nid(name,'Disease'); y=380+i*320;pos[d]=[560,y]
 for e in SEM:
  if e['source']==d and e['relation'] in ['CAUSED_BY','HAS_SYMPTOM']:
   pos[e['target']]=[1020,y+(70 if e['relation']=='CAUSED_BY' else -70)]
for i,name in enumerate(['ถอนต้นที่เป็นโรคไปเผาทำลาย','ควบคุมเพลี้ยอ่อน','กำจัดซากพืชและวัชพืช','ปลูกพืชหมุนเวียน']):pos[nid(name,'Treatment')]=[1910,440+i*260]
register('02_chili_virus_detail','เจาะลึกพริก: ไวรัส 4 ชนิด เชื่อมด้วยเพลี้ยอ่อน','แยก CVMV / PVY / CMV / TEV • แชร์โหนดพาหะและวิธีจัดการ • หลักฐาน PDF หน้า 30–33',pos,subedges(pos),2210,1590)
# Durian: one disease and condition-specific management; shared biocontrol is true graph bridge.
d=nid(V['durian'],'Disease');pos={nid('ทุเรียน','Crop'):[150,740],d:[550,740]}
first=[e for e in SEM if e['source']==d];order={'HAS_SYMPTOM':0,'HAS_ENVIRONMENT':1,'HAS_SEASON':2,'CO_OCCURS_WITH':3,'MANAGED_BY':4}
first.sort(key=lambda e:order[e['relation']])
for i,e in enumerate(first):pos[e['target']]=[1130,300+i*155]
bi=nid('ใช้ชีวภัณฑ์เมื่อโรคไม่รุนแรง','Treatment'); severe=nid('ขูดเปลือกและทาแผลเมื่อโรครุนแรง','Treatment')
for i,e in enumerate([e for e in SEM if e['source']==bi]):pos[e['target']]=[1760,600+i*180]
for i,e in enumerate([e for e in SEM if e['source']==severe]):pos[e['target']]=[1760,1020+i*145]
register('03_durian_detail','เจาะลึกทุเรียน: รากเน่าโคนเน่า','แยกวิธีจัดการตามความรุนแรง • ชีวภัณฑ์เป็นคนละประเภทกับเชื้อสาเหตุ • หลักฐาน PDF หน้า 46–48',pos,subedges(pos),2070,1580)
# Shared biological agent bridge between crops.
root=nid('โรครากเน่าโคนเน่า (พริก)','Disease');bio=nid('ไตรโคเดอร์มา ฮาร์เซียนัม','Biocontrol');bac=nid('บาซิลลัส ซับทีลิส','Biocontrol')
ids=[nid('พริก','Crop'),root,nid('รองก้นหลุมด้วยชีวภัณฑ์','Treatment'),bio,bac,nid('ใช้ชีวภัณฑ์เมื่อโรคไม่รุนแรง','Treatment'),d,nid('ทุเรียน','Crop')]
coords=[[170,460],[570,460],[980,460],[1420,330],[1420,600],[980,850],[570,850],[170,850]]
pos=dict(zip(ids,coords))
register('04_shared_biocontrol','จุดเชื่อมข้ามพืช: ชีวภัณฑ์ร่วมกัน','พริกและทุเรียนเชื่อมผ่านชีวภัณฑ์ที่เอกสารระบุ • ไม่ได้หมายความว่าเป็นโรคจากเชื้อชนิดเดียวกัน • หน้า 29 และ 47',pos,subedges(pos),1750,1150)
# All semantic nodes: fixed reproducible layout for exploration, no artificial edges.
net=nx.Graph();net.add_edges_from((e['source'],e['target']) for e in SEM)
components=sorted(nx.connected_components(net),key=len,reverse=True);pos={}
for i,component in enumerate(components):
 positions=nx.spring_layout(net.subgraph(component),seed=41,k=1.1/math.sqrt(len(component)),iterations=220)
 cx,cy,scale=(1600,1300,1000) if i==0 else (3050+(i%2)*650,350+(i//2)*730,260)
 for k,(x,y) in positions.items():pos[k]=[float(cx+x*scale),float(cy+y*scale)]
register('05_full_semantic','กราฟความสัมพันธ์เชิงเนื้อหาทั้งหมด',f"{meta['semantic_nodes']} โหนด • {meta['semantic_relationships']} ความสัมพันธ์ • {meta['semantic_components']} กลุ่มที่เชื่อมกัน • คลิกโหนดหรือค้นหาเพื่อดูเส้นทาง",pos,SEM,4300,2650)
ED={e['id']:e for e in E}
def esc(x):return html.escape(str(x),quote=True)
def wrap(s,maxunits=20):
 from pythainlp.tokenize import word_tokenize
 from PIL import ImageFont
 font=ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf',18)
 chunks=[];line=''
 for word in word_tokenize(s,engine='newmm',keep_whitespace=True):
  if line and font.getlength(line+word)>240 and word not in [')',']',',']:
   chunks.append(line.strip());line=word.lstrip()
  else:line+=word
 if line:chunks.append(line.strip())
 return chunks

REL={'HAS_SECTION':'มีบท','COVERS':'ครอบคลุม','HAS_DISEASE':'มีโรค','CAUSED_BY':'เกิดจาก','HAS_SYMPTOM':'มีอาการ','MANAGED_BY':'จัดการโดย','TRANSMITTED_BY':'มีพาหะ','USES_BIOCONTROL':'ใช้ชีวภัณฑ์','USES_CHEMICAL':'ใช้สาร','USES':'ใช้','HAS_ENVIRONMENT':'พบในสภาพ','HAS_SEASON':'พบในฤดู','CO_OCCURS_WITH':'พบร่วมกับ','INCLUDES':'ประกอบด้วย','HAS_PREPARATION':'เตรียมโดย'}
def svg(v):
 w,h=v['width'],v['height'];full=v['key']=='05_full_semantic';out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(v["title"])}">', '<style>text{font-family: Noto Sans Thai, Noto Looped Thai, sans-serif}</style><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8394a4"/></marker></defs>',f'<rect width="{w}" height="{h}" fill="#f6f8fc"/>',f'<rect x="0" y="0" width="{w}" height="166" fill="#122b3b"/>',f'<text x="55" y="48" fill="#76d5bf" font-size="18" letter-spacing="2">PLANT KNOWLEDGE GRAPH / SOURCE-REVIEWED</text>',f'<text x="55" y="99" fill="white" font-size="35" font-weight="700">{esc(v["title"])}</text>',f'<text x="55" y="138" fill="#c1d5df" font-size="19">{esc(v["subtitle"])}</text>', '<g class="edges">']
 for eid in v['edges']:
  e=ED[eid]; x1,y1=v['positions'][e['source']];x2,y2=v['positions'][e['target']]
  dx=x2-x1;dy=y2-y1
  if full:
   path=f'M {x1} {y1} L {x2} {y2}';lx=(x1+x2)/2;ly=(y1+y2)/2
  else:
   sign=1 if dx>=0 else -1;sx=x1+sign*137;tx=x2-sign*145
   if abs(dx)<280:sx=x1;tx=x2; y1+=45;y2-=45
   bend=max(70,abs(tx-sx)*.47);path=f'M {sx} {y1} C {sx+sign*bend} {y1}, {tx-sign*bend} {y2}, {tx} {y2}';lx=(sx+tx)/2;ly=(y1+y2)/2
  if v['key']=='02_chili_virus_detail' and e['relation'] in ['MANAGED_BY','TRANSMITTED_BY']:
   path=f'M {sx} {y1} L 1200 {y1} C 1250 {y1}, {tx-50} {y2}, {tx} {y2}'
  dash=' stroke-dasharray="7 5"' if e['layer']!='semantic' else ''
  out.append(f'<g class="edge" data-id="{eid}" data-source="{e["source"]}" data-target="{e["target"]}"><title>{esc(e["relation"])} | PDF {esc(e["pages"])}</title><path d="{path}" fill="none" stroke="#9cacbb" stroke-width="{1.5 if full else 2}" marker-end="url(#arrow)"{dash}/><path d="{path}" fill="none" stroke="transparent" stroke-width="16"/>')
  # Dense virus panel uses relationship labels on hover plus a key, leaving paths legible.
  if not full and (v['key']!='02_chili_virus_detail' or e['relation'] in ['HAS_DISEASE','CAUSED_BY','HAS_SYMPTOM']) and e['relation']!='HAS_SECTION':
   label=REL.get(e['relation'],e['relation']);out.append(f'<text x="{lx}" y="{ly-7}" text-anchor="middle" font-size="14" fill="#536c7d" stroke="#f6f8fc" stroke-width="5" paint-order="stroke">{esc(label)}</text>')
  out.append('</g>')
 out.append('</g><g class="nodes">')
 if v['key']=='02_chili_virus_detail':out.append('<text x="55" y="210" font-size="21" fill="#3a5a6d">โรค → เพลี้ยอ่อน: TRANSMITTED_BY (มีพาหะ)  |  โรค → วิธีจัดการ: MANAGED_BY (จัดการโดย)</text>')
 for k,(x,y) in v['positions'].items():
  n=N[k];color=COLORS.get(n['kind'],'#758da1');name=n['name'];lines=wrap(name,23 if not full else 17)
  if full:
   out.append(f'<g class="node" data-id="{k}" tabindex="0"><title>{esc(name)} ({n["kind"]})</title><circle cx="{x}" cy="{y}" r="12" fill="{color}"/><text x="{x+17}" y="{y+5}" font-size="13" fill="#233d4e">{esc(name)}</text></g>');continue
  height=max(88,42+len(lines)*24)
  out.append(f'<g class="node" data-id="{k}" tabindex="0"><title>{esc(name)} ({n["kind"]})</title><rect x="{x-140}" y="{y-height/2}" width="280" height="{height}" rx="14" fill="white" stroke="{color}" stroke-width="2"/><rect x="{x-140}" y="{y-height/2+14}" width="5" height="{height-28}" rx="2" fill="{color}"/><text x="{x-122}" y="{y-height/2+23}" font-size="12" font-weight="700" fill="{color}">{n["kind"].upper()}</text>')
  for j,line in enumerate(lines):out.append(f'<text x="{x-122}" y="{y-height/2+49+j*24}" font-size="18" font-weight="500" fill="#203b4d">{esc(line)}</text>')
  out.append('</g>')
 out.append('</g>')
 legend=' · '.join(['พืช: เขียว','โรค: แดง','สาเหตุ: ม่วง','อาการ: เหลือง','วิธีจัดการ: น้ำเงิน','ชีวภัณฑ์: ฟ้าอมเขียว'])
 out.append(f'<text x="55" y="{h-61}" font-size="17" fill="#40586a">{legend}</text><text x="55" y="{h-30}" font-size="15" fill="#697e8f">{len(v["positions"])} โหนด / {len(v["edges"])} เส้นในภาพนี้ • เส้นประ = โครงสร้างเอกสาร • ข้อความเป็นข้อมูลที่คู่มือปี 2562 รายงาน</text></svg>')
 return ''.join(out)
svgs={}
for v in views:
 s=svg(v);svgs[v['key']]=s;(R/(v['key']+'.svg')).write_text(s)
(R/'view_manifest.json').write_text(json.dumps(views,ensure_ascii=False,indent=2))
style='''*{box-sizing:border-box}body{margin:0;font-family:"Noto Sans Thai","Noto Looped Thai",sans-serif;color:#213e4d;background:#edf2f7}header{padding:24px 32px;background:#122b3b;color:white}header h1{margin:0 0 8px;font-size:25px}header p{margin:0;color:#bbd3e0;font-size:14px}.stats{display:flex;gap:10px;margin-top:17px;flex-wrap:wrap}.stats span{background:#234655;padding:7px 12px;border-radius:7px}nav{padding:14px 22px;display:flex;gap:8px;flex-wrap:wrap;background:white}button,input{font:inherit;border:1px solid #c5d3df;border-radius:6px;padding:8px 12px;background:white;color:#284b62}button{cursor:pointer}button.active{background:#185d75;color:white}.toolbar{display:flex;gap:8px;padding:10px 22px;align-items:center;flex-wrap:wrap}input{width:280px}main{display:grid;grid-template-columns:minmax(0,1fr) 345px;gap:12px;padding:0 18px 20px}#viewport{height:78vh;overflow:hidden;background:#f6f8fc;border-radius:12px;border:1px solid #c9d6df;touch-action:none}#viewport svg{width:100%;height:100%;font-family:"Noto Sans Thai","Noto Looped Thai",sans-serif;user-select:none}.node,.edge{cursor:pointer}.dim{opacity:.10}.selected rect,.selected circle{stroke:#ff6a00!important;stroke-width:4!important}aside{background:white;border-radius:12px;padding:20px;overflow:auto;height:78vh;font-size:14px}aside h2{font-size:19px;margin:0 0 12px}aside h3{font-size:15px}aside pre{white-space:pre-wrap;line-height:1.65;font-family:inherit;font-size:13px;background:#f3f6fa;padding:10px;border-radius:6px}a{color:#146e88}.relrow{padding:10px 0;border-bottom:1px solid #dce4eb}.pill{font-size:11px;background:#eaf1f7;padding:3px 6px;border-radius:5px}.notice{font-size:12px;line-height:1.7;color:#607787}#results{max-height:160px;overflow:auto}.result{display:block;margin:4px 0;width:100%;text-align:left;font-size:13px}@media(max-width:900px){main{grid-template-columns:1fr}aside{height:45vh}#viewport{height:65vh}}'''
body='''<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Neo4j | ความรู้ในการกำจัดโรคพืช</title><style>STYLE</style><header><h1>กราฟความรู้จากคู่มือการกำจัดโรคพืช</h1><p>ภาพรวม • ภาพเจาะลึก • ตรวจหลักฐานรายเส้น • ชุดข้อมูลเดียวกับไฟล์นำเข้า Neo4j</p><div class="stats">STATS</div></header><nav id="tabs"></nav><div class="toolbar"><input id="search" placeholder="ค้นหาพืช โรค เชื้อ หรือวิธีจัดการ" aria-label="ค้นหาโหนด"><button id="zin">ขยาย +</button><button id="zout">ย่อ −</button><button id="reset">พอดีจอ</button><button id="clear">ยกเลิกเลือก</button><a id="export" download>ดาวน์โหลดภาพ SVG</a><a href="import.cypher" download>Cypher นำเข้า</a></div><main><div id="viewport"></div><aside id="detail"></aside></main><script>const graph=GRAPH, views=VIEWS, svgs=SVGS;SCRIPT</script></html>'''
js='''const N=Object.fromEntries(graph.nodes.map(n=>[n.id,n])),E=Object.fromEntries(graph.relationships.map(e=>[e.id,e]));const viewport=document.getElementById('viewport'),detail=document.getElementById('detail');let current,box,start;const h=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function instructions(){detail.innerHTML='<h2>สำรวจกราฟ</h2><p>คลิกโหนดเพื่อดูความสัมพันธ์ทั้งหมด คลิกเส้นเพื่อดูหน้า PDF และข้อความหลักฐาน</p><p>ลากพื้นหลังเพื่อเลื่อน ใช้ปุ่ม + / − เพื่อซูม ค้นหาได้ทั้งกราฟ</p><div id="results"></div><h3>อ่านกราฟอย่างไร</h3><p>ลูกศรบอกทิศทางความสัมพันธ์ เส้นประเป็นโครงสร้างเอกสาร ส่วนเส้นทึบเป็นความสัมพันธ์เชิงเนื้อหา</p><p>กราฟไวรัส: พริก → HAS_DISEASE → โรค; โรค → CAUSED_BY → ไวรัส; โรค → HAS_SYMPTOM → อาการ; โรค → TRANSMITTED_BY → เพลี้ยอ่อน; โรค → MANAGED_BY → วิธีจัดการ</p><p class="notice">ข้อมูลสะท้อนเนื้อหาคู่มือปี 2562; การทบทวนโดย AI ไม่ใช่การยืนยันประสิทธิผลทางวิชาการ ภาพนี้จัดวางจากชุดข้อมูลสำหรับ Neo4j ไม่ใช่ภาพหน้าจอ Neo4j Browser</p><p><a href="README.md">รายละเอียดและวิธีนำเข้า</a> · <a href="graph.json" download>ข้อมูลทั้งหมด</a> · <a href="graph_report.pdf">รายงาน PDF</a></p>'}function apply(){viewport.querySelector('svg').setAttribute('viewBox',box.join(' '))}function load(v){current=v;viewport.innerHTML=svgs[v.key];box=[0,0,v.width,v.height];apply();document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.key===v.key));document.getElementById('export').href=v.key+'.svg';instructions()}function focus(id){let group=viewport.querySelector('.node[data-id="'+id+'"]');if(!group){const target=views.at(-1).positions[id]?views.at(-1):views.find(v=>v.positions[id]);if(target)load(target);group=viewport.querySelector('.node[data-id="'+id+'"]')}const adj=new Set([id]);graph.relationships.filter(e=>e.source===id||e.target===id).forEach(e=>{adj.add(e.source);adj.add(e.target)});viewport.querySelectorAll('.node').forEach(n=>{n.classList.toggle('dim',!adj.has(n.dataset.id));n.classList.toggle('selected',n.dataset.id===id)});viewport.querySelectorAll('.edge').forEach(e=>e.classList.toggle('dim',e.dataset.source!==id&&e.dataset.target!==id));let n=N[id];let rows=graph.relationships.filter(e=>(e.source===id||e.target===id)&&e.layer==='semantic');detail.innerHTML='<h2>'+h(n.name)+'</h2><span class="pill">'+h(n.kind)+'</span><p>'+rows.length+' ความสัมพันธ์เชิงเนื้อหาในกราฟเต็ม</p>'+rows.map(e=>'<div class="relrow"><b>'+h(N[e.source].name)+'</b><br><button class="result" data-edge="'+e.id+'">'+h(e.relation)+' → '+h(N[e.target].name)+'</button><small>PDF หน้า '+e.pages.join(', ')+'</small></div>').join('');if(current.key==='05_full_semantic'&&current.positions[id]){const [x,y]=current.positions[id];box=[x-650,y-420,1300,840];apply()}}function edge(id){const e=E[id];detail.innerHTML='<h2>'+h(N[e.source].name)+'</h2><b>'+h(e.relation)+'</b><p>→ '+h(N[e.target].name)+'</p><p>ชั้นข้อมูล: '+h(e.layer)+'</p>'+(e.condition?'<p>เงื่อนไข: '+h(e.condition)+'</p>':'')+(e.context?'<p>'+h(e.context)+'</p>':'')+'<p>หน้า PDF: '+e.pages.map(p=>'<a href="../9-file-4786.pdf#page='+p+'" target="_blank">'+p+'</a>').join(' · ')+'</p><p class="notice">หลักฐานเก็บเป็นข้อความทั้งหน้าเพื่อรักษาบริบท รวมหน้าก่อนหน้าในกรณีชื่อพืชหรือหัวข้ออยู่คนละหน้า การมีเลขหน้าไม่ได้เท่ากับการยืนยันข้อกล่าวอ้าง</p><pre>'+h(e.evidence)+'</pre>'}views.forEach(v=>{let b=document.createElement('button');b.textContent=v.title;b.dataset.key=v.key;b.onclick=()=>load(v);document.getElementById('tabs').append(b)});viewport.onclick=e=>{const n=e.target.closest('.node'),r=e.target.closest('.edge');if(n)focus(n.dataset.id);else if(r)edge(r.dataset.id)};detail.onclick=e=>{const b=e.target.closest('[data-edge]');if(b)edge(b.dataset.edge);const n=e.target.closest('[data-node]');if(n)focus(n.dataset.node)};function zoom(f){let [x,y,w,h]=box;box=[x+w*(1-f)/2,y+h*(1-f)/2,w*f,h*f];apply()}document.getElementById('zin').onclick=()=>zoom(.75);document.getElementById('zout').onclick=()=>zoom(1.3);document.getElementById('reset').onclick=()=>{box=[0,0,current.width,current.height];apply()};document.getElementById('clear').onclick=()=>load(current);viewport.onpointerdown=e=>{if(e.target.closest('.node,.edge'))return;start={x:e.clientX,y:e.clientY,b:[...box]};viewport.setPointerCapture(e.pointerId)};viewport.onpointermove=e=>{if(!start)return;box=[start.b[0]-(e.clientX-start.x)*start.b[2]/viewport.clientWidth,start.b[1]-(e.clientY-start.y)*start.b[3]/viewport.clientHeight,start.b[2],start.b[3]];apply()};viewport.onpointerup=()=>start=null;document.getElementById('search').oninput=e=>{instructions();let q=e.target.value.trim().toLowerCase();if(!q)return;document.getElementById('results').innerHTML=graph.nodes.filter(n=>n.kind!=='Page'&&n.name.toLowerCase().includes(q)).slice(0,30).map(n=>'<button class="result" data-node="'+n.id+'">'+h(n.name)+' · '+n.kind+'</button>').join('')};load(views[0]);'''
stats=''.join(f'<span>{v:,} {label}</span>' for v,label in [(meta['semantic_nodes'],'โหนดเนื้อหา'),(meta['semantic_relationships'],'เส้นความสัมพันธ์เนื้อหา'),(10,'บท'),(48,'หน้า PDF')])
body=body.replace('STYLE',style).replace('STATS',stats).replace('GRAPH',json.dumps(G,ensure_ascii=False).replace('</','<\\/')).replace('VIEWS',json.dumps(views,ensure_ascii=False)).replace('SVGS',json.dumps(svgs,ensure_ascii=False)).replace('SCRIPT',js)
(R/'index.html').write_text(body,encoding='utf-8')
# Printable report uses the same SVG data, no rasterized text in vector output.
report='<!doctype html><html lang="th"><meta charset="utf-8"><style>body{font-family:"Noto Sans Thai","Noto Looped Thai",sans-serif;margin:0}section{page-break-after:always}svg{width:100%;height:245mm}h1,p{margin:15px 25px}</style>'
report+='<section><h1>กราฟความรู้สำหรับ Neo4j</h1><p>คู่มือเกษตรกร: ความรู้ในการกำจัดโรคพืช (2562) — PDF 48 หน้า</p><p>เนื้อหา 207 โหนด / 263 ความสัมพันธ์; รวมเอกสาร บท และหลักฐาน 266 โหนด / 879 ความสัมพันธ์</p><p>ครอบคลุม 10 บท; กราฟเนื้อหามี 7 กลุ่มที่แยกจากกัน กลุ่มใหญ่สุด 165 โหนด; รวมโครงสร้างเอกสารแล้วเชื่อมถึงกันทั้งหมด</p><p>ภาพจัดวางจากข้อมูลสำหรับ Neo4j ไม่ใช่ภาพหน้าจอ Neo4j Browser และไม่ได้รัน BERT/SBERT ใหม่</p><p>วิธีสร้าง: อ่าน PDF ทบทวนชื่อและบริบทโดย AI ใช้ชื่อโรคแยกตามพืช/ไวรัส รวมวิธีจัดการที่เป็นการกระทำเดียวกัน และเก็บหลักฐานรายหน้าไว้ตรวจกลับ</p><p>ทุกข้อกล่าวอ้างเป็นสิ่งที่เอกสารปี 2562 รายงาน ไม่ใช่การยืนยันผลทางวิชาการหรือคำแนะนำสารเคมีปัจจุบัน</p><p>ชื่อไฟล์: index.html สำหรับสำรวจ; import.cypher สำหรับนำเข้า; graph.json/CSV สำหรับข้อมูล; validation.json สำหรับผลตรวจ</p></section>'
for v in views[:4]:report+='<section>'+svgs[v['key']]+'</section>'
(R/'report.html').write_text(report,encoding='utf-8')
print('Built interactive HTML and',len(views),'SVG views')
