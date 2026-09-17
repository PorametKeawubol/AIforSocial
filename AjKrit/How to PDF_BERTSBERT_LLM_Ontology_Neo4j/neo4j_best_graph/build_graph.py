"""Source-reviewed graph; no embedding/LLM inference is run by this builder."""
from pathlib import Path
import json, hashlib, csv, re, collections
ROOT=Path(__file__).resolve().parent
PAGES={p['page']:p['text'] for p in json.loads((ROOT/'source_pages.json').read_text())}
SHA='a234df1a468da4ea3fbc48f9617d8a42bfe56948ae9438234c92cc39b6591d49'
NS='PlantGuide2019Reviewed'
nodes={}; edges={}
def uid(text): return hashlib.sha256(text.encode()).hexdigest()[:20]
def node(name,kind,**props):
 key=uid(SHA+'|'+kind+'|'+name)
 nodes.setdefault(key,dict(id=key,name=name,kind=kind,dataset=NS,**props))
 return key
def add(s,sk,r,t,tk,pages,**props):
 pp=sorted(set([pages] if isinstance(pages,int) else pages))
 assert pp and all(p in PAGES for p in pp)
 a=node(s,sk); b=node(t,tk); k=uid(a+'|'+r+'|'+b)
 if k not in edges: edges[k]=dict(id=k,source=a,target=b,relation=r,pages=[],evidence='',source_sha256=SHA,assertion='reported_in_source',**props)
 e=edges[k]; e['pages']=sorted(set(e['pages']+pp));e['evidence']='\n\n'.join(f'[PDF page {p}]\n{PAGES[p]}' for p in e['pages'])
 return e
# Canonical treatment names intentionally share only the same action, not disease identity.
ROT='ปลูกพืชหมุนเวียน'; REMOVE='ถอนต้นที่เป็นโรคไปเผาทำลาย'; SPACE='เว้นระยะปลูกให้อากาศถ่ายเท'; DRAIN='จัดระบบระบายน้ำไม่ให้ท่วมขัง'; SOLAR='ไถพลิกและตากดิน'; CLEAN='กำจัดซากพืชและวัชพืช'; APHID='ควบคุมเพลี้ยอ่อน'; TRIC='ไตรโคเดอร์มา'; BAC='บาซิลลัส ซับทีลิส'
CHAPTERS=[('การห่มดินแก้ดินเค็ม',5,8),('การจัดการดินเปรี้ยว',9,11),('น้ำส้มควันไม้',12,15),('โรคข้าวโพด',16,19),('แคงเกอร์ในมะนาว',20,22),('โรคพริก',23,37),('ราน้ำค้างฤดูหนาว',38,40),('เชื้อราที่มากับฝน',41,42),('สูตรกำจัดวัชพืช',43,45),('รากเน่าโคนเน่าในทุเรียน',46,48)]
def disease(crop,name,page,cause=None,sym=None,manage=(),season=None):
 d=name+' ('+crop+')'
 # Chapter context supplies crop where a continuation page omits it.
 chapter=next((x for x in CHAPTERS if x[1]<=page<=x[2]),None)
 context=[page] if crop in PAGES[page] else [chapter[1],page]
 add(crop,'Crop','HAS_DISEASE',d,'Disease',context)
 if cause:add(d,'Disease','CAUSED_BY',cause,'Pathogen',page)
 if sym:add(d,'Disease','HAS_SYMPTOM',sym,'Symptom',page)
 for action in manage:add(d,'Disease','MANAGED_BY',action,'Treatment',page)
 if season:add(d,'Disease','HAS_SEASON',season,'Season',page)
 return d
# Soil and soil-cover chapter.
add('ดินเค็ม','Soil','MANAGED_BY','การห่มดิน','Treatment',[5,7,8])
for a,p in [('พรวนหน้าดิน',7),('โรยเมล็ดถั่วเขียว',7),('ปลูกพืชทนเค็ม',7),('คลุมดินด้วยวัสดุธรรมชาติ',8),('เติมปุ๋ยอินทรีย์บนวัสดุคลุมดิน',8)]:add('การห่มดิน','Treatment','INCLUDES',a,'Activity',p)
for a in ['ฟางข้าว','เศษใบไม้','พรมใยปาล์ม']:add('การห่มดิน','Treatment','USES',a,'Material',6)
for a in ['รักษาความชื้นในดิน','ลดวัชพืช','เพิ่มอินทรียวัตถุ']:add('การห่มดิน','Treatment','HAS_BENEFIT',a,'Benefit',6)
add('โรยเมล็ดถั่วเขียว','Activity','USES','ถั่วเขียว','Crop',7)
for a in ['ข้าวหอมมะลิ 105','ข้าวแดงน้อย']:add('ปลูกพืชทนเค็ม','Activity','USES',a,'Variety',7)
add('เติมปุ๋ยอินทรีย์บนวัสดุคลุมดิน','Activity','USES','ปุ๋ยอินทรีย์','Fertilizer',8)
for a in ['ลดการใช้ปุ๋ยเคมี','เปลี่ยนมาใช้ปุ๋ยอินทรีย์','ล้างดินด้วยน้ำหมักสมุนไพรรสจืด']:add('ดินเปรี้ยว','Soil','MANAGED_BY',a,'Treatment',[10,11])
add('เปลี่ยนมาใช้ปุ๋ยอินทรีย์','Treatment','USES','ปุ๋ยอินทรีย์','Fertilizer',10)
# Keep each concentration scoped to its own application record.
add('น้ำส้มควันไม้','Material','HAS_PREPARATION','พักให้ตกตะกอนและแยกชั้น','Preparation',13)
add('พักให้ตกตะกอนและแยกชั้น','Preparation','HAS_DURATION','ประมาณ 3 เดือน','Duration',13)
for ratio,purpose in [('1:20','พ่นดินก่อนปลูกเพื่อจัดการจุลินทรีย์และแมลง'),('1:50','พ่นดินเพื่อจัดการจุลินทรีย์ที่ทำลายพืช'),('1:100','ราดโคนเพื่อจัดการโรคราและโรคเน่า'),('1:200','พ่นใบและดินเพื่อไล่แมลงและป้องกันเชื้อรา'),('1:500','พ่นผลอ่อนและก่อนเก็บเกี่ยว'),('1:1000','ใช้เป็นสารจับใบ')]:
 a='การใช้น้ำส้มควันไม้ '+ratio
 add('น้ำส้มควันไม้','Material','HAS_APPLICATION',a,'Application',14)
 add(a,'Application','HAS_PURPOSE',purpose,'Purpose',14)
 nodes[node(a,'Application')]['dilution_as_printed']=ratio
# Maize.
disease('ข้าวโพด','โรคใบไหม้แผลใหญ่',17,'เชื้อรา','แผลยาวสีเทาหรือน้ำตาลอ่อนบนใบข้าวโพด', ['พ่นสารป้องกันกำจัดโรคใบไหม้แผลใหญ่'])
disease('ข้าวโพด','โรคราสนิม',17,sym='จุดนูนแตกเป็นผงสีสนิมบนใบข้าวโพด',manage=['พ่นสารป้องกันกำจัดโรคราสนิม'],season='ฤดูฝน')
md=disease('ข้าวโพด','โรคราน้ำค้าง',18,sym='ใบลายเขียวอ่อนหรือเหลืองสลับเขียวแก่และผงขาวใต้ใบ',manage=['คลุกเมล็ดด้วยสารป้องกันโรคราน้ำค้าง',REMOVE])
nodes[node(md,'Disease')]['aliases']=['โรคใบลาย']
disease('ข้าวโพด','โรคโคนเน่า',18,sym='ท่อลำเลียงสีน้ำตาล มีน้ำเมือกและกลิ่นเหม็น',manage=[REMOVE,DRAIN,ROT,'ใส่ปูนขาวบริเวณที่พบโรค'])
mosaic=disease('ข้าวโพด','โรคใบด่าง',19,'ไวรัส','ใบด่างลายเขียวซีดสลับเขียวเข้มและต้นแคระแกร็น',[APHID,ROT,'ปลูกข้าวโพดพันธุ์ต้านทาน'])
add(mosaic,'Disease','TRANSMITTED_BY','เพลี้ยอ่อน','Pest',19)
# Canker: only lime has the folk-remedy treatment links, not every citrus host.
can=disease('มะนาว','โรคแคงเกอร์',20,'เชื้อแบคทีเรีย',manage=[])
add(can,'Disease','HAS_ENVIRONMENT','ลมฝนแรง','Environment',20)
for a,p in [('ทาน้ำปูนกินหมากบริเวณลำต้น',21),('ฉีดพ่นน้ำปูนใส',22)]:
 add(can,'Disease','MANAGED_BY',a,'Treatment',[20,p],condition='การระบาดระยะแรก ตามภูมิปัญญาที่คู่มือรายงาน')
add('ทาน้ำปูนกินหมากบริเวณลำต้น','Treatment','USES','ปูนกินหมาก','Material',21)
add('ฉีดพ่นน้ำปูนใส','Treatment','USES','ปูนขาวก่อสร้าง','Material',22)
for crop in ['ส้มเขียวหวาน','ส้มโอ']:
 disease(crop,'โรคแคงเกอร์',20,'เชื้อแบคทีเรีย')
# Chili: preserve distinct disease/virus identities.
an=disease('พริก','โรคแอนแทร็กโนส',24,'เชื้อรา','แผลบุ๋มสีน้ำตาล ผลหดย่นคล้ายกุ้งแห้ง',[SPACE,'เลือกเมล็ดพันธุ์ปลอดโรค','แช่เมล็ดในน้ำอุ่น','ตัดกิ่งและผลที่เป็นโรคไปเผาทำลาย'])
nodes[node(an,'Disease')]['aliases']=['โรคกุ้งแห้ง']
add(an,'Disease','HAS_ENVIRONMENT','อุณหภูมิ 27–30 °C','Environment',24)
add('แช่เมล็ดในน้ำอุ่น','Treatment','HAS_CONDITION','50 °C นาน 30 นาที','Condition',[24,29])
wet=disease('พริก','โรคเน่าเปียก',25,'เชื้อรา','ยอด ใบอ่อน และดอกเน่าเละสีน้ำตาลดำ',[SPACE,'ตัดแต่งทรงพุ่มให้โปร่ง','ตัดกิ่งที่เป็นโรคไปเผาทำลาย'])
add(wet,'Disease','HAS_ENVIRONMENT','ความชื้นสูง','Environment',25)
fu=disease('พริก','โรคเหี่ยวจากฟูซาเรียม',26,'ฟูซาเรียม ออกซิสปอรัม','รากเน่า ใบล่างเหลืองลามขึ้นบน เหี่ยวและร่วง',[ROT,SOLAR,'ปรับสภาพดินด้วยปูนขาวและอินทรียวัตถุ'])
add(fu,'Disease','HAS_GROWTH_STAGE','ออกดอกและติดผล','GrowthStage',26)
ph=disease('พริก','โรคเหี่ยวจากไฟทอฟทอรา',27,'ไฟทอฟทอรา แคพไซไค','โคนและรากเน่า ผลฉ่ำน้ำ เนื้อผลสีดำ',[ROT,REMOVE,'ปรับสภาพดินด้วยปูนขาวและอินทรียวัตถุ','รองก้นหลุมด้วยจุลินทรีย์ปฏิปักษ์'])
add(ph,'Disease','HAS_GROWTH_STAGE','ระยะติดผล','GrowthStage',27)
for b in [TRIC,'บาซิลลัส']:add('รองก้นหลุมด้วยจุลินทรีย์ปฏิปักษ์','Treatment','USES_BIOCONTROL',b,'Biocontrol',27)
ba=disease('พริก','โรคเหี่ยวจากแบคทีเรีย',28,'เชื้อแบคทีเรีย','เหี่ยวกลางวัน ฟื้นกลางคืน และเหี่ยวตายทั้งที่ใบเขียว',[REMOVE,SOLAR,ROT,'ใส่ปุ๋ยอินทรีย์'])
rr=disease('พริก','โรครากเน่าโคนเน่า',29,'ราเมล็ดผักกาด','ต้นกล้าเน่า ใบเหลือง เหี่ยว ร่วง และยืนต้นตาย',[CLEAN,SOLAR,ROT,'แช่เมล็ดในน้ำอุ่น',REMOVE,'รองก้นหลุมด้วยชีวภัณฑ์'])
for b in ['ไตรโคเดอร์มา ฮาร์เซียนัม',BAC]:add('รองก้นหลุมด้วยชีวภัณฑ์','Treatment','USES_BIOCONTROL',b,'Biocontrol',29)
viruses=[]
for short,long,p,sym in [('CVMV','Chilli veinal mottle virus',30,'ใบด่างเขียวซีด ใบยอดหดสั้น ผลบิดเบี้ยว'),('PVY','Potato virus Y',31,'เส้นใบบวมใส ใบด่างหดย่น ต้นแคระแกร็น'),('CMV','Cucumber mosaic virus',31,'ใบด่างเขียวสลับเหลือง ผลผิดรูป ไม่ออกดอก'),('TEV','Tobacco etch virus',32,'เส้นใบใส ใบและผลด่างเป็นวงซ้อน')]:
 d=disease('พริก','โรคใบด่าง '+short,p,short,sym)
 nodes[node(short,'Pathogen')]['scientific_name_as_printed']=long
 add(d,'Disease','TRANSMITTED_BY','เพลี้ยอ่อน','Pest',p)
 for a in [REMOVE,APHID,CLEAN,ROT]:add(d,'Disease','MANAGED_BY',a,'Treatment',[30,p] if p!=32 else [30,32,33],context='PVY/CMV/TEV: คู่มืออ้างวิธีเดียวกับ CVMV; TEV เชื่อมต่อหน้า 32–33')
 viruses.append(d)
 for crop in (['ลำโพง','ยาสูบ','มะเขือเทศ'] if short=='CVMV' else ['ยาสูบ','มะเขือเทศ'] if short=='PVY' else []):add(short,'Pathogen','HAS_HOST',crop,'Crop',p)
for material in ['สะเดา','หางไหล','สาบเสือ','ตะไคร้หอม','ดาวเรือง']:add(APHID,'Treatment','USES','สารสกัด'+material,'Material',30)
yellow=disease('พริก','โรคใบหงิกเหลือง',32,'ไวรัสใบหงิกเหลือง','ใบด่างเหลือง ใบโค้งรูปถ้วย กลางใบหงิกย่น',season='ฤดูแล้ง')
add(yellow,'Disease','TRANSMITTED_BY','แมลงหวี่ขาว','Pest',32)
for a in ['ควบคุมแมลงหวี่ขาว','ใช้กับดักกาวเหนียว','ใช้พลาสติกสีบรอนซ์คลุมแปลง',REMOVE,ROT]:add(yellow,'Disease','MANAGED_BY',a,'Treatment',[32,33])
frog=disease('พริก','โรคใบจุดตากบ',34,'เชื้อรา','แผลขอบน้ำตาลเข้ม กลางแผลเทาหรือขาว',[SPACE,'ลดการให้น้ำ','พ่นสารป้องกันกำจัดโรคใบจุดตากบ'])
add(frog,'Disease','HAS_ENVIRONMENT','อากาศร้อนชื้น','Environment',34)
stem=disease('พริก','โรคลำต้นไหม้',35,'เชื้อรา','รอยไหม้สีดำบริเวณลำต้น',[SOLAR,DRAIN,REMOVE,'ใช้จุลินทรีย์ปฏิปักษ์'])
for b in [TRIC,BAC]:add('ใช้จุลินทรีย์ปฏิปักษ์','Treatment','USES_BIOCONTROL',b,'Biocontrol',35)
disease('พริก','โรคกุ้งแห้งเทียม',35,'เชื้อรา','ผลซีดแห้งและเส้นใยดำคล้ายกำมะหยี่',['ให้ธาตุอาหารครบถ้วน'])
root=disease('พริก','โรครากปม',36,'ไส้เดือนฝอย','รากเป็นปมและเน่า ต้นแคระแกร็น ใบเหลือง')
for a in [SOLAR,REMOVE,'แยกเครื่องมือจากพื้นที่ระบาด','ใส่ปุ๋ยอินทรีย์','ปลูกพืชสลับเพื่อตัดวงจรไส้เดือนฝอย']:add(root,'Disease','MANAGED_BY',a,'Treatment',[36,37])
for crop in ['ถั่วลิสง','ปอเทือง','ดาวเรือง','งา']:add('ปลูกพืชสลับเพื่อตัดวงจรไส้เดือนฝอย','Treatment','USES',crop,'Crop',37)
disease('พริก','โรคราแป้ง',36,'เชื้อรา','ผงสีขาวคล้ายแป้งปกคลุมใบแก่',[CLEAN,'พ่นสารป้องกันกำจัดโรคราแป้ง'],season='ฤดูหนาว')
# Winter downy mildew remains a general disease group, not a claim of identical pathogens.
general='กลุ่มโรคราน้ำค้าง'
for crop in ['พืชตระกูลแตง','พืชตระกูลกะหล่ำ','ข้าวโพดหวาน','ไม้ดอก']:add(crop,'Crop','HAS_DISEASE',general,'DiseaseGroup',39)
add(md,'Disease','IS_A',general,'DiseaseGroup',[18,39])
add(general,'DiseaseGroup','HAS_SEASON','ฤดูหนาว','Season',[38,39])
add(general,'DiseaseGroup','HAS_ENVIRONMENT','ความชื้นสูง','Environment',39)
add(general,'DiseaseGroup','HAS_SYMPTOM','ผงขาวหรือเทาบนใบ ใบเหลืองเปลี่ยนเป็นน้ำตาล','Symptom',39)
for a in ['ล้างใบพืชในตอนเช้า','พ่นน้ำส้มควันไม้หลังล้างใบ']:add(general,'DiseaseGroup','MANAGED_BY',a,'Treatment',[39,40],condition='วิธีตามเกษตรกรที่คู่มือรายงาน')
add('พ่นน้ำส้มควันไม้หลังล้างใบ','Treatment','USES','น้ำส้มควันไม้','Material',40)
# Rainy-season chapter.
for path,d in [('พิเทียม','โรครากเน่า โคนเน่า และยอดเน่าในต้นกล้าพืชไร่'),('ไฟทอฟทอรา','โรครากเน่าโคนเน่าในไม้ผล'),('สเคลอโรเทียม','โรคกล้าไหม้ ราเม็ดผักกาด และเหี่ยวในพืชผัก'),('ฟูซาเรียม','โรคเหี่ยวในไม้ดอก'),('ไรซ็อกโทเนีย','โรคเน่าคอดิน')]:add(d,'DiseaseGroup','CAUSED_BY',path,'Pathogen',41)
for a in [DRAIN,'จัดโรงเรือนให้อากาศถ่ายเท','เก็บผลและตัดกิ่งที่เป็นโรคไปทำลาย','ใช้ไตรโคเดอร์มารอบโคนต้น']:add('การจัดการเชื้อราฤดูฝน','Treatment','INCLUDES',a,'Treatment',41 if a in [DRAIN,'จัดโรงเรือนให้อากาศถ่ายเท'] else 42)
add('การจัดการเชื้อราฤดูฝน','Treatment','HAS_SEASON','ฤดูฝน','Season',41)
add('ใช้ไตรโคเดอร์มารอบโคนต้น','Treatment','USES_BIOCONTROL',TRIC,'Biocontrol',42)
add('โรครากเน่าโคนเน่าในไม้ผล','DiseaseGroup','MANAGED_BY','ใช้ไตรโคเดอร์มารอบโคนต้น','Treatment',[41,42])
# Weed recipes represented as source recipes, not universal instructions or safety claims.
for recipe,materials,p in [('สูตรน้ำส้มสายชู',['น้ำส้มสายชู','เกลือแกง','น้ำยาล้างจาน'],44),('สูตรน้ำหัวผักกาด',['น้ำหัวผักกาด','น้ำมะพร้าวอ่อน','ผงกลูโคส'],44),('สูตรเกลือ',['เกลือแกง','ปุ๋ย 21-0-0','น้ำ'],45)]:
 add('วัชพืช','Pest','MANAGED_BY',recipe,'Treatment',p)
 for m in materials:add(recipe,'Treatment','USES',m,'Fertilizer' if m=='ปุ๋ย 21-0-0' else 'Material',p)
add('สูตรน้ำหัวผักกาด','Treatment','APPLIED_TO','ข้าว','Crop',44)
add('สูตรเกลือ','Treatment','APPLIED_TO','ข้าว','Crop',45,condition='ต้นข้าวสูงเหนือหัวเข่า ตามหน้า 45')
# Durian: carry context across pages; do not infer a species absent in this chapter.
dur=disease('ทุเรียน','โรครากเน่าโคนเน่า',46,sym='ใบอ่อนเหี่ยวเหลือง แผลฉ่ำน้ำ เปลือกเน่าและน้ำเยิ้ม',season='ฤดูฝน')
add(dur,'Disease','HAS_ENVIRONMENT','ความชื้นสูง','Environment',46)
add(dur,'Disease','CO_OCCURS_WITH','มอดเจาะลำต้นทุเรียน','Pest',46)
for a,p,c in [('ตัดแต่งส่วนเป็นโรคและเผานอกแปลง',47,'การควบคุมเบื้องต้น'),('ใช้ชีวภัณฑ์เมื่อโรคไม่รุนแรง',47,'โรคไม่รุนแรง'),('ขูดเปลือกและทาแผลเมื่อโรครุนแรง',48,'โรครุนแรง'),('ขุดต้นตายไปเผานอกแปลงและจัดการดิน',48,'โรครุนแรงมากหรือยืนต้นตาย')]:add(dur,'Disease','MANAGED_BY',a,'Treatment',[46,p],condition=c)
for b in ['ไตรโคเดอร์มา ฮาร์เซียนัม',BAC]:add('ใช้ชีวภัณฑ์เมื่อโรคไม่รุนแรง','Treatment','USES_BIOCONTROL',b,'Biocontrol',47)
for chem in ['ฟอสอีทิล-อะลูมิเนียม','เมทาแลกซิล','เมทาแลกซิล+แมนโคเซบ']:add('ขูดเปลือกและทาแผลเมื่อโรครุนแรง','Treatment','USES_CHEMICAL',chem,'Chemical',48)
add('ไตรโคเดอร์มา ฮาร์เซียนัม','Biocontrol','IS_A',TRIC,'Biocontrol',47)
nodes[node(BAC,'Biocontrol')]['aliases']=['บาซิลลัส ซับติลิส']
# Freeze domain layer before provenance links.
domain_ids=set(nodes); domain_edge_ids=set(edges)
DOC='คู่มือเกษตรกร: ความรู้ในการกำจัดโรคพืช (2562)'
node(DOC,'Document',sha256=SHA,filename='9-file-4786.pdf',publication_year=2019,page_count=48)
for title,start,end in CHAPTERS:
 node(title,'Section',page_start=start,page_end=end)
 add(DOC,'Document','HAS_SECTION',title,'Section',4,assertion_layer='document_structure')
 covered=set()
 for e in list(edges.values()):
  if e['id'] in domain_edge_ids and any(start<=p<=end for p in e['pages']):covered|={e['source'],e['target']}
 for k in sorted(covered):add(title,'Section','COVERS',nodes[k]['name'],nodes[k]['kind'],[start],assertion_layer='document_structure',evidence_note='Section index; supporting pages are on semantic relationships')
for p in PAGES:
 name=f'PDF หน้า {p:02d}';node(name,'Page',number=p,text=PAGES[p])
 add(DOC,'Document','HAS_PAGE',name,'Page',p,assertion_layer='document_structure')
 mentioned=set()
 for k in domain_edge_ids:
  e=edges[k]
  if p in e['pages']:mentioned|={e['source'],e['target']}
 for k in sorted(mentioned):add(name,'Page','SUPPORTS_ENTITY',nodes[k]['name'],nodes[k]['kind'],p,assertion_layer='provenance',evidence_note='Entity supported by page plus chapter context; not necessarily an exact-name mention')
for e in edges.values():e['layer']='semantic' if e['id'] in domain_edge_ids else 'provenance'
def dump(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
graph=dict(metadata=dict(dataset=NS,source_sha256=SHA,method='AI source review of all 48 PDF pages; deterministic builder; no new BERT/SBERT/LLM run',publication_year=2019),nodes=list(nodes.values()),relationships=list(edges.values()))
dump('graph.json',graph)
for fname,rows,fields in [('nodes.csv',list(nodes.values()),['id','name','kind','dataset']),('relationships.csv',list(edges.values()),['id','source','relation','target','pages','layer','evidence','source_sha256'])]:
 with (ROOT/fname).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for row in rows:w.writerow({**row,'pages':json.dumps(row.get('pages',[]))})
# Cypher literals escaped with JSON strings, but dict keys emitted as Cypher identifiers.
def cy(v):
 if isinstance(v,dict):return '{'+', '.join('`'+k+'`: '+cy(x) for k,x in v.items())+'}'
 if isinstance(v,list):return '['+', '.join(cy(x) for x in v)+']'
 return json.dumps(v,ensure_ascii=False)
queries=['CREATE CONSTRAINT plant_guide_reviewed_id IF NOT EXISTS FOR (n:PlantGuide2019Reviewed) REQUIRE n.id IS UNIQUE']
for kind in sorted({n['kind'] for n in nodes.values()}):
 rows=[n for n in nodes.values() if n['kind']==kind]
 queries.append(f'UNWIND {cy(rows)} AS row MERGE (n:PlantGuide2019Reviewed {{id: row.id}}) SET n:{kind} SET n += row')
for rel in sorted({e['relation'] for e in edges.values()}):
 rows=[{k:v for k,v in e.items() if k!='evidence'} for e in edges.values() if e['relation']==rel]
 # Full evidence stored once on Page.text; relationship pages link back without duplicating text in Neo4j.
 queries.append(f'UNWIND {cy(rows)} AS row MATCH (s:PlantGuide2019Reviewed {{id:row.source}}), (t:PlantGuide2019Reviewed {{id:row.target}}) MERGE (s)-[r:{rel} {{id:row.id}}]->(t) SET r += row')
(ROOT/'import.cypher').write_text(';\n\n'.join(queries)+';\n',encoding='utf-8')
# Verification separates semantic connectivity from documentary backbone.
import networkx as nx
G=nx.Graph();G.add_nodes_from(domain_ids);G.add_edges_from((edges[k]['source'],edges[k]['target']) for k in domain_edge_ids)
A=nx.Graph();A.add_nodes_from(nodes);A.add_edges_from((e['source'],e['target']) for e in edges.values())
assert all(e['source'] in nodes and e['target'] in nodes for e in edges.values())
assert all(e['pages'] and e['source_sha256']==SHA for e in edges.values())
assert len(set((n['kind'],n['name']) for n in nodes.values()))==len(nodes)
report=dict(nodes=len(nodes),relationships=len(edges),semantic_nodes=len(domain_ids),semantic_relationships=len(domain_edge_ids),semantic_components=len(list(nx.connected_components(G))),semantic_largest_component=max(map(len,nx.connected_components(G))),full_graph_components=nx.number_connected_components(A),isolated_semantic_nodes=len(list(nx.isolates(G))),chapters=10,pdf_pages=48,relationship_page_reference_coverage=1.0,duplicate_node_keys=0,dangling_relationships=0,neo4j_live_import=False)
dump('validation.json',report)
dump('views.json',dict(virus_diseases=viruses,durian=dur,maize_downy=md,general_downy=general))
print(json.dumps(report,ensure_ascii=False,indent=2))
