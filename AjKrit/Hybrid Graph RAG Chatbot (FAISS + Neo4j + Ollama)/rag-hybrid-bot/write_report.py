from html import escape as e
from core import *
def run():
    source=read('source.json');index=read('index_manifest.json');facts=read('facts.json');evaluation=read('retrieval_evaluation.json');tests=read('checkpoints/07/results.json');answers=read('answers.json')
    table='<table><tr><th>วิธีค้นคืน</th><th>Hit@1</th><th>Hit@5</th><th>MRR@5</th></tr>'
    for mode,metrics in evaluation['summary'].items():
        table+='<tr><td>'+e(mode)+'</td>'+''.join(f'<td>{v:.2%}</td>' for v in metrics.values())+'</tr>'
    table+='</table>'
    content=f'''<span class="tag">LOCAL HYBRID GRAPH RAG · DELIVERY REPORT</span><h1>ระบบถาม–ตอบคู่มือโรคพืชภาษาไทย</h1><p>FAISS + Thai BM25 + Neo4j + Ollama</p><article><h2>ผลการประมวลผลจริง</h2><p>{source['pages']} หน้า PDF · {index['vectors']} chunks · {len(facts)} facts</p><p>LLM: {e(MODEL)} · Embedding: {e(index['model'])} ({index['dimension']} dimensions)</p><p>เวลาสร้าง FAISS: {index['seconds']} วินาที · tests exit code: {tests['tests_exit_code']}</p><p><a href="http://127.0.0.1:8090">เปิดเว็บ Chatbot</a> · <a href="http://localhost:7475">เปิด Neo4j Browser</a></p></article><article><h2>ผลเปรียบเทียบการค้นคืน 16 คำถาม</h2>{table}<p>เป็นชุดวินิจฉัยจาก PDF เดียว ไม่ใช่ held-out benchmark; Hit@k คือค้นพบอย่างน้อยหนึ่งหน้าที่ตรง gold และไม่ได้วัดความถูกต้องของคำตอบ</p><a href="retrieval_evaluation.csv">CSV รายคำถาม</a> · <a href="retrieval_evaluation.json">JSON ฉบับเต็ม</a></article>'''
    content+='<article><h2>คำตอบจากการรันจริง</h2>'
    for r in answers:
        content+=f'<h3>{e(r["query"])}</h3><p style="white-space:pre-wrap">{e(r["answer"])}</p><small>{e(r["generation"]["model"])} · {r["generation"]["seconds"]} วินาที</small>'
    content+='</article><article><h2>Checkpoint — ภาพแคปและผล export</h2><table><tr><th>Checkpoint</th><th>HTML</th><th>PDF</th><th>ภาพแคป</th><th>JSON</th></tr>'
    for i in range(1,8):
        p=f'checkpoints/{i:02d}'
        content+=f'<tr><td>{i:02d}</td>'+''.join(f'<td><a href="{p}/{f}">{label}</a></td>' for f,label in [('report.html','รายงาน'),('report.pdf','PDF'),('screenshot.png','PNG'),('results.json','ข้อมูล')])+'</tr>'
    content+='''</table></article><article><h2>กราฟและหลักฐาน</h2><p><a href="neo4j_export.csv">Neo4j CSV</a> · <a href="neo4j_export.json">Neo4j JSON</a> · <a href="knowledge_graph.graphml">GraphML</a> · <a href="neo4j_browser.png">ภาพ Neo4j Browser</a> · <a href="graph_rejected.json">รายการที่คัดออก</a> · <a href="review_applied.json">บันทึกทบทวนกราฟ</a></p><p>ตรวจคำอ้างและชื่อ entity กับข้อความจริง พร้อมกำหนดชนิดความสัมพันธ์ มีการทบทวนและเพิ่มข้อเท็จจริงบางรายการจากต้นฉบับโดย Codex ซึ่งระบุ origin แยกจาก model; ยังไม่ใช่การรับรองความถูกต้องโดยผู้เชี่ยวชาญ</p></article><article><h2>หน้าเว็บจริง</h2><p><a href="chatbot_desktop.png">Desktop PNG</a> · <a href="chatbot_mobile.png">Mobile PNG</a> · <a href="chatbot.pdf">PDF</a></p><img src="chatbot_desktop.png" style="width:100%;border-radius:12px"></article><article><h2>ขอบเขตการทดสอบ</h2><p>เว็บและฐานข้อมูลรันจริงในเครื่อง LINE มี adapter และ signature test แต่ยังไม่ทดสอบรับ–ส่งข้อความจริง เพราะไม่ได้รับ credentials ไม่มี public tunnel</p><p>การเลือกโมเดลคำนึงถึง AMD GPU 8 GB / RAM 30 GB ไม่ได้ทดสอบโมเดลทุกตัวเพื่อยืนยันว่าดีที่สุดทั่วโลก เนื้อหาคำตอบอิงหนังสือปี 2562 ไม่ได้ปรับคำแนะนำเกษตรให้เป็นข้อมูลปัจจุบัน</p><p><a href="test_results.txt">ผลทดสอบระบบ</a> · <a href="answers.json">คำตอบเต็ม JSON</a> · <a href="llm_comparison.json">ผลเปรียบเทียบ LLM ตัวอย่าง</a></p></article>'''
    (DATA/'REPORT.html').write_text(f'<!doctype html><html lang="th"><meta charset="utf-8"><title>Hybrid Graph RAG · Report</title><style>{STYLE}</style><main>{content}</main></html>',encoding='utf-8')
if __name__=='__main__':run()
