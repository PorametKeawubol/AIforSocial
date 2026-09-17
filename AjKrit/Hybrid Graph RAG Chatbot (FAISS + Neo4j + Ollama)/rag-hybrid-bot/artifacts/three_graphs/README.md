# กราฟ Neo4j 3 วิธี

เปิด `index.html` เพื่อดูกราฟและดาวน์โหลดข้อมูลแต่ละแบบ

- **Model I** ใช้ `from langchain_neo4j import Neo4jGraph` เขียนกราฟจริง โดยใช้ผลสกัด Qwen3.5 9B แบบ schema-controlled ที่รันไว้ก่อนหน้าและผ่านตัวกรอง ไม่รวม facts ที่เพิ่มจากการทบทวนต้นฉบับ โค้ดปรับจาก LLMGraphTransformer ของใบงาน ไม่ใช่การอ้างว่ารัน LLMGraphTransformer ตรงตัว
- **Model II** ใช้ `from neo4j import GraphDatabase` และ `from pythainlp.tag.thainer import ThaiNameTagger` รัน ThaiNER 1.4 จริง เก็บ raw BIO/POS tokens และ regex entities สร้างความสัมพันธ์ CO_OCCURS_WITH จากชื่อที่อยู่ใกล้กันใน chunk การปรากฏร่วมกันไม่ใช่หลักฐานว่าอยู่ในสถานที่เดียวกันหรือมีความสัมพันธ์ทางโรคพืช
- **Model III** สกัดใหม่ด้วย schema `crop`, `disease`, `pathogen`, `symptom`, `treatment` ตาม Neo4j_Create.py ในใบงาน ใช้ Qwen3.5 9B และ GraphDatabase สกัดทั้ง 48 หน้า กรองชื่อที่ไม่พบในหน้า ไม่เดาชื่อพืชจากหน้าก่อน จึงอาจได้กราฟน้อยกว่าการสกัดแบบทั่วไป

ทุกแบบเขียนลงฐานข้อมูลแล้ว query กลับมา export ไม่ได้สร้างภาพจำลองแทนฐานข้อมูล ภาพ Neo4j แสดงไม่เกิน 18 เส้นเพื่อให้อ่านได้ CSV/JSON/GraphML มีข้อมูลครบ รวม nodes ที่ไม่มีเส้นเชื่อม กราฟทั้งสามใช้ label ต่างกัน ไม่แก้กราฟ Chatbot เดิม

## ไฟล์ในแต่ละโฟลเดอร์

- `neo4j_graph.png`: ภาพแคปกราฟใน Neo4j Browser
- `nodes.csv`: ทุกโหนด
- `relationships.csv`: ทุกความสัมพันธ์พร้อมหน้าและหลักฐาน (UTF-8 BOM สำหรับ Excel)
- `raw.json`: metadata, nodes, relationships ที่อ่านกลับจาก Neo4j
- `raw.html`, `raw.png`: เปิดอ่าน/ภาพตัวอย่าง raw
- `table.html`, `table.pdf`, `table.png`: ตารางข้อมูลทั้งหมดใน HTML/PDF; PNG เป็นภาพช่วงแรก
- `graph.graphml`: กราฟเต็มสำหรับนำเข้าโปรแกรมกราฟ
- `queries.cypher`: คำสั่งเปิดกราฟ/ตาราง/raw ใน Neo4j Browser
- `summary.json`: จำนวนและวิธีที่ใช้

Model II มี `ner_raw_tokens.json`; Model III มี `page_XX.json`, `extraction_raw.json`, `rejected.json` สำหรับตรวจผลสกัดก่อนและหลังตัวกรอง

Model I/III ยังอาจตีความความสัมพันธ์ผิดได้ การพบชื่อ/ข้อความในต้นฉบับไม่ใช่การยืนยันความหมายทั้งหมด และจำนวน nodes/edges ไม่ใช่คะแนนความแม่นยำ

## รันซ้ำ

ใช้ `.venv/bin/python three_graphs.py model_i` / `model_ii` / `model_iii` ตามลำดับ แล้ว `.venv/bin/python capture_three_graphs.py` ต้องเปิด Neo4j และ Ollama ก่อน Dependencies เพิ่ม: `langchain-neo4j`, `python-crfsuite` และ corpus `thainer-1.4`

เอกสารอ้างอิง API:
- https://docs.langchain.com/oss/python/integrations/graphs/neo4j_cypher
- https://pythainlp.org/docs/5.3.2/api/tag.html

## Native Neo4j image export

`graph_export.svg` และ `graph_export.png` คือไฟล์ที่ดาวน์โหลดผ่านเมนู Download ของ Neo4j Browser โดยตรง โดย query ทุก node และทุก relationship ของ model นั้น รวม isolated nodes ไม่มี LIMIT ต่างจาก `neo4j_graph.png` ซึ่งเป็นภาพแคปหน้าจอเดิม รายละเอียดอยู่ใน `native_exports.json`

ภาพ export ใช้สีตาม `kind` ของโหนด: Crop = เขียว, Disease = แดง, Pathogen = ม่วง, Symptom = เหลือง, Treatment = น้ำเงิน และ Topic = เทา. Model II ใช้สีตามชนิด ThaiNameTagger เช่น PERSON, LOCATION, TIME และ ORGANIZATION. ไฟล์ export ใช้ Neo4j display-copy ที่มี property `visual_variant`; กราฟข้อมูลต้นฉบับที่มี label `WorksheetModelI` ถึง `WorksheetModelIII` ยังคงอยู่และไม่ถูกแก้ไข.
