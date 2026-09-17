# รุ่น BGE-M3 + Recursive Splitter

โค้ดปัจจุบันใช้ BAAI/bge-m3 (1024 มิติ) ทั้งข้อมูลและคำถาม พร้อม index `lab5_bge_m3_recursive_index` และ label `Lab5BGEChunk` แยกจาก MiniLM เดิม ใช้ Graph Entity เดิมได้

```bash
python -m pip install -r requirements.txt
python prepare_data.py --splitter recursive_zwsp --output prepared_recursive.json
python create_embeddings.py --batch-size 8
python graph_rag.py "What mitigations are available for PowerShell?" --retrieve-only
python graph_rag.py --test-questions --compare --output results/comparison_bge.json
python export_results.py --input results/comparison_bge.json --output results/lab5_report_bge.md
```

Recursive แบ่ง description สูงสุด 500 ตัวอักษร มี overlap เป้าหมาย50และเพิ่มheaderของEntity รองรับ zero-width space ที่มีอยู่แล้ว ไม่ตัดคำไทยหรือแทรกตัวคั่นเอง โมเดลใหม่ต้องดาวน์โหลดและคำนวณ embedding ใหม่ จึงใช้เวลามากกว่า MiniLM; ค่า batch8 ตั้งไว้สำหรับลดหน่วยความจำ ส่วนผลเดิมใน comparison.json เป็น MiniLM ไม่ใช่ BGE-M3

รายการด้านล่างเป็นคู่มือรอบ MiniLM เดิม: ชื่อ index, dimension และคำสั่ง embeddings ให้ใช้ส่วนอัปเดตด้านบนแทน

---

**Lab 5 — Graph RAG for Cybersecurity Intelligence**

ทำตาม [lab5.pdf](../lab5.pdf) โดยใช้ `enterprise-attack.json` ชุดเดียวสร้างทั้ง Knowledge Graph และ Text Chunks แล้วค้นหาด้วย Vector Search + Graph Traversal ก่อนให้ LLM ตอบคำถาม

คู่มือนี้อธิบายวิธีรันและหลักฐานที่ต้องเก็บ การมี Source Code อย่างเดียวยังไม่แปลว่าทำ Lab ครบ ให้ตรวจผลลัพธ์จริง บันทึกภาพ และเขียนวิเคราะห์ด้วย

**1. เปิด Terminal และเตรียมเครื่องมือ**

เครื่องนี้มี Python environment และ dependencies พร้อมแล้ว ใช้คำสั่ง:

```bash
cd "/home/porametk/Desktop/AIforSocial/AIforSocial/AjKrit/Graph RAG/graph-rag-lab"
source ../../../.venv/bin/activate
docker start neo4j-GraphRAG
docker ps --filter name=neo4j-GraphRAG
```

ถ้า Docker daemon ยังไม่เปิด ให้เปิด Docker Desktop ก่อน แล้วรันใหม่ ใช้ Container `neo4j-GraphRAG` ที่มีอยู่แล้วสำหรับขั้นตอนนี้ เปิด Neo4j Browser ที่ <http://localhost:7474>

โฟลเดอร์ Lab นี้มี `.env` ส่วนตัวที่ตั้งค่าการเชื่อมต่อไว้แล้ว ให้ใช้ไฟล์เดิม สคริปต์จะอ่านไฟล์นี้อัตโนมัติ

สำหรับการย้ายไปเครื่องอื่น ให้สร้าง Python environment และติดตั้ง dependencies แล้วคัดลอกตัวอย่างเฉพาะเมื่อยังไม่มี `.env`:

```bash
python -m pip install -r requirements.txt
if [ ! -f .env ]; then
  cp .env.example .env
fi
```

จากนั้นแก้ค่าการเชื่อมต่อใน `.env` ให้ตรงกับ Neo4j ของเครื่องนั้น และใช้ Neo4j 5.26 ขึ้นไปสำหรับคำสั่งของ Lab นี้

**2. ดาวน์โหลดและสำรวจ Dataset — Task 1–2**

```bash
python download_data.py
mkdir -p results
python inspect_data.py | tee results/inspect_data.txt
```

`download_data.py` ดาวน์โหลด Enterprise ATT&CK STIX 2.1 จาก repository ของ MITRE และบันทึกวันดาวน์โหลดกับ SHA256 ใน `enterprise-attack.metadata.json` ถ้ามีไฟล์ข้อมูลแล้วจะใช้ไฟล์เดิม เพื่อให้การทดลองซ้ำใช้ข้อมูลชุดเดียวกัน

ดูส่วน `Task 2 answers` ในผลลัพธ์ เพื่อตอบจำนวน Object ทั้งหมด และจำนวน `attack-pattern`, `malware`, `intrusion-set`, `course-of-action`, `relationship` ตัวเลขเหล่านี้มาจาก **ข้อมูลดิบทั้งหมด** จึงรวม Object ที่ถูก revoked/deprecated ด้วย

เก็บไฟล์ Dataset และ metadata ชุดนี้ไว้ตลอด Lab จำนวนข้อมูลอาจแตกต่างจากเพื่อนที่ดาวน์โหลดคนละวัน หากเปลี่ยน Dataset หลังนำเข้าแล้ว สคริปต์จะตรวจพบ SHA256 ที่ไม่ตรงกัน

**3. เตรียมข้อมูลและสร้าง Knowledge Graph — Task 3–6**

```bash
python prepare_data.py
python create_graph.py
```

`prepared.json` จะมี `entities`, `relationships`, `documents`, `chunks` และ `metadata` โดยกำหนดแบบจำลองดังนี้:

| STIX Type | Neo4j Label |
|---|---|
| `intrusion-set` | `Group` |
| `malware` | `Malware` |
| `tool` | `Tool` |
| `attack-pattern` | `Technique` |
| `course-of-action` | `Mitigation` |

ทุก Entity มี Label `AttackEntity` เพิ่มเติมเพื่อระบุชุดข้อมูลของ Lab 5 ความสัมพันธ์ที่นำเข้าคือ `USES`, `MITIGATES`, `SUBTECHNIQUE_OF` และเก็บทิศทางจาก STIX เช่น `Mitigation → MITIGATES → Technique`

ขั้นเตรียมข้อมูลเลือกเวอร์ชันล่าสุดของแต่ละ STIX ID แล้วเก็บเฉพาะ Entity/Relationship ที่ยัง active พร้อมตรวจว่า Endpoint ทั้งสองอยู่ในชุดที่เลือก จึงคาดว่าจำนวนที่นำเข้า Graph จะต่างจากตัวเลขข้อมูลดิบของ Task 2 รายละเอียดการกรองอยู่ใน `prepared.json` ส่วน `metadata`

สคริปต์ใช้ `MERGE` และ ID เดิม จึงรันซ้ำกับข้อมูลชุดเดิมได้ ผลตรวจจำนวนอยู่ใน `results/graph_counts.json` ถ้าพบ Dataset คนละ SHA256 ให้กลับมาใช้ไฟล์เดิม หรือเตรียม Neo4j instance/database แยกสำหรับข้อมูลชุดใหม่

เปิด [lab5_queries.cypher](lab5_queries.cypher) คัดลอกคำสั่ง **ทีละชุด** ไป Neo4j Browser รันชุดที่ 1–4 เพื่อดู Metadata จำนวน Node/Relationship และกราฟ APT29 จากนั้นบันทึกภาพกราฟและตารางจำนวนไว้ส่ง

**4. สร้าง Text Chunks, Embedding และ Vector Index — Task 7–11**

Text Dataset และ Chunks ถูกสร้างแล้วในขั้น `prepare_data.py` จาก `description` ของ Entity ชุดเดียวกับ Graph โดยใช้:

- `chunk_size = 500` ตัวอักษรของ Description และ `chunk_overlap = 50` ตัวอักษร
- เพิ่มหัวข้อความ `Type`, `Name`, `ATT&CK ID` ให้แต่ละ Chunk; หัวข้อนี้อยู่นอกขนาด 500 ตัวอักษร
- แต่ละ Chunk มี `entity_id` สำหรับเชื่อมกลับไปยัง Entity ต้นทาง

สร้างเวกเตอร์และนำเข้า Neo4j:

```bash
python create_embeddings.py
```

ใช้ `sentence-transformers/all-MiniLM-L6-v2` สร้างเวกเตอร์ 384 มิติ ประมวลผลเป็น Batch บน CPU ตามค่าเริ่มต้น การรันครั้งแรกอาจต้องดาวน์โหลดโมเดลก่อน สคริปต์แสดงความคืบหน้าและรันซ้ำเพื่อทำต่อจาก Embedding ที่ข้อความ โมเดล และ Dataset ตรงกันได้

Node มีทั้ง Label `Chunk` และ `Lab5Chunk` เชื่อมด้วย `Chunk → DESCRIBES → AttackEntity` ดัชนีชื่อ `lab5_chunk_embedding_index` ใช้ Label `Lab5Chunk`, Property `embedding`, 384 มิติ และ cosine similarity

ตรวจ `results/embedding_report.json` และรัน Query ชุดที่ 5–6 ใน Neo4j Browser ดัชนีต้องเป็น `ONLINE` ก่อนค้นหา

**5. ทดลองค้นหาก่อนให้ LLM ตอบ — Task 12–15**

```bash
python graph_rag.py "What techniques are associated with APT29?" \
  --retrieve-only \
  --output results/retrieval_apt29.json
```

คำสั่งนี้แสดง Top-K Chunks, Similarity Score, Entity ที่ใช้เริ่มค้นกราฟ และจำนวน Graph Paths เปิด `results/retrieval_apt29.json` เพื่อดูหลักฐานเต็ม:

| Field ใน `results[0]` | ใช้ตรวจอะไร |
|---|---|
| `chunks` | ข้อความที่ค้นเจอ พร้อมคะแนน ชื่อ Entity และ URL |
| `graph.seeds` | Entity ที่ใช้เริ่ม Graph Traversal และวิธีที่พบ |
| `graph.cypher`, `graph.parameters` | Cypher และค่าพารามิเตอร์ที่รันจริง |
| `graph.paths` | Node, Relationship, ทิศทาง และหลักฐานข้อความจาก STIX |
| `vector_context` | ข้อความอ้างอิง `[V1]`, `[V2]`, … |
| `graph_context` | เส้นทางอ้างอิง `[G1]`, `[G2]`, … |
| `context_truncated` | Context ถูกจำกัดจำนวน/ขนาดหรือไม่ |

เส้นทางการทำงานคือ `Question → Embedding → Vector Search → Entity → Graph Traversal → Context Fusion` ถ้าคำถามมีชื่อ Entity, ATT&CK ID หรือ Alias ที่ตรงกับข้อมูล สคริปต์จะใช้ Entity ที่ระบุเป็นจุดเริ่มค้นกราฟ หากไม่พบชื่อที่ตรง จะใช้ Entity จาก Vector Search

ทดลองระบุ Entity เองหรือเพิ่มจำนวนผลได้ เช่น:

```bash
python graph_rag.py "What mitigations are available for this technique?" \
  --entity PowerShell --top-k 5 --max-paths 100 \
  --retrieve-only --output results/retrieval_powershell.json
```

**6. ให้ LLM สร้างคำตอบ — Task 16**

เครื่องนี้มี Ollama และโมเดล `qwen2.5:7b` แล้ว ตรวจบริการด้วย:

```bash
ollama list
```

หากเชื่อมต่อบริการไม่ได้ ให้เปิดอีก Terminal แล้วรัน `ollama serve` ค้างไว้ จากนั้นกลับมาที่ Terminal ของ Lab:

```bash
python graph_rag.py "What techniques are associated with APT29?" \
  --output results/answer_apt29.json
```

เปิดผลใน `results[0].graph_rag.answer` ซึ่งควรอธิบายคำตอบพร้อมอ้างอิง `[V…]` หรือ `[G…]` ตรวจว่าข้อความที่อ้างถึงรองรับความสัมพันธ์นั้นจริง ระบบตรวจได้ว่ารหัสอ้างอิงมีอยู่ แต่การตรวจว่าคำกล่าวถูกต้องตามหลักฐานยังต้องอ่านเทียบเอง

**7. ทดสอบหลายคำถามและเปรียบเทียบ — Task 17–18**

รันห้าคำถามตามใบงานและเพิ่มคำถาม Multi-hop อีกหนึ่งข้อ พร้อมสร้างคำตอบจากทั้งสองระบบ:

```bash
python graph_rag.py --compare --test-questions \
  --output results/comparison.json
```

คำถามที่ใช้:

1. What techniques are associated with APT29?
2. What malware is associated with APT29?
3. Which techniques are related to SUNBURST malware?
4. What mitigations are available for PowerShell?
5. What cybersecurity relationships can be found around APT29?
6. Which mitigations address techniques used by APT29?

`--compare` ใช้ Top-K Chunks ชุดเดียวกัน โมเดลเดียวกัน และค่าการสร้างคำตอบเดียวกันให้ทั้งสองระบบ โดย Vector RAG รับ Vector Context ส่วน Graph RAG รับ Vector Context รวมกับ Graph Context คำถามทั้งหมดมีไว้ทดสอบผลจริง ไม่ได้กำหนดคำตอบล่วงหน้า

คำสั่งนี้เรียก LLM หลายครั้งจึงใช้เวลามากกว่าการค้นหาอย่างเดียว สคริปต์บันทึกผลหลังจบแต่ละคำถาม ตรวจว่ามีครบหกคำถามก่อนสรุปว่าเสร็จ ถ้าต้องการทดลองการค้นหาทุกข้อก่อน ใช้ `--test-questions --retrieve-only --output results/retrieval_all.json`

สำหรับการเปรียบเทียบคำถามเดียว:

```bash
python graph_rag.py "What malware is associated with APT29?" \
  --compare --output results/comparison_apt29_malware.json
```

อ่าน `vector_rag.answer` และ `graph_rag.answer` ของแต่ละคำถาม แล้วทำตารางเปรียบเทียบตามผลจริง เช่น:

| คำถาม | หลักฐาน/ข้อค้นพบจาก Vector RAG | หลักฐาน/ข้อค้นพบจาก Graph RAG | ข้อสังเกตจากการตรวจ |
|---|---|---|---|
| Techniques of APT29 | เติมจากผลรัน | เติมจากผลรัน | ความสัมพันธ์มีหลักฐานรองรับหรือไม่ |
| Malware of APT29 | เติมจากผลรัน | เติมจากผลรัน | แยก Malware กับ Tool ถูกหรือไม่ |
| Mitigations for PowerShell | เติมจากผลรัน | เติมจากผลรัน | ทิศทาง `MITIGATES` ถูกหรือไม่ |
| Relationships around APT29 | เติมจากผลรัน | เติมจากผลรัน | มี Entity ที่เกี่ยวข้องหลุดออกนอกคำถามหรือไม่ |
| Multi-hop mitigations | เติมจากผลรัน | เติมจากผลรัน | อธิบายเส้นทางผ่าน Technique ได้หรือไม่ |

**8. วิเคราะห์ผล — Task 19**

เขียนคำตอบหกข้อตามใบงาน โดยยกตัวอย่างอ้างอิงผลที่รันจริง แนวทางต่อไปนี้ใช้ช่วยคิด:

1. **Vector RAG มีข้อดีอะไร?** ค้นข้อความที่มีความหมายใกล้คำถามได้แม้ถ้อยคำไม่ตรงกัน อธิบายว่าคำถามใดได้ Chunk ที่มีประโยชน์ในผลทดลอง
2. **Graph RAG มีข้อดีอะไร?** เพิ่มความสัมพันธ์ที่ระบุไว้อย่างชัดเจนและตรวจย้อนกลับเป็นเส้นทางได้ ยกตัวอย่าง `[G…]` ที่เพิ่มหลักฐานให้คำตอบ
3. **คำถามใดเหมาะกับ Graph RAG?** พิจารณาคำถามเรื่องความสัมพันธ์และ Multi-hop เช่น Group ใช้ Technique ใด และ Mitigation ใดเชื่อมกับ Technique นั้น เทียบผลทั้งสองระบบก่อนสรุป
4. **Graph ช่วยลดปัญหาอะไร?** ช่วยค้นความสัมพันธ์ที่ Top-K ข้อความอาจไม่ครอบคลุม และทำให้เห็นทิศทางชัดขึ้น แต่ยังขึ้นกับความครบถ้วนของ Dataset, การเลือก Entity และขอบเขตการค้น
5. **ถ้าไม่มี Graph จะค้นความสัมพันธ์อย่างไร?** ต้องค้นความสัมพันธ์จากข้อความ เช่น เพิ่มคำค้นหรือค้นหลายรอบ แล้วตรวจว่าข้อความรองรับความเชื่อมโยงจริง คะแนนความคล้ายเพียงอย่างเดียวไม่ยืนยันความสัมพันธ์
6. **เหตุใดเหมาะกับ Cybersecurity Intelligence?** ข้อมูลมี Group, Software, Technique และ Mitigation ที่เชื่อมต่อกัน อธิบายประโยชน์ของการตรวจความเชื่อมโยงพร้อมหลักฐานจากการทดลอง

ข้อจำกัดที่ควรเขียนให้ตรงกับระบบนี้: การเลือก Cypher เป็น Template ตามคำสำคัญภาษาอังกฤษ ไม่ใช่การแปลงคำถามธรรมชาติได้ทุกรูปแบบ; ค่าเริ่มต้นคือ Top-K 5 และ Graph Paths สูงสุด 100; Graph Context มีขนาดจำกัด; Dataset ที่ใช้กรองเฉพาะ Entity/Relationship บางประเภท; คำตอบ LLM และรหัสอ้างอิงยังต้องตรวจด้วยคน ไม่มีคะแนนความแม่นยำอัตโนมัติใน Lab นี้ และ Similarity Score ไม่ใช่เปอร์เซ็นต์ความมั่นใจของคำตอบ

**9. เช็ก Task ทั้งหมดและสิ่งส่ง — Task 20**

ตารางนี้เป็นแผนผังงานกับไฟล์ ให้ใช้ผลรันจริงตรวจการทำครบแต่ละ Task:

| Task | งาน | สคริปต์/หลักฐาน |
|---:|---|---|
| 1 | Download Dataset | `download_data.py`, Dataset และ metadata |
| 2 | สำรวจจำนวน Object | `inspect_data.py`, `results/inspect_data.txt` |
| 3 | ออกแบบ Cybersecurity KG | ตาราง Label และทิศทางความสัมพันธ์ในคู่มือนี้ |
| 4 | Prepare Graph Data | `prepare_data.py`, `prepared.json` |
| 5 | สร้าง Graph | `create_graph.py` |
| 6 | ตรวจ Knowledge Graph | `results/graph_counts.json`, Query และภาพ Neo4j |
| 7 | สร้าง Text Dataset | `prepared.json` → `documents` |
| 8 | Chunking | `prepared.json` → `chunks`, ค่าขนาดใน `metadata` |
| 9 | Embedding | `create_embeddings.py` |
| 10 | Store Vector + DESCRIBES | Query Chunk/Entity ใน `lab5_queries.cypher` |
| 11 | Vector Index | `results/embedding_report.json`, `SHOW VECTOR INDEXES` |
| 12 | Vector Retrieval | `graph_rag.py --retrieve-only`, `chunks` และ `score` |
| 13 | Graph Retrieval | ผล `graph.cypher`, `graph.parameters`, `graph.paths` |
| 14 | Graph-Augmented Retrieval | ผล Chunks พร้อม Seeds และ Paths |
| 15 | Context Fusion | `vector_context`, `graph_context` |
| 16 | LLM Answer | `graph_rag.answer` พร้อมการตรวจหลักฐาน |
| 17 | ทดสอบอย่างน้อยห้าคำถาม | `--test-questions`, ตรวจจำนวนผลที่บันทึก |
| 18 | เปรียบเทียบสองระบบ | `--compare`, `results/comparison.json` และตารางวิเคราะห์ |
| 19 | วิเคราะห์หกคำถาม | เขียนคำตอบจากผลทดลองของตนเอง |
| 20 | Final System | แสดงกระบวนการตั้งแต่ Dataset จนถึงคำตอบและผลเปรียบเทียบ |

สิ่งส่งตามใบงานประกอบด้วย Source Code, ภาพและจำนวน Graph, ผล Vector Search พร้อมคะแนน, คำถาม/Cypher/ผล Graph Retrieval, Context พร้อมคำตอบ LLM และตารางเปรียบเทียบ เก็บผลวิเคราะห์กับภาพกราฟเพิ่มในโฟลเดอร์ `results/` ได้ โดยไฟล์ JSON ที่สร้างไม่ได้แทนภาพหน้าจอหรือคำตอบวิเคราะห์ที่ยังไม่ได้เขียน

หากเปิดเครื่องใหม่และนำเข้าข้อมูลกับเวกเตอร์ครบแล้ว เริ่มจากขั้นที่ 1 แล้วไปขั้นที่ 5 หรือ 6 ได้ หากเพิ่งสร้าง Graph ครั้งแรก ต้องรันขั้นที่ 4 ให้จบก่อนใช้ `graph_rag.py`

แหล่งอ้างอิง: รูปแบบ STIX และช่องทางข้อมูลอธิบายใน [MITRE ATT&CK Data & Tools](https://attack.mitre.org/resources/attack-data-and-tools/) และ Dataset มาจาก [MITRE ATT&CK STIX Data](https://github.com/mitre-attack/attack-stix-data) โมเดล Embedding อ้างอิง [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) ส่วนการจัดเก็บเวกเตอร์ดู [Neo4j Vector Indexes](https://neo4j.com/docs/cypher-manual/5/indexes/semantic-indexes/vector-indexes/) และการเรียก LLM ดู [Ollama Chat API](https://docs.ollama.com/api/chat)
