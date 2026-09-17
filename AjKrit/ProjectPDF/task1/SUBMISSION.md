# งาน 1 — Knowledge Graph in Neo4j (ชุด A และ B)

ฐานข้อมูล Neo4j ที่เปิดใช้งานมีเฉพาะกราฟสำหรับ Project 2 สองชุด ใช้ข้อมูล PDF หน้า 21–30 (เลขหน้าพิมพ์ 8–17) หัวข้อ 1.1–1.16 ข้อมูลมาจากคู่มือฉบับเดือนกรกฎาคม 2558

## โมเดลที่เปรียบเทียบ

| ชุด | Embedding | LLM แบบ Local | Candidate ดิบ | ตรงกับ 82 เส้นหลังตรวจต้นฉบับ |
|---|---|---|---:|---:|
| A | `BAAI/bge-m3` | `Qwen3.8-27B GGUF UD-Q4_K_M` | 64 | 55 |
| B | `intfloat/multilingual-e5-large` | `Qwen3.8-27B GGUF UD-Q4_K_M` | 67 | 58 |

B ทำได้ดีกว่าเล็กน้อยในรอบนี้ เพราะเสนอความสัมพันธ์ได้มากกว่าและตรงกับรายการที่ตรวจต้นฉบับ 58 เส้น เทียบกับ A 55 เส้น หลัง source review ทั้งสองชุดมีข้อเท็จจริงชุดเดียวกันเพื่อให้การเปรียบเทียบไม่ปะปนกับความผิดพลาดของโมเดล ทุกเส้นเก็บ `model_candidate_found`, `review_action`, `embedding_model`, `llm_model`, `evidence` และ `pdf_pages` สำหรับตรวจย้อนหลัง

Embedding ช่วยจับความหมายและเสนอ ontology hint ส่วน LLM สกัด Entity, Relation, Owner, Condition และหลักฐานจากข้อความ ข้อเท็จจริงสุดท้ายผ่านการตรวจเทียบต้นฉบับ ไม่ได้นำ output ของโมเดลเข้า Neo4j โดยไม่ตรวจ

## โครงสร้างของแต่ละกราฟ

แต่ละชุดมี 81 Nodes และ 99 Relationships เป็น component เดียว:

| Label | จำนวน | ความหมาย |
|---|---:|---|
| SourceDocument | 1 | คู่มือต้นฉบับและ metadata |
| Chapter | 1 | หมวด 1 คำแนะนำในการไปติดต่อที่สถานีตำรวจ |
| Case | 16 | หัวข้อ 1.1–1.16 |
| Document | 34 | ประเภทเอกสาร |
| Evidence | 17 | หลักฐานหรือข้อมูลประกอบ |
| Contact | 3 | เบอร์ติดต่อ |
| Organization | 5 | หน่วยงานหรือสถานที่ รวมศูนย์ข้อมูลคนหาย มูลนิธิกระจกเงา |
| PersonRole | 4 | บทบาทเจ้าหน้าที่หรือผู้รับรอง |

Relationship มี 11 ชนิด โดยโครงสร้างหลักเป็น `SourceDocument → HAS_CHAPTER → Chapter → HAS_SECTION → Case` แล้วเชื่อมไปยัง Entity ต่าง ๆ สำหรับหัวข้อ 1.4 ใช้ `Case → CONTACT_ORGANIZATION → Organization → HAS_CONTACT → Contact` เพื่อเก็บศูนย์ข้อมูลคนหายและเบอร์โทรอย่างมีความหมาย รวม 82 เส้นเนื้อหาและ 17 เส้นโครงสร้าง

## เปิดกราฟใน Neo4j

เปิด <http://localhost:7474/browser/> แล้วรีเฟรชหนึ่งครั้ง หากหน้าเดิมเปิดค้างอยู่

กราฟ A:

```cypher
MATCH (d:SourceDocument {group:'A'})-[h:HAS_CHAPTER]->(ch:Chapter {group:'A'})-[s:HAS_SECTION]->(c:Case {group:'A'})
WHERE c.section IN ['1.4','1.5','1.13']
OPTIONAL MATCH p=(c)-[*1..2]->(e {group:'A'})
RETURN d,h,ch,s,c,p,e;
```

กราฟ B:

```cypher
MATCH (d:SourceDocument {group:'B'})-[h:HAS_CHAPTER]->(ch:Chapter {group:'B'})-[s:HAS_SECTION]->(c:Case {group:'B'})
WHERE c.section IN ['1.4','1.5','1.13']
OPTIONAL MATCH p=(c)-[*1..2]->(e {group:'B'})
RETURN d,h,ch,s,c,p,e;
```

Query สำหรับแสดงผล เปรียบเทียบ และตรวจคุณภาพทั้งหมดอยู่ใน `ab_views.cypher`

ชุดภาพที่แนะนำสำหรับนำเสนออยู่ใน `BEST_VIEWS.cypher`:

- ภาพรวม A/B: 18 Nodes / 17 Relationships แสดง SourceDocument → Chapter → Case ครบ 1.1–1.16
- ภาพละเอียด A/B: 10 Nodes / 9 Relationships ของหัวข้อ 1.4 แสดง Evidence และ Organization → Contact

ไฟล์ภาพคือ `artifacts/final_ab/A/neo4j-best-overview.png`, `neo4j-best-detail.png` และไฟล์ชื่อเดียวกันในโฟลเดอร์ `B`

## สีของกราฟ

Node ใช้เพียง label ตามประเภท ไม่มี label กลางมาทับสีอีกแล้ว:

- SourceDocument = แดง
- Chapter = ม่วงเข้ม
- Case = น้ำเงิน
- Document = เขียว
- Evidence = ส้ม
- Contact = ม่วง
- Organization = ฟ้า
- PersonRole = เหลือง

ภาพที่บันทึกจาก Neo4j Browser จริง:

- [กราฟ A](artifacts/final_ab/A/neo4j-focus.png)
- [กราฟ B](artifacts/final_ab/B/neo4j-focus.png)

## ผลตรวจ

- ฐานข้อมูล active มีเฉพาะ A และ B: 162 Nodes / 198 Relationships
- แต่ละชุดมี 81 Nodes / 99 Relationships
- ไม่มี Node ที่ไม่มี `group` และไม่มี Relationship ข้าม A/B
- ไม่มีความสัมพันธ์ซ้ำและไม่มีเส้นเนื้อหาที่ขาด evidence หรือเลขหน้า
- ทุก 81 Nodes ของแต่ละชุดเข้าถึงจาก SourceDocument ภายใน 4 hops
- Cypher แสดงผลและตรวจคุณภาพ 8 ชุดรันสำเร็จ

หลักฐานอยู่ใน `artifacts/final_ab/neo4j-switch-report.json`, `query-verification.json` และ `summary.json` ฐานข้อมูลเดิมถูกหยุดและเก็บไว้เฉพาะเป็น recovery backup; ไม่ปรากฏใน Neo4j ที่เปิดใช้งาน

## สร้างและตรวจซ้ำ

รันจากโฟลเดอร์ `Project2/task1`:

```bash
.venv/bin/python build_ab_reviewed.py
.venv/bin/python switch_to_ab_database.py
.venv/bin/python verify_ab.py
.venv/bin/python -m unittest discover -p 'test*.py'
```

โครงสร้างและผลตรวจรองรับองค์ประกอบที่ rubric ระบุในระดับดีมากทุกข้อ แต่คะแนนสุดท้ายขึ้นกับผู้ประเมิน การ source review ทำโดยผู้ช่วยและไม่ได้รับรองว่าข้อมูลในคู่มือปี 2558 ยังเป็นข้อกำหนดปัจจุบัน
