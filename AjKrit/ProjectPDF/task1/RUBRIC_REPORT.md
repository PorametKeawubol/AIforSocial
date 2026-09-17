# ตรวจงานตามเกณฑ์ Task 1 — กราฟ A และ B

Neo4j ที่เปิดใช้งานมีเพียงกราฟ A (`BGE-M3`) และ B (`multilingual-e5-large`) โดยใช้ `Qwen3.8-27B GGUF UD-Q4_K_M` เหมือนกัน ทั้งสองผ่าน source review จาก PDF หน้า 21–30 จึงมีกราฟสุดท้ายชุดละ 81 Nodes / 99 Relationships ส่วนผลของโมเดลก่อน review เก็บไว้ใน property เพื่อเปรียบเทียบอย่างตรวจสอบได้

| เกณฑ์ | หลักฐานในงาน | ผลตรวจ |
|---|---|---|
| 1. การออกแบบโครงสร้าง Graph | 8 domain labels และ 11 relationship types; มีชั้น SourceDocument → Chapter → Case และ Case → Organization → Contact | Schema สื่อความหมายและทิศทางผ่าน validator |
| 2. การสร้าง Nodes | มี Chapter, 16 Case ครบ 1.1–1.16 และศูนย์ข้อมูลคนหาย มูลนิธิกระจกเงา; stable ID และ source metadata | ชุดละ 81 Nodes ไม่มีแนวคิดหรือ ID ซ้ำ |
| 3. การสร้าง Relationships | แยกโครงสร้างหมวด เอกสาร หลักฐาน หน่วยงาน ผู้รับรอง ช่องทาง และข้อยกเว้น | ชุดละ 99 เส้น ทิศทางถูกต้อง และเชื่อมเป็น component เดียว |
| 4. ความถูกต้องและสมบูรณ์ | ทุกเส้นเนื้อหามี evidence, PDF page, owner/condition และ review status | Missing-evidence, duplicate และ cross-group query คืน 0 แถว |
| 5. Cypher และการแสดงผล | `ab_views.cypher` มีกราฟ A, กราฟ B, ตารางเปรียบเทียบ สรุป label และ quality queries | 8 queries รันกับ Neo4j จริง; แยกสี 7 ประเภท |

ผลของโมเดลในรอบสกัดทีละหัวข้อ:

| ชุด | Candidate relationships | Candidate ที่ตรงกับ final content relations |
|---|---:|---:|
| A | 64 | 55/82 |
| B | 67 | 58/82 |

B ดีกว่า A เล็กน้อยตามตัวเลขรอบนี้ แต่ final topology เหมือนกันเพราะตรวจและแก้จากต้นฉบับแล้ว ความแตกต่างยังตรวจได้จาก `group`, `embedding_model`, `model_candidate_found`, `model_candidates_json` และ `review_action`

หลักฐานการรันจริง:

- `artifacts/final_ab/neo4j-switch-report.json`: active database มี A/B เท่านั้น รวม 162 Nodes / 198 Relationships
- `artifacts/final_ab/query-verification.json`: 8 Cypher queries รันผ่าน
- `artifacts/final_ab/A/neo4j-focus.png` และ `B/neo4j-focus.png`: ภาพ Neo4j Browser ที่แยกสีตามชนิด Node

หลักฐานเหล่านี้รองรับระดับ “ดีมาก” ตามรายละเอียด rubric แต่คะแนนจริงเป็นดุลยพินิจของผู้ประเมิน
