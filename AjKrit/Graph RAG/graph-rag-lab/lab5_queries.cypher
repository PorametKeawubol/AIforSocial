// Lab 5: คัดลอกไป Neo4j Browser และรันทีละชุดคำสั่ง
// ใช้เฉพาะ Label/Property ของ Lab 5 เพื่อให้จำนวนไม่ปนกับ Lab ก่อนหน้า

// 1. ตรวจ Snapshot และสถานะนำเข้า
MATCH (m:Lab5Metadata {id: 'dataset'})
RETURN m.sha256 AS dataset_sha256,
       m.graph_complete AS graph_complete,
       m.embeddings_complete AS embeddings_complete,
       m.entity_count AS entity_count,
       m.relationship_count AS relationship_count,
       m.chunk_count AS chunk_count,
       m.embedding_model AS embedding_model;

// 2. Task 6 — จำนวน Entity แยกตามประเภท (ไม่รวม Chunk/Metadata)
MATCH (n:AttackEntity {lab: 'lab5'})
RETURN n.entity_type AS type, count(n) AS count
ORDER BY type;

// 3. Task 6 — จำนวน Relationship ใน Knowledge Graph
MATCH (a:AttackEntity {lab: 'lab5'})-[r]->(b:AttackEntity {lab: 'lab5'})
WHERE r.lab = 'lab5'
RETURN a.entity_type AS source_type, type(r) AS relationship,
       b.entity_type AS target_type, count(r) AS count
ORDER BY source_type, relationship, target_type;

// 4. Task 6 — กราฟตัวอย่าง APT29; เลือกมุมมอง Graph และบันทึกภาพ
// LIMIT ใช้จำกัดขนาดภาพ จึงไม่ใช่รายการความสัมพันธ์ทั้งหมดของ APT29
MATCH p=(g:AttackEntity:Group {name: 'APT29', lab: 'lab5'})
        -[:USES]->(n:AttackEntity {lab: 'lab5'})
RETURN p
LIMIT 40;

// 5. Task 10 — แสดงว่า Chunk เชื่อมกลับไป Entity ได้
MATCH p=(c:Chunk:Lab5Chunk {lab: 'lab5'})-[:DESCRIBES]->
        (e:AttackEntity:Group {name: 'APT29', lab: 'lab5'})
RETURN p
LIMIT 10;

// 5b. ตรวจข้อความและจำนวนมิติ โดยไม่แสดงเวกเตอร์ทั้งก้อน
MATCH (c:Chunk:Lab5Chunk {lab: 'lab5'})-[:DESCRIBES]->(e:AttackEntity {lab: 'lab5'})
RETURN c.id AS chunk_id, e.name AS entity, c.text AS text,
       size(c.embedding) AS dimensions, c.embedding_model AS embedding_model
LIMIT 5;

// 5c. จำนวน Chunk และลิงก์กลับ Entity
MATCH (c:Lab5Chunk {lab: 'lab5'})
OPTIONAL MATCH (c)-[r:DESCRIBES]->(:AttackEntity {lab: 'lab5'})
RETURN count(DISTINCT c) AS chunks, count(r) AS describes_relationships;

// 6. Task 11 — Vector Index ต้องเป็น ONLINE, 384 มิติ และ cosine
SHOW VECTOR INDEXES
YIELD name, state, populationPercent, labelsOrTypes, properties, options
WHERE name = 'lab5_chunk_embedding_index'
RETURN name, state, populationPercent, labelsOrTypes, properties, options;

// 7. Task 13 / Question 1 — Technique ที่ APT29 ใช้ พร้อมหลักฐาน STIX
MATCH (g:AttackEntity:Group {name: 'APT29', lab: 'lab5'})
      -[r:USES]->(t:AttackEntity:Technique {lab: 'lab5'})
RETURN g.name AS actor, t.attack_id AS technique_id, t.name AS technique,
       r.id AS relationship_id, r.description AS evidence, t.url AS source_url
ORDER BY technique_id;

// 8. Task 17 / Question 2 — Malware ที่เกี่ยวข้องกับ APT29
MATCH (g:AttackEntity:Group {name: 'APT29', lab: 'lab5'})
      -[r:USES]->(m:AttackEntity:Malware {lab: 'lab5'})
RETURN g.name AS actor, m.attack_id AS malware_id, m.name AS malware,
       r.description AS evidence, m.url AS source_url
ORDER BY malware;

// 9. Task 17 / Question 3 — Technique ที่ SUNBURST ใช้
MATCH (m:AttackEntity:Malware {name: 'SUNBURST', lab: 'lab5'})
      -[r:USES]->(t:AttackEntity:Technique {lab: 'lab5'})
RETURN m.name AS malware, t.attack_id AS technique_id, t.name AS technique,
       r.description AS evidence, t.url AS source_url
ORDER BY technique_id;

// 10. Task 17 / Question 4 — Mitigation ของ PowerShell
// ทิศทางในข้อมูลคือ Mitigation --MITIGATES--> Technique
MATCH (m:AttackEntity:Mitigation {lab: 'lab5'})
      -[r:MITIGATES]->(t:AttackEntity:Technique {name: 'PowerShell', lab: 'lab5'})
RETURN t.name AS technique, m.attack_id AS mitigation_id, m.name AS mitigation,
       r.description AS evidence, m.url AS source_url
ORDER BY mitigation_id;

// 11. Task 17 / Question 5 — ความสัมพันธ์รอบ APT29
// ค้นได้ทั้งสองทิศ แต่แสดงต้นทาง/ปลายทางจริงของ Relationship
MATCH (g:AttackEntity:Group {name: 'APT29', lab: 'lab5'})
      -[r:USES|MITIGATES|SUBTECHNIQUE_OF]-(other:AttackEntity {lab: 'lab5'})
RETURN startNode(r).name AS source, type(r) AS relationship,
       endNode(r).name AS target, r.description AS evidence
ORDER BY relationship, target, source
LIMIT 100;

// 12. Task 18 — Multi-hop: Mitigation ของ Technique ที่ APT29 ใช้
// Group --USES--> Technique <--MITIGATES-- Mitigation
// เส้นทางนี้ไม่ได้ยืนยันว่า Mitigation จะหยุด Group ได้เสมอในทุกสถานการณ์
MATCH (g:AttackEntity:Group {name: 'APT29', lab: 'lab5'})
      -[u:USES]->(t:AttackEntity:Technique {lab: 'lab5'})
      <-[r:MITIGATES]-(m:AttackEntity:Mitigation {lab: 'lab5'})
RETURN g.name AS actor, t.attack_id AS technique_id, t.name AS technique,
       m.attack_id AS mitigation_id, m.name AS mitigation,
       u.description AS use_evidence, r.description AS mitigation_evidence
ORDER BY technique_id, mitigation_id
LIMIT 100;

// 12b. แสดงเส้นทาง Multi-hop เป็นกราฟสำหรับบันทึกภาพ
MATCH p=(g:AttackEntity:Group {name: 'APT29', lab: 'lab5'})
        -[:USES]->(:AttackEntity:Technique {lab: 'lab5'})
        <-[:MITIGATES]-(:AttackEntity:Mitigation {lab: 'lab5'})
RETURN p
LIMIT 20;

// 13. ตรวจความสัมพันธ์ Sub-technique กับ Parent Technique
MATCH (sub:AttackEntity:Technique {name: 'PowerShell', lab: 'lab5'})
      -[r:SUBTECHNIQUE_OF]->(parent:AttackEntity:Technique {lab: 'lab5'})
RETURN sub.attack_id AS subtechnique_id, sub.name AS subtechnique,
       type(r) AS relationship, parent.attack_id AS parent_id, parent.name AS parent;

// Task 12 Vector Search จากคำถามต้องสร้าง Query Embedding ก่อน
// ใช้ python graph_rag.py "What techniques are associated with APT29?" --retrieve-only
// ผล JSON มี vector_query, chunks, score และ Graph Context ให้ตรวจสอบ
