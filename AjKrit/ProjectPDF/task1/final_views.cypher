// 1. ภาพรวมกราฟสุดท้าย เชื่อมจากคู่มือไปทุกหัวข้อและ Entity
MATCH (d:FinalTask1:SourceDocument)-[s:HAS_SECTION]->(c:FinalTask1:Case)
OPTIONAL MATCH (c)-[r]->(e:FinalTask1)
RETURN d,s,c,r,e;

// 2. ภาพแนะนำสำหรับนำเสนอ: คนหาย รถ/เรือหาย และพรากผู้เยาว์
MATCH (d:FinalTask1:SourceDocument)-[s:HAS_SECTION]->(c:FinalTask1:Case)
WHERE c.section IN ['1.4','1.5','1.13']
OPTIONAL MATCH (c)-[r]->(e:FinalTask1)
RETURN d,s,c,r,e;

// 3. เอกสารหรือหลักฐานประเภทเดียวกันที่เชื่อมหลายหัวข้อ
MATCH (c:FinalTask1:Case)-[r:HAS_SUPPORTING_EVIDENCE]->(e:FinalTask1)
WITH e,collect(c) AS cases,collect(r) AS links
WHERE size(cases)>1
RETURN e,cases,links;

// 4. ตัวอย่างทิศทางและข้อยกเว้น: บัตรหายไม่ต้องแจ้งตำรวจตามคู่มือ
MATCH (c:FinalTask1:Case {section:'1.2'})-[r]->(e:FinalTask1)
RETURN c,r,e;

// 5. แสดงว่า B เสนอรายการใด และรายการใดผู้ตรวจเติม/แก้จากต้นฉบับ
MATCH (c:FinalTask1:Case)-[r]->(e:FinalTask1)
RETURN c.section AS section,type(r) AS relation,e.name AS entity,
       r.model_candidate_found AS found_by_B_candidate,
       r.review_action AS review_action,r.condition AS condition,
       r.owner AS owner,r.pdf_pages AS pdf_pages,r.evidence AS evidence
ORDER BY toInteger(split(c.section,'.')[1]),entity;

// 6. จำนวน Nodes แยกประเภท
MATCH (n:FinalTask1)
RETURN n.type AS label,count(*) AS total ORDER BY label;

// 7. ทุกหัวข้อต้องมีความสัมพันธ์เนื้อหาอย่างน้อยหนึ่งเส้น
MATCH (c:FinalTask1:Case)
OPTIONAL MATCH (c)-[r]->(e:FinalTask1)
RETURN c.section,c.name,count(r) AS relations
ORDER BY toInteger(split(c.section,'.')[1]);

// 8. Duplicate concepts: คาดหวัง 0 แถว
MATCH (n:FinalTask1)
WITH n.type AS label,n.name AS name,count(*) AS total
WHERE total>1 RETURN label,name,total;

// 9. Duplicate relationships: คาดหวัง 0 แถว
MATCH (a:FinalTask1)-[r]->(b:FinalTask1)
WITH a.id AS source,type(r) AS relation,b.id AS target,
     coalesce(r.condition,'') AS condition,coalesce(r.owner,'') AS owner,count(*) AS total
WHERE total>1 RETURN source,relation,target,condition,owner,total;

// 10. เส้นเนื้อหาที่ไม่มีหลักฐานหรือเลขหน้า: คาดหวัง 0 แถว
MATCH (c:FinalTask1:Case)-[r]->(e:FinalTask1)
WHERE r.evidence IS NULL OR trim(r.evidence)='' OR size(coalesce(r.pdf_pages,[]))=0
RETURN c,r,e;

// 11. ตรวจว่าทุกโหนดเข้าถึงได้จากคู่มือภายใน 2 hops: คาดหวัง 79
MATCH (root:FinalTask1:SourceDocument)
MATCH p=(root)-[*0..2]-(n:FinalTask1)
RETURN count(DISTINCT n) AS reachable_nodes;
