// 1. ภาพรวมกราฟตรวจแล้ว (78 nodes / 81 relationships)
MATCH (c:Task1Reviewed:Case)-[r]->(e:Task1Reviewed)
RETURN c,r,e;

// 2. มุมมองแนะนำสำหรับภาพส่งงาน: คนหาย + พรากผู้เยาว์ + รถหาย
MATCH (c:Task1Reviewed:Case)-[r]->(e:Task1Reviewed)
WHERE c.section IN ['1.4','1.5','1.13']
RETURN c,r,e;

// 3. เอกสาร/หลักฐานร่วมกัน โดยยังเก็บเจ้าของและเงื่อนไขบนเส้น
MATCH (c:Task1Reviewed:Case)-[r:HAS_SUPPORTING_EVIDENCE]->(e:Task1Reviewed)
WITH e,collect(c) AS cases,collect(r) AS links
WHERE size(cases)>1
RETURN e,cases,links;

// 4. ทิศทางและข้อยกเว้น: บัตรหาย ไม่ต้องแจ้งตำรวจ ตามข้อความในคู่มือ
MATCH (c:Task1Reviewed:Case {section:'1.2'})-[r]->(e:Task1Reviewed)
RETURN c,r,e;

// 5. อย่าสับสนเอกสารที่หายกับเอกสารประกอบ และเอกสารที่ได้รับ
MATCH (c:Task1Reviewed:Case {section:'1.3'})-[r]->(e:Task1Reviewed)
RETURN c,r,e;

// 6. ตารางหลักฐานที่ใช้ตรวจเส้นทุกเส้น
MATCH (c:Task1Reviewed:Case)-[r]->(e:Task1Reviewed)
RETURN c.section AS section,c.name AS topic,type(r) AS relation,e.name AS entity,
       r.owner AS owner,r.condition AS condition,r.pdf_pages AS pdf_pages,
       r.printed_pages AS printed_pages,r.evidence AS source_text,r.review_status AS review
ORDER BY toInteger(split(c.section,'.')[1]),entity;

// 7. จำนวน Nodes แยก Label
MATCH (n:Task1Reviewed) RETURN n.type AS label,count(*) AS total ORDER BY label;

// 8. ความครบทุกหัวข้อ: แต่ละหัวข้อต้องมีอย่างน้อยหนึ่งเส้น
MATCH (c:Task1Reviewed:Case)
OPTIONAL MATCH (c)-[r]->(:Task1Reviewed)
RETURN c.section,c.name,count(r) AS relations ORDER BY toInteger(split(c.section,'.')[1]);

// 9. Duplicate nodes: ผลที่คาดหวัง 0 แถว
MATCH (n:Task1Reviewed) WITH n.type AS label,n.name AS name,count(*) AS total
WHERE total>1 RETURN label,name,total;

// 10. Duplicate relationships: ผลที่คาดหวัง 0 แถว
MATCH (c:Task1Reviewed)-[r]->(e:Task1Reviewed)
WITH c.id AS source,type(r) AS relation,e.id AS target,r.condition AS condition,r.owner AS owner,count(*) AS total
WHERE total>1 RETURN source,relation,target,condition,owner,total;

// 11. หลักฐาน/เลขหน้าหาย หรือทิศทางผิด: ผลที่คาดหวัง 0 แถว
MATCH (c:Task1Reviewed)-[r]->(e:Task1Reviewed)
WHERE NOT c:Case OR e:Case OR r.evidence IS NULL OR trim(r.evidence)='' OR size(coalesce(r.pdf_pages,[]))=0
RETURN c,r,e;
