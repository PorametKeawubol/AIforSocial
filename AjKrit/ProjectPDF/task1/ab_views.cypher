// A: ภาพแนะนำสำหรับนำเสนอ (คนหาย รถ/เรือหาย และพรากผู้เยาว์)
MATCH (d:SourceDocument {group:'A'})-[h:HAS_CHAPTER]->(ch:Chapter {group:'A'})-[s:HAS_SECTION]->(c:Case {group:'A'})
WHERE c.section IN ['1.4','1.5','1.13']
OPTIONAL MATCH p=(c)-[*1..2]->(e {group:'A'})
RETURN d,h,ch,s,c,p,e;

// B: ภาพแนะนำสำหรับนำเสนอ (คนหาย รถ/เรือหาย และพรากผู้เยาว์)
MATCH (d:SourceDocument {group:'B'})-[h:HAS_CHAPTER]->(ch:Chapter {group:'B'})-[s:HAS_SECTION]->(c:Case {group:'B'})
WHERE c.section IN ['1.4','1.5','1.13']
OPTIONAL MATCH p=(c)-[*1..2]->(e {group:'B'})
RETURN d,h,ch,s,c,p,e;

// เปรียบเทียบ A/B: จำนวน Node, Relationship และความสัมพันธ์ที่โมเดลพบก่อนตรวจต้นฉบับ
MATCH (n)
WITH n.group AS group,count(n) AS nodes
MATCH (a {group:group})-[r]->(b {group:group})
RETURN group,nodes,count(r) AS relationships,
       sum(CASE WHEN r.model_candidate_found THEN 1 ELSE 0 END) AS model_candidate_matches
ORDER BY group;

// จำนวน Node แยกชนิดในแต่ละชุด
MATCH (n)
RETURN n.group AS group,labels(n)[0] AS node_type,count(*) AS total
ORDER BY group,node_type;

// ตรวจความสัมพันธ์ซ้ำ: ต้องได้ 0 แถว
MATCH (a)-[r]->(b)
WITH a.group AS group,a.id AS source,type(r) AS relation,b.id AS target,
     coalesce(r.condition,'') AS condition,coalesce(r.owner,'') AS owner,count(*) AS total
WHERE total>1
RETURN group,source,relation,target,condition,owner,total;

// ตรวจข้อมูลที่ไม่มีหลักฐานหรือเลขหน้า: ต้องได้ 0 แถว
MATCH (c:Case)-[r]->(e)
WHERE type(r)<>'HAS_SECTION' AND
      (r.evidence IS NULL OR trim(r.evidence)='' OR size(coalesce(r.pdf_pages,[]))=0)
RETURN c,r,e;

// ตรวจความสัมพันธ์ข้าม A/B: ต้องได้ 0 แถว
MATCH (a)-[r]->(b)
WHERE a.group<>b.group
RETURN a,r,b;

// ทุก Node ต้องเข้าถึงได้จากคู่มือภายใน 4 hops: ต้องได้ A=81 และ B=81
MATCH (root:SourceDocument)
MATCH p=(root)-[*0..4]-(n {group:root.group})
RETURN root.group AS group,count(DISTINCT n) AS reachable_nodes
ORDER BY group;

// ดูจำนวนเต็มของแต่ละกราฟ: ต้องได้ชุดละ 81 Nodes / 99 Relationships
MATCH (n)
WITH n.group AS group,count(n) AS nodes
MATCH (a {group:group})-[r]->(b {group:group})
RETURN group,nodes,count(r) AS relationships
ORDER BY group;

// แสดงกราฟ A เต็มทุก 81 Nodes / 99 Relationships
MATCH (a {group:'A'})-[r]->(b {group:'A'})
RETURN a,r,b;

// แสดงกราฟ B เต็มทุก 81 Nodes / 99 Relationships
MATCH (a {group:'B'})-[r]->(b {group:'B'})
RETURN a,r,b;
