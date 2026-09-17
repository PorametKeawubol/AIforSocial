// 1. All semantic relationships (263 relationships)
MATCH p=(s:PlantGuide2019Reviewed)-[r]->(t:PlantGuide2019Reviewed)
WHERE r.layer = 'semantic'
RETURN p;

// 2. Cross-crop path through shared biological controls, preserving disease identities
MATCH p=(c1:PlantGuide2019Reviewed:Crop {name:'พริก'})-[:HAS_DISEASE]->(d1)
-[:MANAGED_BY]->(t1)-[:USES_BIOCONTROL]->(b:Biocontrol)
<-[:USES_BIOCONTROL]-(t2)<-[:MANAGED_BY]-(d2)<-[:HAS_DISEASE]-(c2:PlantGuide2019Reviewed:Crop {name:'ทุเรียน'})
RETURN p;

// 3. Common vector across different chili virus diseases
MATCH p=(c:PlantGuide2019Reviewed:Crop {name:'พริก'})-[:HAS_DISEASE]->(d)
-[:TRANSMITTED_BY]->(v:Pest {name:'เพลี้ยอ่อน'})
RETURN p;

// 4. Evidence rows: full Page.text + source year/hash, no inferred treatment efficacy
MATCH (s:PlantGuide2019Reviewed)-[r]->(t:PlantGuide2019Reviewed)
WHERE r.layer='semantic' AND s.name CONTAINS 'CVMV'
UNWIND r.pages AS pageNumber
MATCH (p:PlantGuide2019Reviewed:Page {number:pageNumber})
RETURN s.name AS subject, type(r) AS relation, t.name AS object,
       r.condition AS condition, p.number AS page, p.text AS evidence, r.source_sha256 AS source;

// 5. Data-quality checks: no missing page references
MATCH (:PlantGuide2019Reviewed)-[r]->(:PlantGuide2019Reviewed)
WHERE r.layer='semantic'
RETURN count(r) AS relationships, sum(CASE WHEN size(r.pages)>0 THEN 1 ELSE 0 END) AS with_pages;
