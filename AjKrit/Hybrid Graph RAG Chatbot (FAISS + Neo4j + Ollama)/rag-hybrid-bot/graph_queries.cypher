// Run in the dedicated Neo4j Browser at http://localhost:7475.
// Use the source SHA-256 from artifacts/source.json when filtering multiple PDFs.
MATCH p=(s:RagEntity)-[:SUBJECT_OF]->(f:RagFact)-[:OBJECT]->(o:RagEntity)
RETURN p LIMIT 100;

MATCH (s:RagEntity)-[:SUBJECT_OF]->(f:RagFact)-[:OBJECT]->(o:RagEntity)
WHERE s.name CONTAINS 'แคงเกอร์' OR o.name CONTAINS 'แคงเกอร์'
RETURN s.name AS subject,f.relation AS relation,o.name AS object,
       f.page AS pdf_page,f.evidence AS evidence;

MATCH (f:RagFact)-[:EVIDENCED_BY]->(p:RagPage)<-[:PART_OF]-(c:RagChunk)
RETURN f.page AS page,count(DISTINCT f) AS facts,count(DISTINCT c) AS chunks
ORDER BY page;
