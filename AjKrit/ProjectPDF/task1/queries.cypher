// Overview: exclude large text properties in table views.
MATCH (d:Project2:Document)-[:HAS_SECTION]->(s:Section)
RETURN d,s LIMIT 80;

// Example: missing persons and vehicles, with original passage evidence.
MATCH (s:Project2:Section)-[r:HAS_PASSAGE]->(p:Passage)
WHERE s.section_number IN ['1.4','1.5']
OPTIONAL MATCH (p)-[m:MENTIONS]->(e:Entity)
RETURN s,r,p,m,e;

// Explicit source-specific relations; review evidence before accepting.
MATCH (s:Project2:Section)-[r:CONTACT_VIA|HAS_GUIDANCE]->(t)
RETURN s,r,t;

// Shared entities across passages. These edges mean mentions, not requirements.
MATCH (p:Project2:Passage)-[r:MENTIONS]->(e:Entity)
WHERE e.name = 'หนังสือมอบอำนาจ'
RETURN p,r,e LIMIT 50;

// Inspect extraction problems.
MATCH (p:Project2:Page {needs_text_review:true})
RETURN p.pdf_page, p.printed_page, p.text ORDER BY p.pdf_page;
