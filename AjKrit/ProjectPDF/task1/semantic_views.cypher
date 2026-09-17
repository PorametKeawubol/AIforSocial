// Overview, group A. Change A to B for comparison.
MATCH (c:Semantic10:Case {group:'A'})-[r]->(e:Semantic10)
RETURN c,r,e LIMIT 150;

// Missing people and vehicles.
MATCH (c:Semantic10:Case {group:'A'})-[r]->(e:Semantic10)
WHERE c.name STARTS WITH '1.4 ' OR c.name STARTS WITH '1.5 '
RETURN c,r,e;

// Shared evidence. Review whether conditions differ by case.
MATCH (c:Semantic10:Case {group:'A'})-[r]->(e:Semantic10)
WITH e, collect(c) AS cases, collect(r) AS links
WHERE size(cases)>1
RETURN e,cases,links;

// Evidence table.
MATCH (c:Semantic10:Case {group:'A'})-[r]->(e:Semantic10)
RETURN c.name,e.name,type(r),r.condition,r.evidence,r.pdf_page,r.status
ORDER BY r.pdf_page;
