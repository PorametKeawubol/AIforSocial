// V2 group B (change B to A for comparison)
MATCH (c:SectionKG:Case {group:'B'})-[r]->(e:SectionKG {group:'B'}) RETURN c,r,e;

// Connected evidence categories across missing persons/minors/vehicle cases
MATCH (c:SectionKG:Case {group:'B'})-[r]->(e:SectionKG {group:'B'})
WHERE c.section IN ['1.4','1.5','1.13'] RETURN c,r,e;

// Missing model output must be visible, not hidden by scaffolded Case count
MATCH (c:SectionKG:Case {group:'B'})
WHERE NOT (c)-->() RETURN c.section,c.name;

// Semantic review queue
MATCH (c:SectionKG:Case {group:'B'})-[r]->(e:SectionKG)
RETURN c.section,type(r),e.name,e.type,r.owner,r.condition,r.pdf_pages,r.evidence,r.review_flags;

// Duplicate concepts (expect no rows)
MATCH (n:SectionKG {group:'B'}) WITH n.type AS type,n.name AS name,count(*) AS count
WHERE count>1 RETURN type,name,count;
