MATCH p=(s:WorksheetModelII)-[r]->(t:WorksheetModelII) RETURN p LIMIT 80;

MATCH (s:WorksheetModelII)-[r]->(t:WorksheetModelII) RETURN s.name AS subject,type(r) AS relation,t.name AS object,r.page AS page,r.evidence AS evidence;

MATCH (n:WorksheetModelII) RETURN properties(n);
