MATCH p=(s:WorksheetModelI)-[r]->(t:WorksheetModelI) RETURN p LIMIT 80;

MATCH (s:WorksheetModelI)-[r]->(t:WorksheetModelI) RETURN s.name AS subject,type(r) AS relation,t.name AS object,r.page AS page,r.evidence AS evidence;

MATCH (n:WorksheetModelI) RETURN properties(n);
