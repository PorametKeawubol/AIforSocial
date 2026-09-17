MATCH p=(s:WorksheetModelIII)-[r]->(t:WorksheetModelIII) RETURN p LIMIT 80;

MATCH (s:WorksheetModelIII)-[r]->(t:WorksheetModelIII) RETURN s.name AS subject,type(r) AS relation,t.name AS object,r.page AS page,r.evidence AS evidence;

MATCH (n:WorksheetModelIII) RETURN properties(n);
