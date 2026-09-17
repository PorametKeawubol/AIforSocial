// ภาพรวม A: เห็นคู่มือ หมวด 1 และหัวข้อครบ 1.1–1.16 (18 Nodes / 17 Relationships)
MATCH (d:SourceDocument {group:'A'})-[h:HAS_CHAPTER]->
      (ch:Chapter {group:'A'})-[s:HAS_SECTION]->(c:Case {group:'A'})
RETURN d,h,ch,s,c;

// ภาพละเอียด A: หัวข้อ 1.4 มี Evidence และ Organization -> Contact หลายชั้น
MATCH (d:SourceDocument {group:'A'})-[h:HAS_CHAPTER]->
      (ch:Chapter {group:'A'})-[s:HAS_SECTION]->(c:Case {group:'A',section:'1.4'})
OPTIONAL MATCH p=(c)-[*1..2]->(e {group:'A'})
RETURN d,h,ch,s,c,p,e;

// ภาพรวม B: เห็นคู่มือ หมวด 1 และหัวข้อครบ 1.1–1.16 (18 Nodes / 17 Relationships)
MATCH (d:SourceDocument {group:'B'})-[h:HAS_CHAPTER]->
      (ch:Chapter {group:'B'})-[s:HAS_SECTION]->(c:Case {group:'B'})
RETURN d,h,ch,s,c;

// ภาพละเอียด B: หัวข้อ 1.4 มี Evidence และ Organization -> Contact หลายชั้น
MATCH (d:SourceDocument {group:'B'})-[h:HAS_CHAPTER]->
      (ch:Chapter {group:'B'})-[s:HAS_SECTION]->(c:Case {group:'B',section:'1.4'})
OPTIONAL MATCH p=(c)-[*1..2]->(e {group:'B'})
RETURN d,h,ch,s,c,p,e;
