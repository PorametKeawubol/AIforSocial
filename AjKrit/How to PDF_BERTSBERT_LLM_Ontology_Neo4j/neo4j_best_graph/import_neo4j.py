"""Import the reviewed dataset into an authorized Neo4j database without deleting data."""
from pathlib import Path
import os,json,re
from collections import defaultdict
from neo4j import GraphDatabase
R=Path(__file__).resolve().parent

def main():
 uri=os.environ.get('NEO4J_URI','bolt://localhost:7687')
 user=os.environ.get('NEO4J_USERNAME','neo4j')
 password=os.environ.get('NEO4J_PASSWORD')
 if not password:raise SystemExit('Set NEO4J_PASSWORD in your environment before importing.')
 data=json.loads((R/'graph.json').read_text()); groups=defaultdict(list); rels=defaultdict(list)
 for n in data['nodes']:
  if not re.fullmatch('[A-Za-z][A-Za-z0-9_]*',n['kind']):raise ValueError('Invalid label')
  groups[n['kind']].append(n)
 for e in data['relationships']:
  if not re.fullmatch('[A-Z][A-Z0-9_]*',e['relation']):raise ValueError('Invalid relation type')
  rels[e['relation']].append({k:v for k,v in e.items() if k!='evidence'})
 def write(tx):
  for kind,rows in groups.items():
   tx.run(f'UNWIND $rows AS row MERGE (n:PlantGuide2019Reviewed {{id:row.id}}) SET n:{kind} SET n += row',rows=rows).consume()
  for kind,rows in rels.items():
   tx.run(f'UNWIND $rows AS row MATCH (s:PlantGuide2019Reviewed {{id:row.source}}), (t:PlantGuide2019Reviewed {{id:row.target}}) MERGE (s)-[r:{kind} {{id:row.id}}]->(t) SET r += row',rows=rows).consume()
 with GraphDatabase.driver(uri,auth=(user,password)) as db:
  db.verify_connectivity()
  with db.session(database=os.environ.get('NEO4J_DATABASE','neo4j')) as s:
   s.run('CREATE CONSTRAINT plant_guide_reviewed_id IF NOT EXISTS FOR (n:PlantGuide2019Reviewed) REQUIRE n.id IS UNIQUE').consume()
   s.execute_write(write)
   n=s.run('MATCH (n:PlantGuide2019Reviewed) RETURN count(n) AS c').single()['c']
   e=s.run('MATCH (:PlantGuide2019Reviewed)-[r]->(:PlantGuide2019Reviewed) RETURN count(r) AS c').single()['c']
   actual={'nodes':n,'relationships':e}
   assert n==len(data['nodes']) and e==len(data['relationships']),actual
   (R/'neo4j_import_result.json').write_text(json.dumps(actual,indent=2))
   print('Imported and verified:',actual)
if __name__=='__main__':main()
