"""Build and validate the source-reviewed submission graph, separately from raw A/B."""
import hashlib,json,re,argparse,os,subprocess
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent; O=R/'artifacts/semantic10'; DEST=O/'reviewed'
REL_TYPES={'HAS_SUPPORTING_EVIDENCE':{'Document','Evidence'},'CONTACT_VIA':{'Contact'},'HANDLED_BY':{'Organization','PersonRole'},'NOT_HANDLED_BY':{'Organization','PersonRole'},'ISSUES':{'Document'},'ABOUT_DOCUMENT':{'Document'},'CERTIFIED_BY':{'PersonRole'}}
def key(*s):return hashlib.sha256('|'.join(map(str,s)).encode()).hexdigest()[:24]
def norm(s):return re.sub(r'\s+','',s)
def literal(v):
 if isinstance(v,str):return "'"+v.replace('\\','\\\\').replace("'","\\'").replace('\n','\\n').replace('\r','\\r')+"'"
 if isinstance(v,dict):return '{'+','.join(k+':'+literal(x) for k,x in v.items())+'}'
 if isinstance(v,list):return '['+','.join(literal(x) for x in v)+']'
 return str(v).lower()
def build():
 ref=json.loads((O/'reference_inventory.json').read_text());pages=json.loads((O/'source.json').read_text());pmap={p['pdf_page']:p for p in pages}
 pdf=next(R.parent.glob('คู่มือประชาชน*.pdf'));sha=hashlib.sha256(pdf.read_bytes()).hexdigest();dataset='police-manual-selected10-reviewed-v1';nodes={};edges=[]
 for section,name in ref['case_names'].items():
  ns=[p['pdf_page'] for p in pages if '['+section+']' in p['tagged']]
  nid=key(dataset,'Case',section);nodes[nid]=dict(id=nid,type='Case',section=section,name=name,display=section+' '+name,dataset=dataset,pdf_pages=ns,source_sha256=sha,kind='procedure_topic')
 for x in ref['items']:
  nid=key(dataset,x['type'],norm(x['target']));a=key(dataset,'Case',x['section'])
  if nid not in nodes:nodes[nid]=dict(id=nid,type=x['type'],name=x['target'],display=x['target'],dataset=dataset,pdf_pages=[],source_sha256=sha,kind='concept_type' if x['type'] in {'Document','Evidence'} else 'named_reference')
  nodes[nid]['pdf_pages']=sorted(set(nodes[nid]['pdf_pages']+x['pdf_pages']))
  eid=key(dataset,a,x['relation'],nid,x['condition'],x['owner'])
  edges.append(dict(id=eid,source=a,target=nid,relation=x['relation'],section=x['section'],condition=x['condition'],owner=x['owner'],evidence=x['evidence'],pdf_pages=x['pdf_pages'],printed_pages=[int(pmap[n]['printed_page']) for n in x['pdf_pages']],source_sha256=sha,review_status='agent_source_review',provenance='manual_reference_not_llm_output',dataset=dataset))
 graph=dict(dataset=dataset,source_file=pdf.name,source_sha256=sha,scope=ref['scope'],authorship=ref['authorship'],nodes=list(nodes.values()),edges=edges)
 checks=validate(graph,pmap)
 DEST.mkdir(parents=True,exist_ok=True)
 (DEST/'graph.json').write_text(json.dumps(graph,ensure_ascii=False,indent=2))
 (DEST/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
 queries=['CREATE CONSTRAINT task1_reviewed_id IF NOT EXISTS FOR (n:Task1Reviewed) REQUIRE n.id IS UNIQUE']
 for typ in sorted({n['type'] for n in nodes.values()}):
  queries.append('UNWIND '+literal([n for n in nodes.values() if n['type']==typ])+' AS row MERGE (n:Task1Reviewed:'+typ+' {id:row.id}) SET n += row')
 for rel in sorted(REL_TYPES):
  rows=[e for e in edges if e['relation']==rel]
  queries.append('UNWIND '+literal(rows)+' AS row MATCH (a:Task1Reviewed {id:row.source}), (b:Task1Reviewed {id:row.target}) MERGE (a)-[r:'+rel+' {id:row.id}]->(b) SET r += row')
 (DEST/'import.cypher').write_text(';\n\n'.join(queries)+';\n')
 print(json.dumps(checks,ensure_ascii=False));return graph,queries

def validate(g,pmap):
 nodes={n['id']:n for n in g['nodes']};signatures=set();adj={k:set() for k in nodes}
 assert len(nodes)==len(g['nodes']), 'duplicate node ID'
 assert len({(n['type'],norm(n['name'])) for n in g['nodes']})==len(nodes),'duplicate concept'
 assert {n['section'] for n in nodes.values() if n['type']=='Case'}=={'1.'+str(i) for i in range(1,17)}
 assert len({e['id'] for e in g['edges']})==len(g['edges'])
 for e in g['edges']:
  a,b=nodes[e['source']],nodes[e['target']]
  assert a['type']=='Case' and b['type'] in REL_TYPES[e['relation']], 'wrong direction or endpoint type'
  signature=(a['id'],e['relation'],b['id'],e['condition'],e['owner']);assert signature not in signatures;signatures.add(signature)
  assert e['evidence'] and norm(e['evidence']) in norm('\n'.join(pmap[p]['text'] for p in e['pdf_pages'])),'untraceable evidence'
  assert all('['+a['section']+']' in pmap[p]['tagged'] for p in e['pdf_pages']), 'wrong section page'
  adj[a['id']].add(b['id']);adj[b['id']].add(a['id'])
 assert all(adj.values()),'isolated node'
 unseen=set(nodes);components=[]
 while unseen:
  todo=[unseen.pop()];component=[]
  while todo:
   n=todo.pop();component.append(n)
   for m in adj[n]&unseen:unseen.remove(m);todo.append(m)
  components.append(len(component))
 return dict(passed=True,nodes=len(nodes),relationships=len(g['edges']),cases=16,labels=dict(Counter(n['type'] for n in nodes.values())),relationship_types=dict(Counter(e['relation'] for e in g['edges'])),connected_components=sorted(components,reverse=True),checks=['unique IDs and concept names','no repeated semantic relationship','all 16 scoped cases','allowed relation direction and target type','all evidence spans found in referenced pages','section/page consistency','no isolated nodes'],limitation='Agent source review; not independent human evaluation. No claim of full-PDF coverage or current legal validity.')

def connect():
 from neo4j import GraphDatabase
 password=os.environ.get('NEO4J_PASSWORD');user=os.environ.get('NEO4J_USER','neo4j')
 if not password:
  containers=['neo4j-Project2-AB','neo4j-GraphRAG']
  config=None
  for container in containers:
   try:
    candidate=json.loads(subprocess.check_output(['docker','inspect',container],stderr=subprocess.DEVNULL))[0]
   except subprocess.CalledProcessError:
    continue
   if candidate.get('State',{}).get('Running') or config is None:config=candidate
  if config is None:raise RuntimeError('Set NEO4J_PASSWORD or start the project Neo4j container')
  auth=next((v.split('=',1)[1] for v in config['Config']['Env'] if v.startswith('NEO4J_AUTH=')),None)
  if not auth or '/' not in auth:raise RuntimeError('Set NEO4J_PASSWORD in the environment')
  user,password=auth.split('/',1)
 return GraphDatabase.driver(os.environ.get('NEO4J_URI','bolt://localhost:7687'),auth=(user,password))
def load(graph,queries):
 with connect() as driver:
  driver.verify_connectivity()
  with driver.session(database=os.environ.get('NEO4J_DATABASE','neo4j')) as s:
   counts=[]
   for attempt in range(2):
    for q in queries:s.run(q).consume()
    nc=s.run('MATCH (n:Task1Reviewed {dataset:$d}) RETURN count(n) AS n',d=graph['dataset']).single()['n']
    ec=s.run('MATCH (:Task1Reviewed {dataset:$d})-[r]->(:Task1Reviewed {dataset:$d}) RETURN count(r) AS n',d=graph['dataset']).single()['n'];counts.append([nc,ec])
   assert counts[0]==counts[1]==[len(graph['nodes']),len(graph['edges'])]
   examples=[dict(r) for r in s.run("MATCH (c:Task1Reviewed:Case)-[r]->(d:Task1Reviewed) WHERE c.section='1.2' RETURN c.name AS topic,type(r) AS relation,d.name AS target,r.condition AS condition")]
   result=dict(imported=True,twice_imported_counts=counts,idempotent=True,example_query=examples)
   (DEST/'neo4j-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('Neo4j import and repeat import passed:',counts)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--import-neo4j',action='store_true');a=p.parse_args();g,queries=build()
 if a.import_neo4j:load(g,queries)
