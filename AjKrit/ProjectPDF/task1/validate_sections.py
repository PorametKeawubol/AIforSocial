"""Inspect V2 machine graph; check structure without claiming semantic completeness."""
import json,argparse,csv
from pathlib import Path
from collections import Counter
from section_pipeline import OUT,REL_TYPES,validate,sections

def audit(group):
 source={s['section']:s for s in sections()};p=OUT/group/'graph.json'
 if not p.exists():return {'group':group,'status':'not_exported'}
 g=json.loads(p.read_text());nodes={n['id']:n for n in g['nodes']};assert len(nodes)==len(g['nodes'])
 assert len({(n['type'],n['name']) for n in nodes.values()})==len(nodes)
 assert len({e['id'] for e in g['edges']})==len(g['edges'])
 adjacency={k:set() for k in nodes};degree=Counter();flags=[];accepted_count=rejected_count=0;seconds=0
 for e in g['edges']:
  a,b=nodes[e['source']],nodes[e['target']]
  assert a['type']=='Case' and b['type'] in REL_TYPES[e['relation']]
  assert a['section']==e['section']
  sec=source[e['section']];expected='\n'.join(x['text'] for x in sec['lines'] if e['start_line']<=x['line']<=e['end_line'])
  assert expected==e['evidence'] and expected
  degree[a['section']]+=1;adjacency[a['id']].add(b['id']);adjacency[b['id']].add(a['id'])
  if e['review_flags']:flags.append({'id':e['id'],'section':e['section'],'target':b['name'],'flags':e['review_flags']})
 for sec in source:
  f=OUT/group/(sec+'.json')
  if f.exists():
   d=json.loads(f.read_text());accepted_count+=len(d['accepted']);rejected_count+=len(d['rejected']);seconds+=d['seconds']
 unseen=set(nodes);components=[]
 while unseen:
  todo=[unseen.pop()];count=0
  while todo:
   u=todo.pop();count+=1
   for v in adjacency[u]&unseen:unseen.remove(v);todo.append(v)
  components.append(count)
 result={'group':group,'complete_extraction':g['complete'],'structural_checks_passed':True,'semantic_review_complete':False,'nodes':len(nodes),'relationships':len(g['edges']),'labels':dict(Counter(n['type'] for n in nodes.values())),'relationships_by_type':dict(Counter(e['relation'] for e in g['edges'])),'sections_with_edges':len(degree),'sections_without_edges':sorted(set(source)-set(degree)),'connected_component_sizes':sorted(components,reverse=True),'shared_concept_nodes':sum(len(adjacency[n['id']])>1 for n in nodes.values() if n['type']!='Case'),'flags':flags,'accepted_before_deduplication':accepted_count,'rejected':rejected_count,'llm_wall_seconds':round(seconds,1),'note':'All 16 Case nodes are scaffolded from PDF headings. sections_with_edges measures model coverage; case count alone does not prove completeness.'}
 (OUT/group/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 with (OUT/group/'review.csv').open('w',newline='') as f:
  fields=['section','relation','target','type','owner','condition','pdf_pages','start_line','end_line','evidence','review_flags'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for e in g['edges']:writer.writerow({**{k:e[k] for k in fields if k in e},'target':nodes[e['target']]['name'],'type':nodes[e['target']]['type']})
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group',choices=['A','B','both'],default='both');a=p.parse_args()
 for g in ['A','B'] if a.group=='both' else [a.group]:print(json.dumps(audit(g),ensure_ascii=False))
