"""Revalidate raw V2 outputs with auditable grounding repair; validate/import both sets."""
import json
from section_pipeline import OUT,sections,repair_grounding,save,export,import_graph
from validate_sections import audit
ss=sections()
for group in ['A','B']:
 for sec in ss:
  path=OUT/group/(sec['section']+'.json')
  d=json.loads(path.read_text());assert d['parse_ok'] and not d['truncated']
  raw=json.loads(d['response']['message']['content']);accepted=[];rejected=[]
  for row in raw['edges']:
   item,error=repair_grounding(row,sec)
   if error:rejected.append({'row':row,'reason':error})
   else:accepted.append(item)
  if 'initial_validation' not in d:d['initial_validation']={'accepted':d['accepted'],'rejected':d['rejected']}
  d.update(accepted=accepted,rejected=rejected,validator_version='v2.2-unique-nearby-grounding-repair')
  save(path,d)
 export(group,ss);result=audit(group)
 print(json.dumps(result,ensure_ascii=False),flush=True)
 import_graph(group,ss)
 # Repeated import is required to prove idempotence, not just claimed from MERGE.
 import_graph(group,ss)
print('BOTH_GROUPS_VALIDATED_AND_IMPORTED_REQUIRES_SEMANTIC_REVIEW',flush=True)
