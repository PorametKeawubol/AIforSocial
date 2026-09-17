"""Explicit agent review of frozen A/B rows; no extra model calls or graph mutations."""
import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'artifacts/semantic10'
# Row numbers are zero based in accepted arrays. Notes assessed against source.json.
notes={}
def mark(g,p,indices,message):
 for i in indices:notes.setdefault((g,p,i),[]).append(message)
for g in ['A','B']:
 mark(g,21,[0,1,2,3] if g=='A' else [0,1,2],'condition field omits identity-document alternatives; source evidence may retain some wording')
 mark(g,21,[8,9] if g=='A' else [6,7],'omits special circumstance: injured/deceased person unable to act, and relationship-proof purpose')
 mark(g,22,[0,1,2],'condition field omits representative/spouse/legal-entity circumstance')
 mark(g,22,[6] if g=='A' else [9],'PersonRole is a certifier, not a supporting evidence object; compound target; condition field empty. Prompt lacked CERTIFIED_BY relation')
 mark(g,24,[2] if g=='A' else [0],'daily record is created during procedure, not evidence brought by citizen; relation direction in process is misrepresented. Prompt lacked output-document relation')
 mark(g,26,[2] if g=='A' else [3],'availability condition embedded in entity name instead of condition property')
 mark(g,28,[0,1],'court/will document mislabeled Evidence; omits conditional estate-manager circumstance and alternatives')
 mark(g,28,[2],'omits original/copy/alternative-document choice')
 mark(g,28,[3],'standalone สำเนา lacks what is copied; should model copy of hire-purchase contract, not generic document node')
 mark(g,29,[4],'combines victim clothing and accused belongings into one entity; split for graph')
 mark(g,29,[5],'combines identity card and house-register copy into one entity; split for graph')
 mark(g,29,[6],'combines photo/address/other evidence into one entity; retains availability but omits alternative in condition field')
 mark(g,27,[4] if g=='A' else [3],'target combines documents and other fraud evidence; Document-only type narrows a broader source category')
mark('A',21,[4],'condition field omits foreign visitor applicability; evidence retains it')
mark('A',21,[6,7],'condition field omits acting as representative; target partly encodes role')
mark('A',25,[0,1,3,7],'document content mislabeled Evidence; some targets also combine multiple document examples')
mark('A',25,[0,1,2,3],'availability/current-photo condition embedded in entity instead of structured condition')
mark('A',25,[4],'organization name labeled Contact; telephone appears only in evidence, no telephone node')
mark('A',27,[0],'omits preserve-at-scene instruction; supporting evidence relation alone could imply transport')
mark('B',21,[3],'foreign visitor applicability retained, identity-document alternative not structured')
mark('B',22,[6,7,8],'optional photo-ID alternatives present in evidence but omitted from condition property')
mark('B',25,[0,1],'document mislabeled Evidence; availability condition correctly extracted')
mark('B',25,[2],'current-photo condition present in evidence but omitted from condition property')
mark('B',25,[3],'groups birth and military documents under umbrella; examples/availability retained but not separate nodes')
mark('B',25,[4],'CONTACT_VIA points to Organization, inconsistent with reviewed Contact endpoint schema; channel meaning plausible but needs normalization')
mark('B',25,[7],'condition for 1192 repeats 1192 instead of identifying 191 as alternative')
mark('B',25,[9],'receipt/contract alternatives and availability bundled into entity rather than separate nodes and condition')
mark('B',26,[0],'legal-entity condition is on preceding page and omitted')
mark('B',30,[0],'omits source circumstance: scene evidence at risk of loss/damage; does not preserve handling context')
for g in ['A','B']:mark(g,30,[2] if g=='A' else [4],'groups debt evidence and sale-contract example; Evidence/Document type choice depends on whether category or example is represented, not a clear comparative error')
rows=[];stats={};hashes={}
for g in ['A','B']:
 files=sorted((O/g).glob('page-*.json'));assert len(files)==10
 ds=[json.loads(p.read_text()) for p in files];assert all(d['parse_ok'] and not d['truncated'] for d in ds)
 for f in files:hashes[str(f.relative_to(O))]=hashlib.sha256(f.read_bytes()).hexdigest()
 for d in ds:
  for i,row in enumerate(d['accepted']):rows.append({'group':g,'pdf_page':d['pdf_page'],'accepted_index':i,'row':row,'review':'needs_revision' if (g,d['pdf_page'],i) in notes else 'no_specific_issue_identified','notes':notes.get((g,d['pdf_page'],i),[])})
 stats[g]={'pages':len(ds),'accepted':sum(len(d['accepted']) for d in ds),'rejected':sum(len(d['rejected']) for d in ds),'covered_sections':sorted({r[0] for d in ds for r in d['accepted']},key=lambda s:int(s.split('.')[1])),'seconds':round(sum(d['seconds'] for d in ds),1),'nonempty_condition_rows':sum(bool(r[4]) for d in ds for r in d['accepted'])}
 stats[g]['sections_count']=len(stats[g]['covered_sections'])
report={'reviewer':'agent manual source review, not blinded or independent human gold standard','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'All 108 accepted rows reviewed; all 32 rejected rows inspected for filtering effects. Qualitative decision, no precision/recall claims. Strict output checks differ from semantic correctness.','input_sha256':hashes,'stats':stats,'decision':'Prefer B as starting configuration for this 10-page run, with mandatory review; not proof of general embedding superiority. A retains two identity/representation documents omitted by B.','shared_omissions':['PDF23 district office and municipality both absent despite successful empty JSON','Section1.12 no accepted evidence in either run; raw generated rows were rejected by substring validator','Section1.10 entrustment and ownership evidence filtered out in both','PDF26 receipts, vehicle manual and contact documents largely filtered out in both'],'ontology_limitation':'Prompt allowed only 4 relation types while reviewed ontology has 7. CERTIFIED_BY/ISSUES/ABOUT_DOCUMENT omissions cannot fairly be attributed solely to embedding models. Embeddings supply ontology hints, not graph retrieval or entity merging.','rows':rows}
(O/'comparison-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(O/'comparison-status.json').write_text(json.dumps({'status':'both_complete_qualitative_review_complete','stats':stats,'decision':report['decision'],'audit':'comparison-audit.json','displayed_graph':'Task1Reviewed remains manual source-reviewed graph, not raw B'},ensure_ascii=False,indent=2))
print(json.dumps(stats,ensure_ascii=False,indent=2))
