"""Explicit, source-specific Codex review; never silently relabel model output."""
from core import *

def apply_review(facts,pages,source):
    path=ROOT/'graph_review.json'
    if not path.exists():return facts,[]
    review=json.loads(path.read_text())
    if review['source_sha256']!=source:return facts,[]
    excluded=review['exclude'];removed=[];result=[]
    for f in facts:
        if f['id'] in excluded:removed.append({'page':f['page'],'fact':f,'reason':'Codex source review: '+excluded[f['id']]})
        else:result.append({**f,'origin':'model'})
    for f in review['additions']:
        assert f['evidence'] in pages[f['page']], 'Reviewed quote no longer matches source'
        row={**f,'source':source,'origin':'codex_source_review'}
        row['id']=digest(json.dumps(row,ensure_ascii=False,sort_keys=True));result.append(row)
    dump(DATA/'review_applied.json',dict(reviewer='Codex (AI source review, not independent human validation)',excluded=len(removed),additions=len(review['additions']),source_sha256=source))
    return result,removed
