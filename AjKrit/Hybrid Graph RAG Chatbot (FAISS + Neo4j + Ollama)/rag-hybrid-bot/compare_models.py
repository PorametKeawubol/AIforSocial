"""Two paired answers: original worksheet LLM versus selected LLM; no LLM-as-judge score."""
from core import *
from rag_search import Retriever
if __name__=='__main__':
    r=Retriever();selected=read('answers.json')[:2];rows=[]
    for s in selected:
        baseline=r.answer(s['query'],model='llama3.2:1b')
        rows.append(dict(query=s['query'],selected=s,baseline=baseline))
        dump(DATA/'llm_comparison.json',dict(pairs=rows,note='Illustrative paired outputs, not a benchmark. Latencies may include cold model loading. Review factual support manually.'))
        print('Compared',s['query'],flush=True)
