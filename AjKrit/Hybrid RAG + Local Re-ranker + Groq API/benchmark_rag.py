"""Run the worksheet pipeline and a dense-only baseline with measured API usage."""
import json
import os
import time
from pathlib import Path
from dotenv import dotenv_values

output_dir = Path(os.getenv('RAG_RESULTS_DIR', 'results'))
output_dir.mkdir(parents=True, exist_ok=True)

values = dotenv_values('.env')
key = os.getenv('GROQ_API_KEY') or values.get('GROQ_API_KEY') or values.get('Groq_API_KEY')
if key:
    os.environ['GROQ_API_KEY'] = key

start = time.perf_counter()
import hybrid_rag_rerank as rag
setup_seconds = time.perf_counter() - start
# The worksheet model is unavailable to this account; keep the source untouched.
rag.llm = rag.ChatGroq(model='openai/gpt-oss-20b', temperature=0.1, max_retries=2)
query = 'ทำไมการทำ Re-rank และใช้โมเดลเล็กบน Groq ถึงช่วยลด Hallucination ได้?'
report = {'query': query, 'setup_seconds': setup_seconds, 'model': 'openai/gpt-oss-20b', 'runs': []}
original_invoke = rag.ChatGroq.invoke
active = {}

def measured_invoke(self, *args, **kwargs):
    start = time.perf_counter()
    response = original_invoke(self, *args, **kwargs)
    active['generation_seconds'] = time.perf_counter() - start
    active['usage'] = response.usage_metadata
    active['response_metadata'] = response.response_metadata
    return response

rag.ChatGroq.invoke = measured_invoke
original_rerank = rag.LocalReranker.rerank

def measured_rerank(self, query, documents, top_n=3):
    start = time.perf_counter()
    selected = original_rerank(self, query, documents, top_n)
    active['rerank_seconds'] = time.perf_counter() - start
    active['candidate_count'] = len(documents)
    active['contexts'] = [d.page_content for d in selected]
    return selected

rag.LocalReranker.rerank = measured_rerank
for repeat in range(3):
    # Alternate ordering to reduce a fixed ordering bias.
    modes = ['hybrid_rerank', 'dense_no_rerank'] if repeat % 2 == 0 else ['dense_no_rerank', 'hybrid_rerank']
    for mode in modes:
        active = {'mode': mode, 'repeat': repeat + 1}
        start = time.perf_counter()
        try:
            if mode == 'hybrid_rerank':
                answer = rag.run_hybrid_rag_pipeline(query, top_n_rerank=2)
            else:
                docs = rag.dense_retriever.invoke(query)
                active['contexts'] = [d.page_content for d in docs]
                active['candidate_count'] = len(docs)
                context = '\n\n'.join(f'- {d.page_content}' for d in docs)
                chain = rag.prompt | rag.llm | rag.StrOutputParser()
                answer = chain.invoke({'context': context, 'question': query})
            active['answer'] = answer
            active['total_seconds'] = time.perf_counter() - start
            print(json.dumps(active, ensure_ascii=False), flush=True)
        except Exception as exc:
            # Avoid dumping request headers or credentials in error reports.
            active['error_type'] = type(exc).__name__
            print('Run failed:', active['error_type'], flush=True)
        report['runs'].append(active)
        (output_dir / 'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
