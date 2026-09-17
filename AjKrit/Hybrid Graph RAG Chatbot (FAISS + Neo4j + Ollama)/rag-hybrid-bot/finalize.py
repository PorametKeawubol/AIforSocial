import importlib.metadata, subprocess, platform
from core import *
if __name__=='__main__':
    result=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q'],cwd=ROOT,text=True,capture_output=True)
    (DATA/'test_results.txt').write_text(result.stdout+result.stderr)
    versions={name:importlib.metadata.version(name) for name in ['torch','sentence-transformers','faiss-cpu','pymupdf','neo4j','fastapi','playwright','pythainlp']}
    status=dict(tests_exit_code=result.returncode,tests=result.stdout,python=platform.python_version(),packages=versions,
        services={'chatbot':'http://127.0.0.1:8090','neo4j_browser':'http://localhost:7475','ollama':HOST},
        source_pages=read('source.json')['pages'],chunks=len(read('chunks.json')),facts=len(read('facts.json')),
        line='Adapter implemented and signature unit-tested; live integration not run: credentials not supplied.',
        screenshots='Generated separately by capture.py from checkpoint HTML and the live web app.',
        limitations=['Small diagnostic retrieval set; not a held-out evaluation.','Exact evidence matching does not prove semantic correctness.','No exhaustive model benchmark; hardware-constrained model selection.'])
    checkpoint(7,'ตรวจสอบระบบและสรุปผลการส่งมอบ',status)
    if result.returncode:raise SystemExit(result.returncode)
