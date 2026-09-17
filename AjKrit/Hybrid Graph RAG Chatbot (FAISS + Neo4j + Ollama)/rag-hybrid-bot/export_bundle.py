from zipfile import ZipFile, ZIP_DEFLATED
from core import *
if __name__=='__main__':
    dest=ROOT.parent/'hybrid-graph-rag-results.zip'
    with ZipFile(dest,'w',ZIP_DEFLATED) as z:
        for p in ROOT.rglob('*'):
            rel=p.relative_to(ROOT)
            if not p.is_file() or any(part.startswith('.') for part in rel.parts) or '__pycache__' in rel.parts:continue
            if p.suffix=='.log' or p.name.endswith('.tmp'):continue
            z.write(p,str(Path('rag-hybrid-bot')/rel))
        z.write(ROOT/'.env.example','rag-hybrid-bot/.env.example')
        z.write(PDF,PDF.name)
    print(dest, dest.stat().st_size)
