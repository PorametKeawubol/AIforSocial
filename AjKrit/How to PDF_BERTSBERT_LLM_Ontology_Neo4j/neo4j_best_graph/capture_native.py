from pathlib import Path
import os,json,time
from dotenv import load_dotenv
from neo4j import GraphDatabase
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parent;OUT=R/'neo4j_native';OUT.mkdir(exist_ok=True)
load_dotenv(R.parents[1]/'Hybrid Graph RAG Chatbot (FAISS + Neo4j + Ollama)'/'rag-hybrid-bot'/'.env')
with GraphDatabase.driver('bolt://localhost:7688',auth=(os.environ.get('NEO4J_USERNAME','neo4j'),os.environ['NEO4J_PASSWORD'])) as db:
 with db.session() as s:
  nodes=s.run('MATCH (n:PlantGuide2019Reviewed) RETURN properties(n) AS n').value('n')
  edges=s.run('MATCH (a:PlantGuide2019Reviewed)-[r]->(b:PlantGuide2019Reviewed) RETURN properties(r) AS r').value('r')
  version=s.run('CALL dbms.components() YIELD versions RETURN versions').single()[0]
 assert len(nodes)==266 and len(edges)==879
 (OUT/'database_export.json').write_text(json.dumps({'version':version,'nodes':nodes,'relationships':edges},ensure_ascii=False,indent=2))
views=json.loads((R/'view_manifest.json').read_text())
manifest=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 for view in views:
  page=browser.new_page(viewport={'width':2100,'height':1500},accept_downloads=True)
  page.goto('http://localhost:7475/browser/',wait_until='networkidle')
  page.locator('input[name=username]').fill(os.environ.get('NEO4J_USERNAME','neo4j'));page.locator('input[name=password]').fill(os.environ['NEO4J_PASSWORD'])
  page.get_by_role('button',name='Connect',exact=True).click();page.locator('input[name=password]').wait_for(state='hidden');page.wait_for_timeout(1500)
  def run(q):
   page.locator('[contenteditable=true]').first.fill(q);page.get_by_role('button',name='Run',exact=True).first.click();page.wait_for_timeout(800)
  run(':style')
  page.get_by_role('button',name='Reset styles to default',exact=True).click();page.wait_for_timeout(300)
  page.locator('input[type=file]').set_input_files(str(OUT/'native.grass'));page.wait_for_timeout(500)
  if page.get_by_role('dialog').count():page.get_by_role('button',name='Import',exact=True).click();page.wait_for_timeout(300)
  page.get_by_role('button',name='Remove frame',exact=True).first.click()
  # Configure result size to include the entire semantic graph.
  run(':config initialNodeDisplay: 400')
  page.get_by_role('button',name='Remove frame',exact=True).first.click()
  while page.get_by_role('button',name='Remove frame',exact=True).count():page.get_by_role('button',name='Remove frame',exact=True).first.click()
  key=view['key'];query=(R/('query_'+key+'.cypher')).read_text()
  run(query);page.get_by_role('button',name='Download',exact=True).wait_for(timeout=30000);page.wait_for_timeout(4500)
  if page.get_by_role('button',name='Expand frame',exact=True).count():page.get_by_role('button',name='Expand frame',exact=True).click();page.wait_for_timeout(600)
  page.get_by_role('button',name='Zoom to fit',exact=True).click();page.wait_for_timeout(1800)
  body=page.locator('body').inner_text();(OUT/(key+'_browser.txt')).write_text(body)
  assert f"Nodes ({len(view['positions'])})" in body,(key,body[-1800:])
  assert f"Relationships ({len(view['edges'])})" in body,(key,body[-1800:])
  page.screenshot(path=str(OUT/(key+'_browser.png')),full_page=True)
  # Browser owns the layout and exports; no externally redrawn graph.
  for ext in ['svg','png']:
   page.get_by_role('button',name='Download',exact=True).click()
   with page.expect_download(timeout=60000) as info:
    page.get_by_text('Download as '+ext.upper(),exact=True).click()
   dl=info.value;dest=OUT/(key+'.'+ext);dl.save_as(str(dest))
   manifest.append(dict(view=key,title=view['title'],format=ext,file=dest.name,bytes=dest.stat().st_size,original_filename=dl.suggested_filename,method='Neo4j Browser native Download menu',nodes=len(view['positions']),relationships=len(view['edges']),query=query))
  print('Native export',key,len(view['positions']),len(view['edges']),flush=True)
  page.context.close()
 browser.close()
(OUT/'native_exports.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('All native exports complete')
