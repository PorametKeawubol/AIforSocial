from html import escape
from playwright.sync_api import sync_playwright
from three_graphs import *
import xml.etree.ElementTree as E

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1680,'height':1100},accept_downloads=True)
    page.goto('http://localhost:7475/browser/',wait_until='networkidle')
    page.locator('input[name=username]').fill(os.environ['NEO4J_USERNAME']);page.locator('input[name=password]').fill(os.environ['NEO4J_PASSWORD'])
    page.get_by_role('button',name='Connect',exact=True).click();page.locator('input[name=password]').wait_for(state='hidden');page.wait_for_timeout(4000)
    # Neo4j Browser graph stylesheet: readable entity captions.
    page.locator('[contenteditable=true]').first.fill(':style')
    page.get_by_role('button',name='Run',exact=True).first.click();page.wait_for_timeout(1000)
    page.locator('input[type=file]').set_input_files(str(OUT/'graph.grass'));page.wait_for_timeout(1000)
    print(page.locator('body').inner_text()[-2000:],flush=True)
    if page.get_by_role('dialog').count():
        print(page.get_by_role('dialog').inner_text(),flush=True)
        page.get_by_role('button',name='Import',exact=True).click();page.wait_for_timeout(500)
    page.get_by_role('button',name='Remove frame',exact=True).first.click()
    summaries=[]
    import sys
    variants={v:LABELS[v] for v in sys.argv[1:]} if len(sys.argv)>1 else LABELS
    for variant,label in variants.items():
        folder=OUT/variant
        raw=json.loads((folder/'raw.json').read_text());summary=json.loads((folder/'summary.json').read_text());summaries.append(summary)
        query=f'MATCH p=(s:{label})-[r]->(t:{label}) RETURN p ORDER BY r.page DESC LIMIT 18'
        page.locator('[contenteditable=true]').first.fill(query);page.get_by_role('button',name='Run',exact=True).first.click();page.wait_for_timeout(3500)
        body=page.locator('body').inner_text()
        if 'Connection.MissingDatabaseError' in body:raise RuntimeError('Neo4j browser not ready')
        page.screenshot(path=str(folder/'neo4j_graph.png'),full_page=True)
        (folder/'neo4j_browser_text.txt').write_text(body)
        # Table and Raw rendered directly from records read back from Neo4j, all rows included.
        fields=['subject','relation','object','page','method']
        table='<table><thead><tr>'+''.join('<th>'+f+'</th>' for f in fields)+'</tr></thead><tbody>'
        for r in raw['relationships']:
            table+='<tr>'+''.join('<td>'+escape(str(r.get(f,'')))+'</td>' for f in fields)+'</tr>'
        table+='</tbody></table>'
        html=f'<!doctype html><html lang="th"><meta charset="utf-8"><style>{STYLE}td{{font-size:13px;overflow-wrap:anywhere}}table{{table-layout:fixed}}</style><main><h1>{escape(summary["title"])}</h1><p>{summary["nodes"]} nodes · {summary["relationships"]} relationships · PDF 48 pages</p><p>{escape(summary["method"])}</p><p>Graph screenshot shows up to 18 relationships; CSV and JSON contain all records.</p><img src="neo4j_graph.png" style="width:100%"><h2>Table — Neo4j export</h2>{table}</main></html>'
        (folder/'table.html').write_text(html,encoding='utf-8')
        (folder/'raw.html').write_text(f'<!doctype html><meta charset="utf-8"><style>{STYLE}</style><main><h1>{escape(summary["title"])} · Raw JSON</h1><pre>{escape(json.dumps(raw,ensure_ascii=False,indent=2))}</pre></main>',encoding='utf-8')
        report=browser.new_page(viewport={'width':1680,'height':1100})
        report.goto((folder/'table.html').as_uri());report.pdf(path=str(folder/'table.pdf'),format='A3',landscape=True,print_background=True)
        report.screenshot(path=str(folder/'table.png'),full_page=False)
        report.goto((folder/'raw.html').as_uri());report.screenshot(path=str(folder/'raw.png'),full_page=False);report.close()
        # GraphML preserves every node and edge, including isolated NER nodes.
        ns='http://graphml.graphdrawing.org/xmlns';E.register_namespace('',ns)
        root=E.Element('{'+ns+'}graphml')
        for key,scope in [('name','node'),('kind','node'),('relation','edge'),('page','edge'),('evidence','edge')]:E.SubElement(root,'{'+ns+'}key',{'id':key,'for':scope,'attr.name':key,'attr.type':'string'})
        graph=E.SubElement(root,'{'+ns+'}graph',{'id':variant,'edgedefault':'directed'})
        for n in raw['nodes']:
            node=E.SubElement(graph,'{'+ns+'}node',{'id':n['id']})
            for key in ['name','kind']:E.SubElement(node,'{'+ns+'}data',{'key':key}).text=n[key]
        for i,r in enumerate(raw['relationships']):
            edge=E.SubElement(graph,'{'+ns+'}edge',{'id':str(i),'source':r['source'],'target':r['target']})
            for key in ['relation','page','evidence']:E.SubElement(edge,'{'+ns+'}data',{'key':key}).text=str(r[key])
        E.ElementTree(root).write(folder/'graph.graphml',encoding='utf-8',xml_declaration=True)
        # Remove result frame so each graph screenshot stays separate.
        page.get_by_role('button',name='Remove frame',exact=True).first.click()
    browser.close()
content=f'<!doctype html><html lang="th"><meta charset="utf-8"><style>{STYLE}</style><main><h1>กราฟ Neo4j ทั้ง 3 วิธี</h1><p>ข้อมูลจาก PDF เดียวกัน 48 หน้า · กราฟแยกตามวิธี · export อ่านกลับจาก Neo4j จริง</p>'
for variant,summary in zip(variants,summaries):
    content+=f'<article><h2>{escape(summary["title"])}</h2><p>{summary["nodes"]} nodes · {summary["relationships"]} relationships</p><p>{escape(summary["method"])}</p><img src="{variant}/neo4j_graph.png" style="width:100%"><p>'+ ' · '.join(f'<a href="{variant}/{f}">{title}</a>' for f,title in [('neo4j_graph.png','Graph PNG'),('table.html','Table HTML'),('table.pdf','Table PDF'),('nodes.csv','Nodes CSV'),('relationships.csv','Relationships CSV'),('raw.json','Raw JSON'),('raw.html','Raw viewer'),('graph.graphml','GraphML'),('queries.cypher','Cypher')])+'</p></article>'
content+='</main></html>'
(OUT/'index.html').write_text(content,encoding='utf-8');dump(OUT/'summary.json',summaries)
print('Captured all three graph variants')
