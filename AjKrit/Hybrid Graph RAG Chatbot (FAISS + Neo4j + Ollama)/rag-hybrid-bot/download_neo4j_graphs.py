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
    # Reset the persistent Browser stylesheet before importing colors by entity kind.
    page.locator('[contenteditable=true]').first.fill(':style')
    page.get_by_role('button',name='Run',exact=True).first.click();page.wait_for_timeout(1000)
    page.get_by_role('button',name='Reset styles to default',exact=True).click();page.wait_for_timeout(500)
    page.locator('input[type=file]').set_input_files(str(OUT/'graph.grass'));page.wait_for_timeout(1000)
    print(page.locator('body').inner_text()[-2000:],flush=True)
    if page.get_by_role('dialog').count():
        print(page.get_by_role('dialog').inner_text(),flush=True)
        page.get_by_role('button',name='Import',exact=True).click();page.wait_for_timeout(500)
    page.get_by_role('button',name='Remove frame',exact=True).first.click()
    manifest=[]
    for variant,label in LABELS.items():
        folder=OUT/variant
        query=f"MATCH (n {{visual_variant: '{variant}'}}) OPTIONAL MATCH p=(n)-[]->({{visual_variant: '{variant}'}}) RETURN n,p"
        page.locator('[contenteditable=true]').first.fill(query)
        page.get_by_role('button',name='Run',exact=True).first.click()
        page.get_by_role('button',name='Download',exact=True).wait_for()
        page.wait_for_timeout(5000)
        body=page.locator('body').inner_text()
        summary=json.loads((folder/'summary.json').read_text())
        assert f"Nodes ({summary['nodes']})" in body
        assert f"Relationships ({summary['relationships']})" in body
        for ext in ['svg','png']:
            page.get_by_role('button',name='Download',exact=True).click()
            with page.expect_download(timeout=60000) as info:
                page.get_by_text('Download as '+ext.upper(),exact=True).click()
            dl=info.value
            target=folder/f'graph_export.{ext}'
            dl.save_as(str(target))
            assert target.stat().st_size>0
            manifest.append(dict(model=variant,format=ext,filename=target.name,original_filename=dl.suggested_filename,bytes=target.stat().st_size,nodes=summary['nodes'],relationships=summary['relationships'],method='Neo4j Browser Download menu',query=query))
            print(variant,ext,target.stat().st_size,flush=True)
        page.get_by_role('button',name='Remove frame',exact=True).first.click()
    dump(OUT/'native_exports.json',manifest)
    browser.close()
