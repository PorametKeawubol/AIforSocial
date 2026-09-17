from pathlib import Path
import json
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parent
views=json.loads((R/'view_manifest.json').read_text());errors=[]
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1600,'height':1100},device_scale_factor=1)
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((R/'index.html').as_uri());page.wait_for_selector('.node')
 assert page.locator('nav button').count()==5
 counts=[]
 for i,v in enumerate(views):
  page.locator('nav button').nth(i).click()
  assert page.locator('.node').count()==len(v['positions'])
  assert page.locator('.edge').count()==len(v['edges'])
  page.locator('.node').first.click(force=True)
  assert page.locator('#detail h2').inner_text()
  page.locator('#clear').click()
  page.locator('.edge').first.dispatch_event('click')
  assert 'หน้า PDF:' in page.locator('#detail').inner_text()
  counts.append({'view':v['key'],'nodes':len(v['positions']),'edges':len(v['edges'])})
 page.locator('#search').fill('ไตรโคเดอร์มา')
 assert page.locator('#results button').count()>=2
 page.locator('#results button').first.click()
 assert 'ไตรโคเดอร์มา' in page.locator('#detail h2').inner_text()
 page.locator('#search').fill('โรคพริก');page.locator('#results button').first.click()
 assert page.locator('#detail h2').inner_text()=='โรคพริก'
 page.locator('nav button').first.click();page.locator('#search').fill('');page.screenshot(path=str(R/'interactive_preview.png'))
 for v in views[:4]:
  page.set_viewport_size({'width':v['width'],'height':v['height']})
  page.set_content('<html><style>html,body{margin:0;padding:0;overflow:hidden}</style>'+ (R/(v['key']+'.svg')).read_text()+'</html>');page.evaluate('document.fonts.ready')
  page.screenshot(path=str(R/(v['key']+'.png')),full_page=False,timeout=15000)
 page.goto((R/'report.html').as_uri());page.pdf(path=str(R/'graph_report.pdf'),format='A3',landscape=True,print_background=True,margin={'top':'8mm','bottom':'8mm','left':'8mm','right':'8mm'})
 b.close()
assert not errors,errors
(R/'ui_validation.json').write_text(json.dumps({'javascript_errors':errors,'views':counts,'search_test':'passed','section_search_test':'passed','edge_evidence_test':'passed'},indent=2))
print('Browser checks passed; PNGs and PDF exported')
