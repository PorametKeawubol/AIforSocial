from playwright.sync_api import sync_playwright
from core import *
with sync_playwright() as p:
    b=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=b.new_page(viewport={'width':1600,'height':1100})
    page.goto('http://localhost:7475/browser/',wait_until='networkidle')
    page.locator('input[name=username]').fill(os.getenv('NEO4J_USERNAME','neo4j'))
    page.locator('input[name=password]').fill(os.environ['NEO4J_PASSWORD'])
    page.get_by_role('button',name='Connect',exact=True).click()
    page.locator('input[name=password]').wait_for(state='hidden',timeout=30000)
    page.wait_for_timeout(4000)
    editor=page.locator('[contenteditable=true]').first
    editor.fill("MATCH p=(s:RagEntity)-[:SUBJECT_OF]->(f:RagFact)-[:OBJECT]->(o:RagEntity) WHERE s.name CONTAINS 'แคงเกอร์' OR o.name CONTAINS 'แคงเกอร์' RETURN p LIMIT 25")
    page.get_by_role('button',name='Run',exact=True).first.click()
    page.wait_for_timeout(5000)
    print(page.locator('body').inner_text()[:2200])
    page.screenshot(path=str(DATA/'neo4j_browser.png'),full_page=True)
    b.close()
