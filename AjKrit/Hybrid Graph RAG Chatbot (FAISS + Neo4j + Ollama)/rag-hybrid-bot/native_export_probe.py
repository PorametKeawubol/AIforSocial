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
    page.locator('[contenteditable=true]').first.fill('MATCH (n:WorksheetModelI) OPTIONAL MATCH p=(n)-[]->(:WorksheetModelI) RETURN n,p')
    page.get_by_role('button',name='Run',exact=True).first.click();page.wait_for_timeout(5000)
    page.get_by_role('button',name='Download',exact=True).click()
    print(page.locator('body').inner_text()[-3000:])
    browser.close()
