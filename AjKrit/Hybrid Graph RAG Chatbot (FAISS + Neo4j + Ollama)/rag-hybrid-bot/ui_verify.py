from playwright.sync_api import sync_playwright
from core import *
with sync_playwright() as p:
    b=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=b.new_page(viewport={'width':1440,'height':1100},accept_downloads=True)
    errors=[];page.on('pageerror',lambda err:errors.append(str(err)))
    page.goto('http://127.0.0.1:8090');page.wait_for_selector('#export:not([hidden])')
    page.locator('#query').fill('โรครากปมในพริกเกิดจากอะไร')
    with page.expect_response(lambda r:'/api/ask' in r.url,timeout=180000) as response:
        page.locator('#submit').click()
    r=response.value;result=r.json()
    assert r.status==200 and 36 in result['citations'] and not result['warnings']
    page.wait_for_function("document.querySelector('#submit').disabled===false")
    with page.expect_download() as info:page.locator('#export').click()
    download=info.value;download.save_as(str(DATA/'ui_download.json'))
    assert json.loads((DATA/'ui_download.json').read_text())['answer']==result['answer']
    assert not errors
    page.screenshot(path=str(DATA/'chatbot_live_query.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    dump(DATA/'ui_verification.json',dict(status=200,citations=result['citations'],download_matches=True,console_errors=errors,mobile_no_horizontal_overflow=True))
    b.close()
