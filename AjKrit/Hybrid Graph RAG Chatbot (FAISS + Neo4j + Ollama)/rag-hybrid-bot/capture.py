"""Screenshot actual exported checkpoint reports and the running application."""
import asyncio
from playwright.async_api import async_playwright
from core import *
async def main():
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=await browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        for directory in sorted((DATA/'checkpoints').iterdir()):
            if not (directory/'report.html').exists():continue
            await page.goto((directory/'report.html').as_uri());await page.evaluate('document.fonts.ready')
            await page.screenshot(path=str(directory/'screenshot.png'),full_page=True)
            await page.pdf(path=str(directory/'report.pdf'),format='A4',print_background=True)
        await page.goto('http://127.0.0.1:8090');await page.wait_for_selector('#export:not([hidden])')
        await page.screenshot(path=str(DATA/'chatbot_desktop.png'),full_page=True)
        await page.pdf(path=str(DATA/'chatbot.pdf'),format='A4',print_background=True)
        await page.set_viewport_size({'width':390,'height':844})
        await page.screenshot(path=str(DATA/'chatbot_mobile.png'),full_page=True)
        await browser.close()
if __name__=='__main__':asyncio.run(main())
