"""Verify semantic-list deep links and boundary/context playback."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
EXP=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8875/listen_full_source/'
manifest=json.loads((EXP/'review/gap5_pairs/manifest.json').read_text())
errors=[];checked=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100})
    page.on('pageerror',lambda e:errors.append(str(e)))
    for number,d in enumerate(manifest['pairs'],1):
        rows=[r for r in d['rows'] if r['initial'] and r['semantic_review']]
        url=BASE+f'?pair={d["pair"]}&fixed=1&filter=semantic#fixedPairReview'
        page.goto(url)
        page.wait_for_function('document.querySelector("#fixedFilter")?.value==="semantic"')
        assert page.locator('#fixedList button').count()==len(rows)
        assert page.locator('#fixedSemanticHelp').is_visible()
        for r in rows:
            page.locator(f'[data-fixed-id="{r["id"]}"]').click()
            assert page.evaluate('window.fixedPairCurrent.id')==r['id']
            assert '尚未确认' in page.locator('.fixed-semantic').inner_text()
            assert page.locator('.fixed-edge-text').inner_text()
            checked.append(r['id'])
        r=rows[0]
        page.goto(BASE+f'?pair={d["pair"]}&fixed=1&filter=semantic&clip={r["id"]}#fixedPairReview')
        page.wait_for_function('(id)=>window.fixedPairCurrent?.id===id',arg=r['id'])
        assert page.locator('#fixedFilter').input_value()=='semantic'
        page.locator('#fixedHead').click()
        page.wait_for_function('document.querySelector("#fixedPlayer video").currentTime>.1')
        page.locator('#fixedTail').click()
        page.wait_for_function('(t)=>document.querySelector("#fixedPlayer video").currentTime>=t-.1',arg=max(0,r['duration']-3))
        page.locator('#fixedSourceTail').click()
        page.wait_for_function('(t)=>!document.querySelector("#video").paused&&Math.abs(document.querySelector("#video").currentTime-t)<1',arg=max(0,r['end']-3))
        assert page.locator('#fixedPlayer video').evaluate('(v)=>v.paused')
        page.locator('#video').evaluate('(v)=>v.pause()')
        for width in [1440,390]:
            page.set_viewport_size({'width':width,'height':1100})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.locator('#fixedPlayer').scroll_into_view_if_needed()
            page.screenshot(path=str(EXP/f'semantic_{number}_{width}.png'))
        page.set_viewport_size({'width':1440,'height':1100})
    browser.close()
assert not errors,errors
report=dict(checked_ids=checked,count=len(checked),errors=errors,semantic_deep_links=True,
    boundary_playback=True,source_context_playback=True,mobile_no_overflow=True)
(EXP/'semantic_browser_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
