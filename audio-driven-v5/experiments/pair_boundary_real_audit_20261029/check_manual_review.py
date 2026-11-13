"""Check review media, navigation, feedback persistence and mobile layout."""
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
EXP=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8875/listen_full_source/'
manifest=json.loads((EXP/'review/manual_review/manifest.json').read_text())
pairs=list(dict.fromkeys(r['pair'] for r in manifest['rows']))
errors=[];checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100})
    page.on('pageerror',lambda e:errors.append(str(e)))
    for pair in pairs:
        rows=[r for r in manifest['rows'] if r['pair']==pair]
        page.goto(BASE+f'?pair={pair}&review=1#manualReview')
        page.wait_for_function('window.manualReviewCurrent && document.querySelector("#manualReview").open')
        assert page.locator('#manualList button').count()==len(rows)
        assert page.evaluate('window.manualReviewCurrent.category')=='quality'
        for r in rows:
            page.locator(f'[data-review-id="{r["id"]}"]').click()
            page.wait_for_function('(id)=>window.manualReviewCurrent.id===id && document.querySelector("#manualPlayer video").readyState>=1',arg=r['id'])
            assert page.locator('#manualPlayer video').get_attribute('src')==r['media']
            assert page.request.get(BASE+r['media'],headers={'Range':'bytes=0-127'}).status==206
            assert abs(page.locator('#manualPlayer video').evaluate('(v)=>v.duration')-r['context_seconds'])<.09
            assert page.locator('#manualPlayer .manual-evidence').inner_text()
            checks.append(r['id'])
        for category in ['quality','speech','other','tiny']:
            page.locator('#manualFilter').select_option(category)
            assert page.locator('#manualList button').count()==sum(r['category']==category for r in rows)
        page.locator('#manualFilter').select_option('quality')
        page.locator('#manualFocus').click()
        page.wait_for_function('!document.querySelector("#manualPlayer video").paused && document.querySelector("#manualClock").textContent.includes("正在播放需复核区间")')
        for mode in ['a','b','both']:
            page.locator('#manualChannel').select_option(mode)
        page.locator('#manualPlayer video').evaluate('(v)=>v.pause()')
        selected=page.evaluate('window.manualReviewCurrent.id')
        page.locator('#manualDecision').select_option('partial')
        page.locator('#manualNote').fill('浏览器验收记录，不是真实人工判断')
        page.locator('#manualSave').click()
        assert f'已记录 1/{len(rows)}' in page.locator('#manualProgress').inner_text()
        page.reload()
        page.wait_for_function('(id)=>window.manualReviewCurrent?.id===id',arg=selected)
        assert page.locator('#manualDecision').input_value()=='partial'
        assert page.locator('#manualNote').input_value()=='浏览器验收记录，不是真实人工判断'
        for width in [1440,390]:
            page.set_viewport_size({'width':width,'height':1050})
            page.evaluate('document.fonts.ready')
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.locator('#manualReview').scroll_into_view_if_needed()
            page.screenshot(path=str(EXP/f'manual_{pairs.index(pair)+1}_{width}.png'))
        page.set_viewport_size({'width':1440,'height':1100})
    with page.expect_download() as event:
        page.locator('#manualExport').click()
    downloaded=json.loads(Path(event.value.path()).read_text())
    assert len(downloaded['decisions'])==2
    assert set(r['pair'] for r in downloaded['decisions'])==set(pairs)
    assert downloaded['production_modified'] is False
    page.evaluate('localStorage.removeItem("listen-source-false-rejection-review-v1")')
    browser.close()
assert not errors,errors
for r in manifest['rows']:
    prod=ROOT/'data_corpora/seamless_listen_local_full/results'/f'{r["pair"]}.json'
    assert hashlib.sha256(prod.read_bytes()).hexdigest()==r['production_result_sha256']
report=dict(count=len(checks),checked_ids=checks,errors=errors,feedback_persistence=True,
            export_both_sources=True,mobile_no_overflow=True,production_hashes_unchanged=True,
            all_media_decode_verified=all(r['media_check']['decoded'] for r in manifest['rows']))
(EXP/'manual_browser_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
