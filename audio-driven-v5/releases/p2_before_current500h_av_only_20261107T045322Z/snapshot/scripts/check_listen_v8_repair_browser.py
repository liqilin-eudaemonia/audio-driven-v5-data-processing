"""Play all repaired-batch samples and verify reviewer interactions."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parents[1]/'data_previews/seamless_listen_v8_repair'
data=json.loads((OUT/'samples.json').read_text());errors=[];played=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1050})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8875/listen_v8_repair/')
    page.wait_for_function('current !== null && samples.length === 24')
    assert page.locator('#list button').count()==24
    for i,row in enumerate(data['samples']):
        page.evaluate('(i)=>choose(samples[i])',i)
        for mode in ('both','speaker','listener'):
            page.locator('#channel').select_option(mode)
            page.locator('#head').click()
            page.wait_for_function('v.readyState>=2 && !v.error && !v.paused && v.currentTime>.12')
            page.locator('#tail').click()
            page.wait_for_function('v.readyState>=2 && !v.error && !v.seeking && v.currentTime>=v.duration-2.1')
            played.append(dict(sample_id=row['sample_id'],channel=mode))
        if (i+1)%6==0:print('Verified samples',i+1,flush=True)
    for kind,count in [('repaired',6),('candidate',6),('boundary_review',3),('audio_review',3),('long',3),('short',2),('exclude',1)]:
        page.locator('#filter').select_option(kind)
        assert page.locator('#list button').count()==count
    page.locator('#filter').select_option('all')
    page.locator('#decision').select_option('boundary')
    page.locator('#note').fill('Automated browser validation only')
    with page.expect_download() as download:page.locator('#export').click()
    payload=json.loads(Path(download.value.path()).read_text())
    assert payload['review']=='v8-repair-small-batch'
    assert payload['selection_sha256']==data['selection_sha256']
    assert next(iter(payload['notes'].values()))['decision']=='boundary'
    page.locator('#decision').select_option('');page.locator('#note').fill('')
    page.evaluate("localStorage.removeItem('listen-v8-repair-notes')")
    page.wait_for_function('v.readyState>=2')
    page.evaluate('v.pause()');page.evaluate('document.fonts.ready')
    response=page.request.get(page.evaluate('v.src'),headers={'Range':'bytes=0-127'})
    assert response.status==206 and len(response.body())==128
    assert page.locator('#source').evaluate('(img)=>img.complete && img.naturalWidth>0')
    page.screenshot(path=str(OUT/'browser_desktop.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(OUT/'browser_mobile.png'),full_page=True)
    assert not errors,errors
    browser.close()
result=dict(samples=24,playback_checks=played,page_errors=errors,filters=True,
    review_export=True,range_206=True,mobile_overflow=False)
(OUT/'browser_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(samples=24,playback_checks=len(played),page_errors=errors)))
