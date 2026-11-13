"""Check review playback, exclusion labels, crop images and feedback export."""
import collections,json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parents[1]/'data_previews/seamless_listen_v8_recovery'
data=json.loads((OUT/'samples.json').read_text());errors=[];played=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1050})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8875/listen_v8_recovery/')
    page.wait_for_function('(n)=>current !== null && samples.length === n',arg=len(data['samples']))
    assert page.locator('#list button').count()==len(data['samples'])
    for i,row in enumerate(data['samples']):
        page.evaluate('(i)=>choose(samples[i])',i)
        for mode in ('both','speaker','listener'):
            page.locator('#channel').select_option(mode)
            page.locator('#head').click()
            page.wait_for_function('v.readyState>=2 && !v.error && !v.paused && v.currentTime>.12')
            page.locator('#tail').click()
            page.wait_for_function('v.readyState>=2 && !v.error && !v.seeking && v.currentTime>=v.duration-2.1')
            played.append(dict(sample_id=row['sample_id'],channel=mode))
        page.wait_for_function("document.querySelector('#source').complete && document.querySelector('#source').naturalWidth>0")
        if row['review_kind']=='head_shoulders':
            page.wait_for_function("document.querySelector('#oldcrop').complete && document.querySelector('#oldcrop').naturalWidth>0")
            assert page.locator('#cropComparison').is_visible()
        if row['review_kind'].startswith('rejected_'):
            assert '仍被筛掉' in page.locator('#reason').inner_text()
        response=page.request.get(page.evaluate("document.querySelector('#context').src"),headers={'Range':'bytes=0-127'})
        assert response.status==206
        if (i+1)%6==0:print('Verified samples',i+1,flush=True)
    for kind,n in collections.Counter(r['review_kind'] for r in data['samples']).items():
        page.locator('#filter').select_option(kind)
        assert page.locator('#list button').count()==n
    page.locator('#filter').select_option('rejected_content')
    page.locator('#decision').select_option('reject')
    page.locator('#note').fill('Automated browser validation only')
    with page.expect_download() as download:page.locator('#export').click()
    payload=json.loads(Path(download.value.path()).read_text())
    assert payload['selection_sha256']==data['selection_sha256']
    assert any(n['decision']=='reject' for n in payload['notes'].values())
    page.evaluate("localStorage.removeItem('listen-v8-recovery-notes')")
    page.locator('#filter').select_option('head_shoulders')
    page.wait_for_function('v.readyState>=2')
    page.evaluate('v.pause()');page.evaluate('document.fonts.ready')
    response=page.request.get(page.evaluate('v.src'),headers={'Range':'bytes=0-127'})
    assert response.status==206 and len(response.body())==128
    page.screenshot(path=str(OUT/'browser_desktop.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(OUT/'browser_mobile.png'),full_page=True)
    assert not errors,errors
    browser.close()
(OUT/'browser_validation.json').write_text(json.dumps(dict(samples=len(data['samples']),playback_checks=played,
    page_errors=errors,filters=True,review_export=True,range_206=True,mobile_overflow=False),indent=2)+'\n')
print(json.dumps(dict(samples=len(data['samples']),playback_checks=len(played),page_errors=errors)))
