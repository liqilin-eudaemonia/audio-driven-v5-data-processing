"""Validate matched old/new/context playback and source-time switching."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parents[1]/'data_previews/seamless_listen_v8_boundaries'
data=json.loads((OUT/'samples.json').read_text());errors=[];tested=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1050})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8875/listen_v8/')
    page.wait_for_function('current !== null && cases.length === 24')
    for i,c in enumerate(data['cases']):
        page.evaluate('(i)=>choose(cases[i])',i)
        for version in ['old','context']+(['new'] if c['new'] else []):
            page.locator('#version').select_option(version)
            page.locator('#head').click()
            page.wait_for_function('v.readyState>=2 && !v.error && v.currentTime>.12 && !v.paused')
            page.locator('#tail').click()
            page.wait_for_function('v.readyState>=2 && !v.error && !v.seeking && v.currentTime>=v.duration-2.1')
            tested.append(dict(case=c['case_id'],version=version))
        if not c['new']:assert page.locator('#version option[value="new"]').is_disabled()
        if (i+1)%6==0:print('Browser cases verified:',i+1,flush=True)
    page.locator('#filter').select_option('known')
    assert page.locator('#list button').count()==3
    for mode in ['speaker','listener','both']:
        page.locator('#channel').select_option(mode);page.locator('#head').click()
        page.wait_for_function('v.readyState>=2 && !v.error && v.currentTime>.12 && !v.paused')
    page.locator('#filter').select_option('extended')
    assert page.locator('#list button').count()==data['summary']['over_10s']
    page.locator('#filter').select_option('unresolved')
    assert page.locator('#list button').count()==data['summary']['unresolved']
    page.locator('#filter').select_option('known')
    page.locator('#decision').select_option('better');page.locator('#note').fill('Browser check only')
    with page.expect_download() as download:page.locator('#export').click()
    payload=json.loads(Path(download.value.path()).read_text())
    assert payload['review']=='v8-boundary-comparison'
    assert next(iter(payload['notes'].values()))['new_sample_id']
    page.locator('#decision').select_option('');page.locator('#note').fill('')
    page.wait_for_function('v.readyState>=2');page.evaluate('document.fonts.ready');page.evaluate('v.pause()')
    response=page.request.get(page.evaluate('v.src'),headers={'Range':'bytes=0-127'})
    assert response.status==206 and len(response.body())==128
    page.screenshot(path=str(OUT/'browser_desktop.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(OUT/'browser_mobile.png'),full_page=True)
    assert not errors,errors
    browser.close()
result=dict(cases=24,playback_checks=tested,page_errors=errors,channel_switch=True,
    filters=True,review_export=True,range_206=True,mobile_overflow=False)
(OUT/'browser_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(cases=24,versions_played=len(tested),page_errors=errors)))
