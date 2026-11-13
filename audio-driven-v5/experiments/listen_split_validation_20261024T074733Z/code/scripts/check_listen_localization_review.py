"""Browser checks for mapped timing candidates and independent audio playback."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1560,'height':1050},accept_downloads=True)
    page=context.new_page();errors=[];checks=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    response=page.goto('http://127.0.0.1:8876/listen_benchmark_review/localization/')
    assert response.status==200
    page.wait_for_function('window.localizationReview && window.localizationState')
    assert page.locator('#samples option').count()==50
    checks.append('50 parent clips and embedded results load')
    sid=page.evaluate("localizationReview.samples.find(s=>s.id.includes('S0699')).id")
    page.select_option('#samples',sid)
    page.wait_for_function("video.readyState>=2 && audio.readyState>=2")
    assert 'Okay · 3 个窗口支持' in page.locator('#cards').inner_text()
    page.locator('.event').filter(has_text='Okay · 3 个窗口支持').click()
    page.wait_for_timeout(1300)
    s=page.evaluate('({v:video.currentTime,a:audio.currentTime,paused:video.paused,muted:video.muted,loop:localizationState().loop})')
    assert s['loop']==[4.08,4.4] and s['muted'] and not s['paused'],s
    assert 4.04<=s['v']<=4.5 and abs(s['v']-s['a'])<.2,s
    checks.append('window offsets and second okay loop use parent-clip 4.08–4.40 seconds')
    page.locator('#stopLoop').click();assert page.evaluate('localizationState().loop') is None
    page.evaluate('video.pause()')
    page.locator('#diagnostic').evaluate('(e)=>e.open=true')
    assert page.locator('#diagCards article').count()==3
    assert '00:00:02.000' in page.locator('#diagCards').inner_text() or '00:00:02.000' in page.locator('#diagCards').text_content()
    checks.append('three backends preserve raw timestamp-string outputs')
    page.locator('#feedbackOnly').check();assert page.locator('#samples option').count()==4
    page.locator('#note').fill('browser-test-only: event 1.0–1.2')
    with page.expect_download() as download:page.locator('#export').click()
    data=json.loads(Path(download.value.path()).read_text());assert len(data['records'])==1 and data['human_ground_truth'] is False
    checks.append('four feedback clips filter and isolated browser feedback export')
    media=page.evaluate('video.src');r=context.request.get(media,headers={'Range':'bytes=0-1023'});assert r.status==206
    checks.append('media byte-range requests succeed')
    page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),page.evaluate('document.documentElement.scrollWidth')
    checks.append('390px mobile layout fits viewport')
    assert not errors,errors
    (ROOT/'experiments/listen_localization_20261024T070145Z/browser_check.json').write_text(json.dumps({'checks':checks,'errors':errors},ensure_ascii=False,indent=2))
    print(json.dumps({'checks':checks,'errors':errors},ensure_ascii=False))
    browser.close()
