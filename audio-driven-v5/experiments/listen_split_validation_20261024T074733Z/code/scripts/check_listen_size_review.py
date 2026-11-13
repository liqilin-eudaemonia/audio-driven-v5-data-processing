"""Browser verification for real media, synchronized playback and threshold feedback."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_previews/seamless_listen_size_review'
data=json.loads((OUT/'review.json').read_text())
errors=[];checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1680,'height':1200})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8875/listen_size_review/')
    page.wait_for_function('window.reviewData && window.currentSample && document.getElementById("listenerVideo").readyState>=2 && document.getElementById("speakerVideo").readyState>=2')
    assert page.locator('#list button').count()==50
    page.locator('#play').click()
    page.wait_for_function('document.getElementById("listenerVideo").currentTime>.3 && !document.getElementById("speakerVideo").paused')
    for channel in ['listener','speaker','both','mute']:
        page.locator('#channel').select_option(channel)
        states=page.evaluate('({l:document.getElementById("listenerVideo").muted,s:document.getElementById("speakerVideo").muted})')
        assert states==dict(l=channel not in ['listener','both'],s=channel not in ['speaker','both'])
    page.locator('#channel').select_option('listener')
    page.locator('#play').click()
    page.locator('#faceMin').select_option('128')
    expected=data['statistics']['face']['p10_short_side_threshold_seconds']['128']/3600
    assert f'{expected:.1f}' in page.locator('#retention').inner_text()
    page.locator('#cropMin').select_option('512')
    page.locator('#onlyPass').check()
    expected_rows=[r for r in data['items'] if 'random' in r['cohorts'] and
                   r['face_geometry']['listener'].get('face_short_side_px',{}).get('p10',-1)>=128 and
                   r['crop_proposals']['listener'].get('size',-1)>=512]
    assert page.locator('#list button').count()==len(expected_rows)
    page.locator('#onlyPass').uncheck()
    page.locator('#faceMin').select_option('0');page.locator('#cropMin').select_option('0')
    for group in data['groups']:
        page.locator('#group').select_option(group['id'])
        count=sum(group['id'] in r['cohorts'] for r in data['items'])
        assert page.locator('#list button').count()==count
        if count:
            page.wait_for_function('document.getElementById("listenerVideo").readyState>=2 && document.getElementById("speakerVideo").readyState>=2')
            assert page.locator('#error').inner_text()==''
        checks.append(dict(group=group['id'],samples=count))
    page.locator('#group').select_option('face_0')
    page.locator('#faceMin').select_option('512');page.locator('#onlyPass').check()
    assert page.locator('#empty').is_visible()
    assert page.evaluate('document.getElementById("listenerVideo").getAttribute("src")') is None
    page.locator('#onlyPass').uncheck();page.locator('#faceMin').select_option('0')
    page.locator('#group').select_option('face_3')
    page.wait_for_function('document.getElementById("listenerVideo").readyState>=2 && document.getElementById("speakerVideo").readyState>=2')
    page.locator('#inside').click()
    page.wait_for_function('!document.getElementById("listenerVideo").seeking')
    page.evaluate('seek(range()[0]+.3)')
    page.wait_for_function('!document.getElementById("listenerVideo").seeking && !document.getElementById("speakerVideo").seeking')
    page.evaluate('document.fonts.ready')
    page.screenshot(path=str(OUT/'review_desktop.png'),full_page=True)
    page.locator('#decision').select_option('crop');page.locator('#notes').fill('Automated browser validation only')
    page.locator('#save').click()
    with page.expect_download() as download:page.locator('#export').click()
    notes=json.loads(Path(download.value.path()).read_text())
    assert any(r['text']=='Automated browser validation only' for r in notes['notes'])
    current=page.evaluate('window.currentSample')
    response=page.request.get('http://127.0.0.1:8875/listen_size_review/'+current['roles']['listener']['context_video'],headers={'Range':'bytes=0-127'})
    assert response.status==206 and len(response.body())==128
    page.locator('#endBoundary').click()
    page.wait_for_function('document.getElementById("listenerVideo").readyState>=2 && !document.getElementById("listenerVideo").seeking')
    page.evaluate('seek(range()[1]-.12)')
    page.locator('#play').click()
    page.wait_for_function('document.getElementById("listenerVideo").currentTime>range()[1]+.15')
    page.locator('#play').click()
    assert page.evaluate('Math.abs(document.getElementById("listenerVideo").currentTime-document.getElementById("speakerVideo").currentTime)<.16')
    page.locator('#view').select_option('clip')
    page.wait_for_function('document.getElementById("listenerVideo").readyState>=2')
    assert page.evaluate('Math.abs(document.getElementById("listenerVideo").duration-window.currentSample.duration_s)<.1')
    page.evaluate("localStorage.removeItem('seamless-size-review-20261024')")
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(OUT/'review_mobile.png'),full_page=True)
    browser.close()
assert not errors,errors
(OUT/'browser_validation.json').write_text(json.dumps(dict(groups=checks,page_errors=errors,
    random_samples=50,audio_channels_verified=True,thresholds_and_empty_state_verified=True,
    range_seek=True,playback_continues_across_boundary=True,paired_playback_synchronized=True,
    feedback_export=True,mobile_overflow=False),ensure_ascii=False,indent=2)+'\n')
print('Browser checks passed',len(checks),'groups')
