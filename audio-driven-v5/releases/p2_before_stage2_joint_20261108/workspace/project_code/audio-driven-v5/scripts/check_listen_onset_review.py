"""Verify reserve pool labels, source VAD and direct-link playback in the shared UI."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_previews/seamless_listen_size_review'
data=json.loads((OUT/'review.json').read_text());items=[r for r in data['items'] if 'reserve_all' in r['cohorts']]
assert len(items)==32
assert sum('reserve_random' in r['cohorts'] for r in items)==24
for r in items:
    assert r['admission_pool']=='onset_reserve' and r['audio_pool']=='keep_candidate'
    e=r['onset_evidence'];s=r['source_start_s']
    assert any(a<s+.5 and b>s for a,b in e['listener_vad_source'])
    assert e['boundary_activity']['onset_crossing']==any(a<s-1e-7 and b>s+1e-7 for a,b in e['listener_vad_source'])
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1680,'height':1250})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8875/listen_size_review/?group=reserve_all')
    page.wait_for_function('window.currentSample && document.getElementById("listenerVideo").readyState>=2 && document.getElementById("speakerVideo").readyState>=2')
    assert page.locator('#list button').count()==32
    assert page.locator('#channel').input_value()=='listener'
    assert '333.3' in page.locator('#retention').inner_text() and '2002.6' not in page.locator('#retention').inner_text()
    assert '储备池' in page.locator('#sampleInfo').inner_text()
    assert page.locator('#onsetInfo').is_visible()
    for group,expected in [('reserve_random',24),('reserve_crossing',4),('reserve_early',4)]:
        page.locator('#group').select_option(group)
        assert page.locator('#list button').count()==expected
        page.wait_for_function('document.getElementById("listenerVideo").readyState>=2 && document.getElementById("speakerVideo").readyState>=2')
        if group=='reserve_crossing':assert '人声跨过切入点' in page.locator('#onsetInfo').inner_text()
        if group=='reserve_early':assert '前 0.5 秒内起声' in page.locator('#onsetInfo').inner_text()
    page.locator('#faceMin').select_option('128')
    expected=data['statistics_by_pool']['onset_reserve']['face']['p10_short_side_threshold_seconds']['128']/3600
    assert f'{expected:.1f}' in page.locator('#retention').inner_text()
    page.locator('#faceMin').select_option('0')
    page.locator('#group').select_option('reserve_crossing')
    page.wait_for_function('document.getElementById("listenerVideo").readyState>=2 && document.getElementById("speakerVideo").readyState>=2')
    page.evaluate('seek(range()[0]-.12)')
    page.locator('#play').click()
    page.wait_for_function('document.getElementById("listenerVideo").currentTime>range()[0]+.6')
    page.locator('#play').click()
    page.evaluate('document.fonts.ready')
    page.screenshot(path=str(OUT/'onset_samples/review_desktop.png'),full_page=True)
    page.locator('#decision').select_option('onset_keep');page.locator('#notes').fill('Automated onset test')
    page.locator('#save').click()
    with page.expect_download() as download:page.locator('#export').click()
    feedback=json.loads(Path(download.value.path()).read_text())
    assert any(n['decision']=='onset_keep' and n['text']=='Automated onset test' for n in feedback['notes'])
    page.locator('#group').select_option('random')
    assert page.locator('#onsetInfo').is_hidden()
    assert '2002.6' in page.locator('#retention').inner_text()
    assert '主池音质通过候选' in page.locator('#sampleInfo').inner_text()
    page.evaluate("localStorage.removeItem('seamless-size-review-20261024')")
    browser.close()
assert not errors,errors
(OUT/'onset_samples/browser_validation.json').write_text(json.dumps(dict(samples=32,random=24,
    pool_labels_verified=True,statistics_from_reserve=True,source_vad_consistent=True,
    boundary_playback=True,feedback_export=True,primary_regression=True,page_errors=errors),indent=2)+'\n')
print('Onset review passed; 32 samples, reserve/main distinction and source VAD verified')
