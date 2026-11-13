import json
from pathlib import Path
from playwright.sync_api import sync_playwright

RUN=Path(__file__).resolve().parent
URL='http://127.0.0.1:8876/listen_benchmark_review/audio_official_aligned100/'
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1500,'height':1000})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(URL);page.wait_for_function('typeof C!=="undefined" && C && audio.readyState>=2')
    assert page.locator('#samples option').count()==41
    assert page.evaluate('C.aligned_audio_events.length>0')
    page.click('#none');assert page.locator('#samples option').count()==59
    assert '没有剩余发声标注' in page.locator('#panel').inner_text()
    assert page.evaluate('C.gate.full_silence')
    page.screenshot(path=str(RUN/'no_audio_labels.png'))
    # No-label/non-silent clips must not be misrepresented as silent.
    page.evaluate('select(DATA.cases.findIndex(c=>c.label_group==="no_audio_labels"&&!c.gate.full_silence))')
    assert '没有可保留的原模型音频事件' in page.locator('#panel').inner_text()
    page.click('#has');page.evaluate('select(18)');page.wait_for_function('audio.readyState>=2')
    assert '采用官方词时间' in page.locator('#panel').inner_text()
    assert '1.156–1.656' in page.locator('#panel').inner_text()
    left=page.locator('.left').bounding_box();right=page.locator('.right').bounding_box()
    assert left['x']+left['width']<right['x']
    assert left['y']+left['height']<=1001
    assert page.evaluate('document.documentElement.scrollHeight<=innerHeight+1')
    for name in ['listener','removed','speaker','processed']:
        page.evaluate('audio.pause();audio.currentTime=0.5');page.click('#'+name)
        page.wait_for_function('audio.readyState>=2 && Math.abs(audio.currentTime-0.5)<.1')
    page.click('#play');page.wait_for_timeout(400)
    assert page.evaluate('!audio.paused && Math.abs(audio.currentTime-video.currentTime)<.2')
    page.click('#listener');page.wait_for_function('audio.readyState>=2&&!audio.paused');page.evaluate('audio.pause()')
    page.locator('.timing .new button').first.click();page.wait_for_function('audio.paused',timeout=4000)
    assert page.evaluate('audio.currentTime>=1.65 && audio.currentTime<1.9')
    page.screenshot(path=str(RUN/'desktop.png'))
    # Inspect every sample's rendered grouping, cards and official/visual tabs.
    for i in range(100):
        page.evaluate('(i)=>select(i)',i)
        assert page.locator('#panel .card').count()==page.evaluate('C.aligned_audio_events.length')
        for tab in ['excluded','visual','official']:
            page.click('#tab-'+tab)
        assert not page.locator('#status').inner_text()
    page.click('#has');page.evaluate('select(54)');page.wait_for_function('audio.readyState>=2')
    assert 'Um' in page.locator('#panel .description').inner_text()
    assert 'Yeah.' in page.locator('#panel .notice').inner_text()
    page.screenshot(path=str(RUN/'conflicting_text_preserved.png'))
    page.select_option('#decision','good');page.fill('#note','browser test only')
    with page.expect_download() as info:page.click('#export')
    feedback=json.loads(Path(info.value.path()).read_text());assert len(feedback['records'])==1 and feedback['records'][0]['decision']=='good'
    assert feedback['training_applied'] is False
    for name in ['labelDownload','audioDownload']:
        assert page.request.get(URL+page.locator('#'+name).get_attribute('href'),headers={'Range':'bytes=0-127'}).status in (200,206)
    page.evaluate('localStorage.removeItem(STORAGE)')
    page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(400)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    page.screenshot(path=str(RUN/'mobile.png'))
    assert not errors,errors
    report=dict(samples_rendered=100,two_groups=[41,59],no_labels_vs_silence_distinguished=True,
                desktop_video_left_and_labels_right=True,desktop_fits_viewport=True,mobile_no_overflow=True,
                audio_switch_preserves_time_and_play=True,synchronized_video=True,range_play_stops=True,
                original_conflicting_text_preserved=True,feedback_export=True,download_links=True,page_errors=errors)
    (RUN/'browser_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    browser.close()
