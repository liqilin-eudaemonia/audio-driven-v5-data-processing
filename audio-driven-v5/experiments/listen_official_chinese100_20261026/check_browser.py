import json
from pathlib import Path
from playwright.sync_api import sync_playwright
RUN=Path(__file__).resolve().parent;URL='http://127.0.0.1:8876/listen_benchmark_review/audio_official_chinese100/'
with sync_playwright() as p:
    b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1366,'height':768});errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)));page.goto(URL);page.wait_for_function('typeof C!=="undefined"&&C&&audio.readyState>=2')
    assert page.locator('#samples option').count()==38;page.click('#none');assert page.locator('#samples option').count()==62
    for i,text in [(22,'南京地铁'),(57,'我唔记得咗'),(76,'不能')]:
        page.evaluate('(i)=>select(i)',i);page.wait_for_function('audio.readyState>=2')
        assert page.evaluate('C.aligned_audio_events.length')==0
        page.click('#tab-excluded');assert text in page.locator('#panel').inner_text();assert '对应区间静音' in page.locator('#panel').inner_text()
        for role in ['previous_processed','processed','listener','removed']:
            page.evaluate('audio.pause();audio.currentTime=.1');page.click('#'+role);page.wait_for_function('audio.readyState>=2&&Math.abs(audio.currentTime-.1)<.05')
        assert page.request.get(URL+page.locator('#labelDownload').get_attribute('href')).status==200
    page.screenshot(path=str(RUN/'removed_chinese.png'))
    page.click('#has')
    for i,text in [(3,'嗯'),(39,'哎'),(47,'呵'),(78,'哈'),(83,'哈'),(95,'嗯')]:
        page.evaluate('(i)=>select(i)',i);assert text in page.locator('#panel').inner_text()
    for i in range(100):
        page.evaluate('(i)=>select(i)',i);assert page.locator('#panel .card').count()==page.evaluate('C.aligned_audio_events.length')
    page.evaluate('select(47)');page.screenshot(path=str(RUN/'single_character_kept.png'))
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1&&document.documentElement.scrollHeight<=innerHeight+1')
    page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    assert not errors,errors
    report=dict(groups=[38,62],all_100_samples_rendered=True,three_removed_cases_visible=True,single_character_cases_visible=True,comparison_audio_switch=True,downloads=True,desktop_fits=True,mobile_no_overflow=True,errors=errors)
    (RUN/'browser_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));b.close()
