import json
from pathlib import Path
from playwright.sync_api import sync_playwright

run=Path(__file__).resolve().parent
url='http://127.0.0.1:8876/listen_benchmark_review/audio_official_gate100/'
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1500,'height':1050})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(url);page.wait_for_function('typeof C!=="undefined" && C && audio.readyState>=2')
    assert page.locator('#overview tbody tr').count()==100
    assert page.locator('#timeline .lane').count()==9
    decoded=page.evaluate('''async()=>{
      let count=0;const ctx=new AudioContext();
      for(const c of DATA.cases)for(const k of ['processed','removed']){
        const r=await fetch(c.media[k]);if(!r.ok)throw Error(c.sample_id+k);
        const b=await ctx.decodeAudioData(await r.arrayBuffer());
        if(Math.abs(b.duration-c.duration_s)>.001)throw Error('duration');count++;
      }await ctx.close();return count;
    }''')
    assert decoded==200
    counts={}
    for name,expected in [('full_silence',52),('has_audio',48),('words_rescue',7)]:
        page.select_option('#filter',name);counts[name]=page.locator('#overview tbody tr').count();assert counts[name]==expected
    page.select_option('#filter','all')
    page.evaluate('select(DATA.cases.findIndex(c=>!c.gate.full_silence))')
    page.wait_for_function('audio.readyState>=2')
    for name in ['listener','removed','speaker','processed']:
        page.evaluate('audio.pause();audio.currentTime=0.5');page.click('#'+name)
        page.wait_for_function('audio.readyState>=2 && Math.abs(audio.currentTime-0.5)<.1')
        assert page.evaluate('video.muted')
    page.evaluate('audio.play()');page.wait_for_timeout(400)
    assert page.evaluate('!audio.paused && Math.abs(audio.currentTime-video.currentTime)<.2')
    page.click('#listener');page.wait_for_function('audio.readyState>=2 && !audio.paused')
    page.evaluate('audio.pause()')
    for name in ['downloadAudio','downloadLabels','downloadGate']:
        href=page.locator('#'+name).get_attribute('href');assert page.request.get(url+href).status==200
    page.select_option('#inside','reduced');page.select_option('#outside','preserved');page.fill('#reviewNote','browser test only')
    with page.expect_download() as info:page.click('#exportReview')
    feedback=json.loads(Path(info.value.path()).read_text());assert len(feedback['records'])==1
    assert feedback['records'][0]['noise_change']=='reduced' and feedback['training_applied'] is False
    page.reload();page.wait_for_function('typeof C!=="undefined" && C')
    assert '1 / 100' in page.locator('#reviewProgress').inner_text()
    page.evaluate('localStorage.removeItem(STORAGE)');page.reload();page.wait_for_function('typeof C!=="undefined" && C')
    page.evaluate('select(DATA.cases.findIndex(c=>!c.gate.full_silence))')
    page.wait_for_timeout(600);page.screenshot(path=str(run/'desktop.png'),full_page=False)
    page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(300)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    page.screenshot(path=str(run/'mobile.png'),full_page=False)
    assert not errors,errors
    report=dict(new_audio_files_browser_decoded=decoded,filters=counts,nine_tracks=True,switch_keeps_time=True,playback_synchronized=True,download_links=True,feedback_export_and_storage=True,mobile_no_overflow=True,page_errors=errors)
    (run/'browser_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    browser.close()
