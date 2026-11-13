"""Browser + timeline integrity checks for continuous source review."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_previews/seamless_listen_full_source_recovery'
index=json.loads((OUT/'index.json').read_text())
errors=[]
checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8875/listen_full_source/')
    for pair in index['pairs']:
        name=pair['pair']
        page.locator('#source').select_option(name)
        page.wait_for_function('(name)=>window.reviewData?.pair===name && document.querySelector("#video").readyState>=2',arg=name)
        d=json.loads((OUT/name/'review.json').read_text())
        timeline=d['timeline']
        assert timeline[0]['start']==0 and timeline[-1]['end']==d['summary']['source_seconds']
        assert all(abs(a['end']-b['start'])<1e-6 for a,b in zip(timeline,timeline[1:]))
        assert abs(sum(r['end']-r['start'] for r in timeline)-d['summary']['source_seconds'])<1e-5
        assert all(not r['start']<t<r['end'] for r in d['attempts'] for group in d['shots'] for t in group)
        assert len(d['attempts'])>1
        for mode in ['timeline','attempts','old']:
            page.locator('#listMode').select_option(mode)
            assert page.locator('#rows tr').count()==len(d[mode])
        page.locator('#listMode').select_option('timeline')
        for channel in ['both','speaker','listener','a','b']:
            page.evaluate('seek(2)')
            page.locator('#channel').select_option(channel)
            page.locator('#play').click()
            page.wait_for_function('!document.querySelector("#video").paused && document.querySelector("#video").currentTime>2.2')
            page.locator('#play').click()
        # Crossing a candidate's end must not stop or switch the source video.
        end=d['attempts'][0]['end']
        page.evaluate('(t)=>seek(t-.15)',end)
        page.locator('#play').click()
        page.wait_for_function('(t)=>document.querySelector("#video").currentTime>t+.2 && !document.querySelector("#video").paused',arg=end)
        page.locator('#play').click()
        novel=next((r for r in d['attempts'] if r['pool']=='keep_candidate' and r.get('recovery_complete_sentence') is False),d['attempts'][0])
        page.evaluate('(t)=>seek(t)',(novel['start']+novel['end'])/2)
        page.wait_for_function('!document.querySelector("#video").seeking')
        page.evaluate('document.fonts.ready')
        page.screenshot(path=str(OUT/name/'review_desktop.png'),full_page=True)
        response=page.request.get('http://127.0.0.1:8875/listen_full_source/'+d['media'],headers={'Range':'bytes=0-127'})
        assert response.status==206 and len(response.body())==128
        page.locator('#mark').click()
        page.locator('#note').fill('Automated validation only')
        page.locator('#saveNote').click()
        with page.expect_download() as download:page.locator('#exportNotes').click()
        payload=json.loads(Path(download.value.path()).read_text())
        assert any(n['pair']==name and n['text']=='Automated validation only' for n in payload['notes'])
        checks.append(dict(pair=name,full_timeline=True,multiple_pairs=len(d['attempts']),
                           audio_modes=5,continuous_across_boundary=True,range_206=True,feedback_export=True))
    page.evaluate("localStorage.removeItem('listen-full-source-notes')")
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(OUT/'review_mobile.png'),full_page=True)
    browser.close()
assert not errors,errors
(OUT/'browser_validation.json').write_text(json.dumps(dict(checks=checks,page_errors=errors,mobile_overflow=False),indent=2)+'\n')
print(json.dumps(dict(checked_pairs=len(checks),page_errors=errors)))
