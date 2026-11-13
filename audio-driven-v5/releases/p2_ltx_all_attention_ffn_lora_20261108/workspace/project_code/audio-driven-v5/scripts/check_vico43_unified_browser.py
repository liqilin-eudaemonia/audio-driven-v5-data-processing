"""Exercise real Chromium video playback, switching, seek and mobile layout."""
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'compare/deployment'))
from vico_avatarforcing import WORK,read,save,sha


def check():
    os.environ['LD_LIBRARY_PATH']='/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')
    from playwright.sync_api import sync_playwright
    url='http://127.0.0.1:8876/listen_benchmark_review/vico43_unified_20261107/'
    data=read(ROOT/'data_previews/listen_benchmark_review/vico43_unified_20261107/data.json')
    cases=data['cases'];errors=[];responses=[];checks=[]
    # Check every displayed media URL exists; actual complete decoding is in the
    # separate numerical/media audit. Browser exercises representative cases.
    for c in cases:
        for media in [c['speaker'],c['gt'],c['speaker_audio'],c['our_audio'],*c['native'].values(),*c['score'].values()]:
            target='http://127.0.0.1:8876'+media if media.startswith('/') else url+media
            with urllib.request.urlopen(urllib.request.Request(target,method='HEAD'),timeout=10) as r:
                assert r.status==200,target
            responses.append(target)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox','--disable-gpu'])
        page=browser.new_page(viewport={'width':1600,'height':1050})
        page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(url).status==200
        page.wait_for_function('window.reviewDebug&&reviewDebug().ready',timeout=60000)
        assert page.locator('#clip option').count()==43
        assert page.locator('#image tbody tr').count()==page.locator('#deca tbody tr').count()==3
        assert page.locator('#l2l tbody tr').count()==4
        assert page.locator('#variance tbody tr').count()==4
        assert page.locator('#variance tbody tr').nth(1).locator('td').nth(1).inner_text()==f"{data['deca']['joint']['metrics']['V_D']['exp']:.8f}"
        assert page.locator('#bfm tbody tr').count()==sum(len(v) for v in data['bfm']['joint']['metrics_raw'].values())
        page.locator('details:has(#bfm) > summary').click()
        assert page.locator('#bfm tbody tr').nth(0).locator('td').nth(3).inner_text()==f"{data['bfm']['avatarforcing']['metrics_raw']['FD']['exp']:.5f}"
        picks=[0,min(range(43),key=lambda i:cases[i]['frames']),max(range(43),key=lambda i:cases[i]['frames'])]
        dup=next(i for i,c in enumerate(cases) if c['id'].startswith('yTInFHb_o1Q_000516_000551.'))
        picks.append(dup)
        for i in dict.fromkeys(picks):
            page.select_option('#clip',str(i))
            page.wait_for_function(f'window.reviewDebug&&reviewDebug().ready&&reviewDebug().index=={i}',timeout=60000)
            assert not page.evaluate('reviewDebug().errors')
            page.click('#play');page.wait_for_timeout(650)
            state=page.evaluate('reviewDebug()')
            assert min(state['times'])>.15,state
            assert max(state['times'])-min(state['times'])<.22,state
            page.click('#pause')
            checks.append(dict(id=cases[i]['id'],playback=state))
        page.select_option('#clip','0');page.wait_for_function('reviewDebug().ready&&reviewDebug().index===0')
        page.select_option('#geometry','score');page.wait_for_function('reviewDebug().ready')
        assert page.evaluate("['joint','dystream','avatarforcing'].every(n=>document.getElementById(n).videoWidth===256)")
        page.locator('#seek').evaluate("e=>{e.value='2.5';e.dispatchEvent(new Event('input',{bubbles:true}))}")
        page.wait_for_timeout(300)
        seek=page.evaluate('reviewDebug()');assert max(abs(t-2.5) for t in seek['times'])<.12,seek
        page.select_option('#audio','ours')
        page.click('#play');page.wait_for_timeout(550)
        audio_ours=page.evaluate("({t:document.getElementById('sound').currentTime,error:document.getElementById('sound').error?.message})")
        assert audio_ours['t']>2.6 and not audio_ours['error'],audio_ours
        page.click('#pause');page.select_option('#audio','mute')
        assert page.evaluate("document.getElementById('sound').paused")
        page.screenshot(path=str(WORK/'review_desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
        page.screenshot(path=str(WORK/'review_mobile.png'),full_page=True)
        assert page.goto(url+'blind.html').status==200
        page.wait_for_function('window.blindDebug&&blindDebug().loaded===86',timeout=30000)
        body=page.inner_text('body')
        assert not any(s in body for s in ['Avatar Forcing','DyStream','联合8000'])
        page.fill('#rater','browser-validation-only')
        page.click('#play');page.wait_for_timeout(600)
        assert page.evaluate("['A','B','context'].every(n=>document.getElementById(n).currentTime>.1)")
        page.click('#stop')
        with page.expect_download() as d:page.click('#export')
        d.value.save_as(str(WORK/'human_review/browser_export_validation_only.csv'))
        # No human score is fabricated: all rating cells remain blank.
        text=(WORK/'human_review/browser_export_validation_only.csv').read_text()
        assert 'browser-validation-only' in text
        browser.close()
    assert not errors,errors
    out=dict(passed=True,at=time.time(),media_HEAD_200=len(responses),case_count=43,
        representative_cases=checks,seek=seek,scoring_resolution_verified=True,mobile_no_horizontal_overflow=True,
        generated_listener_audio_switch=audio_ours,mute_switch_verified=True,
        blind_tasks=86,blind_labels_hidden=True,blind_playback_and_CSV_export=True,human_scores_collected=0,js_errors=errors,
        scope='All media URLs checked; actual playback sampled at first, shortest, longest and shared-audio cases; not human review of all videos')
    save(WORK/'BROWSER_AUDIT.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['representative_cases']},ensure_ascii=False,indent=2))


if __name__=='__main__':check()
