#!/usr/bin/env python3
"""Validate exact source preservation, blind review persistence and audio playback."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/listen_visual_input50_20261029'
OUT=ROOT/'data_previews/listen_benchmark_review/prompt_versions_20261029/fps_decision'
URL='http://127.0.0.1:8876/listen_benchmark_review/prompt_versions_20261029/fps_decision/'
libs=Path('/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
from playwright.sync_api import sync_playwright


def main():
    d=json.loads((OUT/'data.json').read_text());checks=[]
    assert len(d['samples'])==50 and len(set(d['queue']))==15
    assert sum(s['task']['group']=='key' for s in d['samples'])==9
    assert sum(s['task']['group']=='random' for s in d['samples'])==6
    for s in d['samples']:
        source=json.loads((RUN/'results'/(s['id']+'.json')).read_text())
        for arm,rs in s['series'].items():
            for r in rs:
                original=next(x for x in source['records'] if x['arm']==arm and x['view']==r['view'])
                assert all(r[k]==original.get(k) for k in r),(s['id'],arm,r['view'])
        assert sorted(s['assignment'])==['video16','video8']
        if s['task']['group'] in ('key','random'):
            assert (ROOT/s['task']['evidence']).exists(),s['id']
    checks.append('All 586 original requests preserve raw text, parsed events, windows, prompts and invalid status; 9+6 review cases have frame evidence')
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        ctx=browser.new_context(viewport={'width':1536,'height':1050},accept_downloads=True)
        page=ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('window.nativeFPSReview && !document.querySelector("#app").hidden')
        page.wait_for_function('reviewAudioState().ready && video.readyState>=2')
        assert page.locator('#sampleSelect option').count()==15
        assert page.locator('.card').count()==2
        assert '8 fps' not in page.locator('#columns').inner_text()
        assert not page.locator('#candidateBox').get_attribute('open')
        assert page.locator('[name=decision]:checked').count()==0
        initial=page.evaluate('nativeFPSState().sample')
        page.locator('#saveNext').click();assert page.evaluate('nativeFPSState().sample')==initial
        page.locator('[name=decision]').first.check()
        page.locator('#preference').select_option('A')
        page.locator('#optionalNote summary').click()
        page.locator('#note').fill('SYNTHETIC CHECK — not human feedback')
        page.locator('#save').click()
        assignment=next(s['assignment'] for s in d['samples'] if s['id']==initial)
        with page.expect_download() as download:page.locator('#export').click()
        saved=json.loads(Path(download.value.path()).read_text())
        r=saved['records'][0];assert r['preferred_arm']==assignment[0] and not r['names_revealed']
        assert saved['human_ground_truth'] is False and saved['training_admitted'] is False
        page.reload();page.wait_for_function('window.nativeFPSReview')
        assert page.locator('#note').input_value().startswith('SYNTHETIC CHECK')
        assert page.locator('[name=decision]:checked').count()==1
        page.locator('#saveNext').click();assert page.evaluate('nativeFPSState().sample')!=initial
        assert page.locator('[name=decision]:checked').count()==0
        checks.append('Default blind slots, no preselected answer, answer required to advance, autosave/reload/export preserve correct arm mapping; synthetic browser context only')
        page.locator('#queue').select_option('random')
        assert page.locator('#sampleSelect option').count()==6
        assert page.locator('.event.focus').count()==2
        page.locator('#queue').select_option('all')
        assert page.locator('#sampleSelect option').count()==50
        # Check exact rendered descriptions for every full-clip result without media scanning.
        for s in d['samples']:
            page.locator('#sampleSelect').select_option(s['id'])
            for slot,arm in enumerate(s['assignment']):
                rs=next(r for r in s['series'][arm] if r['view']=='full')
                assert page.locator('.card').nth(slot).locator('.behavior').all_text_contents()==[e['behavior'] for e in rs['parsed']['events']]
        checks.append('All 100 full-clip rendered results match source descriptions; random matched entries highlighted and all50 accessible')
        s=next(s for s in d['samples'] if any(r['start']>0 and r['status']=='valid' and r['parsed']['events'] for r in s['series']['video8']))
        page.locator('#sampleSelect').select_option(s['id'])
        r=next(r for r in s['series']['video8'] if r['start']>0 and r['status']=='valid' and r['parsed']['events'])
        page.locator('#view').select_option(r['view'])
        card=page.locator('.card').nth(s['assignment'].index('video8'))
        assert abs(float(card.locator('.seek').first.get_attribute('data-a'))-r['parsed']['events'][0]['start']-r['start'])<1e-7
        page.wait_for_function('reviewAudioState().ready && video.readyState>=2')
        card.locator('.seek').first.click();page.wait_for_timeout(500)
        assert page.evaluate('!video.paused && !listenerAudio.paused && Math.abs(video.currentTime-listenerAudio.currentTime)<.2')
        page.locator('#speed').select_option('0.5')
        assert page.evaluate('video.playbackRate===.5 && listenerAudio.playbackRate===.5')
        page.locator('#audioMode').select_option('dialogue')
        page.wait_for_function('reviewAudioState().ready && speakerAudio.readyState>=2');page.wait_for_timeout(400)
        assert page.evaluate('!speakerAudio.paused && !listenerAudio.paused')
        page.evaluate('video.pause()');assert page.evaluate('speakerAudio.paused && listenerAudio.paused')
        page.locator('#audioMode').select_option('processed');page.wait_for_function('reviewAudioState().ready')
        checks.append('Window offsets, seek/loop, audible listener, dual reference, processed audio, half speed and pause synchronization')
        invalid=next((s,a,r) for s in d['samples'] for a,rs in s['series'].items() for r in rs if r['status']!='valid')
        s,a,r=invalid;page.locator('#sampleSelect').select_option(s['id']);page.locator('#view').select_option(r['view'])
        card=page.locator('.card').nth(s['assignment'].index(a))
        assert card.locator('.bad').count()==1 and card.locator('.event').count()==0
        assert card.locator('pre').first.text_content()==r['raw']
        checks.append('Invalid output is inspectable but never rendered as usable events')
        for s in d['samples']:
            for path in [s['video'],s['audio']['original'],s['audio']['speaker']]:
                r=ctx.request.get(URL+path,headers={'Range':'bytes=0-127'})
                assert r.status==206 and len(r.body())==128,(s['id'],path,r.status)
        assert ctx.request.get(URL+'../?comparison=inputs').status==200
        checks.append('All150 existing media endpoints support seek ranges; original review remains reachable')
        page.locator('#queue').select_option('review');page.locator('#audioMode').select_option('original')
        page.wait_for_function('video.readyState>=2 && reviewAudioState().ready')
        page.evaluate('localStorage.clear()');page.reload();page.wait_for_function('window.nativeFPSReview && video.readyState>=2 && reviewAudioState().ready')
        page.screenshot(path=str(OUT/'desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(OUT/'mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors,errors
        checks.append('Desktop and mobile rendering without overflow or JavaScript errors')
        browser.close()
    report={'passed':True,'checks':checks,'page_errors':errors,'synthetic_feedback_only':True}
    (RUN/'native_fps_decision/browser_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
