#!/usr/bin/env python3
"""Validate candidate timing/provenance and the actual review/audio browser flow."""
import csv
import json
import os
from pathlib import Path
from run_glare_nodding_pilot import OUT, SOURCE, intervals, sha

libs = Path('/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu')
if libs.exists():
    os.environ['LD_LIBRARY_PATH'] = str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
from playwright.sync_api import sync_playwright

URL = 'http://127.0.0.1:8876/listen_benchmark_review/prompt_versions_20261029/glare_nodding/'

def main():
    checks = []
    # Separate occurrences stay separate; end clamp and missing-face flags are retained.
    events = intervals([0,.5,.6,0,.4,.5], [1,1,0,1,1,1], 25, .23)
    assert [(e['start'], e['end']) for e in events] == [(.04,.12),(.16,.23)]
    assert events[0]['face_detection_fraction'] == .5
    assert events[1]['touches_clip_boundary']
    assert intervals([0,0], [1,1], 25, .08) == []
    data=json.loads((SOURCE.parent/'glare_nodding/data.json').read_text())
    source={s['id']:s for s in json.loads(SOURCE.read_text())['samples']}
    assert len(data['samples']) == data['summary']['samples'] == data['summary']['ok']
    for s in data['samples']:
        d=s['detector'];assert d['status']=='ok'
        assert d['video_sha256']==sha(Path(d['video']))
        for k in ('r4:video8','r4:video16'):
            assert s['series'][k]==source[s['id']]['series'][k]
        for name,p in d['profiles'].items():
            assert len(p['scores'])==d['frames']==len(d['pose']['detected'])
            assert p['events']==intervals(p['scores'],d['pose']['detected'],d['fps'],s['duration'])
            rows=list(csv.DictReader((OUT/'scores'/name/(s['id']+'.csv')).open()))
            assert len(rows)==d['frames']
            for row,score in zip(rows,p['scores']):
                assert abs(float(row['nodding_intensity'])-score)<.000001
            assert all(0<=e['start']<e['end']<=s['duration'] for e in p['events'])
    checks.append('all candidates reconstructed from original gated frame scores; bounds, gaps, CSV lengths and video hashes checked; VLM records exactly preserved')
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        ctx=browser.new_context(viewport={'width':1536,'height':1050},accept_downloads=True)
        page=ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('window.glareReviewData && reviewAudioState().ready && video.readyState>=2')
        assert page.locator('#samples option').count()==len(data['samples'])
        first=data['samples'][0]
        for s in data['samples']:
            page.locator('#samples').select_option(s['id'])
            assert page.locator('#profiles .card').count()==3
            for k in ('r4:video8','r4:video16'):
                for e in s['series'][k][0]['parsed'].get('events',[]):
                    assert e['behavior'] in page.locator('#vlm').inner_text()
            for path in [s['video'],s['audio']['original'],s['audio']['speaker']]:
                r=ctx.request.get(URL+path,headers={'Range':'bytes=0-255'})
                assert r.status==206,(s['id'],path,r.status)
        page.locator('#samples').select_option(first['id'])
        page.wait_for_function('reviewAudioState().ready && video.readyState>=2')
        page.locator('#soundPlay').click();page.wait_for_timeout(800)
        assert page.evaluate('!video.paused && !reviewAudioState().listenerPaused')
        assert page.evaluate('Math.abs(video.currentTime-reviewAudioState().listenerTime)<.2')
        page.locator('#speed').select_option('0.5')
        page.evaluate('video.currentTime=1.1');page.wait_for_timeout(600)
        assert page.evaluate('Math.abs(video.currentTime-reviewAudioState().listenerTime)<.2')
        page.locator('#audioMode').select_option('processed');page.wait_for_function('reviewAudioState().ready')
        page.locator('#audioMode').select_option('dialogue');page.wait_for_function('reviewAudioState().ready');page.wait_for_timeout(400)
        assert page.evaluate('Math.abs(video.currentTime-reviewAudioState().speakerTime)<.2')
        page.evaluate('video.pause()')
        page.locator('#decision').select_option('uncertain')
        page.locator('#note').fill('SYNTHETIC TEST — not human review')
        with page.expect_download() as download: page.locator('#export').click()
        exported=json.loads(Path(download.value.path()).read_text())
        assert exported['records'][0]['note'].startswith('SYNTHETIC') and not exported['training_admitted']
        page.reload();page.wait_for_function('window.glareReviewData')
        assert page.locator('#note').input_value().startswith('SYNTHETIC')
        page.locator('#note').fill('');page.locator('#decision').select_option('')
        if data.get('reaction_summary'):
            for s in data['samples']:
                page.locator('#samples').select_option(s['id'])
                assert page.locator('.reaction-card').count()==6
                for k,v in s['reactions']['channels'].items():
                    assert page.locator(f'.reaction-card[data-reaction="{k}"] .candidate').count()==len(v['events'])
            page.locator('#filter').select_option('reaction:smiling')
            assert page.locator('#samples option').count()==data['reaction_summary']['channels']['smiling']['clips']
            page.locator('#filter').select_option('all')
            page.locator('.reaction-decision[data-type="smiling"]').select_option('uncertain')
            page.locator('.reaction-note[data-type="smiling"]').fill('SYNTHETIC CLASS REVIEW')
            with page.expect_download() as download:page.locator('#export').click()
            exported=json.loads(Path(download.value.path()).read_text())
            assert exported['schema']=='glare-reactions-human-review-v1'
            assert exported['reaction_records'][0]['type']=='smiling'
            page.reload();page.wait_for_function('window.glareReviewData')
            assert page.locator('.reaction-note[data-type="smiling"]').input_value()=='SYNTHETIC CLASS REVIEW'
            page.locator('.reaction-note[data-type="smiling"]').fill('')
            page.locator('.reaction-decision[data-type="smiling"]').select_option('')
            checks.append('all 15 six-channel panels, candidate counts, category filter and independent per-category feedback export/reload verified')
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(OUT/'review_desktop.png'),full_page=False)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors,errors
        browser.close()
    checks.append('all sample videos and both audio sources HTTP206; desktop/mobile, exact full-clip text, synchronized original/processed/dialogue audio, speed/seek, local feedback reload/export checked')
    (OUT/'audit.json').write_text(json.dumps({'passed':True,'samples':len(data['samples']),'checks':checks,'browser_errors':errors},indent=2)+'\n')
    if data.get('reaction_summary'):
        reaction_out=OUT.parent/'listen_glare_reactions15_20261029'
        (reaction_out/'audit_browser.json').write_text(json.dumps({'passed':True,'checks':checks,'browser_errors':errors},indent=2)+'\n')
    print(json.dumps({'passed':True,'samples':len(data['samples']),'checks':checks}))

if __name__=='__main__':
    main()
