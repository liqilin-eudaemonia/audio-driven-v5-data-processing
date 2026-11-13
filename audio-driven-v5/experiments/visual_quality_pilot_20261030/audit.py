#!/usr/bin/env python3
"""Validate frozen source identity, measured support, media, and real browser review."""
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
WEB=ROOT/'data_previews/visual_quality_pilot_20261030'
URL='http://127.0.0.1:8876/visual_quality_pilot_20261030/'
libs=Path('/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
if Path('/tmp/native-fps-fonts.conf').exists():os.environ['FONTCONFIG_FILE']='/tmp/native-fps-fonts.conf'
from playwright.sync_api import sync_playwright


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    data=read(WEB/'data.json');selection=read(OUT/'selection.json');assert len(data['results'])==24
    assert len({r['sample_id'] for r in data['results']})==24
    sources={r['sample_id']:r for r in selection['rows']}
    for inp in selection['inputs']:assert sha(inp['path'])==inp['sha256']
    frame_count=0;total_s=0
    for r in data['results']:
        s=r['source'];assert s==sources[r['sample_id']]['source'];assert not r['automatic_rejection'] and not r['training_admitted']
        assert sha(s['source_result'])==s['source_result_sha256']==r['source_result_sha256']
        original=read(s['source_result'])['clips'][s['source_clip_index']]
        assert original['sample_id']==r['sample_id']
        assert original['start']==s['source_start_s'] and original['end']==s['source_end_s']
        m=r['metrics'];frames=r['crop_detection'];assert len(frames)==m['frames']==math.ceil(s['duration_s']*25-1e-9)
        assert sum(f['detected'] for f in frames)==m['detected_frames']
        covered=sum(min(s['duration_s'],(i+1)/25)-i/25 for i,f in enumerate(frames) if not f['detected'])
        assert abs(covered-sum(z['end']-z['start'] for z in m['missing_intervals']))<1e-6
        meta=read(OUT/'context_media_metadata'/f"{r['sample_id']}.json")
        ts=meta['source_frame_times_s'];assert len(ts)==len(frames)
        assert all(a<=b for a,b in zip(ts,ts[1:]))
        assert s['source_start_s']-1e-5<=ts[0]<s['source_start_s']+.1 and ts[-1]<s['source_end_s']+1e-5
        for point in r['raw_measurements']:
            assert 0<=point['t']<s['duration_s']
            if point['detected']:
                x0,y0,x1,y1=point['bbox_native']
                assert abs((y1-y0)-point['face_height_native'])<1e-5
                assert abs((x1-x0)-point['face_width_native'])<1e-5
                assert 0<=point['bbox_retained_fraction']<=1.0000001
                assert point['detector_input'] in ('expanded_context','full_frame')
        for f in r['findings']:assert 0<=f['start']<=f['end']<=s['duration_s']
        probe=read_probe(WEB/r['media']);video=next(t for t in probe['streams'] if t['codec_type']=='video')
        assert video['height']==512 and video['avg_frame_rate']=='25/1' and int(video['nb_frames'])==m['frames']
        assert any(t['codec_type']=='audio' for t in probe['streams'])
        assert abs(float(probe['format']['duration'])-s['duration_s'])<.09
        frame_count+=len(frames);total_s+=s['duration_s']
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        ctx=browser.new_context(viewport={'width':1440,'height':1100},accept_downloads=True)
        page=ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('typeof data!=="undefined" && data && current && document.getElementById("video").readyState>=2')
        assert page.locator('#list .entry').count()==24
        page.locator('#flagged').click();assert page.locator('#list .entry').count()==data['summary']['review']
        page.locator('#all').click()
        # Every pair's identity and media are reachable, including unflagged controls.
        for i,r in enumerate(data['results']):
            page.locator('#list .entry').nth(i).click()
            assert r['sample_id'] in page.locator('#source').inner_text()
            response=ctx.request.get(URL+r['media'],headers={'Range':'bytes=0-255'})
            assert response.status==206 and len(response.body())==256
            assert ctx.request.get(URL+r['poster']).status==200
        flagged=next((i for i,r in enumerate(data['results']) if r['status']=='review'),0)
        page.locator('#list .entry').nth(flagged).click()
        page.wait_for_function('document.getElementById("video").readyState>=2')
        page.locator('#whole').click();page.wait_for_timeout(800)
        assert page.evaluate('!document.getElementById("video").paused && document.getElementById("video").currentTime>0')
        page.locator('#focus').click();page.wait_for_timeout(2500)
        assert page.evaluate('document.getElementById("video").paused')
        page.locator('#half').click();assert page.evaluate('document.getElementById("video").playbackRate')==.5
        page.locator('#normal').click()
        page.locator('#judgment').select_option('uncertain');page.locator('#note').fill('AUTOMATED QA ONLY — temporary browser context')
        with page.expect_download() as info:page.locator('#export').click()
        download=info.value;download.save_as(OUT/'qa_export.json')
        exported=read(OUT/'qa_export.json');assert len(exported['records'])==1
        # This export is explicitly a UI test, never a human review.
        page.screenshot(path=str(OUT/'review_desktop.png'),full_page=True)
        mobile=browser.new_context(viewport={'width':390,'height':844});mp=mobile.new_page()
        mp.goto(URL);mp.wait_for_function('typeof data!=="undefined" && data && current')
        assert mp.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        mp.screenshot(path=str(OUT/'review_mobile.png'),full_page=True)
        assert not errors,errors
        browser.close()
    report=dict(status='passed',clips=24,measured_frames=frame_count,seconds=total_s,
                checks=['frozen job/source hashes unchanged; same pair IDs and start/end times',
                        'frame coverage and missing intervals agree; raw-coordinate dimensions and sample times checked',
                        'all 24 comparison videos: 25 fps, expected frame count, audio track, duration, HTTP range seeking',
                        'real browser: all clips, review filter, full playback, focused playback auto-stop, speed, export, mobile layout'],
                browser_errors=errors,human_review_performed=False)
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


def read_probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))


if __name__=='__main__':main()
