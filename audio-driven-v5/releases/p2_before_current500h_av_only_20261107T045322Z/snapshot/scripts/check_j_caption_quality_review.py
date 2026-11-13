#!/usr/bin/env python3
"""Verify source/media identity, browser playback, filters, and feedback export."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/listen_j_caption_quality_review_20261105'
OUT=ROOT/'data_previews/listen_benchmark_review/j_caption_quality_20261105'
URL='http://127.0.0.1:8876/listen_benchmark_review/j_caption_quality_20261105/'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    data=json.loads((OUT/'data.json').read_text());samples=data['samples']
    assert len(samples)==60 and len({s['pair'] for s in samples})==60
    assert sum(s['group']=='pass' for s in samples)==24
    assert all(s['generation_checks_passed'] and not s['caption_review_flags'] for s in samples if s['group']=='pass')
    assert all(not s['generation_checks_passed'] and s['caption_review_flags'] for s in samples if s['group']=='review')
    checks=[]
    for s in samples:
        dest=OUT/s['media'];cap=json.loads((dest/'caption_original.json').read_text())
        assert sha(dest/'caption_original.json')==s['audit']['caption_sha256']==sha(s['source_paths']['caption'])
        assert cap['caption']==s['caption'] and cap['review_flags']==s['caption_review_flags']
        assert cap['source']['video_transform']['crop']==s['final_crop']
        assert cap['input_evidence']['audio_identity']['processed_waveform_sha256']==s['listener_audio']['waveform_sha256']
        for role in ('listener','speaker'):
            info=s[role+'_audio'];raw=(dest/info['raw']).read_bytes()
            assert len(raw)==info['samples']*4 and hashlib.sha256(raw).hexdigest()==info['raw_sha256']
        assert sha(dest/'listener.mp4')==s['audit']['preview_sha256']
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(dest/'listener.mp4')]))
        video=next(r for r in probe['streams'] if r['codec_type']=='video')
        audio=next(r for r in probe['streams'] if r['codec_type']=='audio')
        assert (video['width'],video['height'],video['avg_frame_rate'])==(512,512,'25/1')
        assert int(video['nb_frames'])==s['video']['frames']
        assert abs(float(video['duration'])-s['duration'])<=.041
        assert abs(float(audio['duration'])-s['duration'])<.004
        checks.append(dict(id=s['id'],review_id=s['review_id'],passed=True))
    libs=Path('/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu')
    if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
    from playwright.sync_api import sync_playwright
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',
            headless=True,args=['--no-sandbox','--disable-gpu'])
        context=browser.new_context(viewport={'width':1450,'height':1050},accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL+'?group=pass').status==200
        page.wait_for_function('window.reviewData && !document.querySelector("#play").disabled && document.querySelector("#video").readyState>=2')
        assert page.locator('#samples option').count()==24
        page.select_option('#groupFilter','review');page.wait_for_function('!document.querySelector("#play").disabled')
        assert page.locator('#samples option').count()==36
        for reason in ['invented_vocal_on_silence','coverage_uncertain','missing_reaction_cue','missing_nod_cue','missing_vocal_cue','generation_anomaly']:
            page.select_option('#reasonFilter',reason)
            count=page.locator('#samples option').count()
            assert count>=6,(reason,count)
        page.select_option('#reasonFilter','all');page.select_option('#groupFilter','all')
        assert page.locator('#samples option').count()==60
        playback_ready=[]
        for sample in samples:
            page.select_option('#samples',sample['id'])
            page.wait_for_function('id=>reviewCurrent().id===id&&!document.querySelector("#play").disabled&&document.querySelector("#video").readyState>=2',arg=sample['id'])
            assert page.locator('#caption').text_content()==sample['caption']
            assert page.locator('#mediaError').text_content()==''
            playback_ready.append(sample['review_id'])
        pcm=page.evaluate('''async()=>{const out=[];for(const s of reviewData.samples){for(const role of ['listener','speaker']){const a=s[role+'_audio'],r=await fetch(s.media+a.raw);if(!r.ok)throw Error('PCM HTTP '+r.status);const b=await r.arrayBuffer();const h=[...new Uint8Array(await crypto.subtle.digest('SHA-256',b))].map(x=>x.toString(16).padStart(2,'0')).join('');out.push({id:s.review_id,role,pass:h===a.raw_sha256&&b.byteLength===a.samples*4})}}return out}''')
        assert len(pcm)==120 and all(r['pass'] for r in pcm)
        page.select_option('#samples',samples[0]['id']);page.wait_for_function('!document.querySelector("#play").disabled')
        page.click('#play');page.wait_for_timeout(800)
        playback=page.evaluate('({audio:reviewAudio(),time:document.querySelector("#video").currentTime})')
        assert playback['audio']['context']=='running' and playback['audio']['playing']==1 and playback['time']>.2
        page.click('[data-mode="speaker"]');assert page.evaluate('reviewAudio().playing')==1
        page.click('[data-mode="both"]');assert page.evaluate('reviewAudio().playing')==2
        page.locator('#video').evaluate('(v)=>v.currentTime=1.0');page.wait_for_timeout(200)
        assert page.evaluate('reviewAudio().playing')==2
        page.click('#pause');assert page.evaluate('reviewAudio().playing')==0
        page.select_option('#verdict','uncertain');page.select_option('#checkerVerdict','false_positive')
        page.locator('.issue').first.check();page.fill('#note','Automated feedback persistence check')
        with page.expect_download() as download:page.click('#export')
        feedback=json.loads(Path(download.value.path()).read_text())
        assert feedback['manifest_sha256']==data['manifest_sha256']
        record=next(r for r in feedback['records'] if r['sample_id']==samples[0]['id'])
        assert record['verdict']=='uncertain' and record['checker_verdict']=='false_positive'
        assert record['caption_sha256']==samples[0]['audit']['caption_sha256']
        page.reload();page.wait_for_function('window.reviewData&&!document.querySelector("#play").disabled')
        assert page.locator('#note').input_value()=='Automated feedback persistence check'
        page.evaluate("localStorage.removeItem('j-caption-quality-review-20261105-v1')")
        page.reload();page.wait_for_function('window.reviewData&&!document.querySelector("#play").disabled')
        assert page.locator('#note').input_value()=='' and page.locator('#verdict').input_value()==''
        page.select_option('#groupFilter','review');page.wait_for_function('!document.querySelector("#play").disabled')
        page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(WORK/'review_desktop.png'))
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200)
        overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
        page.screenshot(path=str(WORK/'review_mobile.png'),full_page=True)
        assert not overflow
        assert not errors,errors
        browser.close()
    request=urllib.request.Request(URL+samples[0]['media']+'listener.mp4',headers={'Range':'bytes=0-31'})
    with urllib.request.urlopen(request) as response:assert response.status==206 and len(response.read())==32
    report=dict(passed=True,samples=60,media_checks=checks,browser_samples_loaded=playback_ready,
        browser_pcm_hashes=pcm,playback=playback,filters_verified=True,feedback_export_and_reload=True,
        http_range_verified=True,mobile_overflow=overflow,page_errors=errors,data_sha256=sha(OUT/'data.json'))
    (WORK/'review_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,samples=60,browser_pcm_hashes=120,all_videos_loaded=True,page_errors=errors)))


if __name__=='__main__':main()
