"""Verify real preview media, scored waveforms and Chromium playback controls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
PAGE=ROOT/'data_previews/listen_benchmark_review/p2_seamless200_20261105'
OUT=ROOT/'experiments/p2_seamless200_analysis_20261105'
RUN=ROOT/'experiments/p2_seamless200_tenway_24gpu_20261105'
URL='http://127.0.0.1:8876/listen_benchmark_review/p2_seamless200_20261105/'


def main():
    d=json.loads((PAGE/'data.json').read_text());assert len(d['cases'])==16 and len(d['models'])==10
    checks=[]
    for case in d['cases']:
        raw=[case['gt_pcm'],case['speaker_pcm'],*[x['pcm'] for x in case['models'].values()]]
        for item in raw:
            b=(PAGE/item['url']).read_bytes()
            assert len(b)==item['samples']*4 and hashlib.sha256(b).hexdigest()==item['sha256']
        assert case['media_audit']['gt_audio_sha256']==case['gt_pcm']['sha256']
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(PAGE/case['gt_video'])]))['streams'][0]
        assert (info['width'],info['height'],info['avg_frame_rate'])==(512,512,'25/1')
        for model in d['models']:
            key=model['key'];entry=case['models'][key];p=PAGE/entry['video']
            assert p.is_symlink() and p.resolve()==(RUN/'generated'/key/'seamless'/case['id']/'generated.mp4').resolve()
            meta=json.loads((RUN/'generated'/key/'seamless'/case['id']/'metadata.json').read_text())
            assert meta['arm']==key and meta['checkpoint']==model['checkpoint'] and meta['step']==model['step'] and meta['seed']==137
        checks.append(dict(id=case['id'],media_and_PCM_pass=True))
    libs=Path('/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu')
    if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
    fonts=Path('/tmp/native-fps-browser-libs/extracted/usr/share/fonts/opentype/noto')
    if fonts.exists():
        fc=OUT/'fontconfig.xml'
        fc.write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><dir>/usr/share/fonts</dir><dir>{fonts}</dir><cachedir>/tmp/seamless-review-fontcache</cachedir></fontconfig>')
        os.environ['FONTCONFIG_FILE']=str(fc)
    from playwright.sync_api import sync_playwright
    errors=[];playback={}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox','--disable-gpu'])
        context=browser.new_context(viewport={'width':1580,'height':1100},accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('window.reviewData && reviewDebug().loaded && !document.querySelector("#play").disabled',timeout=60000)
        assert page.locator('#samples option').count()==16 and page.locator('#modelA option').count()==10
        assert page.locator('#summaryTable tbody tr').count()==10
        page.click('[data-mode="both"]');page.click('[data-focus="a"]');page.click('#play');page.wait_for_timeout(900)
        playback=page.evaluate('reviewDebug()')
        assert playback['context']=='running' and playback['sources']==2 and playback['times']['gt']>.3
        assert max(playback['times'].values())-min(playback['times'].values())<.15
        page.locator('#seek').evaluate('(el)=>{el.value=1.2;el.dispatchEvent(new Event("input"))}');page.wait_for_timeout(200)
        assert page.evaluate('reviewDebug().times.gt')>1.1
        page.click('[data-focus="b"]');assert page.evaluate('reviewDebug().focus')=='b'
        page.select_option('#speed','0.5');assert page.evaluate('sources.every(s=>s.playbackRate.value===.5)')
        page.click('#pause');assert page.evaluate('reviewDebug().sources')==0
        page.select_option('#speed','1');page.click('[data-mode="listener"]')
        # Actual codec loading and PCM verification for each displayed model of
        # every disclosed case, not just the initial screenshot.
        for case in d['cases']:
            page.select_option('#samples',case['id'])
            page.wait_for_function('(id)=>reviewDebug().sample===id&&reviewDebug().loaded',arg=case['id'],timeout=60000)
            assert page.evaluate('Object.values(videos).every(v=>v.videoWidth===512&&v.videoHeight===512)')
        for model in d['models']:
            page.select_option('#modelB',model['key'])
            page.wait_for_function('reviewDebug().loaded',timeout=60000)
        # Exact browser-fetched PCM on every panel, including models not loaded
        # into the two currently visible comparison slots.
        hashes=page.evaluate('''async()=>{let n=0;for(const c of reviewData.cases){for(const a of [c.gt_pcm,c.speaker_pcm,...Object.values(c.models).map(m=>m.pcm)]){const r=await fetch(a.url);if(!r.ok)throw Error('PCM '+r.status);const b=await r.arrayBuffer();const h=[...new Uint8Array(await crypto.subtle.digest('SHA-256',b))].map(x=>x.toString(16).padStart(2,'0')).join('');if(h!==a.sha256||b.byteLength!==a.samples*4)throw Error('PCM identity mismatch');n++;}}return n;}''')
        assert hashes==192
        page.select_option('#samples',d['cases'][4]['id']);page.wait_for_function('reviewDebug().loaded')
        page.get_by_text('保存本条人工复核意见（可选）',exact=True).click()
        page.select_option('#decision','b');page.locator('#note').fill('QA audio and video check')
        with page.expect_download() as event:page.click('#exportReview')
        review=json.loads(Path(event.value.path()).read_text());assert any(x['note']=='QA audio and video check' for x in review['records'])
        page.evaluate("localStorage.removeItem('seamless200-review-20261105')")
        page.get_by_text('保存本条人工复核意见（可选）',exact=True).click()
        page.click('#reset');page.evaluate('seek(3.4)');page.wait_for_timeout(250)
        page.evaluate('scrollTo(0,0)');page.screenshot(path=str(OUT/'desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200)
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth'),'Mobile overflow'
        page.screenshot(path=str(OUT/'mobile.png'),full_page=True)
        assert not errors,errors
        browser.close()
    # Range seeking through the existing Cursor-forwarded server.
    request=urllib.request.Request(URL+d['cases'][0]['gt_video'],headers={'Range':'bytes=0-31'})
    with urllib.request.urlopen(request,timeout=10) as f:assert f.status==206 and len(f.read())==32
    result=dict(success=True,cases=16,systems=10,video_and_audio_source_checks=checks,
                browser_PCM_SHA_checks=192,synchronized_playback=playback,
                source_switch_seek_speed=True,all_case_and_model_selectors=True,
                human_review_export=True,range_requests=True,page_errors=errors,
                no_mobile_overflow=True)
    (OUT/'BROWSER_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(success=True,cases=16,systems=10,PCM_hashes=192,page_errors=errors)))


if __name__=='__main__':main()
