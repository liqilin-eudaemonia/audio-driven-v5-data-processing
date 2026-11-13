"""Media provenance and real Chromium checks for the 24-sample final-crop review."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_previews/listen_benchmark_review/j_face224_training_20261101'
WORK=ROOT/'experiments/listen_j_face224_training_review_20261101'
URL='http://127.0.0.1:8876/listen_benchmark_review/j_face224_training_20261101/'


def main():
    data=json.loads((OUT/'data.json').read_text())
    assert len(data['samples'])==24
    checks=[]
    for s in data['samples']:
        assert s['split'] in ('train','dev') and s['face_p10']>=224
        dest=OUT/s['media'];meta=json.loads((dest/'crop_provenance.json').read_text())
        assert meta['final']['actual_crop']==s['final_crop']
        q=json.loads(Path(s['source_paths']['measurement']).read_text())
        assert q['crop']['new']==s['final_crop']
        assert (dest/'caption_original.json').read_bytes()==Path(s['source_paths']['caption']).read_bytes()
        assert json.loads((dest/'caption_original.json').read_text())['caption']==s['caption']
        for role in ('listener','speaker'):
            a=s[role+'_audio'];raw=(dest/a['raw']).read_bytes()
            assert len(raw)==a['samples']*4
            assert hashlib.sha256(raw).hexdigest()==a['raw_sha256']
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(dest/'listener.mp4')]))
        v=next(x for x in probe['streams'] if x['codec_type']=='video')
        a=next(x for x in probe['streams'] if x['codec_type']=='audio')
        assert (v['width'],v['height'],v['avg_frame_rate'])==(512,512,'25/1')
        assert int(v['nb_frames'])==s['video']['frames']
        assert abs(float(a['duration'])-s['duration'])<.003
        assert abs(float(v['duration'])-s['duration'])<=1/25+.001
        checks.append(dict(id=s['id'],passed=True))
    libs=Path('/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu')
    if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
    from playwright.sync_api import sync_playwright
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',
            headless=True,args=['--no-sandbox','--disable-gpu'])
        context=browser.new_context(viewport={'width':1450,'height':1050},accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('window.reviewData && !document.querySelector("#play").disabled && document.querySelector("#video").readyState>=2')
        assert page.locator('#samples option').count()==24
        # Confirm browser fetches exact raw PCM, not a second processing recipe.
        pcm=page.evaluate('''async()=>{const out=[];for(const s of reviewData.samples){for(const role of ['listener','speaker']){const a=s[role+'_audio'],r=await fetch(s.media+a.raw);if(!r.ok)throw Error('PCM HTTP '+r.status);const b=await r.arrayBuffer();const h=[...new Uint8Array(await crypto.subtle.digest('SHA-256',b))].map(x=>x.toString(16).padStart(2,'0')).join('');out.push({id:s.id,role,pass:h===a.raw_sha256&&b.byteLength===a.samples*4})}}return out}''')
        assert len(pcm)==48 and all(x['pass'] for x in pcm)
        page.click('[data-mode="speaker"]');page.click('#play');page.wait_for_timeout(1100)
        first=page.evaluate('({audio:reviewAudio(),t:document.querySelector("#video").currentTime,w:document.querySelector("#video").videoWidth})')
        assert first['audio']['context']=='running' and first['audio']['playing']==1 and first['t']>.3 and first['w']==512
        page.click('[data-mode="both"]');assert page.evaluate('reviewAudio().playing')==2
        page.locator('#video').evaluate('(v)=>v.currentTime=1.5');page.wait_for_timeout(200)
        assert page.evaluate('reviewAudio().playing')==2
        page.click('#pause');page.wait_for_function('reviewAudio().playing===0')
        page.select_option('#filter','voice');page.wait_for_function('!document.querySelector("#play").disabled')
        voices=page.locator('#samples option').count();assert voices==sum(not s['listener_audio']['full_silence'] for s in data['samples'])
        page.click('[data-mode="listener"]');page.click('#play');page.wait_for_timeout(500)
        assert page.evaluate('reviewAudio().playing')==1
        page.click('#pause')
        page.locator('.issue').first.check();page.fill('#note','Automated persistence check')
        with page.expect_download() as download:page.click('#export')
        feedback=json.loads(Path(download.value.path()).read_text());assert any(r['note']=='Automated persistence check' for r in feedback['records'])
        page.reload();page.wait_for_function('window.reviewData && !document.querySelector("#play").disabled')
        assert page.locator('#note').input_value()=='Automated persistence check'
        page.evaluate("localStorage.removeItem('j-face224-final-review-20261101')")
        page.locator('.issue').first.uncheck();page.fill('#note','');page.evaluate("localStorage.removeItem('j-face224-final-review-20261101')")
        page.evaluate('window.scrollTo(0,0)')
        page.evaluate('document.fonts.ready')
        page.screenshot(path=str(WORK/'review_desktop.png'))
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200)
        overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
        page.screenshot(path=str(WORK/'review_mobile.png'),full_page=True)
        assert not overflow,'Mobile horizontal overflow'
        assert not errors,errors
        browser.close()
    result=dict(passed=True,media_checks=checks,raw_pcm_browser_hashes=pcm,
        playback=first,voice_samples=voices,both_audio_tracks_switch_and_seek=True,
        feedback_export_and_reload=True,mobile_overflow=overflow,page_errors=errors,
        training_manifest_or_cache_verified=False,
        note='Verified J source-material replay and existing geometry convention, not an already admitted/cached training batch.')
    (WORK/'review_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,samples=24,pcm_hashes=48,voice_samples=voices,page_errors=errors)))


if __name__=='__main__':main()
