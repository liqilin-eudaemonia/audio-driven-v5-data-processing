"""All43 DyStream media and172 L2L motion traces in the reused viewer."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/p2_compare_vico43_20261106'
PAGE=ROOT/'data_previews/listen_benchmark_review/p2_vico43_compare_20261106'
URL='http://127.0.0.1:8876/listen_benchmark_review/p2_vico43_compare_20261106/'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    d=json.loads((PAGE/'data.json').read_text());assert len(d['cases'])==43
    assert d['source_results_sha256']==sha(WORK/'dystream/MOTION_RESULTS.json')
    assert d['source_image_sha256']==sha(WORK/'dystream/IMAGE_RESULTS.json')
    assert d['l2l_results_sha256']==sha(WORK/'learning2listen/RESULTS.json')
    traces=0
    for c in d['cases']:
        uid=c['id'];assert c['generated_wav'] is None and c['audio']['generated_wav_sha256'] is None
        assert sha(PAGE/c['scoring_video'])==c['source_sha256']['prediction']
        assert (PAGE/c['native_video']).resolve()==(WORK/'dystream/generated'/uid/'generated.mp4').resolve()
        for name,r in c['l2l'].items():
            arr=np.load(PAGE/r['download'])['motion'].astype(float);assert arr.shape==(c['frames'],56)
            delta=arr-arr[:1]
            for key,sl in [('expression',slice(0,50)),('head',slice(50,53)),('jaw',slice(53,56))]:
                expected=np.sqrt(np.square(delta[:,sl]).mean(1)).round(7)
                assert np.array_equal(expected,np.array(r['curves'][key]))
            traces+=1
    assert traces==172
    os.environ['LD_LIBRARY_PATH']='/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')
    from playwright.sync_api import sync_playwright
    errors=[];cases=[]
    with sync_playwright() as p:
        b=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox','--disable-gpu'])
        page=b.new_page(viewport={'width':1500,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('window.reviewData&&reviewDebug().loaded',timeout=60000)
        assert page.locator('#list button').count()==43 and page.locator('#l2lModel option').count()==4
        assert page.locator('#audioMode option').count()==2
        page.click('#play');page.wait_for_timeout(800);playback=page.evaluate('reviewDebug()')
        assert playback['sources']==1 and playback['context']=='running' and min(playback['times'].values())>.2
        assert max(playback['times'].values())-min(playback['times'].values())<.15
        page.select_option('#audioMode','silent');assert page.evaluate('reviewDebug().sources')==0
        page.select_option('#audioMode','speaker');assert page.evaluate('reviewDebug().sources')==1
        page.select_option('#speed','0.5');assert page.evaluate('sources.every(s=>s.playbackRate.value===.5)')
        page.click('#pause');page.select_option('#speed','1')
        for c in d['cases']:
            page.evaluate('id=>show(reviewData.cases.findIndex(c=>c.id===id)).catch(fail)',c['id'])
            page.select_option('#resolution','native');page.wait_for_function('reviewDebug().loaded&&reviewDebug().sizes.prediction[0]===512',timeout=60000)
            page.select_option('#resolution','scoring');page.wait_for_function('reviewDebug().loaded&&reviewDebug().sizes.prediction[0]===256',timeout=60000)
            assert page.evaluate('Object.values(videos).every(v=>v.readyState>=2&&!v.error)')
            for name in c['l2l']:
                page.select_option('#l2lModel',name)
                assert name in page.locator('#l2lMetrics').inner_text()
            assert c['id'] in page.locator('#jointLink').get_attribute('href')
            cases.append(c['id']);print('verified',len(cases),c['id'],flush=True)
        page.evaluate('show(0).catch(fail)');page.wait_for_function('reviewDebug().loaded')
        canvas=page.locator('[data-motion="head"]');canvas.click(position={'x':100,'y':75})
        assert page.evaluate('reviewDebug().times.gt')>.3
        page.select_option('#resolution','native');page.wait_for_function('reviewDebug().loaded&&reviewDebug().sizes.prediction[0]===512')
        page.locator('#overview summary').click();page.click('#reset');page.screenshot(path=str(WORK/'visualization_desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(150)
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
        page.screenshot(path=str(WORK/'visualization_mobile.png'),full_page=True)
        assert not errors,errors;b.close()
    request=urllib.request.Request(URL+d['cases'][0]['native_video'],headers={'Range':'bytes=0-31'})
    with urllib.request.urlopen(request,timeout=10) as f:assert f.status==206 and len(f.read())==32
    report=dict(passed=True,cases=len(cases),l2l_series_sets=traces,both_video_resolutions=True,all_four_motion_selectors=True,
        synchronized_playback=playback,seek_speed_audio_and_curve_seek=True,no_fabricated_listener_audio=True,
        no_mobile_overflow=True,range_requests=True,page_errors=errors,data_sha256=sha(PAGE/'data.json'),html_sha256=sha(PAGE/'index.html'))
    (WORK/'VISUALIZATION_BROWSER_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS43 video pairs,172 motion trace sets, both resolutions and playback')


if __name__=='__main__':main()
