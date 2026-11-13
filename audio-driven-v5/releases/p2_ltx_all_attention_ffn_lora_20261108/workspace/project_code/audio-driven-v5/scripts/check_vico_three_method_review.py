"""Verify source scores and four-player synchronization on all43 pairs."""
import csv
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/p2_compare_vico43_20261106'
ART=WORK/'three_method_review'
PAGE=ROOT/'data_previews/listen_benchmark_review/p2_vico43_compare_20261106'
URL='http://127.0.0.1:8876/listen_benchmark_review/p2_vico43_compare_20261106/'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def local(url):return ROOT/'data_previews'/url.lstrip('/') if url.startswith('/') else PAGE/url


def main():
    d=read(PAGE/'data.json');assert d['method_families']==3 and d['video_methods']==2 and len(d['cases'])==43
    for path,h in d['source_sha256'].items():assert sha(Path(path))==h
    assert d['summaries']['joint']['image']==read(ROOT/'experiments/p2_vico256_combined8000_20261106/RESULTS.json')['metrics']
    assert d['summaries']['dystream']['image']==read(WORK/'dystream/IMAGE_RESULTS.json')['metrics']
    assert d['summaries']['learning2listen']['image'] is None and not d['summaries']['learning2listen']['video']
    for c in d['cases']:
        for m in ['joint','dystream']:
            assert local(c[m]['native_video']).is_file()
            assert sha(local(c[m]['scoring_video']))==c[m]['source_prediction_sha256']
        assert c['dystream']['generated_wav'] is None
        for n,r in c['l2l'].items():assert local(r['download']).is_file()
    # Raw CSV values times the declared display multiplier must reproduce the
    # corresponding displayed source values exactly (including FD/FID_delta).
    csv_rows=list(csv.DictReader((PAGE/'all_methods_metrics.csv').open()))
    for r in csv_rows:
        raw=float(r['raw_value']);scale=float(r['display_multiplier']);name=r['method'];key=r['metric'];group=r['group']
        if name=='learning2listen':value=d['summaries'][name]['variants'][r['variant']]['metrics'][key][group]
        elif r['space']=='image':value=d['summaries'][name]['image'][key]
        else:value=d['summaries'][name]['motion'][key][group]
        assert abs(raw*scale-value)<1e-12
    os.environ['LD_LIBRARY_PATH']='/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')
    from playwright.sync_api import sync_playwright
    errors=[];checked=[]
    with sync_playwright() as p:
        b=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox','--disable-gpu'])
        page=b.new_page(viewport={'width':1800,'height':1100},accept_downloads=True);page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('window.reviewData&&reviewDebug().loaded',timeout=60000)
        assert page.locator('video').count()==4 and page.locator('#list button').count()==43
        assert page.locator('#imageSummary tbody tr').count()==6
        for name in ['joint','dystream']:
            for key,v in d['summaries'][name]['image'].items():
                assert page.locator(f'td[data-method="{name}"][data-metric="{key}"]').inner_text()==f'{v:.5f}'
        assert page.locator('#imageSummary td.na').count()==20
        assert '暂无' in page.locator('#l2lVideoStatus').inner_text()
        page.click('#play');page.wait_for_timeout(1000);playback=page.evaluate('reviewDebug()')
        assert playback['context']=='running' and playback['sources']==1
        assert min(playback['times'].values())>.3
        assert max(playback['times'].values())-min(playback['times'].values())<.16
        for mode,n in [('joint',1),('both',2),('silent',0),('speaker',1)]:
            page.select_option('#audioMode',mode);assert page.evaluate('reviewDebug().sources')==n
        page.select_option('#speed','0.5');assert page.evaluate('sources.every(x=>x.playbackRate.value===.5)')
        page.locator('#seek').evaluate("el=>{el.value=1.25;el.dispatchEvent(new Event('input'))}")
        page.wait_for_timeout(180);assert min(page.evaluate('Object.values(reviewDebug().times)'))>1.2
        page.click('#pause');page.select_option('#speed','1')
        for c in d['cases']:
            page.evaluate('id=>show(reviewData.cases.findIndex(c=>c.id===id)).catch(fail)',c['id'])
            page.select_option('#resolution','native');page.wait_for_function('reviewDebug().loaded&&reviewDebug().sizes.joint[0]===512&&reviewDebug().sizes.dystream[0]===512',timeout=60000)
            assert page.evaluate('Object.values(videos).every(v=>v.readyState>=2&&!v.error)')
            page.select_option('#resolution','scoring');page.wait_for_function('reviewDebug().loaded&&Object.values(reviewDebug().sizes).every(x=>x[0]===256)',timeout=60000)
            for name in c['l2l']:
                page.select_option('#l2lModel',name);assert name in page.locator('#l2lMetrics').inner_text()
            if c['duration']==1:assert page.locator('#caseMetrics').inner_text().count('长度不足')==4
            if c['duration']==3:assert page.locator('#caseMetrics').inner_text().count('长度不足')==2
            checked.append(c['id']);print('checked',len(checked),c['id'],flush=True)
        page.evaluate('show(0).catch(fail)');page.wait_for_function('reviewDebug().loaded')
        page.locator('[data-motion="head"]').click(position={'x':110,'y':75});assert min(page.evaluate('Object.values(reviewDebug().times)'))>.3
        page.click('#play');page.wait_for_timeout(600);assert min(page.evaluate('Object.values(reviewDebug().times)'))>.5;page.click('#pause')
        page.select_option('#resolution','native');page.wait_for_function('reviewDebug().loaded&&reviewDebug().sizes.joint[0]===512')
        page.click('#reset');page.evaluate('scrollTo(0,0)');page.screenshot(path=str(ART/'desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(150)
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
        page.screenshot(path=str(ART/'mobile.png'),full_page=True)
        page.reload();page.wait_for_function('reviewDebug().loaded',timeout=60000)
        assert page.evaluate('reviewDebug().sample')==d['cases'][0]['id']
        assert not errors,errors;b.close()
    for m in ['joint','dystream']:
        url=d['cases'][0][m]['native_video'];url='http://127.0.0.1:8876'+url if url.startswith('/') else URL+url
        request=urllib.request.Request(url,headers={'Range':'bytes=0-31'})
        with urllib.request.urlopen(request,timeout=10) as f:assert f.status==206 and len(f.read())==32
    report=dict(passed=True,method_families=3,actual_video_methods=2,cases=len(checked),four_visible_video_players=True,
        actual_synchronized_playback=playback,both_resolutions_for_both_methods=True,audio_modes_seek_speed=True,
        l2l_variants=4,l2l_video_unavailable_explicit=True,l2l_image_metrics_not_fabricated=True,
        all_summary_csv_values_verified=len(csv_rows),mobile_no_overflow=True,range_requests=True,page_errors=errors,
        data_sha256=sha(PAGE/'data.json'),html_sha256=sha(PAGE/'index.html'))
    (ART/'BROWSER_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: three methods,43 pairs, four-player playback and all score tables')


if __name__=='__main__':main()
