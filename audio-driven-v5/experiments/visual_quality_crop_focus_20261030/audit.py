#!/usr/bin/env python3
import json,hashlib,os,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
WEB=ROOT/'data_previews/visual_quality_pilot_20261030/crop_focus'
URL='http://127.0.0.1:8876/visual_quality_pilot_20261030/crop_focus/'
os.environ['FONTCONFIG_FILE']='/tmp/native-fps-fonts.conf'
os.environ['LD_LIBRARY_PATH']='/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')
from playwright.sync_api import sync_playwright

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    data=read(WEB/'data.json');assert len(data['results'])==24
    assert data['protocol']['runner_sha256']==sha(OUT/'run.py')
    prior=read(ROOT/'data_previews/visual_quality_pilot_20261030/data.json')
    assert sha(ROOT/'data_previews/visual_quality_pilot_20261030/data.json')==data['protocol']['input_sha256']
    for r,old in zip(data['results'],prior['results']):
        assert r['source']==old['source'] and r['id']==old['sample_id']
        s=r['source'];assert sha(s['source_result'])==s['source_result_sha256']
        c=r['crop']['new'];w,h=r['crop']['source_size']
        assert c['width']==c['height'] and c['x']>=0 and c['y']>=0 and c['x']+c['width']<=w and c['y']+c['height']<=h
        assert r['before']['frames']==r['after']['frames']==old['metrics']['frames']
        assert not r['production_applied'] and not r['training_admitted']
        for x in r['focus_intervals']:assert 0<=x['start']<x['end']<=s['duration_s'] and x['duration']>=.6-1e-8
        for x in r['focus_samples']:
            assert 0<=x['t']<s['duration_s']
            if x['measured']:
                a,b,c,d=x['native_patch'];assert 0<=a<c<=w and 0<=b<d<=h
                assert x['laplacian_variance']>=0 and x['gradient_energy']>=0
            else:assert x['lap_ratio'] is None and x['gradient_ratio'] is None
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(WEB/r['media'])]))
        video=next(x for x in probe['streams'] if x['codec_type']=='video')
        assert video['height']==512 and video['avg_frame_rate']=='25/1' and int(video['nb_frames'])==r['before']['frames']
        assert any(x['codec_type']=='audio' for x in probe['streams'])
        assert abs(float(probe['format']['duration'])-s['duration_s'])<.09
    errors=[]
    with sync_playwright() as p:
        b=p.chromium.launch(headless=True,args=['--no-sandbox'])
        ctx=b.new_context(viewport={'width':1500,'height':1080},accept_downloads=True)
        page=ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('typeof data!=="undefined" && data && current && document.getElementById("video").readyState>=2')
        assert page.locator('#list .entry').count()==24
        for i,r in enumerate(data['results']):
            page.locator('#list .entry').nth(i).click()
            assert r['id'] in page.locator('#source').inner_text()
            for img in ['sharp','soft']:
                src=page.locator('#'+img).get_attribute('src');assert src and ctx.request.get(URL+src).status==200
            assert ctx.request.get(URL+r['media'],headers={'Range':'bytes=0-255'}).status==206
            poster=page.locator('#video').get_attribute('poster');assert ctx.request.get(URL+poster).status==200
        page.locator('#improved').click();assert page.locator('#list .entry').count()==len(data['summary']['geometric_improvements'])
        page.locator('#focusOnly').click();assert page.locator('#list .entry').count()==len(data['summary']['focus_review_clips'])
        page.locator('#all').click();page.locator('#list .entry').nth(12).click()
        page.wait_for_function('document.getElementById("video").readyState>=2')
        page.locator('#whole').click();page.wait_for_timeout(700)
        assert page.evaluate('!document.getElementById("video").paused && document.getElementById("video").currentTime>0')
        page.locator('#cropPoint').click();page.wait_for_timeout(2500);assert page.evaluate('document.getElementById("video").paused')
        page.locator('#slow').click();assert page.evaluate('document.getElementById("video").playbackRate')==.5
        page.locator('#normal').click();page.locator('#chart').click(position={'x':120,'y':50});page.wait_for_timeout(400)
        assert page.evaluate('!document.getElementById("video").paused')
        page.evaluate('document.getElementById("video").pause()')
        page.screenshot(path=str(OUT/'review_desktop.png'),full_page=True)
        page.locator('#cropVote').select_option('similar');page.locator('#focusVote').select_option('uncertain');page.locator('#note').fill('AUTOMATED UI TEST ONLY')
        with page.expect_download() as info:page.locator('#export').click()
        info.value.save_as(OUT/'qa_export.json');assert len(read(OUT/'qa_export.json')['records'])==1
        mobile=b.new_page(viewport={'width':390,'height':844});mobile.goto(URL)
        mobile.wait_for_function('typeof data!=="undefined" && data && current')
        assert mobile.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        mobile.screenshot(path=str(OUT/'review_mobile.png'),full_page=True)
        assert not errors,errors;b.close()
    audit=dict(status='passed',clips=24,source_snapshots_unchanged=True,production_mutations=0,
               checks=['source IDs, start/end times, hashes and crop bounds','focus measurement support, unknown-frame handling and interval bounds',
                       '24 synchronized triple-view media with audio, frame counts and HTTP range seeking',
                       'browser filters, images, full/focused playback, speed, clickable clarity chart, review export and mobile layout'],
               browser_errors=errors,human_review=False)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
