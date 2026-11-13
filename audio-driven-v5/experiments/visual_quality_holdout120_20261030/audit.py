#!/usr/bin/env python3
import json,hashlib,math,os,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];WEB=ROOT/'data_previews/visual_quality_holdout120_20261030'
URL='http://127.0.0.1:8876/visual_quality_holdout120_20261030/'
os.environ['FONTCONFIG_FILE']='/tmp/native-fps-fonts.conf'
os.environ['LD_LIBRARY_PATH']='/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')
from playwright.sync_api import sync_playwright
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    d=read(WEB/'data.json');selection=read(OUT/'selection.json');rows=d['results'];assert len(rows)==120
    assert len({r['source']['pair'] for r in rows})==40
    assert not ({r['source']['pair'] for r in rows}&set(selection['excluded_tuning_sources']))
    for mode in ['naturalistic','improvised']:assert sum(r['source']['pair'].startswith(mode+'_') for r in rows)==60
    assert sha(OUT/'selection.json')==d['protocol']['selection_sha256']
    assert sha(OUT/'run.py')==d['protocol']['runner_sha256']
    for job in selection['jobs']:assert sha(job['path'])==job['sha256']
    candidates=[];source_hashes={};rendered=[]
    for r,original in zip(rows,selection['rows']):
        assert r['status']=='ok' and r['id']==original['sample_id'] and r['source']==original['source']
        s=r['source'];q=r['focus'];duration=s['duration_s']
        if s['source_result'] not in source_hashes:source_hashes[s['source_result']]=sha(s['source_result'])
        assert source_hashes[s['source_result']]==s['source_result_sha256']
        assert len(r['focus_samples'])==math.ceil(duration*5-1e-9)==q['total_samples']
        assert sum(x['measured'] for x in r['focus_samples'])==q['valid_samples']
        assert all(x['lap_ratio'] is None and x['gradient_ratio'] is None for x in r['focus_samples'] if not x['measured'])
        assert abs(sum(x['duration'] for x in q['intervals'])/duration-q['low_fraction'])<1e-8
        assert q['candidate_exclusion']==(q['longest_low_s']>=2-1e-9 and q['low_fraction']>=.25-1e-9)
        if q['candidate_exclusion']:candidates.append(r['ordinal'])
        c=r['crop'];a,b=c['old'],c['new'];w,h=s['video_transform']['source_size']
        assert b['width']<=math.floor(a['width']*1.05) and b['width']>=a['width']
        assert 0<=b['x']<=w-b['width'] and 0<=b['y']<=h-b['height'] and b['height']==b['width']
        if not c['changed']:assert a==b
        if r['review_media']:
            m=r['review_media'];rendered.append(r)
            assert sha(WEB/m['media'])==m['sha256']
            assert m['frames']==math.ceil(duration*25-1e-9)
            # Direct 5 Hz measurements use exactly the same original frames as review playback.
            for i,x in enumerate(r['focus_samples']):assert abs(x['source_time']-m['source_frame_times_s'][i*5])<1e-6
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(WEB/m['media'])]))
            v=next(x for x in probe['streams'] if x['codec_type']=='video')
            assert int(v['nb_frames'])==m['frames'] and v['avg_frame_rate']=='25/1' and v['height']==512
            assert any(x['codec_type']=='audio' for x in probe['streams'])
            assert abs(float(probe['format']['duration'])-duration)<.09
    assert candidates==d['summary']['exclusion_candidates']
    assert all(r['review_media'] for r in rows if r['focus']['candidate_exclusion'])
    errors=[]
    with sync_playwright() as p:
        b=p.chromium.launch(headless=True,args=['--no-sandbox']);ctx=b.new_context(viewport={'width':1500,'height':1080},accept_downloads=True)
        page=ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('typeof data!=="undefined" && data && current && document.getElementById("video").readyState>=2')
        assert page.locator('#list .entry').count()==len(rendered)
        assert page.locator('#overview tr').count()==121
        for i,r in enumerate([next(r for r in rows if r['id']==sid) for sid in d['review_order']]):
            page.locator('#list .entry').nth(i).click();assert r['id'] in page.locator('#source').inner_text()
            m=r['review_media'];assert ctx.request.get(URL+m['media'],headers={'Range':'bytes=0-255'}).status==206
            for key in ['poster','sharp_face','soft_face']:
                if m[key]:assert ctx.request.get(URL+m[key]).status==200
        page.locator('#candidate').click();assert page.locator('#list .entry').count()==len(candidates)
        page.locator('#incomplete').click();assert page.locator('#list .entry').count()==min(4,len(d['summary']['measurement_incomplete']))
        page.locator('#all').click();page.locator('#list .entry').first.click()
        page.wait_for_function('document.getElementById("video").readyState>=2')
        page.locator('#whole').click();page.wait_for_timeout(700);assert page.evaluate('!document.getElementById("video").paused && document.getElementById("video").currentTime>0')
        page.locator('#focus').click();page.wait_for_timeout(2500);assert page.evaluate('document.getElementById("video").paused')
        page.locator('#slow').click();assert page.evaluate('document.getElementById("video").playbackRate')==.5
        page.locator('#normal').click();page.locator('#chart').click(position={'x':120,'y':50});page.wait_for_timeout(400)
        assert page.evaluate('!document.getElementById("video").paused');page.evaluate('document.getElementById("video").pause()')
        page.screenshot(path=str(OUT/'review_desktop.png'),full_page=True)
        page.locator('#qualityVote').select_option('uncertain');page.locator('#note').fill('AUTOMATED UI CHECK ONLY')
        with page.expect_download() as info:page.locator('#export').click()
        info.value.save_as(OUT/'qa_export.json');assert len(read(OUT/'qa_export.json')['records'])==1
        mobile=b.new_page(viewport={'width':390,'height':844});mobile.goto(URL);mobile.wait_for_function('typeof data!=="undefined" && data && current')
        assert mobile.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        mobile.screenshot(path=str(OUT/'review_mobile.png'),full_page=True)
        assert not errors,errors;b.close()
    report={'status':'passed','holdout_clips':120,'source_groups':40,'rendered_reviews':len(rendered),'source_hashes_unchanged':True,
            'checks':['selection excludes tuning sources; 60 naturalistic / 60 improvised','frozen combined threshold and missing measurement handling',
                      'crop growth bounded at 5%, valid native coordinates','5 fps metrics and 25 fps previews use identical original frames',
                      'all review media streams, audio, timestamps, hashes and HTTP seeking','browser filters, all 120 rows, playback, focus jump, chart, export, mobile layout'],
            'human_quality_validated':False,'production_mutations':0,'browser_errors':errors}
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
