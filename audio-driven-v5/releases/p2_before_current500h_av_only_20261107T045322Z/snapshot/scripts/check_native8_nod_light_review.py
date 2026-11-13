#!/usr/bin/env python3
import json
import os
from pathlib import Path
from prepare_native8_nod_semantic_pilot import ROOT,OUT
from listen_visual_native8_nod import save
libs=Path('/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu')
os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
from playwright.sync_api import sync_playwright

URL='http://127.0.0.1:8876/listen_benchmark_review/prompt_versions_20261029/glare_nodding/semantic_light/'

def main():
    path=ROOT/'data_previews/listen_benchmark_review/prompt_versions_20261029/glare_nodding/semantic_light/data.json'
    data=json.loads(path.read_text());assert len(data['samples'])==15 and data['llm_executed'] is False
    assert data['raw_count']==365 and data['light_count']==359
    assert sum(s['stats']['nod_candidates'] for s in data['samples'])==16
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox']);ctx=browser.new_context(viewport={'width':1500,'height':1000},accept_downloads=True);page=ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('window.lightReviewData && reviewAudioState().ready && video.readyState>=2')
        assert page.locator('#samples option').count()==15
        for sample in data['samples']:
            page.locator('#samples').select_option(sample['id'])
            assert page.locator('#before .event').count()==sample['stats']['original']
            assert page.locator('#after .event').count()==sample['stats']['light_dedup']
            for e in sample['record']['events']:assert e['description'] in page.locator('#after').inner_text()
            for path in [sample['video'],sample['audio']['original'],sample['audio']['speaker']]:assert ctx.request.get(URL+path,headers={'Range':'bytes=0-255'}).status==206
        duplicated=next(s for s in data['samples'] if s['stats']['exact_duplicates'])
        page.locator('#samples').select_option(duplicated['id']);page.locator('#filter').select_option('duplicates')
        assert page.locator('#before .event').count()-page.locator('#after .event').count()==duplicated['stats']['exact_duplicates']
        page.locator('#filter').select_option('glare_nod_recall');assert page.locator('#after .event').count()==duplicated['stats']['nod_candidates']
        page.locator('#filter').select_option('all');page.wait_for_function('reviewAudioState().ready && video.readyState>=2')
        page.locator('#soundPlay').click();page.wait_for_timeout(700);assert page.evaluate('!video.paused && !reviewAudioState().listenerPaused && Math.abs(video.currentTime-reviewAudioState().listenerTime)<.2')
        page.locator('#speed').select_option('0.5');page.evaluate('video.currentTime=1.2');page.wait_for_timeout(500);assert page.evaluate('Math.abs(video.currentTime-reviewAudioState().listenerTime)<.2')
        page.evaluate('video.pause()')
        with page.expect_download() as dl:page.locator('#download').click()
        saved=json.loads(Path(dl.value.path()).read_text());assert saved['llm_executed'] is False and saved['events']==duplicated['record']['events']
        page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors;browser.close()
    report={'passed':True,'samples':15,'raw':365,'light':359,'exact_duplicates_collapsed':6,'nod_candidates':16,
            'llm_results_used':False,'checks':['all original descriptions and counts shown','duplicate grouping and source filtering','all media HTTP206','audio playback/seek/slow speed sync','JSON download preserves records','mobile overflow and JS errors']}
    save(OUT/'audit_light_review.json',report);print(json.dumps(report))

if __name__=='__main__':main()
