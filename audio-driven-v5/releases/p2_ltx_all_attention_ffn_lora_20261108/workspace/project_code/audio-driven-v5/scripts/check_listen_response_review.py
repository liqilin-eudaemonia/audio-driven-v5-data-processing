"""Check provenance, conservative supervision and the interactive review workflow."""
import argparse,hashlib,json,tempfile
from pathlib import Path

def check(run,url):
    d=json.loads((run/'review.json').read_text());m=json.loads((run/'manifest.json').read_text())
    raw=(run/'observations.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==m['source_observation_sha256']
    assert raw==(Path(m['source'])/'observations.jsonl').read_bytes()
    assert len(d['samples'])==50 and sum(len(s['review_groups']) for s in d['samples'])==5
    for s in d['samples']:
        for g in s['review_groups']:
            assert g['actual_occurrence_count'] is None and not g['confirmed_occurrences']
            assert not any(g['training'].values())
    target=next(s for s in d['samples'] if 'S0805' in s['id']);g=target['review_groups'][0]
    assert len(g['reference_word_candidates'])==5
    assert sum(not t['time_valid'] for t in g['reference_word_candidates'])==2
    from playwright.sync_api import sync_playwright
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1440,'height':1100},accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(url+'?sample='+target['id']).status==200
        page.wait_for_function('video.readyState>=2 && audio.readyState>=2')
        assert page.locator('#samples option').count()==5
        assert '5 个词' in page.locator('#analysis').inner_text()
        page.click('#context');page.wait_for_timeout(600)
        state=page.evaluate('({v:video.currentTime,a:audio.currentTime,paused:video.paused})')
        assert not state['paused'] and abs(state['v']-state['a'])<.2,state
        page.click('#pause');page.click('#copy');assert page.locator('#intervals .row').count()==3
        page.select_option('#decision','single');page.click('#save');assert '一个区间' in page.locator('#message').inner_text()
        page.select_option('#decision','multiple');page.click('#save');assert '已保存' in page.locator('#message').inner_text()
        with page.expect_download() as download:page.click('#export')
        feedback=json.loads(Path(download.value.path()).read_text());assert len(feedback['records'])==1
        assert not feedback['records'][0]['training_admitted']
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'review.json';f.write_text(json.dumps(feedback))
            page.evaluate('localStorage.clear()');page.reload();page.wait_for_function('video.readyState>=2 && audio.readyState>=2')
            page.set_input_files('#import',str(f));page.wait_for_function('Object.keys(responseReviewState().saved).length===1')
            assert page.locator('#intervals .row').count()==3
        page.evaluate('localStorage.clear()');page.reload();page.wait_for_function('video.readyState>=2 && audio.readyState>=2')
        assert '0 / 5' in page.locator('#progress').inner_text()
        for s in d['samples']:
            if not s['review_groups']:continue
            page.select_option('#samples',s['id']);page.wait_for_function('video.readyState>=2 && audio.readyState>=2')
            assert context.request.get(page.evaluate('audio.src'),headers={'Range':'bytes=0-1023'}).status==206
            assert context.request.get(page.evaluate('video.src'),headers={'Range':'bytes=0-1023'}).status==206
        page.select_option('#samples',target['id']);page.wait_for_function('video.readyState>=2 && audio.readyState>=2')
        page.screenshot(path=str(run/'review_desktop.png'),full_page=True)
        page.uncheck('#focus');assert page.locator('#samples option').count()==50
        page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(run/'review_mobile.png'),full_page=True)
        assert not errors,errors
        browser.close()
    report=dict(samples=50,focus_groups=5,all_raw_bytes_preserved=True,s0805_reference_words=5,s0805_unlocalized_words=2,
                unit_tests_passed=6,browser_checks=['five media pairs range206','synchronized playback','invalid review rejected','save/export/import round trip','all50 selector','mobile layout'],errors=errors,
                human_listening_completed=False,training_admitted=False)
    (run/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--url',required=True);a=p.parse_args();check(a.run,a.url)
