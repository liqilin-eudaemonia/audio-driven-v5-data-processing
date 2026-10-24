"""Browser checks of independent modalities and interval-preserving integration."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
RUN=Path((ROOT/'experiments/latest_listen_split_run.txt').read_text())
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1560,'height':1050},accept_downloads=True)
    page=context.new_page();errors=[];checks=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    assert page.goto('http://127.0.0.1:8876/listen_benchmark_review/split_validation/').status==200
    page.wait_for_function('window.splitReview')
    data=page.evaluate('splitReview')
    assert len(data['samples'])==50 and data['alignment_completed']==307
    assert sum(s.get('schema_pass',0)+s.get('failed',0) for s in data['stats'].values())==614
    assert not any(s.get('pending',0) for s in data['stats'].values())
    assert 'all visible backchannel' in data['prompts']['visual_windows']
    assert 'all audible backchannel' in data['prompts']['audio_windows']
    for prompt in data['prompts'].values():assert 'each occurrence its own start time, end time' in prompt
    checks.append('921 requests completed; both exact prompts request every occurrence with start/end timestamps')
    for s in data['samples']:
        events={e['id']:e for e in s['integrated']['events']}
        assert all(0<=e['start']<e['end']<=s['duration'] and not e['human_verified'] for e in events.values())
        for link in s['integrated']['links']:
            v,a=events[link['visual_id']],events[link['audio_id']]
            assert v['modality']=='visual' and a['modality']=='audio'
            assert link['overlap_start']==max(v['start'],a['start'])
            assert link['overlap_end']==min(v['end'],a['end'])
    checks.append('all integrated candidate boundaries valid; cross-modal associations preserve independent events')
    sid=next(s['id'] for s in data['samples'] if 'S0699' in s['id'])
    page.select_option('#samples',sid);page.wait_for_function('video.readyState>=2 && audio.readyState>=2')
    assert page.locator('#cards article').count()==4
    page.locator('.event').filter(has_text='Okay · 3 个窗口支持').click()
    page.wait_for_timeout(1300)
    state=page.evaluate('({v:video.currentTime,a:audio.currentTime,paused:video.paused,muted:video.muted,loop:localizationState().loop})')
    assert state['loop']==[4.08,4.4] and not state['paused'] and state['muted'],state
    assert 4.04<=state['v']<=4.5 and abs(state['v']-state['a'])<.2,state
    checks.append('second okay loops independently synchronized WAV/video at 4.08–4.40 seconds')
    page.locator('#stopLoop').click();page.evaluate('video.pause()')
    page.select_option('#windowSelector',sid+'__w004')
    assert page.locator('#cards .window').count()==2
    checks.append('same-window selector compares visual and audio for exactly the same 4–7 second window')
    page.locator('#feedbackOnly').check();assert page.locator('#samples option').count()==4
    page.locator('#note').fill('isolated-browser-test: nod 4.2–4.8')
    with page.expect_download() as d:page.locator('#export').click()
    feedback=json.loads(Path(d.value.path()).read_text());assert feedback['schema']=='listen-split-validation-feedback-v1' and feedback['human_ground_truth'] is False and len(feedback['records'])==1
    response=context.request.get('http://127.0.0.1:8876/listen_benchmark_review/split_validation/candidate_timelines.json')
    assert response.status==200 and len(response.json()['samples'])==50
    checks.append('candidate JSON and isolated human-feedback exports preserve non-ground-truth status')
    response=context.request.get(page.evaluate('video.src'),headers={'Range':'bytes=0-1023'});assert response.status==206
    page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    assert not errors,errors
    checks.append('media range playback and 390px mobile layout pass without JavaScript errors')
    (RUN/'browser_check.json').write_text(json.dumps({'checks':checks,'errors':errors},ensure_ascii=False,indent=2))
    print(json.dumps({'checks':checks,'errors':errors},ensure_ascii=False));browser.close()
