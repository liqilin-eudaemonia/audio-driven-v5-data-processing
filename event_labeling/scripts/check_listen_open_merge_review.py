"""Check exported open labels, occurrence preservation, provenance, and playback."""
import json
from pathlib import Path
from collections import Counter
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data_previews/listen_benchmark_review/open_merge';RUN=Path((ROOT/'experiments/latest_listen_open_merge_run.txt').read_text())
d=json.loads((OUT/'review.json').read_text());labels=[json.loads(x) for x in (OUT/'labels.jsonl').read_text().splitlines()]
assert len(labels)==len(d['samples'])==50
checks=[];words=0
for s,label in zip(d['samples'],labels):
    assert s['id']==label['sample_id'] and label['unlabeled_time_policy']=='unknown' and label['training_admitted'] is False
    sources={e['id']:e for e in s['before']};represented={i for t in s['trace'].values() for i in t['source_candidate_ids']};excluded={x['source_candidate_id'] for x in s['excluded']}
    assert represented|excluded==set(sources) and not represented&excluded
    assert all(0<=e['start']<e['end']<=s['duration'] and 'type' not in e for e in s['after'])
    for source in s['before']:
        if source['source']=='asr_aligner':
            target=[e for e in s['after'] if source['id'] in s['trace'][e['event_id']]['source_candidate_ids']]
            assert len(target)==1 and target[0]['text']==source['behavior']
            assert (target[0]['start'],target[0]['end'])==(source['start'],source['end']);words+=1
assert words==42
checks.append('50 open-label records; every original candidate accounted for; all 42 ASR words retain exact original boundaries')
checks.append('no action-type enumeration; omissions remain unknown; no automatic training admission')
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox']);context=browser.new_context(viewport={'width':1500,'height':1050},accept_downloads=True);page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    assert page.goto('http://127.0.0.1:8876/listen_benchmark_review/open_merge/').status==200
    page.wait_for_function('window.openMergeReview && window.openMergeState');assert page.locator('#samples option').count()==50
    sid=next(s['id'] for s in d['samples'] if 'S0699' in s['id']);page.select_option('#samples',sid);page.wait_for_function('video.readyState>=2 && audio.readyState>=2')
    expected=next(s for s in d['samples'] if s['id']==sid);assert page.locator('#before .event').count()==len(expected['before']) and page.locator('#after .event').count()==len(expected['after'])
    target=next(e for e in expected['after'] if e.get('text')=='Okay' and e['start']>4)
    page.locator(f'#after [data-id="{target["event_id"]}"]').click();page.wait_for_timeout(1200)
    state=page.evaluate('({v:video.currentTime,a:audio.currentTime,muted:video.muted,paused:video.paused,loop:openMergeState().loop})')
    assert state['loop']==[4.08,4.4] and state['muted'] and not state['paused'],state
    assert 4.04<=state['v']<=4.5 and abs(state['v']-state['a'])<.2,state
    page.locator('#sourceOnly').check();assert page.locator('#before .event').count()==len(expected['trace'][target['event_id']]['source_candidate_ids'])
    checks.append('before/after comparison, source highlighting/filtering, and second okay synchronized interval loop pass')
    page.locator('#stop').click();page.evaluate('video.pause()');page.locator('#feedbackOnly').check();assert page.locator('#samples option').count()==4
    page.locator('#note').fill('isolated test: keep separate word occurrences')
    with page.expect_download() as dl:page.locator('#exportFeedback').click()
    feedback=json.loads(Path(dl.value.path()).read_text());assert feedback['schema']=='listen-open-merge-feedback-v1' and len(feedback['records'])==1
    for name in ['labels.json','labels.jsonl','provenance.json','summary.json']:assert context.request.get('http://127.0.0.1:8876/listen_benchmark_review/open_merge/'+name).status==200
    assert context.request.get(page.evaluate('video.src'),headers={'Range':'bytes=0-1023'}).status==206
    checks.append('feedback export, label/provenance downloads, and media byte ranges pass')
    page.screenshot(path=str(RUN/'review_desktop.png'))
    page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200);assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    assert not errors,errors;checks.append('390px mobile layout and JavaScript error checks pass')
    browser.close()
(RUN/'browser_check.json').write_text(json.dumps({'checks':checks,'errors':errors},ensure_ascii=False,indent=2));print(json.dumps({'checks':checks,'errors':errors},ensure_ascii=False))
