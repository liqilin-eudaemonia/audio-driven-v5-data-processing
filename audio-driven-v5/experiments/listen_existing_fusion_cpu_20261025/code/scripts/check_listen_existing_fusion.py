"""Evaluate automatic outputs after generation; feedback expectations never enter fusion."""
import argparse,hashlib,json
from pathlib import Path

def check(run,url):
 data=json.loads((run/'review.json').read_text());samples=data['samples'];manifest=json.loads((run/'manifest.json').read_text());assert manifest['human_feedback_input'] is None and manifest['gpu_used'] is False
 assert hashlib.sha256((run/'observations.jsonl').read_bytes()).hexdigest()==manifest['raw_observations_sha256']
 assert (run/'observations.jsonl').read_bytes()==(Path(manifest['source'])/'observations.jsonl').read_bytes()
 for s in samples:
  assert not s['automatic_fusion']['human_feedback_read'] and not s['automatic_fusion']['media_read'] and not s['automatic_fusion']['gpu_used']
  assigned={x['observation_id'] for x in s['evidence_links']+s['unresolved_observations']};assigned.update(i for e in s['events'] for i in e['source_observation_ids']);assert assigned=={e['event_id'] for e in s['original_events']}
  for e in s['events']:
   assert 0<=e['start']<e['end']<=s['duration_s'];assert not any(e['training'].values())
   if 'timing_evidence' in e:
    c=e['timing_evidence'];assert (e['start'],e['end'])==(c['start'],c['end']);assert any((x['start'],x['end'])==(e['start'],e['end']) for x in c['members'])
 def get(code):return next(s for s in samples if code in s['id'])
 def spoken(code):return [e for e in get(code)['events'] if e.get('text')]
 y=spoken('S0805');assert [e['text'].lower() for e in y]==['right','right']+['yeah']*5
 assert all(a['end']<=b['start']+1e-7 for a,b in zip(y,y[1:]))
 assert y[2]['start']<3.6 and y[-1]['end']<5.1
 okay=[e for e in spoken('S0699') if e['text'].lower()=='okay'];assert len(okay)==2
 assert abs(okay[0]['start']-1.78)<=.2 and abs(okay[1]['start']-4.08)<=.2 and okay[0]['end']<okay[1]['start']
 laugh=[e for e in get('S2043')['events'] if e['modality']=='audio'];assert len(laugh)==1 and laugh[0].get('phonetic_hint') and not laugh[0].get('text')
 assert [e['text'] for e in spoken('S0835')]==['Yeah'];assert spoken('S0835')[0]['end']<4.5
 assert [e['text'] for e in spoken('S3358')]==['嗯','嗯']
 assert [e['text'] for e in spoken('S5193')]==['Uh huh','Yeah']
 from playwright.sync_api import sync_playwright
 errors=[]
 with sync_playwright() as p:
  browser=p.chromium.launch(headless=True,args=['--no-sandbox']);context=browser.new_context(viewport={'width':1500,'height':1100},accept_downloads=True);page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
  sid=get('S0805')['id'];assert page.goto(url+'?sample='+sid).status==200;page.wait_for_function('video.readyState>=2&&audio.readyState>=2&&video.videoWidth>0')
  assert page.locator('#samples option').count()==50;page.select_option('#modality','audio');assert page.locator('#events .event').count()==7
  assert '时间待定位' not in page.locator('#events').inner_text();assert '边界：待确定' not in page.locator('#events').inner_text()
  page.click('#events [data-play]');page.wait_for_timeout(500);state=page.evaluate('({v:video.currentTime,a:audio.currentTime,paused:video.paused})');assert not state['paused'] and abs(state['v']-state['a'])<.2;page.click('#pause')
  page.select_option('#decision','accept_content');page.click('#save');assert '已保存' in page.locator('#message').inner_text()
  with page.expect_download() as result:page.click('#export')
  fb=json.loads(Path(result.value.path()).read_text());assert len(fb['records'])==1 and not fb['training_admitted']
  page.evaluate('localStorage.clear()');page.reload();page.wait_for_function('video.readyState>=2&&audio.readyState>=2')
  page.select_option('#scope','human');assert page.locator('#samples option').count()==5
  for code in ['S2043','S0805','S0835','S3358','S5193']:
   sid=get(code)['id'];page.select_option('#samples',sid);page.wait_for_function('(sid)=>video.currentSrc.includes(sid)&&video.readyState>=2&&audio.readyState>=2',arg=sid)
   assert context.request.get(page.evaluate('audio.src'),headers={'Range':'bytes=0-1023'}).status==206
  page.select_option('#scope','all');page.select_option('#samples',get('S0805')['id']);page.select_option('#modality','audio');page.click('#events [data-play]');page.wait_for_timeout(500);page.click('#pause');page.evaluate('document.fonts.ready');page.screenshot(path=str(run/'review_desktop.png'),full_page=True)
  page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=str(run/'review_mobile.png'),full_page=True);assert not errors,errors;browser.close()
 report=dict(samples=50,raw_observations_preserved=682,all_emitted_events_have_observed_positive_intervals=True,unlocalized_units=data['summary']['unlocalized_units'],regressions=['two_separate_okay','right2_yeah5_monotonic_observed_intervals','laughter_not_lexical_hei','localized_yeah_over_unstable_Ed','two_um','uh_huh_is_one_response'],human_feedback_used_for_generation=False,gpu_used=False,browser_errors=errors,boundary_accuracy_not_human_verified=True)
 (run/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--url',required=True);a=p.parse_args();check(a.run,a.url)
