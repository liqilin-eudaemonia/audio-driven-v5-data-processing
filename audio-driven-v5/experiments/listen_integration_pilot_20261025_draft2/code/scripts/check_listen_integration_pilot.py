"""Real 50-sample regression cases, provenance and browser review checks."""
import argparse,hashlib,json,tempfile
from pathlib import Path

def check(run,url):
 data=json.loads((run/'review.json').read_text());samples=data['samples'];assert len(samples)==50
 manifest=json.loads((run/'manifest.json').read_text());raw=(run/'observations.jsonl').read_bytes()
 assert hashlib.sha256(raw).hexdigest()==manifest['raw_observations_sha256']
 assert raw==(Path(manifest['source'])/'observations.jsonl').read_bytes()
 assert hashlib.sha256((run/'human_feedback.original.json').read_bytes()).hexdigest()==manifest['feedback_sha256']
 for s in samples:
  ids={e['event_id'] for e in s['events']};obs={e['event_id'] for e in s['original_events']}
  assigned={i for e in s['events'] for i in e['source_observation_ids']}
  assigned.update(x['observation_id'] for x in s['evidence_links']+s['unresolved_observations']);assert obs==assigned
  assert len(ids)==len(s['events'])
  assert all(not any(e['training'].values()) for e in s['events'])
  assert all(set(g['member_event_ids'])<=ids and g['training_mask'] is None and not g['continuous_activity_claim'] for g in s['response_groups'])
  assert s['negative_coverage']=={'audio':[],'visual':[]}
  tokens=[t['id'] for tr in s['alignment_tracks'] for t in tr['tokens']]
  restored=[t['id'] for c in s['token_clusters'] for t in c['members']];assert sorted(tokens)==sorted(restored)
  assert all(len(c['members'])==len({t['pass_id'] for t in c['members']}) for c in s['token_clusters'])
 def sample(code):return next(s for s in samples if code in s['id'])
 def audio(code):return [e for e in sample(code)['events'] if e['modality']=='audio']
 laugh=audio('S2043');assert len(laugh)==1 and laugh[0]['phonetic_hint']=='嘿嘿' and laugh[0]['text'] is None and laugh[0]['status']['count']=='unknown'
 yeah=audio('S0805');assert [e['text'].lower() for e in yeah]==['right','right']+['yeah']*5;assert all(e['start'] is None and e['end'] is None for e in yeah)
 ed=audio('S0835');assert len(ed)==1 and ed[0]['text']=='Yeah' and ed[0]['end'] is None;assert any(e.get('text')=='Ed' for e in sample('S0835')['original_events'])
 um=audio('S3358');assert len(um)==2 and [(e['start'],e['end']) for e in um]==[(.32,1.44),(1.44,1.52)]
 uh=audio('S5193');assert [e['text'] for e in uh]==['Uh huh','Yeah'];assert [(e['start'],e['end']) for e in uh]==[(3,4),(7,8)]
 okay=[e for e in audio('S0699') if (e.get('text') or '').lower()=='okay'];assert [(e['start'],e['end']) for e in okay]==[(1.78,2.1),(4.08,4.4)]
 assert data['summary']['asr_cross_window_duplicate_observations']>0
 from playwright.sync_api import sync_playwright
 errors=[]
 with sync_playwright() as p:
  browser=p.chromium.launch(headless=True,args=['--no-sandbox']);ctx=browser.new_context(viewport={'width':1500,'height':1100},accept_downloads=True);page=ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
  target=sample('S0805')['id'];assert page.goto(url+'?sample='+target).status==200
  page.wait_for_function('video.readyState>=2&&audio.readyState>=2&&video.videoWidth>0');assert page.locator('#samples option').count()==50
  page.select_option('#modality','audio');assert page.locator('#events .event').count()==7
  assert '时间待定位' in page.locator('#events').inner_text()
  page.click('#events [data-play]');page.wait_for_timeout(500);state=page.evaluate('({v:video.currentTime,a:audio.currentTime,paused:video.paused})');assert not state['paused'] and abs(state['v']-state['a'])<.2;page.click('#pause')
  page.select_option('#decision','accept_event');page.click('#save');assert '有效起止时间' in page.locator('#message').inner_text()
  page.select_option('#decision','accept_content');page.click('#save');assert '已保存' in page.locator('#message').inner_text()
  with page.expect_download() as result:page.click('#export')
  feedback=json.loads(Path(result.value.path()).read_text());assert feedback['records'][0]['start'] is None and not feedback['records'][0]['boundary_confirmed']
  with tempfile.TemporaryDirectory() as tmp:
   f=Path(tmp)/'feedback.json';f.write_text(json.dumps(feedback));page.evaluate('localStorage.clear()');page.reload();page.wait_for_function('video.readyState>=2&&audio.readyState>=2');page.set_input_files('#import',str(f));page.wait_for_function('Object.keys(integrationState().saved).length===1')
  page.evaluate('localStorage.clear()');page.reload();page.wait_for_function('video.readyState>=2&&audio.readyState>=2')
  page.select_option('#scope','human');assert page.locator('#samples option').count()==5
  for code in ['S2043','S0805','S0835','S3358','S5193']:
   sid=sample(code)['id'];page.select_option('#samples',sid);page.wait_for_function('(sid)=>video.currentSrc.includes(sid)&&video.readyState>=2&&audio.readyState>=2',arg=sid)
   assert ctx.request.get(page.evaluate('video.src'),headers={'Range':'bytes=0-1023'}).status==206
   assert ctx.request.get(page.evaluate('audio.src'),headers={'Range':'bytes=0-1023'}).status==206
  page.select_option('#scope','all');page.select_option('#samples',sample('S0699')['id']);page.select_option('#modality','audio');assert page.locator('#events .event').count()==4
  page.select_option('#samples',target);page.wait_for_function('(sid)=>video.currentSrc.includes(sid)&&video.readyState>=2',arg=target);page.click('#events [data-play]');page.wait_for_timeout(500);page.click('#pause');page.evaluate('document.fonts.ready');page.screenshot(path=str(run/'review_desktop.png'),full_page=True)
  page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=str(run/'review_mobile.png'),full_page=True)
  assert not errors,errors;browser.close()
 report=dict(samples=50,original_observations=682,regressions=['laughter_not_three_lexical_words','right2_yeah5_without_fabricated_times','yeah_tail_not_Ed_to10s','two_um_preserved','uh_huh_one_response_plus_yeah','two_separate_okay','all_original_and_ASR_evidence_retained','same_pass_words_not_merged','no_group_activity_mask'],
             asr_before=data['summary']['asr_token_observations'],asr_after=data['summary']['asr_token_clusters'],unit_tests=7,browser_errors=errors,training_admitted=False,full_corpus_applied=False)
 (run/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--url',required=True);a=p.parse_args();check(a.run,a.url)
