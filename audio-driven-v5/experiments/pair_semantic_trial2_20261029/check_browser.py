"""Serial review playback, three-version labels, counts and retained historical URLs."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
EXP=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8875/listen_full_source/'
m=json.loads((EXP/'review/manifest.json').read_text());errors=[];played=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu','--renderer-process-limit=1'])
 page=browser.new_page(viewport={'width':1600,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
 for number,d in enumerate(m['pairs'],1):
  page.goto(BASE+f'?pair={d["pair"]}&fixed=1#fixedPairReview')
  page.wait_for_function('(pair)=>window.reviewData?.pair===pair && document.querySelector("#fixedVersion")',arg=d['pair'])
  assert page.locator('#fixedVersion').input_value()==d['revision']
  assert page.locator('#fixedFilter').input_value()=='all'
  assert page.locator('#fixedList button').count()==20
  assert page.locator('#sevenValidationNote').inner_text()==d['seven_validation_note']
  tables=page.locator('#productionIntervals');tables.locator('details').evaluate_all('(ds)=>ds.forEach(d=>d.open=true)');text=tables.inner_text()
  for stage in d['interval_comparison']['stages']:
   for g in stage['groups']:
    for r in g['before']+g['after']:
     assert f'{r["start"]:.6f}–{r["end"]:.6f}' in text and r['text'] in text
  for c in d['comparison_cases']:
   article=page.locator('#compare-'+c['id'])
   assert article.locator('.before-after-grid > section').count()==3
   for i,side in enumerate(['production','before','after']):
    section=article.locator('.before-after-grid > section').nth(i)
    assert section.locator('strong').first.inner_text()==c[side+'_label']
    for r in c[side]:
     button=section.locator(f'[data-comparison-replay="{r["id"]}"]');button.click();video=button.locator('..').locator('video')
     video.evaluate('(v)=>new Promise((resolve,reject)=>{const until=Date.now()+20000;const poll=()=>{if(!v.paused&&v.currentTime>.1)resolve();else if(Date.now()>until)reject("play timeout");else setTimeout(poll,100)};poll()})')
     assert abs(video.evaluate('(v)=>v.duration')-r['duration'])<.09
     assert page.evaluate('[...document.querySelectorAll("video")].filter(v=>!v.paused).length')==1
     assert page.request.get(BASE+r['media'],headers={'Range':'bytes=0-127'}).status==206
     video.evaluate('(v)=>v.pause()');played.append(dict(pair=d['pair'],case=c['id'],side=side,id=r['id']))
   if c['id'] in ['trial_long_sentence','trial_he_is','trial_during']:
    article.screenshot(path=str(EXP/(c['id']+'.png')))
  for mode,count in [('all',d['initial_count']),('terminal',d['terminal_count']),('semantic',d['semantic_review_count'])]:
   page.locator('#fixedFilter').select_option(mode);assert page.locator('#fixedList button').count()==count
  held=next(r for r in d['rows'] if r.get('semantic_recovery_hold'))
  page.locator(f'[data-new-candidate="{held["id"]}"]').first.click()
  assert page.evaluate('window.fixedPairCurrent.id')==held['id']
  assert '本次保留整段供人工复核' in page.locator('#fixedPlayer').inner_text()
  for revision,prefix in [('local-feedback-head-attachment-5','H'),('seamless-full-source-no-scene-crop-statistics-2','P')]:
   page.locator('#fixedVersion').select_option(revision);assert page.evaluate('(prefix)=>window.fixedPairCurrent.id.startsWith(prefix)',prefix)
  page.locator('#fixedVersion').select_option(d['revision'])
  page.set_viewport_size({'width':390,'height':1000});page.evaluate('document.fonts.ready')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  page.set_viewport_size({'width':1600,'height':1100})
  print('checked source',number,flush=True)
 page.goto(BASE+f'?pair={m["pairs"][0]["pair"]}&fixed=1&compare=production_diff5#fixedPairReview')
 page.wait_for_function('document.getElementById("compare-production_diff5")')
 assert page.locator('#fixedVersion').input_value()=='local-feedback-head-attachment-5'
 # Explicit new comparison links choose J and land on the requested case.
 page.goto(BASE+f'?pair={m["pairs"][0]["pair"]}&fixed=1&compare=trial_he_is#fixedPairReview')
 page.wait_for_function('(()=>{const r=document.getElementById("compare-trial_he_is")?.getBoundingClientRect();return r&&r.top>=-1&&r.top<100})()')
 assert page.locator('#fixedVersion').input_value()==m['pairs'][0]['revision']
 assert page.locator('#source option').count()==7
 browser.close()
assert not errors,errors
report=dict(serial_browser=True,sources=2,played=played,three_version_labels_verified=True,all_times_and_transcripts_verified=True,review_candidates_not_mislabeled_kept=True,range_206=True,mobile_no_overflow=True,no_page_errors=True,old_H_production_comparison_links_preserved=True,full_run_started=False)
(EXP/'browser_validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(played),'comparison videos')
