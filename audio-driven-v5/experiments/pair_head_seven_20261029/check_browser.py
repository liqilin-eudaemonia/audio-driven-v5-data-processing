"""Serial browser checks of all seven sources and every new comparison."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
EXP=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8875/listen_full_source/'
manifest=json.loads((EXP/'review/manifest.json').read_text());errors=[];played=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu','--renderer-process-limit=1'])
 page=browser.new_page(viewport={'width':1440,'height':1050});page.on('pageerror',lambda e:errors.append(str(e)))
 for index,d in enumerate(manifest['pairs']):
  page.goto(BASE+f'?pair={d["pair"]}&fixed=1#fixedPairReview')
  page.wait_for_function('(pair)=>window.reviewData?.pair===pair && document.querySelector("#fixedVersion")',arg=d['pair'])
  assert page.locator('#fixedVersion').input_value()==d['revision']
  assert page.locator('#sevenValidationNote').inner_text()==d['seven_validation_note']
  interval=page.locator('#adjacentIntervals')
  interval.locator('details').evaluate_all('(rows)=>rows.forEach(r=>r.open=true)')
  assert interval.locator('tbody tr').count()==sum(len(stage['groups']) for stage in d['adjacent_comparison']['stages'])
  for stage in d['adjacent_comparison']['stages']:
   for group in stage['groups']:
    for row in group['before']+group['after']:
     assert f'{row["start"]:.6f}–{row["end"]:.6f}' in interval.inner_text()
     assert row['text'] in interval.inner_text()
  for c in d['comparison_cases']:
   article=page.locator('#compare-'+c['id'])
   for side in ['before','after']:
    section=article.locator('.before-after-grid > section').nth(0 if side=='before' else 1)
    for r in c[side]:
     button=section.locator(f'[data-comparison-replay="{r["id"]}"]');button.click()
     video=button.locator('..').locator('video')
     video.evaluate('(v)=>new Promise((resolve,reject)=>{const deadline=Date.now()+20000;const poll=()=>{if(!v.paused&&v.currentTime>.1)resolve();else if(Date.now()>deadline)reject("play timeout");else setTimeout(poll,100)};poll()})')
     assert abs(video.evaluate('(v)=>v.duration')-r['duration'])<.09
     assert page.evaluate('[...document.querySelectorAll("video")].filter(v=>!v.paused).length')==1
     assert page.request.get(BASE+r['media'],headers={'Range':'bytes=0-127'}).status==206
     video.evaluate('(v)=>v.pause()');played.append(dict(pair=d['pair'],case=c['id'],id=r['id']))
   if index<2 and c['id']==('seven_diff1' if index==0 else 'seven_diff2'):
    article.screenshot(path=str(EXP/f'comparison_source{index+1}.png'))
  for mode,expected in [('all',d['initial_count']),('terminal',d['terminal_count']),('semantic',d['semantic_review_count'])]:
   page.locator('#fixedFilter').select_option(mode);assert page.locator('#fixedList button').count()==expected
  prior='published-full-source-recovery-20261023' if index<2 else 'local-feedback-sentence-priority-4'
  page.locator('#fixedVersion').select_option(prior)
  assert page.evaluate('window.fixedPairCurrent.id.startsWith("B")' if index<2 else 'window.fixedPairCurrent.id.startsWith("L")')
  page.locator('#fixedVersion').select_option('local-feedback-sentence-priority-4')
  assert page.evaluate('window.fixedPairCurrent.id.startsWith("L")')
  page.locator('#fixedVersion').select_option(d['revision'])
  if index<2:
   page.set_viewport_size({'width':390,'height':1050});page.evaluate('document.fonts.ready')
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
   page.set_viewport_size({'width':1440,'height':1050})
  print('checked',d['pair'],flush=True)
 # Explicit comparison links must retain historical versions.
 pair=manifest['pairs'][4]['pair']
 page.goto(BASE+f'?pair={pair}&fixed=1&compare=change3#fixedPairReview')
 page.wait_for_function('document.getElementById("compare-change3")')
 assert page.locator('#fixedVersion').input_value()=='local-feedback-sentence-priority-4'
 pair=manifest['pairs'][5]['pair']
 for clip in ['L1-16','G1-16','F1-13']:
  page.goto(BASE+f'?pair={pair}&fixed=1&clip={clip}#fixedPairReview')
  page.wait_for_function('(id)=>window.fixedPairCurrent?.id===id',arg=clip)
 assert page.locator('#source option').count()==7
 browser.close()
assert not errors,errors
report=dict(serial_browser=True,sources=7,all_interval_tables_and_text_verified=True,comparison_clips_played=played,range_206=True,no_page_errors=True,historical_versions_preserved=True,mobile_no_overflow=True,full_run_started=False)
(EXP/'browser_validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(played),'comparison videos')
