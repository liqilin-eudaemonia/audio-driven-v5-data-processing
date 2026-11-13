"""Check the actual production-to-H comparison, including excluded evidence separately."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
EXP=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8875/listen_full_source/'
m=json.loads((EXP/'review/manifest.json').read_text());errors=[];played=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu','--renderer-process-limit=1'])
 page=browser.new_page(viewport={'width':1440,'height':1050});page.on('pageerror',lambda e:errors.append(str(e)))
 for index,d in enumerate(m['pairs'],1):
  page.goto(BASE+f'?pair={d["pair"]}&fixed=1#fixedPairReview')
  page.wait_for_function('(pair)=>window.reviewData?.pair===pair && document.getElementById("productionIntervals")',arg=d['pair'])
  assert page.locator('#fixedVersion').input_value()==d['revision']
  assert page.locator('#sevenValidationNote').inner_text()==d['seven_validation_note']
  interval=page.locator('#productionIntervals');interval.locator('details').evaluate_all('(ds)=>ds.forEach(d=>d.open=true)')
  assert interval.locator('tbody tr').count()==sum(len(s['groups']) for s in d['interval_comparison']['stages'])
  text=interval.inner_text()
  for stage in d['interval_comparison']['stages']:
   for g in stage['groups']:
    for r in g['before']+g['after']:
     assert f'{r["start"]:.6f}–{r["end"]:.6f}' in text
     assert r['text'] in text
  for c in d['comparison_cases']:
   if c.get('archived'):continue
   article=page.locator('#compare-'+c['id'])
   for side in ['before','after','excluded_proposals']:
    rows=c.get(side,[])
    if side=='excluded_proposals':
     if not rows:continue
     section=article.locator('.excluded-proposals');section.evaluate('(d)=>d.open=true')
    else:
     section=article.locator('.before-after-grid > section').nth(0 if side=='before' else 1)
     assert section.locator('video').count()==len(rows)
     assert all(r['pool']=='keep_candidate' for r in rows)
    for r in rows:
     button=section.locator(f'[data-comparison-replay="{r["id"]}"]');button.click();video=button.locator('..').locator('video')
     video.evaluate('(v)=>new Promise((resolve,reject)=>{const until=Date.now()+20000;const poll=()=>{if(!v.paused&&v.currentTime>.1)resolve();else if(Date.now()>until)reject("play timeout");else setTimeout(poll,100)};poll()})')
     assert abs(video.evaluate('(v)=>v.duration')-r['duration'])<.09
     assert page.evaluate('[...document.querySelectorAll("video")].filter(v=>!v.paused).length')==1
     assert page.request.get(BASE+r['media'],headers={'Range':'bytes=0-127'}).status==206
     video.evaluate('(v)=>v.pause()');played.append(dict(source=index,case=c['id'],side=side,id=r['id']))
   if index in [1,6] and c['id']==('production_diff1' if index==1 else 'production_diff2'):
    article.screenshot(path=str(EXP/f'production_source{index}.png'))
  for mode,expected in [('all',d['initial_count']),('terminal',d['terminal_count']),('semantic',d['semantic_review_count'])]:
   page.locator('#fixedFilter').select_option(mode);assert page.locator('#fixedList button').count()==expected
  page.locator('#fixedVersion').select_option('seamless-full-source-no-scene-crop-statistics-2')
  assert page.evaluate('window.fixedPairCurrent.id.startsWith("P")')
  page.locator('#fixedVersion').select_option(d['revision'])
  page.set_viewport_size({'width':390,'height':1050});page.evaluate('document.fonts.ready')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  page.set_viewport_size({'width':1440,'height':1050})
  print('checked production source',index,flush=True)
 # Archive links must open their containing details and retain historical version resolution.
 pair=m['pairs'][4]['pair']
 for case in ['okay_head','change3']:
  page.goto(BASE+f'?pair={pair}&fixed=1&compare={case}#fixedPairReview')
  page.wait_for_function('(id)=>{const r=document.getElementById("compare-"+id)?.getBoundingClientRect();return r&&r.top>=-1&&r.top<100}',arg=case)
  assert page.locator('#fixedVersion').input_value()==('local-feedback-sentence-priority-4' if case=='change3' else 'local-feedback-head-attachment-5')
 assert page.locator('#source option').count()==7
 browser.close()
assert not errors,errors
report=dict(sources=7,comparison='actual production output to latest H',main_case_count=sum(not c.get('archived') for d in m['pairs'] for c in d['comparison_cases']),played=played,all_interval_times_and_text_verified=True,both_main_columns_only_final_kept=True,excluded_proposals_separate=True,source_videos_range_206=True,mobile_no_overflow=True,no_page_errors=True,historical_links_preserved=True,full_run_started=False)
(EXP/'production_browser_validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(played),'videos',flush=True)
