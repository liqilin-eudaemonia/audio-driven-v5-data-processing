"""Serial browser acceptance for changed clips and retained historical examples."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
EXP=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8875/listen_full_source/'
manifest=json.loads((EXP/'review/head_attach/manifest.json').read_text());errors=[];played=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu','--renderer-process-limit=1'])
 page=browser.new_page(viewport={'width':1440,'height':1050});page.on('pageerror',lambda e:errors.append(str(e)))
 for d in manifest['pairs']:
  page.goto(BASE+f'?pair={d["pair"]}&fixed=1#fixedPairReview')
  page.wait_for_function('(pair)=>window.reviewData?.pair===pair && document.querySelector("#fixedVersion")',arg=d['pair'])
  assert page.locator('#fixedVersion').input_value()==d['revision']
  for c in d['comparison_cases']:
   for r in c['before']+c['after']:
    button=page.locator(f'[data-comparison-replay="{r["id"]}"]');button.click()
    page.wait_for_function('(id)=>{const v=document.querySelector(`[data-comparison-replay="${id}"]`).parentElement.querySelector("video");return !v.paused&&v.currentTime>.1}',arg=r['id'])
    video=button.locator('..').locator('video')
    assert abs(video.evaluate('(v)=>v.duration')-r['duration'])<.09
    assert page.evaluate('[...document.querySelectorAll("video")].filter(v=>!v.paused).length')==1
    assert page.request.get(BASE+r['media'],headers={'Range':'bytes=0-127'}).status==206
    video.evaluate('(v)=>v.pause()');played.append(r['id'])
  for mode,expected in [('all',d['initial_count']),('terminal',d['terminal_count']),('semantic',d['semantic_review_count'])]:
   page.locator('#fixedFilter').select_option(mode);assert page.locator('#fixedList button').count()==expected
  page.locator('#fixedFilter').select_option('all');page.locator('#fixedReplay').click()
  page.wait_for_function('!document.querySelector("#fixedPlayer video").paused&&document.querySelector("#fixedPlayer video").currentTime>.1')
  for channel in ['a','b','both']:page.locator('#fixedChannel').select_option(channel)
  page.locator('#fixedPlayer video').evaluate('(v)=>v.pause()')
  page.locator('#fixedVersion').select_option('local-feedback-sentence-priority-4')
  assert page.evaluate('window.fixedPairCurrent.id.startsWith("L")')
  page.locator('#fixedVersion').select_option(d['revision'])
 last=manifest['pairs'][-1]
 for key in ['okay_head','yeah_head','right_head']:
  page.goto(BASE+f'?pair={last["pair"]}&fixed=1&compare={key}#fixedPairReview')
  page.wait_for_function('(id)=>{const r=document.getElementById("compare-"+id)?.getBoundingClientRect();return r&&r.top>=-1&&r.top<100}',arg=key)
  case=next(c for c in last['comparison_cases'] if c['id']==key)
  if key=='right_head':
   attempted,restored=case['after'];old=case['before'][0]
   assert attempted['pool']=='audio_review' and restored['pool']=='keep_candidate'
   assert (old['start'],old['end'],old['text'])==(restored['start'],restored['end'],restored['text'])
   assert '恢复原完整' in page.locator('#compare-right_head').inner_text()
  else:
   assert case['after'][0]['pool']=='keep_candidate'
   assert case['after'][0]['end']==case['before'][0]['end']
  page.locator('#compare-'+key).screenshot(path=str(EXP/(key+'.png')))
 page.goto(BASE+f'?pair={last["pair"]}&fixed=1&compare=okay_head#fixedPairReview')
 page.wait_for_function('document.getElementById("compare-okay_head")')
 for width in [1440,390]:
  page.set_viewport_size({'width':width,'height':1050});page.evaluate('document.fonts.ready')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  page.locator('#compare-okay_head').scroll_into_view_if_needed();page.screenshot(path=str(EXP/f'page_{width}.png'))
 page.set_viewport_size({'width':1440,'height':1050})
 target=next(c for c in last['comparison_cases'] if c['id']=='okay_head')['after'][0]
 page.locator(f'[data-new-candidate="{target["id"]}"]').click()
 assert page.evaluate('window.fixedPairCurrent.id')==target['id']
 page.locator('#fixedDecision').select_option('uncertain');page.locator('#fixedNote').fill('浏览器验收，不是人工结论');page.locator('#fixedSave').click()
 with page.expect_download() as event:page.locator('#fixedExport').click()
 feedback=json.loads(Path(event.value.path()).read_text());assert feedback['rows'][0]['revision']==last['revision']
 # Old comparison URLs select their historical version, rather than silently
 # showing a different comparison under the new default.
 page.goto(BASE+f'?pair={last["pair"]}&fixed=1&compare=change3#fixedPairReview')
 page.wait_for_function('document.getElementById("compare-change3")')
 assert page.locator('#fixedVersion').input_value()=='local-feedback-sentence-priority-4'
 first=manifest['pairs'][0]
 for clip in ['L1-16','G1-16','F1-13']:
  page.goto(BASE+f'?pair={first["pair"]}&fixed=1&clip={clip}#fixedPairReview')
  page.wait_for_function('(id)=>window.fixedPairCurrent?.id===id',arg=clip)
 assert page.locator('#source option').count()==7
 browser.close()
assert not errors,errors
report=dict(serial_browser=True,comparison_clips_played=played,range_206=True,no_page_errors=not errors,head_ends_unchanged=True,right_failed_extension_falls_back_to_original=True,historical_comparison_links_preserved=True,mobile_no_overflow=True,feedback_revision_isolated=True)
(EXP/'browser_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
