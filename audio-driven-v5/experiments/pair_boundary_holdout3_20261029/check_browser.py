"""Validate actual published media, all new clips, comparisons, and historical links."""
import ctypes
assert ctypes.CDLL(None,use_errno=True).prctl(41,1,0,0,0)==0
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
EXP=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8875/listen_full_source/'
manifest=json.loads((EXP/'review/localbc/manifest.json').read_text())
errors=[];checked=[];played=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1440,'height':1100})
 page.on('pageerror',lambda e:errors.append(str(e)))
 for i,d in enumerate(manifest['pairs'],1):
  pair=d['pair'];page.goto(BASE+f'?pair={pair}&fixed=1#fixedPairReview')
  page.wait_for_function('(pair)=>window.reviewData?.pair===pair && document.querySelector("#fixedVersion")',arg=pair)
  assert page.locator('#fixedVersion').input_value()==d['revision']
  assert page.locator('#fixedVersion option').count()==2
  assert '三条新来源' in page.locator('#fixedBody').inner_text()
  assert '前两处修复继续保留' not in page.locator('#fixedBody').inner_text()
  assert page.locator('#localBcComparisons article').count()==len(d['comparison_cases'])
  for c in d['comparison_cases']:
   for r in c['before']+c['after']:
    b=page.locator(f'[data-comparison-replay="{r["id"]}"]');b.click()
    page.wait_for_function('(id)=>{const v=document.querySelector(`[data-comparison-replay="${id}"]`).parentElement.querySelector("video");return v.currentTime>.12&&!v.paused}',arg=r['id'])
    v=b.locator('..').locator('video');assert abs(v.evaluate('(v)=>v.duration')-r['duration'])<.09
    assert page.evaluate('[...document.querySelectorAll("video")].filter(v=>!v.paused).length')==1
    v.evaluate('(v)=>v.pause()');played.append(r['id'])
  for mode in ['all','terminal']:
   page.locator('#fixedFilter').select_option(mode)
   expected=[r for r in d['rows'] if r['initial']] if mode=='all' else [r for r in d['rows'] if r['id'] in d['terminal_ids']]
   assert page.locator('#fixedList button').count()==len(expected)
   for r in expected:
    if r['id'] in checked:continue
    page.locator(f'[data-fixed-id="{r["id"]}"]').click()
    page.wait_for_function('(id)=>window.fixedPairCurrent?.id===id && document.querySelector("#fixedPlayer video").readyState>=1',arg=r['id'])
    assert abs(page.locator('#fixedPlayer video').evaluate('(v)=>v.duration')-r['duration'])<.09
    assert page.request.get(BASE+r['media'],headers={'Range':'bytes=0-127'}).status==206
    assert r['added_context_seconds']==0;checked.append(r['id'])
  for mode,count in [('withheld',sum(r['initial'] and r['pool']!='keep_candidate' for r in d['rows'])),('semantic',d['semantic_review_count'])]:
   page.locator('#fixedFilter').select_option(mode);assert page.locator('#fixedList button').count()==count
  page.locator('#fixedFilter').select_option('all');page.locator('#fixedReplay').click()
  page.wait_for_function('!document.querySelector("#fixedPlayer video").paused && document.querySelector("#fixedPlayer video").currentTime>.1')
  for channel in ['a','b','both']:page.locator('#fixedChannel').select_option(channel)
  page.locator('#fixedPlayer video').evaluate('(v)=>v.pause()')
  page.locator('#fixedDecision').select_option('uncertain');page.locator('#fixedNote').fill('浏览器验收记录，非人工判断');page.locator('#fixedSave').click()
  page.locator('#fixedVersion').select_option('local-listening-boundary-repair-gap5-3');assert page.evaluate('window.fixedPairCurrent?.id?.startsWith("G")')
  page.locator('#fixedVersion').select_option(d['revision'])
  for width in [1440,390]:
   page.set_viewport_size({'width':width,'height':1100});page.evaluate('document.fonts.ready');assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
   page.locator('#localBcComparisons article').first.scroll_into_view_if_needed();page.screenshot(path=str(EXP/f'source{i}_{width}.png'))
  page.set_viewport_size({'width':1440,'height':1100})
  page.locator('#fixedFilter').select_option('all')
 with page.expect_download() as event:page.locator('#fixedExport').click()
 feedback=json.loads(Path(event.value.path()).read_text());assert len(feedback['rows'])==3 and feedback['production_modified'] is False
 last=manifest['pairs'][-1]
 page.goto(BASE+f'?pair={last["pair"]}&fixed=1&compare=change3#fixedPairReview')
 page.wait_for_function('document.querySelector("#compare-change3")?.getBoundingClientRect().top<100')
 assert '213.343–213.963' in page.locator('#compare-change3').inner_text()
 assert '发现退步' in page.locator('#compare-change3').inner_text()
 c=next(c for c in last['comparison_cases'] if c['id']=='change3');assert c['before'][0]['text'].startswith('Okay.') and c['after'][0]['text'].startswith('Well,')
 page.locator('#compare-change3').screenshot(path=str(EXP/'okay_regression.png'))
 # Ensure the original reviewed sources still resolve both historical and latest IDs.
 original='naturalistic_V02_S3579_I00000175'
 for clip,revision in [('G1-16','local-listening-boundary-repair-gap5-3'),('L1-16','local-feedback-sentence-priority-4')]:
  page.goto(BASE+f'?pair={original}&fixed=1&clip={clip}#fixedPairReview');page.wait_for_function('(id)=>window.fixedPairCurrent?.id===id',arg=clip)
  assert page.locator('#fixedVersion').input_value()==revision
  page.locator('#fixedReplay').click();page.wait_for_function('document.querySelector("#fixedPlayer video").currentTime>.1 && !document.querySelector("#fixedPlayer video").paused')
  page.locator('#fixedPlayer video').evaluate('(v)=>v.pause()')
 assert page.locator('#source option').count()==7
 browser.close()
assert not errors,errors
report=dict(new_clips_checked=len(checked),comparison_playbacks=played,page_errors=errors,range_206=True,no_extra_context=True,mobile_no_overflow=True,feedback_export=True,old_review_preserved=True,regression_highlight_and_deep_link=True)
(EXP/'browser_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
