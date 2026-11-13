import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'review';m=json.loads((OUT/'samples.json').read_text());checks=[];errors=[]
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True,args=['--no-sandbox']);p=b.new_page(viewport={'width':1536,'height':1100});p.set_default_timeout(25000);p.on('pageerror',lambda e:errors.append(str(e)));p.goto('http://127.0.0.1:8797/');p.wait_for_function('selected!==undefined');assert p.locator('#list button').count()==len(m['pairs']);assert p.locator('#timeline .span').count()==len(m['segments'])
 for i,r in enumerate(m['pairs']):
  p.evaluate('(i)=>choose(pairs[i],false)',i);p.locator('#crop').click();p.wait_for_function('v.readyState>=2 && !v.error && (v.currentTime>0 || v.ended)');checks.append({'pair':r['pair_id'],'crop_played':True})
  if (i+1)%30==0:print('pairs',i+1,flush=True)
 for f in ['role_unknown','black','all']:
  p.locator('#filter').select_option(f);assert p.locator('#list button').count()==sum(f=='all' or r['kind']==f for r in m['segments']);p.locator('#list button').first.click();p.wait_for_function('v.readyState>=2 && !v.paused')
 p.locator('#filter').select_option('pair')
 # Across a real cut, the same continuous source player passes from one pair to the next.
 p.locator('#cuts').select_option('57.52');assert 'sentence_012' in p.locator('#cutInfo').inner_text() and 'sentence_013' in p.locator('#cutInfo').inner_text();p.locator('#cutPlay').click();p.wait_for_function('mode==="annotated" && v.readyState>=2 && v.currentTime>=55.52 && !v.paused');p.wait_for_function('v.currentTime>57.6 && currentSegment.pair_id==="sentence_013"');checks.append({'cut':57.52,'both_sides_continuous':True})
 for mode in ['raw','annotated']:
  p.locator('#'+mode).click();p.wait_for_function('v.readyState>=2 && !v.paused');r=p.request.get(p.evaluate('v.src'),headers={'Range':'bytes=0-127'});assert r.status==206
 p.evaluate('choose(pairs[11],false)');p.locator('#decision').select_option('keep');p.locator('#note').fill('Automated test');p.reload();p.wait_for_function('selected!==undefined');p.evaluate('choose(pairs[11],false)');assert p.locator('#note').input_value()=='Automated test'
 with p.expect_download() as d:p.locator('#export').click()
 q=json.loads(Path(d.value.path()).read_text());assert q['notes']['sentence_012']['decision']=='keep';assert q['manifest_sha256']==json.loads((OUT/'contract.json').read_text())['manifest_sha256'];p.evaluate('localStorage.removeItem(key);notes={};choose(pairs[11],false)');p.wait_for_function('v.readyState>=2 && !v.seeking');p.evaluate('v.currentTime=56.8');p.wait_for_function('!v.seeking');p.evaluate('document.fonts.ready');p.screenshot(path=str(OUT/'desktop.png'),full_page=True);p.locator('video').screenshot(path=str(OUT/'before_cut.png'));p.evaluate('v.currentTime=58.0');p.wait_for_function('!v.seeking');p.locator('video').screenshot(path=str(OUT/'after_cut.png'))
 p.set_viewport_size({'width':390,'height':844});p.wait_for_timeout(300);assert p.evaluate('document.documentElement.scrollWidth<=innerWidth');p.screenshot(path=str(OUT/'mobile.png'),full_page=True);assert not errors;b.close()
(OUT/'browser_validation.json').write_text(json.dumps({'checks':checks,'all_pair_playback':len(m['pairs']),'complete_timeline':True,'feedback':True,'mobile':True,'errors':errors},indent=2));print('PASS browser',len(m['pairs']),'pairs',flush=True)
