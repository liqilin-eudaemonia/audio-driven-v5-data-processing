import json,sys,subprocess,concurrent.futures,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'experiments/p2_av_review_20261027/reintegrated_20261028'
data=json.loads((OUT/'data.json').read_text());d=data['datasets'][0]
assert len(d['samples'])==24 and sum(len(s['outputs']) for s in d['samples'])==288
videos=set()
for s in d['samples']:
 assert s['asr'] and len(s['outputs'])==12
 videos.add((OUT/s['gt_video']).resolve())
 for key in ['image','speaker','gt_audio','gt_video','metric_gt_audio']:assert (OUT/s[key]).exists()
 for o in s['outputs']:
  for key in ['video','audio','mixed_video']:assert (OUT/o[key]).exists()
  videos.update((OUT/o[k]).resolve() for k in ['video','mixed_video'])
def probe(p):
 d=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type','-of','json',str(p)],text=True))
 assert {s['codec_type'] for s in d['streams']}=={'video','audio'},str(p)
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(probe,sorted(videos)))
errors=[]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
 page=b.new_page(viewport={'width':1440,'height':1080});page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:8876/p2_av_review/');page.wait_for_function('window.reviewReady===true');assert page.locator('#reintegrated-batch a').count()==1
 page.locator('#reintegrated-batch a').click();page.wait_for_function('window.reviewReady===true');assert page.locator('.sample').count()==24
 assert page.locator('.pipeline-card').count()==144
 for mode in ['strict','unfiltered']:
  page.locator('#glare-protocol').select_option(mode);tr=page.locator('#metric-rows tr');assert tr.count()==6
  for i,row in enumerate(d['glare_protocols'][mode]['metrics']):
   cells=tr.nth(i).locator('td').all_text_contents();assert cells[1]=='48'
   for j,k in enumerate(['FVD-32','UTMOSv2','LSE-D','LSE-C','R-F1','R-tIoU','R-ATD','R-FID']):
    v=row['values'][k]
    if v is None:assert cells[j+2]=='—'
    else:assert abs(float(cells[j+2])-v)<(.000051 if k=='R-F1' else .000501)
 page.locator('#seed').select_option('all');assert page.locator('.pipeline-card').count()==288
 page.locator('#variant').select_option('fixed');assert page.locator('.pipeline-card').count()==192
 page.locator('#variant').select_option('guided');assert page.locator('.pipeline-card').count()==96
 page.locator('#variant').select_option('all');page.locator('#seed').select_option('137')
 page.locator('[data-mode=mixed]').click()
 assert page.locator('video[data-listener]').evaluate_all('(vs)=>vs.every(v=>v.getAttribute("src")===v.dataset.mixed&&!v.muted)')
 v=page.locator('video[data-listener]').first;v.evaluate('(v)=>{v.load()}');page.wait_for_function('document.querySelector("video[data-listener]").readyState>=1');assert v.evaluate('(v)=>v.duration>0')
 page.locator('[data-mode=listener]').click();assert page.locator('video[data-listener]').evaluate_all('(vs)=>vs.every(v=>v.getAttribute("src")===v.dataset.listener)')
 page.locator('#glare-protocol').select_option('unfiltered');page.screenshot(path=str(OUT/'desktop.png'))
 mobile=b.new_page(viewport={'width':390,'height':844},is_mobile=True);mobile.on('pageerror',lambda e:errors.append(str(e)));mobile.goto('http://127.0.0.1:8876/p2_av_review/reintegrated_20261028/');mobile.wait_for_function('window.reviewReady===true');assert mobile.evaluate('document.documentElement.scrollWidth<=innerWidth+1');mobile.screenshot(path=str(OUT/'mobile.png'));b.close()
assert not errors,errors
frozen=json.loads((ROOT/'experiments/p2_full50h_duration384_20261028/training_queue/frozen.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
result=dict(status='passed',samples=24,outputs=288,av_videos=len(videos),desktop_and_mobile=True,metric_tables_verified=True,audio_switch_verified=True,training_frozen_files_unchanged=len(frozen),page_errors=errors)
(OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');(HERE/'HTML_VALIDATED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
