from pathlib import Path
import json,subprocess,concurrent.futures
from playwright.sync_api import sync_playwright
E=Path(__file__).resolve().parent;R=E.parents[1];OUT=R/'experiments/p2_av_review_20261027/autonomous_preview_20261028';d=json.loads((OUT/'data.json').read_text());samples=d['datasets'][0]['samples']
assert len(samples)==3 and sum(len(s['outputs']) for s in samples)==12
videos=set()
for s in samples:
 assert len(s['outputs'])==4 and s['asr']
 videos.add((OUT/s['gt_video']).resolve())
 for o in s['outputs']:
  for k in ['video','mixed_video']:videos.add((OUT/o[k]).resolve())
  assert (OUT/o['audio']).exists() and (OUT/o['plan_record']).exists()
def probe(p):
 value=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type','-of','json',str(p)],text=True));assert {s['codec_type'] for s in value['streams']}=={'audio','video'}
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(probe,videos))
errors=[]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
 page=b.new_page(viewport={'width':1440,'height':1080});page.on('pageerror',lambda e:errors.append(str(e)));page.goto('http://127.0.0.1:8876/p2_av_review/autonomous_preview_20261028/');page.wait_for_function('window.reviewReady===true')
 assert page.locator('.sample').count()==3 and page.locator('.pipeline-card').count()==6
 assert '尚未计算评测指标' in page.locator('.metrics-panel').inner_text()
 page.locator('#seed').select_option('all');assert page.locator('.pipeline-card').count()==12
 page.locator('[data-mode=mixed]').click();assert page.locator('video[data-listener]').evaluate_all('(vs)=>vs.every(v=>v.getAttribute("src")===v.dataset.mixed&&!v.muted)')
 v=page.locator('video[data-listener]').first;v.evaluate('(v)=>v.load()');page.wait_for_function('document.querySelector("video[data-listener]").readyState>=1')
 assert v.evaluate('(v)=>v.duration>0')
 page.locator('[data-mode=listener]').click();assert page.locator('video[data-listener]').evaluate_all('(vs)=>vs.every(v=>v.getAttribute("src")===v.dataset.listener)')
 page.locator('#seed').select_option('137');page.screenshot(path=str(OUT/'desktop.png'))
 m=b.new_page(viewport={'width':390,'height':844},is_mobile=True);m.on('pageerror',lambda e:errors.append(str(e)));m.goto('http://127.0.0.1:8876/p2_av_review/autonomous_preview_20261028/');m.wait_for_function('window.reviewReady===true');assert m.evaluate('document.documentElement.scrollWidth<=innerWidth+1');m.screenshot(path=str(OUT/'mobile.png'));b.close()
assert not errors,errors
result=dict(status='passed',samples=3,outputs=12,av_streams_verified=len(videos),browser_errors=errors,desktop_mobile=True,audio_toggle=True)
(OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
