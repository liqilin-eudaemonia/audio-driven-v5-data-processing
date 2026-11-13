from pathlib import Path
import json,subprocess,collections
from playwright.sync_api import sync_playwright
r=Path('/private/liqi_workspace/realtalk_processing/full_20261023/reports/resolution_review');s=json.loads((r/'samples.json').read_text())['samples'];assert len(s)==33
for x in s:
 for kind in ['native','review']:
  f=r/'media'/f"{x['sample_id']}_{kind}.mp4";v=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(f)]))['streams'];vid=next(z for z in v if z['codec_type']=='video');assert abs(float(vid['duration'])-min(6,x['pair']['duration']))<.09
  if kind=='native':assert vid['width']==vid['height']==(x['pair']['crop']['native_edge']+1)//2*2
errors=[]
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True,args=['--no-sandbox']);p=b.new_page(viewport={'width':1440,'height':1000});p.on('pageerror',lambda e:errors.append(str(e)));p.goto('http://127.0.0.1:8798/resolution_review/');assert p.locator('article').count()==33
 p.locator('#filter').select_option(label='128–160px');assert p.locator('article:visible').count()==3
 p.locator('textarea').first.fill('验证用临时意见');p.reload();assert p.locator('textarea').first.input_value()=='验证用临时意见';p.locator('textarea').first.fill('')
 p.locator('video').first.evaluate('(v)=>{v.muted=true;v.play()}');p.wait_for_function('document.querySelector("video").currentTime>0.3');p.locator('video').first.evaluate('(v)=>v.pause()');p.screenshot(path=str(r/'desktop.png'))
 p.set_viewport_size({'width':390,'height':844});assert p.evaluate('document.documentElement.scrollWidth<=innerWidth');p.screenshot(path=str(r/'mobile.png'));assert not errors; b.close()
(r/'validation.json').write_text(json.dumps({'samples':len(s),'video_files':66,'bins':dict(collections.Counter(x['bin'] for x in s)),'native_dimensions_and_duration':'pass','browser_filter_persistence_playback_mobile':'pass','page_errors':errors},ensure_ascii=False,indent=2));print('passed')
