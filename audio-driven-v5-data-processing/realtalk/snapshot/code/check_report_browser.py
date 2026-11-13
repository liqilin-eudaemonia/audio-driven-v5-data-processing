import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports';m=json.loads((OUT/'summary.json').read_text());errors=[]
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True,args=['--no-sandbox']);p=b.new_page(viewport={'width':1440,'height':1100});p.on('pageerror',lambda e:errors.append(str(e)));p.goto('http://127.0.0.1:8798/');p.wait_for_function('data!==undefined');assert p.locator('#thresholdTable tr').count()==len(m['thresholds']);assert p.locator('#faceChart rect').count()==len(m['face_resolution_bins'])
 for criterion in ['crop_edge','face_p10','face_median','face_min']:
  p.locator('#criterion').select_option(criterion)
  for index in [0,5,8,14]:
   p.locator('#threshold').fill(str(index));p.locator('#threshold').dispatch_event('input');r=m['thresholds'][index];prefix=criterion+('_coverage95' if criterion!='crop_edge' else '');assert f"{r[prefix+'_hours']:.2f}" in p.locator('#retained').inner_text()
 p.locator('#criterion').select_option('face_p10');p.locator('#coverage').uncheck();p.locator('#threshold').fill('6');p.locator('#threshold').dispatch_event('input');assert f"{m['thresholds'][6]['face_p10_hours']:.2f}" in p.locator('#retained').inner_text()
 for f in ['pairs.csv','pairs.parquet','threshold_retention.csv','pre_crop_exact_sizes.csv','resolution_duration.png','resolution_duration.svg']:
  r=p.request.get('http://127.0.0.1:8798/'+f,headers={'Range':'bytes=0-127'});assert r.status in [200,206],(f,r.status)
 p.locator('#criterion').select_option('crop_edge');p.locator('#threshold').fill('8');p.locator('#threshold').dispatch_event('input');p.evaluate('document.fonts.ready');p.screenshot(path=str(OUT/'desktop.png'),full_page=True);p.set_viewport_size({'width':390,'height':844});p.wait_for_timeout(200);assert p.evaluate('document.documentElement.scrollWidth<=innerWidth');p.screenshot(path=str(OUT/'mobile.png'),full_page=True);assert not errors,errors;b.close()
(OUT/'browser_validation.json').write_text(json.dumps(dict(criterion_threshold_combinations=16,coverage_toggle=True,downloads=True,mobile=True,errors=errors),indent=2));print('report browser passed')
