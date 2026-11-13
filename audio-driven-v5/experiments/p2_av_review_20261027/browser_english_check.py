"""Validate complete English outputs and both published GLARE protocols."""
import json,unicodedata
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parent
data=json.loads((OUT/'data.json').read_text());errors=[];checks=[];guided=0
for d in data['datasets']:
 assert d['guidance_language']=='English' and set(d['glare_protocols'])=={'strict','unfiltered'}
 for s in d['samples']:
  assert len(s['outputs'])==16
  for o in s['outputs']:
   if o['variant']=='guided':
    guided+=1
    assert o['guidance_language']=='English'
    assert all(not c.isalpha() or 'LATIN' in unicodedata.name(c,'') for c in o['prompt'])
    record=json.loads((OUT/o['guidance_record']).read_text())
    assert record['prompt']==o['prompt'] and record['translations']==o['translation_records']
    assert 'p2_guided_en_' in str((OUT/o['video']).resolve())
   assert o['metric_status']!='not_evaluated'
 assert d['glare_protocols']['unfiltered']['coverage']['glare']==24
 for row1,row2 in zip(d['glare_protocols']['strict']['metrics'],d['glare_protocols']['unfiltered']['metrics']):
  for key in ['FVD-32','UTMOSv2','LSE-D','LSE-C']:
   assert row1['values'][key]==row2['values'][key]
assert guided==384
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':1080});page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:8876/p2_av_review/');page.wait_for_function('window.reviewReady === true')
 assert page.locator('#glare-protocol').input_value()=='unfiltered'
 for d in data['datasets']:
  page.locator(f'[data-dataset={d["id"]}]').click()
  for protocol in ['strict','unfiltered']:
   page.locator('#glare-protocol').select_option(protocol)
   expected=d['glare_protocols'][protocol]
   assert page.locator('#report').get_attribute('href')==expected['report']
   for scope in ['paired','within']:
    page.locator('#metric-scope').select_option(scope)
    rows=page.locator('#metric-rows tr');assert rows.count()==8
    assert '进行中' not in rows.all_inner_texts().__str__()
    for i,r in enumerate(expected['metrics']):
     cells=rows.nth(i).locator('td').all_text_contents();assert cells[1]=='48'
     values=r['values' if scope=='paired' else 'within_values']
     for j,key in enumerate(['FVD-32','UTMOSv2','LSE-D','LSE-C','R-F1','R-tIoU','R-ATD','R-FID']):
      value=values[key]
      if value is None:assert cells[j+2]=='—'
      else:assert abs(float(cells[j+2])-value)<=.5001*10**(-4 if key=='R-F1' else -3)
    checks.append(dict(dataset=d['id'],protocol=protocol,scope=scope,rows=8,outputs_per_row=48))
   if protocol=='unfiltered':assert 'GLARE 24/24' in page.locator('#metric-notes').inner_text()
  page.locator('#variant').select_option('guided');assert page.locator('.actual-prompt').count()==96
  assert all('（全英文）' in x for x in page.locator('.actual-prompt summary').all_text_contents())
  assert page.locator('.translation-audit').count()==96
 page.locator('#metric-scope').select_option('paired');page.locator('#glare-protocol').select_option('unfiltered');page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(OUT/'english_unfiltered_preview.png'))
 mobile=b.new_page(viewport={'width':390,'height':844},is_mobile=True);mobile.on('pageerror',lambda e:errors.append(str(e)))
 mobile.goto('http://127.0.0.1:8876/p2_av_review/');mobile.wait_for_function('window.reviewReady === true');assert mobile.evaluate('document.documentElement.scrollWidth<=innerWidth+1');mobile.screenshot(path=str(OUT/'english_mobile_preview.png'));b.close()
assert not errors,errors
result=dict(status='passed',guided_english_outputs=guided,checks=checks,page_errors=errors,mobile_no_overflow=True)
(OUT/'browser_english_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
