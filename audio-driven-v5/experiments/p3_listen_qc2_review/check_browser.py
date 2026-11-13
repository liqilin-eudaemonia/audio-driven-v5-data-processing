import sys,json
from pathlib import Path
sys.path.insert(0,'/tmp/listen_preview_browser')
from playwright.sync_api import sync_playwright
out=Path('/private/liqi_workspace/project_code/audio-driven-v5/data_previews/seamless_listen_qc2_review')
samples=json.loads((out/'samples.json').read_text())['samples'];errors=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:8875/listen_qc2/',wait_until='domcontentloaded');page.wait_for_function('samples.length>0')
 predicates={'voice':lambda r:r['screen_decision']=='keep' and r['listener_human_evidence'],'review':lambda r:r['screen_decision']=='review','suspect_noise':lambda r:r['screen_decision']=='suspect_noise','quiet':lambda r:r['screen_decision']=='keep' and not r['listener_human_evidence'],'all':lambda r:True}
 for category,predicate in predicates.items():
  page.locator('#filter').select_option(category);n=sum(predicate(r) for r in samples)
  assert page.locator('#list .item').count()==n,(category,n)
  assert page.locator('#empty').is_visible()==(n==0)
 page.locator('#filter').select_option('voice' if any(predicates['voice'](r) for r in samples) else 'all')
 page.evaluate('v.play()');page.wait_for_function('v.currentTime>.4');page.evaluate('v.pause();v.currentTime=1')
 for mode in ['listener','speaker','both']:
  page.locator('#channel').select_option(mode);page.wait_for_function('Math.abs(v.currentTime-1)<.12')
  assert page.evaluate('v.src').endswith(mode+'.mp4')
 if page.locator('#events button').count():
  page.locator('#events button').first.click();page.wait_for_function('!v.paused');assert page.locator('#loop').is_checked()
  page.wait_for_timeout(800)
  assert page.evaluate('v.currentTime<=interval[1]+.3')
 page.evaluate('v.pause()');page.locator('#decision').select_option('real_reaction');page.locator('#note').fill('browser validation')
 assert page.evaluate('notes[current.sample_id].note')=='browser validation'
 page.reload();page.wait_for_function('samples.length>0');assert page.locator('#note').input_value()=='browser validation'
 page.locator('#note').fill('');page.locator('#decision').select_option('')
 with page.expect_download() as d:page.locator('#export').click()
 assert d.value.suggested_filename=='listen_qc2_review_notes.json'
 page.screenshot(path=str(out/'browser_desktop.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 page.screenshot(path=str(out/'browser_mobile.png'),full_page=True)
 response=page.request.get(page.locator('#download').get_attribute('href'),headers={'Range':'bytes=0-127'})
 assert response.status==206 and len(response.body())==128
 assert not errors,errors
 browser.close()
(out/'browser_validation.json').write_text(json.dumps(dict(samples=len(samples),playback=True,switch_preserves_time=True,event_loop=True,notes_persistence=True,notes_export=True,decision_filter=True,empty_group=True,mobile_layout=True,range_requests=True,page_errors=errors),indent=2)+'\n');print('browser PASS')
