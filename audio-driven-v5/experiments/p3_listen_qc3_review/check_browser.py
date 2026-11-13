import sys,json
from pathlib import Path
sys.path.insert(0,'/tmp/listen_preview_browser')
from playwright.sync_api import sync_playwright
out=Path('/private/liqi_workspace/project_code/audio-driven-v5/data_previews/seamless_listen_qc3_review');samples=json.loads((out/'samples.json').read_text())['samples'];errors=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1100});page.set_default_timeout(60000);page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:8875/listen_qc3/',wait_until='domcontentloaded');page.wait_for_function('typeof samples!=="undefined"&&samples.length>0')
 for k in ['voice','previous','exclude','review','controls','quiet','all']:
  page.locator('#filter').select_option(k);n=page.evaluate('samples.filter(r=>matches(r,$("#filter").value)).length')
  assert page.locator('#list .item').count()==n;assert page.locator('#empty').is_visible()==(n==0)
 page.locator('#filter').select_option('controls');assert page.locator('#list .item').count()==3;assert '用户已确认正常回应' in page.locator('#badge').inner_text()
 page.locator('#filter').select_option('previous');assert page.locator('#list .item').count()==16;assert 'S0065' in page.locator('#sid').inner_text();assert '建议排除' in page.locator('#change').inner_text()
 page.locator('#filter').select_option('voice');page.wait_for_function('v.readyState>=2');page.evaluate('v.play()');page.wait_for_function('v.currentTime>.4');page.evaluate('v.pause();v.currentTime=1')
 for mode in ['both','speaker','listener']:
  page.locator('#channel').select_option(mode);page.wait_for_function('Math.abs(v.currentTime-1)<.12');assert page.evaluate('v.src').endswith(mode+'.mp4')
 page.locator('#events button').first.click();page.wait_for_function('!v.paused');assert page.locator('#loop').is_checked();page.evaluate('v.pause()')
 page.locator('#decision').select_option('clean_reaction');page.locator('#note').fill('browser verification')
 page.reload();page.wait_for_function('samples.length>0');assert page.locator('#note').input_value()=='browser verification'
 page.locator('#note').fill('');page.locator('#decision').select_option('')
 with page.expect_download() as download:page.locator('#export').click()
 assert download.value.suggested_filename=='listen_qc3_notes.json'
 page.wait_for_function('v.readyState>=2');page.screenshot(path=str(out/'browser_desktop.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=str(out/'browser_mobile.png'),full_page=True)
 response=page.request.get(page.locator('#download').get_attribute('href'),headers={'Range':'bytes=0-127'});assert response.status==206 and len(response.body())==128
 page.goto('http://127.0.0.1:8875/listen_qc3/annotation.html');page.wait_for_function('rows.length>0');assert page.evaluate('rows.length')==240
 page.locator('#audio').evaluate('(a)=>a.play()');page.wait_for_function('document.querySelector("audio").currentTime>.2');page.locator('#audio').evaluate('(a)=>a.pause()')
 page.locator('#label').select_option('uncertain');page.locator('#note').fill('annotation UI verification');page.locator('#next').click();page.locator('#prev').click();assert page.locator('#note').input_value()=='annotation UI verification';page.locator('#label').select_option('');page.locator('#note').fill('')
 with page.expect_download() as download:page.locator('#export').click()
 assert download.value.suggested_filename=='listen_qc3_event_labels.json'
 assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 page.goto('http://127.0.0.1:8875/listen_qc3/known_examples.html');assert page.locator('audio').count()==11
 page.locator('audio').nth(7).evaluate('(a)=>a.play()');page.wait_for_function('document.querySelectorAll("audio")[7].currentTime>.2');page.locator('audio').nth(7).evaluate('(a)=>a.pause()')
 assert not errors,errors;browser.close()
(out/'browser_validation.json').write_text(json.dumps(dict(samples=len(samples),playback=True,channel_switch=True,event_loop=True,filter=True,positive_controls=3,previous_clips=16,notes=True,notes_export=True,annotation_queue=240,annotation_audio_playback=True,annotation_notes=True,known_examples_playback=True,range_requests=True,mobile_layout=True,page_errors=errors),indent=2)+'\n');print('browser PASS')
