import json
from pathlib import Path
from playwright.sync_api import sync_playwright
r=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_official_random100_20261026');url='http://127.0.0.1:8876/listen_benchmark_review/audio_official_random100/';errors=[];decode_checks=[]
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1500,'height':1050});page.on('pageerror',lambda e:errors.append(str(e)));response=page.goto(url);assert response.status==200;page.wait_for_selector('[data-case="99"]');assert page.locator('#overview tbody tr').count()==100
 data=page.request.get(url+'review.json').json()
 for i,c in enumerate(data['cases']):
  page.evaluate('(i)=>select(i)',i);page.wait_for_function('audio.readyState>=2');assert page.locator('.tracklabel').count()==8
  assert abs(page.evaluate('audio.duration')-c['duration_s'])<.001
  # Decode speaker too, without waiting for full playback of all 100.
  page.locator('#speaker').click();page.wait_for_function('audio.readyState>=2');assert page.evaluate('role')=='speaker';assert abs(page.evaluate('audio.duration')-c['duration_s'])<.001
  page.locator('#listener').click();page.wait_for_function('audio.readyState>=2')
  if i in [0,24,49,74,99]:
   page.locator('#restart').click();page.wait_for_timeout(300);assert page.evaluate('audio.currentTime')>.1;assert page.evaluate('Math.abs(audio.currentTime-video.currentTime)')<.2;page.evaluate('audio.pause()')
  decode_checks.append(dict(sample_id=c['sample_id'],listener_and_speaker_decode=True,duration_matches=True))
 # Check exact VAD/gap buttons and pending/reviewed export workflow.
 page.evaluate('select(0)');page.wait_for_function('audio.readyState>=2');page.locator('#intervalButtons button').first.click();page.wait_for_timeout(250);assert page.evaluate('stopAt') is not None;page.evaluate('audio.pause()')
 page.locator('#inside').select_option('uncertain');page.locator('#outside').select_option('uncertain');page.locator('#reviewNote').fill('QA only temporary');assert '1 / 100' in page.locator('#reviewProgress').inner_text()
 page.locator('#filter').select_option('reviewed');assert page.locator('#overview tbody tr').count()==1
 page.locator('#filter').select_option('no_vad');assert page.locator('#overview tbody tr').count()==59
 page.locator('#filter').select_option('all');page.locator('#next').click();assert page.evaluate('idx')==1;assert page.locator('#reviewNote').input_value()=='';page.locator('#prev').click();assert page.locator('#reviewNote').input_value()=='QA only temporary'
 with page.expect_download() as x:page.locator('#exportReview').click()
 exported=json.loads(Path(x.value.path()).read_text());assert len(exported['records'])==1 and exported['records'][0]['note']=='QA only temporary';assert exported['training_applied'] is False
 page.evaluate('localStorage.clear()');page.reload();page.wait_for_function('audio.readyState>=2');assert '0 / 100' in page.locator('#reviewProgress').inner_text();page.screenshot(path=str(r/'review_desktop.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200);assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=str(r/'review_mobile.png'),full_page=True)
 assert not errors,errors
 (r/'browser_check.json').write_text(json.dumps(dict(page_status=200,cases=100,decoded_tracks=200,decode_checks=decode_checks,play_sync_checked_indices=[0,24,49,74,99],feedback_export_verified=True,filters_verified=True,local_storage_roundtrip_verified=True,errors=errors,mobile_no_horizontal_overflow=True),indent=2));b.close()
print('PASS: 100 cases, 200 audio decodes, sync, review persistence/export, filters, mobile')
