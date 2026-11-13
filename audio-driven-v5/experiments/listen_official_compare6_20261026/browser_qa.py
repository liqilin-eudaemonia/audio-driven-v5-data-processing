import json
from pathlib import Path
from playwright.sync_api import sync_playwright
r=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_official_compare6_20261026');url='http://127.0.0.1:8876/listen_benchmark_review/audio_official_compare_6/';errors=[];checks=[]
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1500,'height':1050});page.on('pageerror',lambda e:errors.append(str(e)));res=page.goto(url);assert res.status==200;page.wait_for_selector('[data-case="5"]')
 for i in range(6):
  page.locator(f'[data-case="{i}"]').click();page.wait_for_function('audio.readyState>=2');assert page.locator('.tracklabel').count()==7
  expected=[2,2,1,1,2,2][i];assert page.locator('#events tbody tr').count()==expected
  page.locator('#restart').click();page.wait_for_timeout(350);assert page.evaluate('audio.currentTime')>.1;assert page.evaluate('Math.abs(audio.currentTime-video.currentTime)')<.2
  page.locator('#speaker').click();page.wait_for_timeout(400);assert page.evaluate('role')=='speaker';assert not page.evaluate('audio.paused');page.evaluate('audio.pause()');page.locator('#listener').click();page.wait_for_function('audio.readyState>=2')
  page.locator('#events .exact').first.click();page.wait_for_timeout(1200)
  expected_end=page.evaluate('C.ours_audio_events[0].end');assert page.evaluate('audio.paused');assert abs(page.evaluate('audio.currentTime')-expected_end)<.15
  page.locator('#events .context').first.click();page.wait_for_timeout(100);assert '前后文' in page.locator('#range').inner_text();page.evaluate('audio.pause()')
  checks.append(dict(case=i,audio_events=expected,listener_and_speaker_playable=True,event_stop_verified=True))
 page.locator('[data-case="2"]').click();page.wait_for_function('audio.readyState>=2');page.screenshot(path=str(r/'review_desktop.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(250);assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=str(r/'review_mobile.png'),full_page=True)
 assert not errors,errors
 (r/'browser_check.json').write_text(json.dumps(dict(page_status=res.status,checks=checks,errors=errors,mobile_no_horizontal_overflow=True),indent=2));b.close()
print('PASS: 6 cases, 12 audio tracks, 7 timeline lanes, event playback, speaker/listener switch, mobile')
