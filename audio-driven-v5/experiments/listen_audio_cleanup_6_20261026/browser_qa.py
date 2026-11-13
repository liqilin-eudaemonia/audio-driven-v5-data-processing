import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
base='http://127.0.0.1:8876/listen_benchmark_review/audio_cleanup_6/'
r=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_audio_cleanup_6_20261026');errors=[];checks=[]
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
 response=page.goto(base);assert response.status==200;page.wait_for_selector('[data-method="av"]');page.wait_for_function('document.querySelector("audio").readyState>=1')
 for i in range(6):
  page.locator(f'[data-case="{i}"]').click();page.wait_for_function('document.querySelector("audio").readyState>=1')
  for m in ['original','df_gentle','df_full','av']:
   page.locator(f'[data-method="{m}"]').click();page.wait_for_function('document.querySelector("audio").readyState>=2');page.locator('#restart').click();page.wait_for_timeout(450)
   a=page.evaluate('''()=>({time:audio.currentTime,duration:audio.duration,paused:audio.paused,error:audio.error?.message,video:video.currentTime,ready:audio.readyState})''');assert a['time']>.1 and not a['error'],(i,m,a)
   assert abs(a['video']-a['time'])<.2,(i,m,a)
   page.evaluate('audio.pause()');checks.append(dict(case=i,method=m,**a))
  page.locator('#removed').click();page.wait_for_function('document.querySelector("audio").readyState>=2');assert page.locator('#compareLabel').inner_text().endswith('移除部分');page.locator('#restart').click();page.wait_for_timeout(300);assert page.evaluate('audio.currentTime')>.1;page.evaluate('audio.pause()')
 # Switching methods retains time and playing state.
 page.locator('[data-case="0"]').click();page.wait_for_function('audio.readyState>=2');page.evaluate('audio.currentTime=1.5');page.locator('[data-method="df_full"]').click();page.wait_for_function('audio.readyState>=2');assert abs(page.evaluate('audio.currentTime')-1.5)<.05
 page.locator('#restart').click();page.wait_for_timeout(500);t=page.evaluate('audio.currentTime');page.locator('[data-method="av"]').click();page.wait_for_timeout(500);assert page.evaluate('audio.currentTime')>t;page.evaluate('audio.pause()');page.locator('[data-method="df_gentle"]').click();page.wait_for_function('audio.readyState>=2')
 # Feedback is per sample and method, and exported as actual JSON.
 page.locator('#noise').select_option('better');page.locator('#retention').select_option('uncertain');page.locator('#note').fill('QA temporary');page.locator('[data-method="av"]').click();assert page.locator('#note').input_value()=='';page.locator('[data-method="df_gentle"]').click();assert page.locator('#note').input_value()=='QA temporary'
 with page.expect_download() as d:page.locator('#export').click()
 data=json.loads(Path(d.value.path()).read_text());assert data['records'][0]['note']=='QA temporary'
 page.evaluate('localStorage.clear()');page.reload();page.wait_for_function('audio.readyState>=2');page.screenshot(path=str(r/'review_desktop.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(400);assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=str(r/'review_mobile.png'),full_page=True)
 d=page.request.get(base+'review.json').json();fn=d['cases'][0]['methods']['original']['file'];rr=page.request.get(base+fn,headers={'Range':'bytes=0-99'});assert rr.status==206 and len(rr.body())==100
 assert not errors,errors
 (r/'browser_check.json').write_text(json.dumps(dict(page_status=response.status,audio_checks=checks,errors=errors,range_status=rr.status,mobile_no_horizontal_overflow=True,switch_preserves_time=True,feedback_export_verified=True),indent=2));b.close()
print('PASS: 24 versions decode/play, video sync, residual, switch time, feedback/export, mobile, HTTP range')
