import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parent;URL='http://127.0.0.1:8876/p2_av_review/'
errors=[];result={}
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'],ignore_default_args=['--mute-audio'])
 page=b.new_page(viewport={'width':1440,'height':1080});page.on('pageerror',lambda e:errors.append(str(e)));page.goto(URL);page.wait_for_function('window.reviewReady===true')
 assert page.locator('.sample').count()==24
 assert page.locator('#metric-rows tr').count()==8
 assert page.locator('.asr').all_text_contents() and all(t not in ('待补充','') for t in page.locator('.asr').all_text_contents())
 assert page.locator('button',has_text='下一条').count()==0
 result['train_samples']=24;result['train_asr']=24;result['metric_rows_train']=8
 gt=page.locator('.gt-video').first;gt.scroll_into_view_if_needed();gt.evaluate('(v)=>v.play()');page.wait_for_function("document.querySelector('.gt-video').currentTime > 0.25")
 result['gt_playback']=gt.evaluate('(v)=>({muted:v.muted,volume:v.volume,readyState:v.readyState,audioBytes:v.webkitAudioDecodedByteCount,videoWidth:v.videoWidth})');assert result['gt_playback']['muted'] is False and result['gt_playback']['volume']==1;assert result['gt_playback']['audioBytes']>0
 speaker=page.locator('.speaker-area > audio').first;speaker.evaluate('(a)=>a.play()');page.wait_for_timeout(200);assert gt.evaluate('(v)=>v.paused')
 gen=page.locator('video[data-listener]').first;gen.evaluate('(v)=>v.play()');page.wait_for_function("document.querySelector('video[data-listener]').currentTime > 0.2");assert speaker.evaluate('(v)=>v.paused')
 gen.evaluate('(v)=>{v.pause();v.currentTime=1.25}');page.wait_for_timeout(100);page.locator('[data-mode=mixed]').click();page.wait_for_function("(()=>{let v=document.querySelector('video[data-listener]');return v.readyState>=1&&Math.abs(v.currentTime-1.25)<.1&&v.currentSrc.endsWith(v.dataset.mixed)})()")
 result['seek_preserved_on_mix']=gen.evaluate('(v)=>v.currentTime');assert gen.evaluate('(v)=>v.paused && !v.muted')
 gen.evaluate('(v)=>v.play()');page.wait_for_timeout(250);page.locator('[data-mode=listener]').click();page.wait_for_function("(()=>{let v=document.querySelector('video[data-listener]');return !v.paused && v.currentSrc.endsWith(v.dataset.listener) && v.currentTime>1})()")
 result['playback_continues_on_switch']=True;page.locator('#pause-all').click();assert page.locator('video:visible').evaluate_all('(vs)=>vs.every(v=>v.paused)')
 gen.evaluate('(v)=>{v.currentTime=1.1}');page.evaluate("changeMode('mixed');changeMode('listener');changeMode('mixed');pauseAll()");page.wait_for_function("(()=>{let v=document.querySelector('video[data-listener]');return v.readyState>=1&&Math.abs(v.currentTime-1.1)<.1&&v.paused})()");result['rapid_switch_and_pause']=True
 page.locator('#seed').select_option('all');assert page.locator('.pipeline-card[data-variant=fixed]').count()==192
 page.locator('#seed').select_option('137');page.locator('#variant').select_option('guided');assert page.locator('.sample').count()==3;assert page.locator('.pipeline-card').count()==9
 result['guided_outputs']=9;page.locator('.sample').first.scroll_into_view_if_needed();page.screenshot(path=str(OUT/'guided_preview.png'))
 page.locator('#seed').select_option('251');assert page.locator('.sample').count()==0;assert '没有已完成输出' in page.locator('#samples').inner_text()
 page.locator('#variant').select_option('all');page.locator('#seed').select_option('137');page.locator('[data-dataset=test]').click();assert page.locator('.sample').count()==24;assert page.locator('#metric-rows tr').count()==4
 assert all('未用于训练' in t for t in page.locator('.gt h3').all_text_contents());assert all(t!='待补充' for t in page.locator('.asr').all_text_contents())
 result['test_samples']=24;result['test_asr']=24;result['metric_rows_test']=4
 last=page.locator('.sample').last;last.scroll_into_view_if_needed();assert last.is_visible();result['last_sample_scrollable']=True
 page.locator('[data-dataset=train]').click();page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(OUT/'desktop_preview.png'))
 mobile=b.new_page(viewport={'width':390,'height':844},is_mobile=True);mobile.on('pageerror',lambda e:errors.append(str(e)));mobile.goto(URL);mobile.wait_for_function('window.reviewReady===true');assert mobile.evaluate('document.documentElement.scrollWidth <= innerWidth+1');mobile.screenshot(path=str(OUT/'mobile_preview.png'));result['mobile_no_horizontal_overflow']=True
 b.close()
assert not errors,errors
result.update(status='passed',page_errors=errors)
(OUT/'browser_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
