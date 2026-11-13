import asyncio,json,os
from pathlib import Path
from playwright.async_api import async_playwright
OUT=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_native_caption_nod_direct50_20261030')
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
  page=await browser.new_page(viewport={'width':1450,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto('http://localhost:8876/listen_benchmark_review/native_simple/natural_caption_nod_direct/',wait_until='networkidle')
  await page.wait_for_function('window.reviewAudioState && window.reviewAudioState().ready')
  assert await page.locator('#priorPanel').is_hidden()
  assert '不输入旧段落' in await page.locator('#pageIntro').inner_text()
  await page.click('#soundPlay');await page.wait_for_timeout(900)
  play=await page.evaluate('({state:window.reviewAudioState(),videoTime:document.getElementById("video").currentTime,videoWidth:document.getElementById("video").videoWidth})')
  await page.locator('#video').evaluate('(v)=>v.pause()')
  nod_count=await page.locator('[data-nod-id]').count();assert nod_count>0
  await page.locator('[data-nod-id]').first.click();await page.wait_for_timeout(350)
  nod_seek=await page.locator('#video').evaluate('(v)=>v.currentTime');assert nod_seek>1
  await page.locator('#video').evaluate('(v)=>v.pause()')
  digests=await page.evaluate('''async()=>{let data=await (await fetch('review.json')).json(),results=[];for(let s of data.samples){let pcm=await window.reviewProcessedPCM(s,new AbortController().signal),prefix=new TextEncoder().encode(`float32le:${s.audio.sample_rate}:(${pcm.length}, 1)`),b=new Uint8Array(prefix.length+pcm.byteLength);b.set(prefix);b.set(new Uint8Array(pcm.buffer),prefix.length);let h=[...new Uint8Array(await crypto.subtle.digest('SHA-256',b))].map(x=>x.toString(16).padStart(2,'0')).join('');results.push({id:s.id,pass:h===s.audio.processed_pcm_sha256});}return results;}''')
  await page.select_option('#filter','audible');audible=await page.locator('#sample option').count();await page.select_option('#filter','silent');silent=await page.locator('#sample option').count()
  await page.set_viewport_size({'width':390,'height':844});overflow=await page.evaluate('document.documentElement.scrollWidth>innerWidth')
  result={'playback':play,'audio_pcm_hash_checks':digests,'audible':audible,'silent':silent,'page_errors':errors,'mobile_overflow':overflow,'nod_buttons':nod_count,'nod_seek_time':nod_seek}
  (OUT/'browser_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(playback=play,hashes_passed=sum(x['pass'] for x in digests),audible=audible,silent=silent,errors=errors,mobile_overflow=overflow)))
  await browser.close()
asyncio.run(main())
