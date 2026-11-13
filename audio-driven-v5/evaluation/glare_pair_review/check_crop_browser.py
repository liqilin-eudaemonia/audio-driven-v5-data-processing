from pathlib import Path
import requests,websocket,json,base64,time
out=Path(__file__).resolve().parent
page=requests.put('http://127.0.0.1:19225/json/new?http://127.0.0.1:8876/glare_reaction_annotation/',timeout=10).json();ws=websocket.create_connection(page['webSocketDebuggerUrl'],timeout=50);seq=0;errors=[]
def cmd(method,params={}):
 global seq
 seq+=1;ws.send(json.dumps({'id':seq,'method':method,'params':params}))
 while True:
  v=json.loads(ws.recv())
  if v.get('method')=='Runtime.exceptionThrown':errors.append(v['params'])
  if v.get('id')==seq:
   if 'error' in v:raise RuntimeError(v['error'])
   return v.get('result',{})
def js(code):
 r=cmd('Runtime.evaluate',dict(expression=code,awaitPromise=True,returnByValue=True,userGesture=True))
 if 'exceptionDetails' in r:raise RuntimeError(r['exceptionDetails'])
 return r.get('result',{}).get('value')
cmd('Runtime.enable');cmd('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1200,deviceScaleFactor=1,mobile=False))
wait="""new Promise((resolve,reject)=>{let n=0;const t=setInterval(()=>{const v=[...document.querySelectorAll('.pair-video')];if(v.length===2&&v.every(x=>x.readyState>=2)){clearInterval(t);resolve(true);}else if(++n>180){clearInterval(t);reject(Error(document.querySelector('#pair-detail').innerText));}},150);})"""
js(wait)
def state():return js("({title:document.querySelector('#pair-detail h2').innerText,text:document.querySelector('#pair-detail').innerText,videos:[...document.querySelectorAll('.pair-video')].map(v=>({src:v.currentSrc,width:v.videoWidth,height:v.videoHeight,duration:v.duration,error:v.error?.message})),boxes:[...document.querySelectorAll('.crop-coordinates')].map(e=>e.innerText)})")
initial=state();assert all(v['width']==512 and v['height']==512 for v in initial['videos']),initial
assert all(abs(v['duration']-97/25)<.08 for v in initial['videos']);assert 'move in that movement' in initial['text']
js("document.querySelector('#pair-play').click();new Promise(r=>setTimeout(r,1000))")
play=js("[...document.querySelectorAll('.pair-video')].map(v=>({time:v.currentTime,paused:v.paused,muted:v.muted}))");assert all(v['time']>.2 for v in play) and abs(play[0]['time']-play[1]['time'])<.3,play
js("document.querySelector('#pair-pause').click();let n=document.querySelector('#pair-frame-number');n.value=32;n.dispatchEvent(new Event('change'));new Promise(r=>setTimeout(r,250))")
frame=js("({label:document.querySelector('#frame-index').innerText,values:document.querySelector('#frame-values').innerText,times:[...document.querySelectorAll('.pair-video')].map(v=>v.currentTime)})");assert '第 32 帧' in frame['label'] and all(abs(t-1.28)<.05 for t in frame['times']),frame
js("document.querySelectorAll('.original-view').forEach(d=>d.open=true);document.querySelector('#pair-detail h2').scrollIntoView();new Promise(r=>setTimeout(r,250))")
rect=js("[...document.querySelectorAll('.source-stage rect')].map(e=>[e.getAttribute('x'),e.getAttribute('y'),e.getAttribute('width'),e.getAttribute('height')])");assert all(float(r[2])>100 for r in rect),rect
(out/'crop_seamless_browser.png').write_bytes(base64.b64decode(cmd('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data']))
js("let s=document.querySelector('#pair-variant');s.value='rectangle_1p5_resize360';s.dispatchEvent(new Event('change'))")
js(wait);alternative=state();assert all('矩形' in b or '扩大' in b for b in alternative['boxes'])
js("document.querySelector('[data-dataset=realtalk][data-example]').click()")
js(wait);rt=state();assert '57.560' in rt['text'] and 'thin air' in rt['text'],rt
assert '[3, 103, 368, 468]' in rt['boxes'][0];assert '[925, 106, 1245, 426]' in rt['boxes'][1]
assert all(abs(v['duration']-147/25)<.08 for v in rt['videos']),rt
js("document.querySelector('#pair-detail h2').scrollIntoView();new Promise(r=>setTimeout(r,300))")
(out/'crop_realtalk_browser.png').write_bytes(base64.b64decode(cmd('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data']))
js("document.querySelector('[data-example^=V00_S1127]').click()")
js(wait);second=state();assert 'Tomorrow night.' in second['text'] and 'ASR 词记录' in second['text']
response=requests.get(initial['videos'][0]['src'],headers={'Range':'bytes=0-1023'},timeout=10);assert response.status_code==206 and len(response.content)==1024
assert not errors,errors
report=dict(initial=initial,play=play,frame=frame,rectangles=rect,alternative=alternative['boxes'],realtalk=rt,second=second['title'],http_range=response.status_code,runtime_errors=errors)
(out/'crop_browser_verified.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(dict(ok=True,initial=initial['title'],boxes=initial['boxes'],realtalk_boxes=rt['boxes'],frame=frame,http_range=response.status_code),ensure_ascii=False));ws.close()
