from pathlib import Path
import requests,websocket,json,base64,time
out=Path(__file__).resolve().parent
page=requests.put('http://127.0.0.1:19223/json/new?http://127.0.0.1:8876/glare_reaction_annotation/',timeout=10).json()
ws=websocket.create_connection(page['webSocketDebuggerUrl'],timeout=40);seq=0;errors=[]
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
wait="""new Promise((resolve,reject)=>{let n=0;const t=setInterval(()=>{const v=[...document.querySelectorAll('.pair-video')];if(v.length===2&&v.every(x=>x.readyState>=1)){clearInterval(t);resolve(true);}else if(++n>100){clearInterval(t);reject(Error(document.querySelector('#pair-detail').innerText));}},150);})"""
js(wait)
initial=js("({title:document.querySelector('#pair-detail h2').innerText,count:document.querySelector('#pair-count').innerText,videos:[...document.querySelectorAll('.pair-video')].map(v=>({src:v.currentSrc,width:v.videoWidth,height:v.videoHeight,duration:v.duration,error:v.error?.message})),training:document.querySelector('#pair-detail .missing').innerText})")
assert len(initial['videos'])==2 and all(v['width']>0 for v in initial['videos'])
assert '未公开' in initial['training']
js("document.querySelector('#pair-play').click();new Promise(r=>setTimeout(r,1500))")
play=js("[...document.querySelectorAll('.pair-video')].map(v=>({time:v.currentTime,paused:v.paused,muted:v.muted}))")
assert all(v['time']>.2 for v in play),play
assert abs(play[0]['time']-play[1]['time'])<.3,play
js("document.querySelector('#pair-pause').click()")
image=cmd('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False));(out/'pairs_browser.png').write_bytes(base64.b64decode(image['data']))
js("document.querySelector('[data-dataset=realtalk][data-example]').click();new Promise(r=>setTimeout(r,1800))")
rt=js("({text:document.querySelector('#pair-detail').innerText.slice(0,700),mixed:document.querySelectorAll('.mixed-line').length})")
assert '无法确定' in rt['text'] and rt['mixed']>0,rt
js("document.querySelector('[data-example^=V00_S1127]').click()")
js(wait)
second=js("({title:document.querySelector('#pair-detail h2').innerText,text:document.querySelector('#pair-detail').innerText})")
assert 'Tomorrow night.' in second['text'];assert 'ASR 词记录' in second['text']
assert not errors,errors
report={'initial':initial,'play':play,'realtalk':rt,'second_example':second['title'],'runtime_errors':errors}
(out/'browser_verified.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,ensure_ascii=False),flush=True)
ws.close()
