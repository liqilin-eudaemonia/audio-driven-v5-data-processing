import json,subprocess,concurrent.futures,hashlib,math,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'review';(OUT/'media/pairs').mkdir(parents=True,exist_ok=True);m=json.loads((OUT/'samples.json').read_text());FONT='/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto';src=ROOT.parent/(m['video_id']+'.avi')
for r in m['pairs']:r['media']='media/pairs/'+r['pair_id']+'.mp4'
# Preserve the global partition; pair dictionaries are independently deserialized.
lookup={r['pair_id']:r for r in m['pairs']}
oldout=ROOT.parent/'full_video_v1/review'
old=json.loads((oldout/'samples.json').read_text())
def interval_key(r):return (r['start_frame'],r['end_frame'],json.dumps(r['crop'],sort_keys=True))
cache={interval_key(r):oldout/r['media'] for r in old['pairs']}
for r in m['pairs']:
 dst=OUT/r['media'];src_cached=cache.get(interval_key(r))
 if src_cached and not dst.exists():os.link(src_cached,dst)
if not (OUT/m['raw_media']).exists():os.link(oldout/old['raw_media'],OUT/m['raw_media'])
for r in m['segments']:
 if r['kind']=='pair':r['media']=lookup[r['pair_id']]['media']
(OUT/'samples.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
header='''[Script Info]
ScriptType: v4.00+
PlayResX: 960
PlayResY: 720
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans CJK SC,22,&H00FFFFFF,&H000000FF,&H0010191B,&H0010191B,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
events=[]
def ts(t):
 cent=round(t*100);return f'{cent//360000}:{cent//6000%60:02d}:{cent//100%60:02d}.{cent%100:02d}'
def emit(a,b,s,layer=1):events.append(f'Dialogue: {layer},{ts(a)},{ts(b)},Default,,0,0,0,,{s}')
def text(a,b,s,x,y,size=22,color='FFFFFF'):
 emit(a,b,r'{\an7\pos('+f'{x},{y}'+r')\fs'+str(size)+r'\1c&H'+color+r'&}'+s)
def rect(a,b,x,y,w,h,color,layer=0):
 emit(a,b,r'{\an7\pos(0,0)\p1\bord0\1c&H'+color+r'&}'+f'm {x} {y} l {x+w} {y} {x+w} {y+h} {x} {y+h}',layer)
D=m['duration'];rect(0,D,24,666,912,12,'564A41')
for r in m['segments']:
 a,b=r['start'],r['end'];x=24+round(912*a/D);w=max(1,round(912*b/D)-(x-24));kind=r['kind'];color='9FEF42' if kind=='pair' else '70D2FF' if kind=='role_unknown' else '767676';rect(0,D,x,666,w,12,color)
 if kind=='pair':
  text(a,b,f"{r['pair_id']}   speaker {r['speaker']} → listener {r['listener']}   镜头 {r['shot']}",18,8,22,'9FEF42')
  text(a,b,'当前候选保留区间 · 绿框为裁剪范围'+(' · 裁剪待确认' if not r['crop'] else ''),18,40,21,'9FEF42')
  c=r['crop']
  if c:
   xx=round(c['x']*.75);yy=76+round(c['y']*.75);sz=round(c['size']*.75)
   for x0,y0,w0,h0 in [(xx,yy,sz,3),(xx,yy+sz-3,sz,3),(xx,yy,3,sz),(xx+sz-3,yy,3,sz)]:rect(a,b,x0,y0,w0,h0,'9FEF42',1)
  suffix='句末/停顿优先；连续区间不丢帧' if r.get('sentence_revision') else ('短片段仍保留' if r['duration']<1 else '切镜头或角色变化后继续提取')
  text(a,b,f"本 pair：{a:.2f}–{b:.2f}s  /  {r['duration']:.2f}s    {suffix}",18,625,21)
 else:
  text(a,b,'未生成 pair：'+('黑屏区间' if kind=='black' else '说话角色未确定'),18,8,23,color)
  text(a,b,'此区间保留在时间轴中，可直接听看复核',18,40,21,color)
  text(a,b,f"区间 {a:.2f}–{b:.2f}s   "+('黑屏检测' if kind=='black' else '可能是停顿、非语音或角色漏标；不判为废片'),18,625,19)
for e in sorted(set(m['scene_cuts']+m['geometry_cuts'])):
 if 0<e<D:
  x=24+round(912*e/D);rect(0,D,x,662,1,20,'FFFFFF');text(max(0,e-.08),min(D,e+.32),f'镜头/布局边界 {e:.2f}s',290,583,23,'70D2FF')
text(0,D,'整片时间轴：绿色＝候选 pair   黄色＝角色待定   灰色＝黑屏   白线＝镜头边界',18,691,17)
emit(0,D,r'{\an7\move(18,646,930,646,0,'+str(round(D*1000))+r')\fs20\1c&HFFFFFF&}▼',3)
ass=ROOT/'full_overlay.ass';ass.write_text(header+'\n'.join(events)+'\n')
clock=f"drawtext=fontfile={FONT}/NotoSansCJK-Regular.ttc:text='原视频 %{{eif\\:floor(t)\\:d}}.%{{eif\\:mod(round(t*100)\\,100)\\:d\\:2}}s':x=730:y=42:fontsize=20:fontcolor=white"
jobs=[]
for r in m['pairs']:
 c=r['crop'];vf=f"crop={c['size']}:{c['size']}:{c['x']}:{c['y']},scale=512:512" if c else 'scale=960:540';jobs.append(dict(path=str(OUT/r['media']),start=r['start'],end=r['end'],filter=vf))
jobs.extend([dict(path=str(OUT/m['media']),start=0,end=D,filter=f"scale=960:540,pad=960:720:0:76:color=0x10191b,ass={ass}:fontsdir={FONT},"+clock),dict(path=str(OUT/m['raw_media']),start=0,end=D,filter='scale=960:540')])
def encode(j):
 dest=Path(j['path'])
 if dest.exists():return
 subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','1','-ss',str(j['start']),'-i',str(src),'-t',str(j['end']-j['start']),'-map','0:v:0','-map','0:a:0','-vf',j['filter']+',setsar=1','-r','25','-c:v','libx264','-preset','fast','-crf','20','-threads','2','-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart','-y',str(dest)],check=True)
 print(dest.name,flush=True)
# Start full-source renders first so they do not wait behind hundreds of small clips.
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(encode,jobs[-2:]+jobs[:-2]))
(ROOT/'jobs.json').write_text(json.dumps(jobs,indent=2))
contract={'manifest_sha256':hashlib.sha256((OUT/'samples.json').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'annotations_sha256':hashlib.sha256((ROOT.parent/(m['video_id']+'_annotations.json')).read_bytes()).hexdigest()};(OUT/'contract.json').write_text(json.dumps(contract,indent=2));print('published',len(m['pairs']),'pairs',flush=True)
