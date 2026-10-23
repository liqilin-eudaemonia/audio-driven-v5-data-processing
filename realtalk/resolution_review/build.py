import json, subprocess, concurrent.futures, random, html
from pathlib import Path
ROOT=Path('/private/liqi_workspace/realtalk_processing/full_20261023'); OUT=ROOT/'reports/resolution_review';OUT.mkdir(exist_ok=True);(OUT/'media').mkdir(exist_ok=True)
bounds=[128,160,192,224,256,320,384,448,512,640,768,1024];groups=[[] for _ in bounds[:-1]];unknown=[]
for f in sorted((ROOT/'manifests').glob('*.json')):
 m=json.loads(f.read_text())
 for r in m['pairs']:
  if not r['crop']:continue
  e=r['crop']['native_edge']
  for i,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
   if lo<=e<hi:
    groups[i].append({'video_id':m['video_id'],'source':m['source']['source'],'pair':r,'source_size':[m['width'],m['height']],'bin':f'{lo}–{hi}px'})
    break
chosen=[];rng=random.Random(20261023)
for group in groups:
 rng.shuffle(group);used=set();picks=[]
 # Favor reviewable duration and complete faces, then fill sparse bins without dropping the bin.
 group.sort(key=lambda x:(x['pair']['duration']<2,x['pair']['face']['valid_frames']/max(1,x['pair']['end_frame']-x['pair']['start_frame'])<.95))
 for distinct in [True,False]:
  for x in group:
   if len(picks)>=3:break
   if x in picks or distinct and x['video_id'] in used:continue
   picks.append(x);used.add(x['video_id'])
 chosen+=picks
for i,x in enumerate(chosen):x['sample_id']=f'{i+1:02d}'
(OUT/'samples.json').write_text(json.dumps({'selection':'seed=20261023; 3 per occupied native crop bin; distinct video preferred; duration>=2 and face coverage>=95% preferred; no quality or flags exclusion; clips capped at6s for review','samples':chosen},ensure_ascii=False,indent=2))
def render(x):
 r=x['pair'];c=r['crop'];sid=x['sample_id'];dur=min(6,r['duration']);native=OUT/'media'/f'{sid}_native.mp4';review=OUT/'media'/f'{sid}_review.mp4'
 if native.exists() and review.exists():return sid
 vf=f"[0:v]fps=25,split=2[a][b];[a]drawbox=x={c['x']}:y={c['y']}:w={c['width']}:h={c['height']}:color=lime:t=4,scale=768:432:force_original_aspect_ratio=decrease,pad=768:512:(ow-iw)/2:(oh-ih)/2[orig];[b]crop={c['width']}:{c['height']}:{c['x']}:{c['y']}:exact=1,split=2[n][p];[n]pad=ceil(iw/2)*2:ceil(ih/2)*2[native];[p]scale=512:512[preview];[orig][preview]hstack[review]"
 cmd=['ffmpeg','-nostdin','-v','error','-threads','1','-filter_complex_threads','1','-ss',str(r['start']),'-i',x['source'],'-filter_complex',vf]
 for label,path in [('native',native),('review',review)]:cmd+=['-map',f'[{label}]','-map','0:a:0?','-t',str(dur),'-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-movflags','+faststart','-y',str(path)]
 p=subprocess.run(cmd,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(sid+':'+p.stderr[-2000:])
 return sid
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for sid in pool.map(render,chosen):print('rendered',sid,flush=True)
cards=[]
for x in chosen:
 r=x['pair'];c=r['crop'];sid=x['sample_id'];face=r['face']['short_side'];coverage=r['face']['valid_frames']/(r['end_frame']-r['start_frame']);flags=html.escape(', '.join(r['flags']) or '无额外标记')
 cards.append(f'''<article data-bin="{x['bin']}"><h2>{sid} · {x['bin']} · 原生裁剪 {c['native_edge']}×{c['native_edge']}px</h2><p>{html.escape(x['video_id'])} · listener {r['listener']} · {r['start']:.2f}–{r['end']:.2f}s · 完整pair {r['duration']:.2f}s（展示前{min(6,r['duration']):.2f}s）</p><p>人脸短边 P10 / 中位数：{face['p10']} / {face['median']}px · 有效框覆盖 {coverage:.1%} · 原画 {x['source_size'][0]}×{x['source_size'][1]}</p><video controls preload="none" src="media/{sid}_review.mp4"></video><p>左：原画与绿色裁剪框；右：该裁剪统一显示为512px。<a href="media/{sid}_native.mp4" target="_blank">打开原生尺寸裁剪</a>（仅为编码偶数尺寸最多补1px，不缩放）</p><p>复核标记：{flags}</p><label>审核意见 <textarea placeholder="例如：清晰度可接受／框偏移／脸太小／建议阈值" data-id="{sid}"></textarea></label></article>''')
opts=''.join(f'<option>{bounds[i]}–{bounds[i+1]}px</option>' for i in range(len(bounds)-1))
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RealTalk 分辨率分档审核</title><style>@font-face{font-family:Review;src:url(review.woff2)}body{background:#111b24;color:#e5edf4;font:16px/1.65 Review,sans-serif;max-width:1320px;margin:auto;padding:24px}a{color:#74d6e7}article{background:#1c2b38;border:1px solid #395065;border-radius:12px;padding:20px;margin:24px 0}h2{font-size:21px}video{width:100%;max-width:1280px}textarea{display:block;width:96%;min-height:60px;background:#101b25;color:white;padding:12px}select,button,textarea{font-family:inherit}select,button{padding:10px;margin:10px}nav{position:sticky;top:0;background:#111b24;padding:8px;z-index:2}</style><h1>RealTalk 原生裁剪分辨率审核</h1><p>每个有数据的预裁剪框尺寸档抽3条，共33条；优先不同源视频及≥2秒、有效脸框覆盖≥95%的片段，稀少档按实际情况补齐。固定随机种子，不按画质挑选。预裁剪框大小不等于人脸大小，因此同时展示人脸短边。当前不应用阈值。</p><p>视频将原画、裁剪框和裁剪结果放在同一画面同步播放，保留原混合音轨（未做人声分离）。每条展示最多6秒；这是清晰度抽样，不代表完整pair或该档所有数据质量。无可靠预裁剪框的8.56小时不强行分档。128px以下和1024px以上没有已知框候选。审核意见保存在当前浏览器，可导出JSON。</p><nav>分辨率 <select id="filter"><option value="all">全部</option>'''+opts+'''</select><button id="export">导出审核意见</button><a href="samples.json">抽样清单</a> · <a href="../">全量统计</a></nav>'''+''.join(cards)+'''<script>const key='realtalk-resolution-review-20261023';let notes=JSON.parse(localStorage.getItem(key)||'{}');document.querySelectorAll('textarea').forEach(t=>{t.value=notes[t.dataset.id]||'';t.oninput=()=>{notes[t.dataset.id]=t.value;localStorage.setItem(key,JSON.stringify(notes))}});document.querySelector('#filter').onchange=e=>document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.bin!==e.target.value);document.querySelector('#export').onclick=()=>{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(notes,null,2)],{type:'application/json'}));a.download='resolution_review_notes.json';a.click();URL.revokeObjectURL(a.href)}</script></html>'''
(OUT/'index.html').write_text(page)
print('DONE',len(chosen),flush=True)
