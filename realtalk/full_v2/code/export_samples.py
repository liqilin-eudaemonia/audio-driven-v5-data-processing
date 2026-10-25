from pathlib import Path
import json,hashlib,subprocess,html,collections
import soundfile as sf
R=Path(__file__).resolve().parents[1];O=R.parent/'full_20261023/reports/full_v2';M=O/'media';M.mkdir(parents=True,exist_ok=True)
from pipeline import atomic
buckets=collections.defaultdict(list)
for f in (R/'results').glob('*/*.json'):
 r=json.loads(f.read_text())
 if not all(k in r for k in ['raw_audio','optimized_audio','effective_crop']):continue
 s=r.get('optimized_sync',{});label='SyncNet通过' if s.get('robust_pass') else '短片段暂定' if r['frames']<=50 else 'SyncNet未通过或不可评分'
 buckets[label].append((hashlib.sha256(r['pair_id'].encode()).hexdigest(),r))
selected=[]
for label,rows in buckets.items():
 # Diverse sources, then fill remaining slots, deterministic samples.
 rows.sort(key=lambda x:x[0]);used=set();chosen=[]
 for _,r in rows:
  if r['video_id'] not in used:chosen.append(r);used.add(r['video_id'])
  if len(chosen)>=8:break
 for _,r in rows:
  if len(chosen)>=8:break
  if r not in chosen:chosen.append(r)
 selected.extend((label,r) for r in chosen)
def ff(args):subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-threads','1','-filter_threads','1',*map(str,args)],check=True)
cards=[];exports=[]
for label,r in selected:
 p=r['canonical_pair'];ident=p['pair_id'];c=r['effective_crop'];dur=r['frames']/25;source=R/'videos'/r['video_id']/'source_video.mp4';crop=M/(ident+'_crop.mp4');paths={}
 ff(['-ss',str(p['start']),'-i',source,'-frames:v',str(r['frames']),'-vf',f"crop={c['width']}:{c['height']}:{c['x']}:{c['y']}:exact=1,pad=ceil(iw/2)*2:ceil(ih/2)*2",'-an','-c:v','libx264','-preset','fast','-crf','18',crop])
 n=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','csv=p=0',str(crop)],text=True));assert n==r['frames']
 for mode in ['raw','optimized']:
  a,sr=sf.read(r[mode+'_audio']);assert sr==16000 and len(a)==r['frames']*640;wav=M/(ident+'_'+mode+'.wav');sf.write(wav,a*r['common_preview_gain'],16000,subtype='PCM_24');dest=M/(ident+'_'+mode+'.mp4')
  ff(['-i',crop,'-i',wav,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-t',str(dur),'-movflags','+faststart',dest]);paths[mode]='media/'+dest.name
 crop.unlink();cards.append(f'<article><h2>{html.escape(label)} · {html.escape(ident)}</h2><p>{p["start"]:.2f}–{p["end"]:.2f}s · speaker {p["speaker"]} → listener {p["listener"]} · 原生框 {c["width"]}px</p><div class="cols">'+''.join(f'<section><h3>{title}</h3><video controls preload="none" src="{paths[mode]}"></video><pre>{html.escape(json.dumps(r.get(mode+"_sync",{}),ensure_ascii=False,indent=2))}</pre></section>' for mode,title in [('raw','listener 分离音轨'),('optimized','listener 进一步优化音轨')])+'</div></article>');exports.append({'pair_id':ident,'category':label,'media':paths,'result_file':str(R/'results'/r['video_id']/(ident+'.json'))})
(O/'samples.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:17px sans-serif;max-width:1200px;margin:30px auto;padding:20px;background:#15232d;color:#eee}.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%}pre{white-space:pre-wrap}article{border:1px solid #666;margin:20px 0;padding:16px}a{color:cyan}</style><h1>新切分 · listener 音视频前后对照</h1><p>两列均为同一 listener 画面；左为模型分离音轨，右为进一步优化音轨。通过、未通过、短片段分别抽样。SyncNet 分数不是完全无串音的保证。</p><a href="index.html">全量进度</a>'+''.join(cards)+'<script>document.querySelectorAll("video").forEach(v=>v.onplay=()=>document.querySelectorAll("video").forEach(o=>{if(o!==v)o.pause()}))</script>');atomic(R/'sample_exports.json',exports);print('EXPORTED',len(exports))
