"""Target 10s, strict 11s cap, sentence-preferred partition with zero lost frames."""
import json,copy,hashlib,re,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'full_video_v1/review';OUT=ROOT/'review';FPS=25;MAX=275
m=json.loads((OLD/'samples.json').read_text());oldpairs=m['pairs'];merged=[]
for original in m['segments']:
 r=copy.deepcopy(original);r.pop('media',None)
 if r['kind']=='pair':r['old_pair_ids']=[r['pair_id']]
 if merged and r['kind']=='pair' and merged[-1]['kind']=='pair' and merged[-1]['end_frame']==r['start_frame'] and merged[-1]['shot']==r['shot'] and merged[-1]['speaker']==r['speaker']:
  q=merged[-1];assert q['crop']==r['crop'];q['end_frame']=r['end_frame'];q['end']=r['end'];q['duration']=(q['end_frame']-q['start_frame'])/FPS;q['old_pair_ids'].extend(r['old_pair_ids']);q['flags']=list(dict.fromkeys(q['flags']+r['flags']))
 else:merged.append(r)
asrs=[json.loads(p.read_text()) for p in sorted(ROOT.glob('w*_asr.json'))];hints=json.loads((ROOT/'semantic_boundary_hints.json').read_text());wave=np.load(ROOT.parent/'pilot_v2'/(m['video_id']+'_mono16k.npy'));ann=json.loads((ROOT.parent/(m['video_id']+'_annotations.json')).read_text());changes=[];segments=[]
for run in merged:
 if run['kind']!='pair' or run['end_frame']-run['start_frame']<=MAX:
  segments.append(run);continue
 a,b=run['start_frame'],run['end_frame'];d=next(q for q in asrs if q['window']['continuous_interval']==[run['start'],run['end']]);off=d['window']['start'];words=[dict(start=round(off+w['timestamp'][0],4),end=round(off+w['timestamp'][1],4),text=w['text'].strip()) for w in d['chunks'] if None not in w['timestamp']]
 candidates={}
 def add(frame,reason,cost,evidence):
  if a<frame<b and (frame not in candidates or cost<candidates[frame]['cost']):candidates[frame]=dict(reason=reason,cost=cost,evidence=evidence,time=frame/FPS)
 for w in words:
  frame=math.ceil(w['end']*FPS-1e-7)
  add(frame,'word_boundary',6,{'word':w,'rounding_seconds':round(frame/FPS-w['end'],4)})
  if re.search(r'[.!?]["\u201d\u2019\']?$',w['text']):add(frame,'asr_sentence_end',0,{'word':w,'rounding_seconds':round(frame/FPS-w['end'],4)})
 for h in hints:
  if h['asr_file']==d['window']['window_id']+'_asr.json':add(math.ceil(h['time']*FPS-1e-7),'semantic_sentence_end',0,h)
 for prev,nxt in zip(words,words[1:]):
  if nxt['start']-prev['end']>=.12:
   fs=range(max(a+1,math.ceil(prev['end']*FPS)),min(b-1,math.floor(nxt['start']*FPS))+1)
   def energy(i):
    x=wave[max(0,round((i/FPS-.02)*16000)):round((i/FPS+.02)*16000)].astype(float);return float(np.mean(x*x)) if len(x) else 1.
   if fs:
    f=min(fs,key=energy);add(f,'natural_pause',2,{'previous_word':prev,'next_word':nxt,'pause_seconds':round(nxt['start']-prev['end'],4)})
 # Dynamic programming across all frames ensures an 11s upper bound even if ASR fails.
 # Fallback frame cuts have high cost; no frame is deleted to meet the bound.
 cost={a:0.};back={}
 for end in range(a+1,b+1):
  boundary_cost=0 if end==b else candidates.get(end,{'cost':40})['cost'];best=(float('inf'),None)
  for start in range(max(a,end-MAX),end):
   dur=(end-start)/FPS;v=cost[start]+2+boundary_cost+.03*(dur-10)**2+(5 if dur<1 else 0)
   if v<best[0]:best=(v,start)
  cost[end]=best[0];back[end]=best[1]
 bounds=[b]
 while bounds[-1]!=a:bounds.append(back[bounds[-1]])
 bounds=list(reversed(bounds));chosen=[]
 for t in bounds[1:-1]:chosen.append(candidates.get(t,dict(reason='frame_fallback',cost=40,evidence='ASR无法提供可行切点',time=t/FPS)))
 oldrows=[q for q in oldpairs if q['pair_id'] in run['old_pair_ids']];oldcuts=[q['end'] for q in oldrows[:-1]]
 for j,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
  r=copy.deepcopy(run);r.update(start_frame=lo,end_frame=hi,start=lo/FPS,end=hi/FPS,duration=(hi-lo)/FPS,sentence_revision=True,continuous_interval=[a/FPS,b/FPS],partition_rule='target10_max11_sentence_preferred')
  r['flags']=[f for f in r['flags'] if f!='不足1秒，仍保留供审核' and not f.startswith('倾听者框缺失')]
  fs=[ann[str(i)] for i in range(lo,hi)];missing=sum(r['listener'] not in f['people'] for f in fs)
  if missing:r['flags'].append(f'倾听者框缺失 {missing} 帧，保留候选待确认真实可见性')
  if hi-lo<25:r['flags'].append('不足1秒，仍保留供审核')
  r['listener_positive_fraction']=sum((f['people'].get(r['listener'],{}).get('score') or 0)>.2 for f in fs)/len(fs)
  r['start_boundary']=dict(reason='hard_interval_start',time=lo/FPS) if j==0 else chosen[j-1]
  r['end_boundary']=dict(reason='hard_interval_end',time=hi/FPS) if j==len(bounds)-2 else chosen[j]
  r['sentence_evidence']=dict(asr_file=d['window']['window_id']+'_asr.json',text=d['text'],chosen_internal_boundaries=chosen)
  r['flags'].append('目标10秒、最长11秒；内部切口优先句末/停顿，所有余段保留')
  segments.append(r)
 newcuts=[t/FPS for t in bounds[1:-1]]
 changes.append(dict(start=a/FPS,end=b/FPS,old_pair_ids=run['old_pair_ids'],old_internal_boundaries=oldcuts,new_internal_boundaries=newcuts,boundary_evidence=chosen,preview_time=(newcuts[0] if newcuts else a/FPS),summary=f'{a/FPS:.2f}–{b/FPS:.2f}s：按句末/停顿优先拆成 '+ ' + '.join(f'{(y-x)/FPS:.2f}秒' for x,y in zip(bounds,bounds[1:]))+'，每段不超过11秒。'))
pairs=[]
for r in segments:
 if r['kind']=='pair':r['pair_id']=f'sentence_{len(pairs)+1:03d}';pairs.append(r)
oldmask=[False]*m['frames'];newmask=[False]*m['frames']
for r in oldpairs:oldmask[r['start_frame']:r['end_frame']]=[True]*(r['end_frame']-r['start_frame'])
for r in pairs:newmask[r['start_frame']:r['end_frame']]=[True]*(r['end_frame']-r['start_frame'])
assert oldmask==newmask;assert all(r['duration']<=11 for r in pairs)
m.update(pairs=pairs,segments=segments,sentence_changes=changes,scope='整片提取；目标10秒、最长11秒，优先句末/停顿，必要时词间切分；所有余段保留。',sentence_policy={'target_seconds':10,'mandatory_max_seconds':11,'priority':['sentence_end','natural_pause','word_boundary','frame_fallback'],'incomplete_sentence_is_not_rejection':True,'hard_boundaries_preserved':True,'semantic_hint_provenance':'semantic_boundary_hints.json；本批转写语义复核补充句末候选，非ASR原生标点'},baseline_manifest_sha256=hashlib.sha256((OLD/'samples.json').read_bytes()).hexdigest())
m['coverage_comparison']={'old_pairs':len(oldpairs),'new_pairs':len(pairs),'old_candidate_frames':sum(oldmask),'new_candidate_frames':sum(newmask),'lost_candidate_frames':0,'added_candidate_frames':0,'same_frame_mask':True,'maximum_pair_seconds':max(r['duration'] for r in pairs)}
(OUT/'samples.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');print(json.dumps(m['coverage_comparison'],ensure_ascii=False));print(json.dumps(changes,ensure_ascii=False,indent=2))
