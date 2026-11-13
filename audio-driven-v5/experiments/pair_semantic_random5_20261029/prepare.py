"""Freeze a random unseen sample, then snapshot only those five sources."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import json,random,sys,hashlib
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
PROD=ROOT/'data_corpora/seamless_listen_local_full'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if not (EXP/'selection.json').exists():
 rows=read(PROD/'catalog.json')['rows'];random.Random(20261031).shuffle(rows)
 excluded={r['pair'] for r in read(ROOT/'data_previews/seamless_listen_full_source_recovery/index.json')['pairs']}
 selected=[];skipped=[]
 for item in rows:
  pair=item['pair']
  if pair in excluded:continue
  needed=[PROD/k/(pair+'.json') for k in ['audit','results','normalized_pairs']]+[ROOT/'data_corpora/seamless_listen_v8_audio_final_full/results'/(pair+'.json')]
  missing=[str(p) for p in needed if not p.exists()]
  if missing:skipped.append(dict(pair=pair,missing=missing));continue
  selected.append(item)
  if len(selected)==5:break
 assert len(selected)==5
 write(EXP/'selection.json',dict(seed=20261031,criteria='Uniform seeded shuffle of original catalog; first five unseen sources with existing production/audio artifacts. No duration or improvement filter. Freeze before segmentation.',excluded=sorted(excluded),skipped=skipped,items=selected,production_results={r['pair']:sha(PROD/'results'/(r['pair']+'.json')) for r in selected}))
for item in read(EXP/'selection.json')['items']:
 print('SELECTED',item['pair'],read(PROD/'audit'/(item['pair']+'.json'))['source_seconds'],flush=True)
import listen_local_recovery as base
for item in read(EXP/'selection.json')['items']:
 path=EXP/(item['pair']+'.json')
 if path.exists():continue
 audit=read(PROD/'audit'/(item['pair']+'.json'))
 ctx=base.SourceContext(item,shot_times=audit['shots'])
 write(path,dict(pair=item['pair'],item=item,people=ctx.people,rms=[x.tolist() for x in ctx.rms],limit=ctx.limit,sources=ctx.sources,shots=ctx.shots,shots_by_track=ctx.shots_by_track,regions=ctx.regions,source_gaps=ctx.source_gaps))
 print('PREPARED',item['pair'],ctx.limit,flush=True)
