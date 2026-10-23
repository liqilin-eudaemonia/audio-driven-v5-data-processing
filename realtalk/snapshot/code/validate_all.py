"""Independent manifest/statistics consistency checks, without changing candidates."""
import json,hashlib,random,collections
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports';index=json.loads((ROOT/'index.json').read_text());files=sorted((ROOT/'manifests').glob('*.json'));assert {p.stem for p in files}=={r['video_id'] for r in index}
frames=collections.Counter();pair_ids=set();checked=0;reference=None;examples=[]
for f in files:
 m=json.loads(f.read_text());assert m['geometry_audit_version']=='v3';ledger=m['ledger'];assert ledger[0]['start_frame']==0 and ledger[-1]['end_frame']==m['frames'];assert all(a['end_frame']==b['start_frame'] for a,b in zip(ledger,ledger[1:]));assert sum(r['end_frame']-r['start_frame'] for r in ledger)==m['frames'];frames['source_frames']+=m['frames']
 for r in ledger:frames[r['kind']]+=r['end_frame']-r['start_frame']
 rolehist=m['listener_face_short_side_frame_histogram_by_role'];assert sum(sum(h.values()) for h in rolehist.values())==m['face_valid_frames']
 for r in m['pairs']:
  assert r['pair_id'] not in pair_ids;pair_ids.add(r['pair_id']);n=r['end_frame']-r['start_frame'];assert 0<n<=275 and abs(r['duration']-n/25)<1e-9;assert r['face']['valid_frames']+r['face']['missing_or_invalid_frames']==n
  c=r['crop']
  if c:assert c['width']==c['height']==c['native_edge'] and 0<=c['x']<=m['width']-c['width'] and 0<=c['y']<=m['height']-c['height']
  if 'stacked_layout_crop_needs_review' in r['flags']:assert c is None
 if m['video_id']=='-1wCGAxqT_4':reference=m
 # Independent sample of annotation dimensions, not the histogram implementation.
 if len(examples)<24 and m['pairs'] and checked%25==0:examples.append((m,m['pairs'][len(m['pairs'])//2]))
 checked+=1
for m,r in examples:
 ann=json.loads(Path(m['source']['annotation']).read_text());lengths=[]
 for frame in range(r['start_frame'],r['end_frame']):
  b=ann.get(str(frame),{}).get('people',{}).get(r['listener'],{}).get('bbox')
  if b and len(b)==4 and all(np.isfinite(v) for v in b) and b[2]>b[0] and b[3]>b[1]:
   w=max(0,min(m['width'],b[2])-max(0,b[0]));h=max(0,min(m['height'],b[3])-max(0,b[1]));
   if w>0 and h>0:lengths.append(min(w,h))
 assert len(lengths)==r['face']['valid_frames']
 if lengths:assert abs(float(np.median(lengths))-r['face']['short_side']['median'])<=.001
old=json.loads(Path('/private/liqi_workspace/realtalk_inspection/full_video_v2_sentence/review/samples.json').read_text());assert [(r['start_frame'],r['end_frame']) for r in old['pairs']]==[(r['start_frame'],r['end_frame']) for r in reference['pairs']]
s=json.loads((OUT/'summary.json').read_text());assert s['complete'] and s['completed_videos']==692;assert s['pair_count']==len(pair_ids)
for k in ['pair','role_unknown','black','source_frames']:assert s['frame_totals'][k]==frames[k]
pq=pd.read_parquet(OUT/'pairs.parquet');assert len(pq)==len(pair_ids) and set(pq.pair_id)==pair_ids;assert abs(pq.duration_seconds.sum()-frames['pair']/25)<1e-6
role=pd.read_csv(OUT/'per_video_listener.csv');assert abs(role.total_seconds.sum()-frames['pair']/25)<1e-6
joint=pd.read_csv(OUT/'pre_crop_size_by_pair_duration.csv');assert abs(joint.seconds.sum()-s['pre_crop_known_hours']*3600)<1e-6
for key in ['crop_edge','face_min','face_p10','face_median']:
 values=[r[key+'_seconds'] for r in s['thresholds']];assert all(a>=b-1e-9 for a,b in zip(values,values[1:]));assert all(v<=s['candidate_hours']*3600+1e-6 for v in values)
for t in s['thresholds']:assert t['face_min_seconds']<=t['face_p10_seconds']+1e-9<=t['face_median_seconds']+2e-9;assert t['face_p10_coverage95_seconds']<=t['face_p10_seconds']+1e-9
result=dict(videos=checked,pairs=len(pair_ids),whole_source_frame_partition=True,maximum_11_seconds=True,reference_152_pairs_unchanged=True,independent_native_face_samples=len(examples),role_histogram_conservation=True,csv_parquet_counts=True,threshold_monotonic=True,pre_crop_native_bounds=True,stacked_layout_separated=True)
(OUT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
