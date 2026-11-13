import json,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
rows=[];probe=[];seen=set()
for kind in ['train','test']:
 for variant in ['fixed','guided']:
  base=ROOT/f'experiments/p2_{"guided_" if variant=="guided" else ""}{kind}24_fourway_eval_20261027'
  for arm in ['E0','E1','E0F','E1F']:
   p=base/arm/'glare/protocol.json'
   if not p.exists():continue
   protocol=read(p)
   for row in protocol['samples']:
    for side in ['gt','generated']:
     video=row[side+'_video'];h=row['files'][video];quality=base/arm/'glare/detections'/h/'quality.json'
     if not quality.exists():continue
     q=read(quality);bad=q['invalid_frames'];counts=collections.Counter('mp_missing' if not b['mediapipe_valid'] else 'fer_missing' if not b['fer_valid'] and b['fer_faces']==0 else 'fer_multiple_or_invalid' if not b['fer_valid'] else 'association' for b in bad)
     item=dict(dataset=kind,variant=variant,arm=arm,sample_id=row['sample_id'],seed=row['sample_seed'],side=side,video=video,video_sha256=h,quality=str(quality),frames=q['frames'],mp_valid=q['mediapipe_valid_frames'],fer_valid=q['fer_valid_frames'],joint_valid=q['jointly_valid_frames'],bad_counts=dict(counts));rows.append(item)
     if kind=='test' and h not in seen and counts['fer_missing']>0 and sum(x['side']==side for x in probe)<4:
      indexes=[b['frame_index'] for b in bad if b['mediapipe_valid'] and not b['fer_valid'] and b['fer_faces']==0]
      probe.append(dict(**item,indexes=indexes[::max(1,len(indexes)//8)][:8]));seen.add(h)
summary={}
for kind in ['train','test']:
 for side in ['gt','generated']:
  unique={r['video_sha256']:r for r in rows if r['dataset']==kind and r['side']==side};values=list(unique.values());count=collections.Counter()
  for r in values:count.update(r['bad_counts'])
  summary[kind+'_'+side]=dict(videos=len(values),strict_complete=sum(v['joint_valid']==v['frames'] for v in values),bad_frames=dict(count))
(OUT/'failure_analysis.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2));(OUT/'probe_inputs.json').write_text(json.dumps(probe,indent=2));print(json.dumps(summary,indent=2))
