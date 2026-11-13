"""Post-extraction annotation/crop consistency; flag only, never filter."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,concurrent.futures
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def audit(f):
 m=json.loads(f.read_text())
 if m.get('geometry_audit_version')=='v3':return m['video_id']
 ann=json.loads(Path(m['source']['annotation']).read_text());lookup={};rolehist={'p0':{},'p1':{}}
 for r in m['pairs']:
  a,b=r['start_frame'],r['end_frame'];role=r['listener'];bs=np.array([ann.get(str(i),{}).get('people',{}).get(role,{}) .get('bbox') or [np.nan]*4 for i in range(a,b)],dtype=float);valid=np.isfinite(bs).all(1)&(bs[:,2]>bs[:,0])&(bs[:,3]>bs[:,1]);bs=bs[valid];c=r['crop']
  if len(bs):
   clipped=bs.copy();clipped[:,[0,2]]=np.clip(clipped[:,[0,2]],0,m['width']);clipped[:,[1,3]]=np.clip(clipped[:,[1,3]],0,m['height']);ss=np.minimum(clipped[:,2]-clipped[:,0],clipped[:,3]-clipped[:,1]);vals,counts=np.unique(np.floor(ss[ss>0]).astype(int),return_counts=True)
   for value,count in zip(vals,counts):rolehist[role][str(int(value))]=rolehist[role].get(str(int(value)),0)+int(count)

  other='p1' if role=='p0' else 'p0';otherbs=np.array([ann.get(str(i),{}).get('people',{}).get(other,{}) .get('bbox') or [np.nan]*4 for i in range(a,b)],dtype=float);otherbs=otherbs[np.isfinite(otherbs).all(1)]
  if c and len(bs) and len(otherbs):
   l=np.median(bs,axis=0);o=np.median(otherbs,axis=0);overlap=max(0,min(l[2],o[2])-max(l[0],o[0]));minwidth=min(l[2]-l[0],o[2]-o[0]);dy=abs((l[1]+l[3]-o[1]-o[3])/2)
   if minwidth>0 and overlap/minwidth>.5 and dy>.22*m['height']:
    r['proposed_crop_before_layout_review']=c;r['crop']=None;c=None
    r['flags']=list(dict.fromkeys(r['flags']+['stacked_layout_crop_needs_review','pre_crop_unresolved']))

  if c and len(bs):
   bs[:,[0,2]]=np.clip(bs[:,[0,2]],0,m['width']);bs[:,[1,3]]=np.clip(bs[:,[1,3]],0,m['height']);bs=bs[(bs[:,2]>bs[:,0])&(bs[:,3]>bs[:,1])];x,y,w,h=c['x'],c['y'],c['width'],c['height'];inside=(bs[:,0]>=x)&(bs[:,1]>=y)&(bs[:,2]<=x+w)&(bs[:,3]<=y+h);cx=(bs[:,0]+bs[:,2])/2;cy=(bs[:,1]+bs[:,3])/2;center=(cx>=x)&(cx<=x+w)&(cy>=y)&(cy<=y+h)
   r['crop_box_consistency']={'evaluated_face_frames':len(bs),'full_face_box_inside_fraction':float(np.mean(inside)) if len(bs) else None,'face_center_inside_fraction':float(np.mean(center)) if len(bs) else None}
   if len(bs) and np.mean(inside)<.95:r['flags'].append('annotation_box_crop_mismatch_review')
  else:r['crop_box_consistency']={'evaluated_face_frames':len(bs),'full_face_box_inside_fraction':None,'face_center_inside_fraction':None}
  r['flags']=list(dict.fromkeys(r['flags']));lookup[r['pair_id']]=r
 assert sum(sum(v.values()) for v in rolehist.values())==m['face_valid_frames']
 m['listener_face_short_side_frame_histogram_by_role']=rolehist
 m['crop_unresolved_pair_frames']=sum(r['end_frame']-r['start_frame'] for r in m['pairs'] if r['crop'] is None)
 m['ledger']=[lookup[r['pair_id']] if r['kind']=='pair' else r for r in m['ledger']];m['geometry_audit_version']='v3';tmp=f.with_suffix('.audit.tmp');tmp.write_text(json.dumps(m,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');tmp.replace(f);return m['video_id']
if __name__=='__main__':
 files=sorted((ROOT/'manifests').glob('*.json'))
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:results=list(pool.map(audit,files))
 print('geometry evidence',len(results),flush=True)
