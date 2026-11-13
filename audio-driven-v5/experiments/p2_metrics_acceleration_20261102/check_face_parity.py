import argparse,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/private/liqi_workspace/project_code/OmniResponse')
from evaluation.video.paired_metrics.backends import Faces
from evaluation.video.paired_metrics.media import load_pair
p=argparse.ArgumentParser();p.add_argument('--device');a=p.parse_args()
work=Path(__file__).parent
rows=[json.loads(x) for x in (work/'parity_pairs.jsonl').read_text().splitlines()]
assets=json.loads(Path('/private/liqi_workspace/project_code/OmniResponse/evaluation/video/paired_metrics/assets.local.json').read_text())
f=Faces(assets,True,4,64,device=a.device,emotion_device=a.device);arrays={};records=[];start=time.time()
for i,r in enumerate(rows):
 gt,gen,_=load_pair(r)
 for side,frames in [('gt',gt),('gen',gen)]:
  indices=np.unique(np.linspace(0,len(frames)-1,8,dtype=int));out=f.extract(frames[indices])
  for key,value in zip(['landmarks','boxes','emotion','count'],out):arrays[f'{i}_{side}_{key}']=np.asarray(value)
 records.append(dict(id=r['id'],frames=len(indices)));print(i,r['id'],flush=True)
name='cpu' if a.device=='cpu' else 'gpu'
np.savez(work/f'face_parity_{name}.npz',**arrays)
(work/f'face_parity_{name}.json').write_text(json.dumps(dict(records=records,seconds=time.time()-start,execution=f.execution),indent=2))
