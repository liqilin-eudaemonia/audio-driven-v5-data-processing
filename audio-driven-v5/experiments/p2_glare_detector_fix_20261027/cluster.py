"""Re-detect all 816 distinct GT/fixed/guided videos under one corrected policy."""
import sys,os,json,time,subprocess,concurrent.futures
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments/p2_guided_full_eval_20261027')]
from generate_cluster import remote,PY
from p2_glare_eval.common import read,sha,atomic_json
from p2_glare_rgb.detector import code_identity,policy
DETECTOR_PY=ROOT/'.venv-p2-glare-detector/bin/python';WORKERS=72

def prepare():
 videos={};pairs={}
 for kind in ['train','test']:
  pairs[kind]={}
  for arm in ['E0','E1','E0F','E1F']:
   for variant in ['fixed','guided']:
    base=ROOT/f'experiments/p2_{"guided_" if variant=="guided" else ""}{kind}24_fourway_eval_20261027';p=read(base/arm/'glare/protocol.json');pairs[kind][arm+'-'+variant]=p['samples']
    for row in p['samples']:
     for side in ['gt','generated']:
      v=row[side+'_video'];h=row['files'][v];videos.setdefault(h,dict(video=v,sha256=h,frames=None))
 items=list(videos.values())
 atomic_json(OUT/'protocol.json',dict(policy=policy(),code=code_identity(),videos=items,pairs=pairs,planned_pairs=768,workers=WORKERS),immutable=True)
 (OUT/'logs').mkdir(exist_ok=True);(OUT/'workers').mkdir(exist_ok=True);print('unique videos',len(items))

def worker(rank):
 from p2_glare_rgb.detector import Engine
 p=read(OUT/'protocol.json');assert p['code']==code_identity();engine=Engine();records=[]
 for item in p['videos'][rank::WORKERS]:
  try:r=engine.detect(item['video'],OUT/'detections'/item['sha256'],item['sha256'])
  except Exception as error:
   import traceback;traceback.print_exc();r=dict(status='error',error=str(error))
  records.append(dict(sha256=item['sha256'],**r));atomic_json(OUT/'workers'/f'{rank:03d}.json',dict(finished=False,results=records,time=time.time()))
 atomic_json(OUT/'workers'/f'{rank:03d}.json',dict(finished=True,results=records,time=time.time()))

def node(rank):
 children=[];env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TF_ENABLE_ONEDNN_OPTS':'0','TF_CPP_MIN_LOG_LEVEL':'2'}
 for local in range(24):
  r=rank*24+local
  with (OUT/'logs'/f'worker-{r:03d}.log').open('a') as log:children.append(subprocess.Popen([str(DETECTOR_PY),str(__file__),'worker',str(r)],stdout=log,stderr=subprocess.STDOUT,env=env))
  time.sleep(.2)
 atomic_json(OUT/f'node-{rank}.json',dict(codes=[p.wait() for p in children],finished=True,time=time.time()))

if __name__=='__main__':
 cmd=sys.argv[1]
 if cmd=='prepare':prepare()
 elif cmd=='worker':worker(int(sys.argv[2]))
 elif cmd=='node':node(int(sys.argv[2]))
 elif cmd=='detach':
  rank=int(sys.argv[2])
  with (OUT/f'node-{rank}.log').open('a') as log:p=subprocess.Popen([str(PY),str(__file__),'node',str(rank)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
  print(json.dumps(dict(rank=rank,pid=p.pid)))
 elif cmd=='launch':
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
   for v in pool.map(lambda r:remote(r,[PY,__file__,'detach',r]),range(3)):print(v)
