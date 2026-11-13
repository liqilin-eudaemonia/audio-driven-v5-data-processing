import sys,os,subprocess,time,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_omni_eval.common import read,atomic_json,check_device
runtime=read(ROOT/'configs/p2_omni_runtime.lock.json');PY=runtime['paths']['metrics_python']
rank=int(sys.argv[1])
if len(sys.argv)>2:
 with (OUT/f'metric-node-{rank}.log').open('a') as log:p=subprocess.Popen([sys.executable,str(__file__),str(rank)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
 print(json.dumps({'node':rank,'pid':p.pid}));raise SystemExit
children=[]
for local in range(8):
 check_device('cuda:'+str(local));r=rank*8+local
 env={**os.environ,'CUDA_VISIBLE_DEVICES':str(local),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
 with (OUT/'logs'/f'omni-{r:02d}.log').open('a') as log:children.append(subprocess.Popen([PY,str(OUT/'omni_shard.py'),str(r)],env=env,stdout=log,stderr=subprocess.STDOUT,close_fds=True))
 time.sleep(.2)
atomic_json(OUT/f'metric-node-{rank}.json',dict(codes=[p.wait() for p in children],finished=True,updated=time.time()))
