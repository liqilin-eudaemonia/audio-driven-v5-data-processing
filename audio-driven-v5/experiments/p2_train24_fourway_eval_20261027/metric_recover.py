"""Move the eight unlaunched shards from busy node1 onto node0 after its first wave."""
import sys,os,time,subprocess,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_omni_eval.common import read,atomic_json,check_device
if len(sys.argv)>1:
 with (OUT/'metric-recovery.log').open('a') as log:p=subprocess.Popen([sys.executable,str(__file__)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
 print(json.dumps({'pid':p.pid}));raise SystemExit
while not (OUT/'metric-node-0.json').exists():time.sleep(5)
assert not any(read(OUT/'metric-node-0.json')['codes'])
MP=read(ROOT/'configs/p2_omni_runtime.lock.json')['paths']['metrics_python'];children=[]
for local,rank in enumerate(range(8,16)):
 check_device('cuda:'+str(local))
 env={**os.environ,'CUDA_VISIBLE_DEVICES':str(local),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
 with (OUT/'logs'/f'omni-{rank:02d}.log').open('a') as log:children.append(subprocess.Popen([MP,str(OUT/'omni_shard.py'),str(rank)],env=env,stdout=log,stderr=subprocess.STDOUT,close_fds=True))
atomic_json(OUT/'metric_relocation.json',dict(original_node=1,execution_node=0,ranks=list(range(8,16)),reason='Node1 GPUs busy at metric start; no other tasks terminated',started=time.time()))
codes=[p.wait() for p in children];atomic_json(OUT/'metric_relocation_finished.json',dict(codes=codes,finished=True,time=time.time()))
