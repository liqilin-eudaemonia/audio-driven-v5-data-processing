import json,os,socket,signal,sys,time
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');w=ROOT/'experiments/listen_j_omni_20261030';q=w/'visual_quality'
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from listen_j_annotation_supervisor import alive,spawn
from listen_benchmark.j_annotation import read,write
node=sys.argv[1];workers=int(sys.argv[2]);cfg=read(w/'config.json');assert socket.gethostname()==cfg['nodes'][node]['hostname']
local=q if node=='node4' else q/'nodes'/node
stopped=[]
for name in ['watchdog','controller']:
 p=local/f'{name}.json';s=read(p) if p.exists() else None
 if alive(s):os.kill(s['pid'],signal.SIGTERM);stopped.append(s)
for _ in range(0 if node=='node4' else 1800):
 if all(not alive(s) for s in stopped):break
 time.sleep(1)
assert node=='node4' or all(not alive(s) for s in stopped),'Old controller still owns node-local lock'
# node4 migrates from a root controller to node-local ownership; source queue
# claim locks prevent any duplicate measurement. Early nod screening remains
# disabled until the old unguarded controller has drained completely.
new=q/'nodes'/node;new.mkdir(parents=True,exist_ok=True)
child=spawn(new,'watchdog-launch',[str(w/'visual_quality_rollout/runtime_env/bin/python'),str(ROOT/'scripts/listen_j_visual_gate_cluster.py'),'watchdog','--work',str(q),'--node',node,'--workers',str(workers)],dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',GLOG_minloglevel='2'))
r=dict(at=time.time(),node=node,workers=workers,child=child,previous=stopped);write(w/'visual_quality_rollout'/f'controller-roll-{node}.json',r);print(json.dumps(r))
