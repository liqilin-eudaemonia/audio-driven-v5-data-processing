"""Explicit latest-complete recovery, without changing the sealed training source."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,fcntl,json,subprocess
G=Path(__file__).resolve().parent;W=G.parent/'p2_all_lora_ablations28_20261111'
sys.path.insert(0,str(W))
import runner as Q
R=Q.R
from p2_listen.contracts import read,atomic_json

def latest(out):
 paths=sorted(out.glob('step_*.complete.json'),reverse=True)
 for marker in paths:
  cp=marker.with_suffix('').with_suffix('.pt')
  if cp.exists():return cp
 if (out/'run.json').exists():raise RuntimeError('Existing run has no complete checkpoint; no reinitialization allowed')
 return None

def run():
 import torch
 torch.set_num_threads(2)
 from listen_cluster import launch
 with (W/'queue.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for arm,target in R.TARGETS.items():
   if (W/(arm+'_ACCEPTED.json')).exists():continue
   out=R.output(arm)
   if (W/'STOP').exists()or(out/'STOP').exists():raise RuntimeError('User STOP preserved')
   m=R.noevent_runtime().load()if arm=='no_event'else R.load()
   cp=latest(out)
   if cp:
    v=m.checkpoint.checked(cp)
    identity=read(out/'run.json')
    assert v['identity']==identity and len(v['rank_rng'])==28
    assert identity['source']==(R.noevent_runtime().source() if arm=='no_event' else R.source())
    assert v['step']<=target
    step=v['step'];del v
   else:step=0
   if step<target:
    if not Q.resources():raise RuntimeError('Selected training GPUs occupied; preserve other tasks')
    c=Q.cluster(arm);cmds=[]
    for i,n in enumerate(c['nodes']):
     cmd=['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v)for k,v in c['network'].items()],Q.PY,'-u',str(W/'runner.py'),'node','--arm',arm,'--index',str(i)]
     if cp:cmd+=['--resume',str(cp)]
     cmds.append(cmd)
    atomic_json(W/'STATUS.json',dict(stage='training',arm=arm,target=target,resume=str(cp)if cp else None,at=time.time(),recovery_entry=str(Path(__file__))))
    launch(c,cmds,G/(arm+'_recovery_nodes_'+str(time.time_ns())))
   path=out/f'step_{target:06d}.pt';v=m.checkpoint.checked(path)
   assert v['step']==target and len(v['rank_rng'])==28;del v
   atomic_json(W/(arm+'_ACCEPTED.json'),dict(success=True,step=target,checkpoint=str(path),sha256=read(path.with_suffix('.complete.json'))['sha256'],at=time.time(),cpu_threads=2))
  atomic_json(W/'COMPLETE.json',dict(success=True,at=time.time(),targets=R.TARGETS))
if __name__=='__main__':run()
