"""Gracefully resize this pipeline's CPU quality pool on the executing host."""
import os,sys,time,signal,socket,json
from pathlib import Path
W=Path(__file__).resolve().parent.parent;R=W.parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from listen_j_annotation_supervisor import alive,spawn
from listen_benchmark.j_annotation import read,write,locked,acquire
n=sys.argv[1];count=int(sys.argv[2]);c=read(W/'config.json');assert socket.gethostname()==c['nodes'][n]['hostname'];assert 16<=count<=96
local=W/'visual_quality/nodes'/n;owner=acquire(W/'nod_cpu_share'/n/'quality_resize.lock',False);assert owner is not None
stopped=[]
for role in ['watchdog','controller']:
 s=read(local/f'{role}.json')
 if alive(s):
  cmd=Path(f"/proc/{s['pid']}/cmdline").read_bytes().decode();assert 'listen_j_visual_gate_cluster.py' in cmd and str(W/'visual_quality') in cmd
  os.kill(s['pid'],signal.SIGTERM);stopped.append(s)
state=dict(node=n,requested_workers=count,started_at=time.time(),previous=stopped,stage='draining')
while any(alive(s) for s in stopped):
 state['heartbeat']=time.time();state['controller']=read(local/'controller.json');write(W/'nod_cpu_share'/n/'quality_resize.json',state);time.sleep(5)
with locked(W/'config-edit.lock'):
 c=read(W/'config.json');c['nodes'][n]['visual_quality_workers']=count;write(W/'config.json',c)
child=spawn(local,'watchdog-launch',[str(W/'visual_quality_rollout/runtime_env/bin/python'),str(R/'scripts/listen_j_visual_gate_cluster.py'),'watchdog','--work',str(W/'visual_quality'),'--node',n,'--workers',str(count)],dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',GLOG_minloglevel='2'))
state.update(stage='restarted',child=child,heartbeat=time.time());write(W/'nod_cpu_share'/n/'quality_resize.json',state);print(json.dumps(state))
