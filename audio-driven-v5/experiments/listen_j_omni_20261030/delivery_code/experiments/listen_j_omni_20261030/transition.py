"""Validated handover: retain CPU nod workers, drain own VLM workers, start Omni."""
from pathlib import Path
import argparse,json,subprocess
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
REMOTE=r'''
import json,os,signal,subprocess,time,socket,sys
from pathlib import Path
root=Path(ROOT);work=Path(WORK);sys.path.insert(0,str(root/'scripts'));sys.path.insert(0,str(root/'src'))
from listen_benchmark.j_annotation import read,write
from listen_j_annotation_supervisor import alive,identity,spawn
cfg=read(work/'config.json');node=cfg['nodes'][NODE];old=Path(node['old_work'])
assert socket.gethostname()==node['hostname']
assert read(work/'validation/canary_audit.json')['passed']
record=work/'transitions'/f'{NODE}.json'
if record.exists():raise SystemExit('Transition already recorded; inspect rather than repeat')
prev=read(old/'config.json');supervisor=read(old/'supervisor.json');watchdog=read(old/'watchdog.json')
assert supervisor['phase']=='annotation' and alive(supervisor) and alive(watchdog)
children=supervisor['children'];nods={n:c for n,c in children.items() if n.startswith('nod-') and alive(c)}
gpus={n:c for n,c in children.items() if n.startswith('gpu-') and alive(c)}
assert len(nods)==prev['nod_workers'] and len(gpus)==8,(len(nods),len(gpus))
write(work/'transitions'/f'{NODE}-before.json',dict(config=prev,supervisor=supervisor,watchdog=watchdog,at=time.time()))
(old/'STOP_SUPERVISION').touch()
def stop_owner(rec,script):
 if not alive(rec):return
 cmd=Path(f"/proc/{rec['pid']}/cmdline").read_bytes().decode().replace('\x00',' ')
 assert script in cmd and str(old) in cmd,cmd
 os.kill(rec['pid'],signal.SIGTERM)
 for _ in range(40):
  if not alive(rec):return
  time.sleep(1)
 raise RuntimeError('Owner did not stop gracefully: '+script)
stop_owner(watchdog,'listen_j_annotation_watchdog.py')
stop_owner(supervisor,'listen_j_annotation_supervisor.py')
updated=dict(prev,visual_enabled=False,visual_paused_reason='user-approved Omni first_plus_evidence pipeline; keep CPU nod branch',visual_paused_at=time.time())
write(old/'config.json',updated)
s=read(old/'supervisor.json');s.pop('rolling_restart',None);s.pop('awaiting_warm_worker',None);s['warming_gpus']=[];write(old/'supervisor.json',s)
env=dict(os.environ,LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
python='/private/software/conda/envs/vllm/bin/python';upstream=prev['upstream']
new_supervisor=spawn(old,'supervisor-cpu-only',[python,str(root/'scripts/listen_j_annotation_supervisor.py'),'--work',str(old),'--upstream',str(upstream)],env)
for _ in range(45):
 now=read(old/'supervisor.json')
 if now.get('pid')==new_supervisor['pid'] and now.get('gpu_ids')==[] and now.get('error') is None:break
 time.sleep(1)
else:raise RuntimeError('CPU-only supervisor not ready')
assert all(alive(c) for c in nods.values()),'An existing nod worker exited during transition'
(old/'STOP_SUPERVISION').unlink()
new_watchdog=spawn(old,'watchdog-cpu-only',[python,str(root/'scripts/listen_j_annotation_watchdog.py'),'--work',str(old),'--upstream',str(upstream)],env)
signaled=[]
for n,c in sorted(gpus.items()):
 if alive(c):
  cmd=Path(f"/proc/{c['pid']}/cmdline").read_bytes().decode().replace('\x00',' ')
  assert 'listen_j_annotation.py' in cmd and 'worker' in cmd and str(old) in cmd
  os.kill(c['pid'],signal.SIGTERM);signaled.append(n)
 time.sleep(.25)
nodework=work/'nodes'/NODE
new_pipeline=spawn(nodework,'watchdog-bootstrap',[python,str(root/'scripts/listen_j_omni.py'),'watchdog','--work',str(work),'--node',NODE],env)
write(record,dict(at=time.time(),host=socket.gethostname(),nod_workers_preserved=len(nods),old_vlm_signaled=signaled,
 old_supervisor=new_supervisor,old_watchdog=new_watchdog,new_omni_watchdog=new_pipeline,status='CPU nod preserved; VLM graceful drain; Omni starts as cards free'))
print(json.dumps(read(record)))
'''
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--node',required=True,choices=['node17','node8']);a=p.parse_args()
 cfg=json.loads((WORK/'config.json').read_text());node=cfg['nodes'][a.node]
 code='ROOT='+repr(str(ROOT))+'\nWORK='+repr(str(WORK))+'\nNODE='+repr(a.node)+'\n'+REMOTE
 command=['ssh','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=8','root@'+node['host'],'python3 -']
 result=subprocess.run(command,input=code,text=True,capture_output=True,timeout=180)
 print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)
