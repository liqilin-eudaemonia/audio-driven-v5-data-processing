"""Prepare an exact PID/GPU pause plan. Execution requires recorded user approval."""
from pathlib import Path
import json,concurrent.futures,subprocess,time,argparse
import common as C
W=C.WORK
REMOTE=r'''from pathlib import Path
import json,subprocess,os,signal
x=json.loads(PAYLOAD);gpus={}
for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines():
 a=[p.strip()for p in row.split(',')]
 if int(a[0])in x['gpus']:gpus[a[1]]=int(a[0])
apps=[];all_apps=[]
for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines():
 a=[p.strip()for p in row.split(',')]
 all_apps.append((int(a[0]),a[1]))
 if a[1]in gpus:apps.append((int(a[0]),a[1]))
records=[]
for pid,uuid in apps:
 d=Path('/proc',str(pid))
 try:
  comm=(d/'comm').read_text().strip()
  if not comm.startswith('VLLM::'):continue
  if any(p==pid and u not in gpus for p,u in all_apps):raise RuntimeError('VLLM also owns reserved GPU; refuse process-wide pause')
  stat=(d/'stat').read_text().rsplit(')',1)[1].split()
  records.append(dict(pid=pid,start_ticks=stat[19],state=stat[0],gpu=gpus[uuid],uuid=uuid,comm=comm))
 except FileNotFoundError:pass
if x['execute']:
 expected=x['records']
 if {(p['pid'],p['start_ticks'],p['uuid'])for p in records}!={(p['pid'],p['start_ticks'],p['uuid'])for p in expected}:raise RuntimeError('VLLM process/GPU identity changed; refuse signal')
 for p in expected:
  d=Path('/proc',str(p['pid']));s=(d/'stat').read_text().rsplit(')',1)[1].split()
  if s[19]!=p['start_ticks']or not(d/'comm').read_text().startswith('VLLM::'):raise RuntimeError('PID reused; refuse signal')
  os.kill(p['pid'],signal.SIGSTOP)
print(json.dumps(dict(records=records,paused=x['execute'],training_not_signaled=True)))
'''
def main(execute=False):
 if execute:
  decision=C.read(W/'USER_CONTENTION_POLICY.json')
  assert decision.get('pause_training_gpu_vllm')is True and decision.get('user_approved')is True
 cl=C.read(C.OUT/'cluster.json');nodes=[n for n in cl['nodes']if n['name']in ['node4','node11']]
 # Read-only pass; exact selected masks exclude node11 reserved4-7.
 def query(n):return dict(node=n['name'],**C.C.remote(cl,n,REMOTE.replace('PAYLOAD',repr(json.dumps(dict(gpus=n['gpus'],execute=False))))))
 with concurrent.futures.ThreadPoolExecutor(2)as pool:plan=list(pool.map(query,nodes))
 C.atomic_json(W/'VLLM_PAUSE_PLAN.json',dict(at=time.time(),nodes=plan,execution_requires_user_approval=True))
 if not execute:return plan
 result=[]
 for n,item in zip(nodes,plan):
  payload=dict(gpus=n['gpus'],execute=True,records=item['records']);code=REMOTE.replace('PAYLOAD',repr(json.dumps(payload)))
  p=subprocess.run(C.transport(cl,n,[cl['python'],'-c',code]),capture_output=True,text=True,timeout=35,check=True)
  result.append(dict(node=n['name'],**json.loads(p.stdout)))
 C.atomic_json(W/'VLLM_PAUSED.json',dict(at=time.time(),nodes=result,preserve_gpu4_7=True,reversible_sigstop=True))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();print(json.dumps(main(a.execute),ensure_ascii=False))
