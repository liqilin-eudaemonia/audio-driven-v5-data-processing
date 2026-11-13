"""User-authorized VLLM termination, only on the current training GPU allocation."""
import json,subprocess,time,concurrent.futures
import common as C
import pause_competitors as P
W=C.WORK
policy=C.read(W/'USER_CONTENTION_POLICY.json');assert policy.get('user_approved')is True and policy.get('kill_training_gpu_vllm')is True
cluster=C.read(C.OUT/'cluster.json')
# Fresh NVML/proc ownership pass reuses the reviewed, cross-reserved-GPU-safe detector.
def query(n):
 payload=dict(gpus=n['gpus'],execute=False)
 return dict(node=n['name'],**C.C.remote(cluster,n,P.REMOTE.replace('PAYLOAD',repr(json.dumps(payload)))))
with concurrent.futures.ThreadPoolExecutor(4)as pool:plan=list(pool.map(query,cluster['nodes']))
C.atomic_json(W/'VLLM_KILL_PLAN.json',dict(at=time.time(),nodes=plan,policy=policy))
KILL=r'''from pathlib import Path
import json,os,signal,time,subprocess
x=json.loads(PAYLOAD);expected=x['records'];selected=set(x['gpus'])
uuids={int(a.split(',')[0]):a.split(',')[1].strip()for a in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
apps=[tuple(a.strip()for a in r.split(','))for r in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()]
allowed={uuids[g]for g in selected};results=[]
def alive(r):
 try:
  s=Path('/proc',str(r['pid']),'stat').read_text().rsplit(')',1)[1].split()
  return s[0]!='Z'and s[19]==r['start_ticks']
 except FileNotFoundError:return False
for r in expected:
 if not alive(r):results.append(dict(pid=r['pid'],already_exited=True));continue
 d=Path('/proc',str(r['pid']))
 if not(d/'comm').read_text().startswith('VLLM::'):raise RuntimeError('Command changed; refuse termination')
 occupied={u for p,u in apps if int(p)==r['pid']}
 if not occupied or not occupied<=allowed or r['uuid']not in occupied:raise RuntimeError('GPU ownership changed or overlaps reserved GPU')
 os.kill(r['pid'],signal.SIGTERM);results.append(dict(pid=r['pid'],sigterm=True))
deadline=time.time()+12
while any(alive(r)for r in expected)and time.time()<deadline:time.sleep(.5)
for r in expected:
 if alive(r):
  if not Path('/proc',str(r['pid']),'comm').read_text().startswith('VLLM::'):raise RuntimeError('PID identity changed before SIGKILL')
  os.kill(r['pid'],signal.SIGKILL)
  next(v for v in results if v['pid']==r['pid'])['sigkill']=True
for _ in range(20):
 if not any(alive(r)for r in expected):break
 time.sleep(.5)
if any(alive(r)for r in expected):raise RuntimeError('Selected VLLM processes have not exited')
print(json.dumps(dict(results=results,all_selected_vllm_exited=True,training_not_signaled=True,reserved_gpus_not_signaled=True)))
'''
results=[]
# Mutations are deliberately sequential and never retried after an ambiguous SSH result.
for n,item in zip(cluster['nodes'],plan):
 if not item['records']:continue
 code=KILL.replace('PAYLOAD',repr(json.dumps(dict(gpus=n['gpus'],records=item['records']))))
 p=subprocess.run(C.transport(cluster,n,[cluster['python'],'-c',code]),capture_output=True,text=True,timeout=45,check=True)
 result=dict(node=n['name'],**json.loads(p.stdout));results.append(result)
 C.atomic_json(W/'VLLM_KILL_PROGRESS.json',dict(at=time.time(),nodes=results))
C.atomic_json(W/'VLLM_KILLED.json',dict(success=True,at=time.time(),nodes=results,plan=plan,training_not_restarted=True,reserved_gpu4_7_preserved=True))
print(json.dumps(dict(success=True,killed=sum(len(n['records'])for n in plan),nodes=results)),flush=True)
