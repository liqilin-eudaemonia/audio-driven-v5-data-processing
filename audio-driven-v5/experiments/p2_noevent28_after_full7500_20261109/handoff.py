"""Owned exact7500 native save/exit, then independent noEvent660-to28 resume."""
from pathlib import Path
import time,os,signal,subprocess,fcntl,traceback,concurrent.futures
import manager as X
C=X.C;W=X.W;OUT=X.OUT;FULL=W.parent/'p2_ltx_all_lora_full28_batch28_20261109';FW=W.parent/'p2_full28_batch28_20261109';TARGET=7500

def emit(stage,**kw):C.write(W/'HANDOFF_STATE.json',dict(stage=stage,at=time.time(),**kw))
def owned_workers_exited(out):
 cl=C.read(out/'cluster.json')
 def one(n):
  offset=sum(len(v['gpus'])for v in cl['nodes'][:cl['nodes'].index(n)])
  code='out='+repr(str(out))+'\nranks='+repr(list(range(offset,offset+len(n['gpus']))))+'\n'+'''from pathlib import Path
import json
alive=[]
for rank in ranks:
 r=json.loads((Path(out)/'rank_logs'/f'worker_{rank:02d}.json').read_text())
 try:
  s=Path('/proc',str(r['pid']),'stat').read_text().rsplit(')',1)[1].split()
  if s[0]!='Z'and s[19]==r['start_ticks']:alive.append(r['pid'])
 except FileNotFoundError:pass
print(json.dumps(dict(alive=alive)))
'''
  return dict(node=n['name'],**C.remote(cl,n,code))
 with concurrent.futures.ThreadPoolExecutor(4)as pool:return list(pool.map(one,cl['nodes']))
def resources():
 cl=C.read(OUT/'cluster.json')
 def one(n):
  code='selected='+repr(n['gpus'])+'\n'+'''import json,subprocess
uuids={int(x.split(',')[0]):x.split(',')[1].strip()for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
chosen={uuids[g]for g in selected};apps=[]
for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
 a=[v.strip()for v in x.split(',')]
 if a[1]in chosen:apps.append(dict(pid=int(a[0]),uuid=a[1],used=a[2]))
print(json.dumps(dict(apps=apps)))
'''
  return dict(node=n['name'],**C.remote(cl,n,code))
 with concurrent.futures.ThreadPoolExecutor(4)as pool:r=list(pool.map(one,cl['nodes']))
 return r

def main():
 with(W/'handoff.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  ready=C.read(W/'READY.json');controller=C.read(FULL/'CONTROLLER.json');fullrun=C.read(FULL/'run.json');configsha=C.sha(FULL/'config.json')
  assert ready['source']==X.R.source()
  assert C.read(FULL/'status.json')['step']<TARGET and not(FULL/'STOP').exists()and C.alive(controller)
  C.write(W/'HANDOFF_CONTROLLER.json',dict(**C.record(),target=TARGET,full_controller=controller))
  emit('waiting_full7500',target=TARGET)
  while True:
   assert C.sha(FULL/'config.json')==configsha
   step=C.read(FULL/'status.json')['step']
   if step>=TARGET:
    C.request_stop(FULL/'STOP','user-full7500-to-noevent28-20261109',target_step=TARGET,reason='Save full7500, exit, resume noEvent660 with batch28 on same28 GPUs')
    if step!=TARGET:raise RuntimeError('Missed exact7500; STOP preserved, successor blocked')
    break
   if not C.alive(controller):raise RuntimeError('Full trainer exited before7500; successor blocked')
   time.sleep(.2)
  emit('waiting_complete_full7500_and_exit');deadline=time.time()+2400;checkpoint=FULL/'step_007500.pt'
  while not checkpoint.with_suffix('.complete.json').exists()or C.alive(controller):
   if time.time()>deadline:raise TimeoutError('Complete full7500 save/exit timed out')
   time.sleep(2)
  assert C.read(FULL/'status.json')['step']==TARGET
  workers=owned_workers_exited(FULL);assert all(not n['alive']for n in workers)
  import torch
  torch.set_num_threads(8);C.activate();v=C.native('full','checkpoint').checked(checkpoint)
  assert v['step']==TARGET and v['identity']==fullrun and len(v['rank_rng'])==28
  del v
  C.write(W/'FULL7500_STOP_ACCEPTED.json',dict(success=True,step=TARGET,workers=workers,checkpoint_sha256=C.sha(checkpoint),model_adam_rng_checked=True))
  old=C.read(FW/'HOURLY_CONTROLLER.json')
  if C.alive(old):
   assert str(FW/'watch_hourly.py')in Path('/proc',str(old['pid']),'cmdline').read_bytes().replace(b'\0',b' ').decode()
   os.kill(old['pid'],signal.SIGTERM)
  emit('waiting_migration_ready');deadline=time.time()+2400
  while not(W/'MIGRATION_READY.json').exists():
   if time.time()>deadline:raise TimeoutError('NoEvent migration not accepted; successor blocked')
   time.sleep(5)
  assert C.read(W/'MIGRATION_READY.json')['success']
  emit('checking_free28');deadline=time.time()+900
  for attempt in range(2):
   while True:
    r=resources();C.write(W/'PRELAUNCH_RESOURCES.json',dict(at=time.time(),nodes=r))
    if all(not n['apps']for n in r):break
    if time.time()>deadline:raise RuntimeError('Selected28 GPUs occupied; no other jobs stopped')
    time.sleep(10)
   time.sleep(2)
  assert C.read(W/'READY.json')==ready and ready['source']==X.R.source()
  with(W/'launch.log').open('a')as f:p=subprocess.Popen([C.PY,'-u',str(W/'manager.py'),'launch'],cwd=X.ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  emit('starting_noevent28',controller_pid=p.pid,resume_step=660);deadline=time.time()+2400
  while len(C.legacy.rows_after(OUT,660))<8:
   if p.poll()is not None:raise RuntimeError('NoEvent startup controller exited; inspect logs, checkpoint retained')
   if time.time()>deadline:raise TimeoutError('NoEvent first8 updates timed out')
   time.sleep(10)
  subprocess.run([C.PY,'-u',str(W/'accept.py')],cwd=X.ROOT,check=True)
  a=C.read(W/'ACCEPTED.json');assert a['world']==a['global_batch']==28 and a['resumed_from']==660
  C.write(X.S/'ACTIVE_TRAINING.json',dict(at=time.time(),full='paused7500',noevent=str(OUT),state='noevent28_training',monitor=str(W/'HOURLY_CONTROLLER.json'),reserved='node11 GPU4-7'))
  emit('noevent28_verified',step=a['step'],mean_seconds=a['mean_resumed_seconds'])
if __name__=='__main__':
 try:main()
 except BaseException as e:
  C.write(W/'HANDOFF_ERROR.json',dict(at=time.time(),error=repr(e),traceback=traceback.format_exc(),automatic_restart=False));raise
