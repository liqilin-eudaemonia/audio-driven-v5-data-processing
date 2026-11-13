"""Request native STOP during checkpoint6500 writing, then verify save and exit."""
from pathlib import Path
import time,os,json,fcntl,traceback,concurrent.futures
import control as X
C=X.C;W=X.W;OUT=X.OUT;TARGET=6500

def main():
 with(W/'stop6500.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  ready=C.read(W/'READY.json');controller=C.read(OUT/'CONTROLLER.json')
  assert C.sha(OUT/'config.json')==ready['config_sha256']
  assert C.read(OUT/'config.json')['training']['save_every']==500
  assert C.read(OUT/'status.json')['step']<TARGET and not(OUT/'STOP').exists()
  C.write(W/'STOP6500_CONTROLLER.json',dict(**C.record(),target=TARGET,output=str(OUT),training_controller=controller))
  C.write(W/'STOP6500_STATUS.json',dict(stage='armed',target=TARGET,at=time.time()))
  while True:
   step=C.read(OUT/'status.json')['step']
   if step>=TARGET:
    assert C.sha(OUT/'config.json')==ready['config_sha256']
    from gate import request_stop
    created=request_stop(OUT/'STOP','user-stop-at6500-20261109',target_step=TARGET,reason='User requests complete checkpoint6500 then stop; no successor')
    C.write(W/'STOP6500_STATUS.json',dict(stage='stop_requested_during_native_save',observed_step=step,created=created,at=time.time()))
    if step!=TARGET:raise RuntimeError('Target passed; STOP requested, exact6500 not achieved')
    break
   if not C.alive(controller):raise RuntimeError('Trainer exited before6500; not claiming completion')
   time.sleep(.2)
  checkpoint=OUT/'step_006500.pt';deadline=time.time()+1800
  while not checkpoint.with_suffix('.complete.json').exists()or C.alive(controller):
   if time.time()>deadline:raise TimeoutError('Waiting complete6500 save and launcher exit; STOP preserved')
   time.sleep(2)
  assert C.read(OUT/'status.json')['step']==TARGET
  import torch
  torch.set_num_threads(8);C.activate();saved=C.native('full','checkpoint').checked(checkpoint)
  assert saved['step']==TARGET and len(saved['rank_rng'])==31 and saved['identity']==C.read(OUT/'run.json')
  cl=C.read(OUT/'cluster.json')
  def check(node):
   offset=sum(len(n['gpus'])for n in cl['nodes'][:cl['nodes'].index(node)])
   code='out='+repr(str(OUT))+'\nranks='+repr(list(range(offset,offset+len(node['gpus']))))+'\n'+'''from pathlib import Path
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
   return dict(node=node['name'],**C.remote(cl,node,code))
  with concurrent.futures.ThreadPoolExecutor(4)as pool:workers=list(pool.map(check,cl['nodes']))
  assert all(not n['alive']for n in workers)
  C.write(W/'STOP6500_ACCEPTED.json',dict(success=True,at=time.time(),step=TARGET,checkpoint=str(checkpoint),checkpoint_sha256=C.read(checkpoint.with_suffix('.complete.json'))['sha256'],model_adam_31rng_checked=True,all_workers_exited=True,workers=workers,successor_launched=False,foreign_jobs_not_signaled=True))
  C.write(W/'STOP6500_STATUS.json',dict(stage='saved_and_stopped',step=TARGET,at=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  C.write(W/'STOP6500_ERROR.json',dict(at=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
