from pathlib import Path
import time,os,signal,concurrent.futures,traceback
import control as X
C=X.C;W=X.W;OUT=X.OUT

def main():
 request=C.read(W/'STOP_NOW_REQUEST.json');deadline=time.time()+1800
 while C.alive(request['controller']):
  if time.time()>deadline:raise TimeoutError('Waiting native save and exit; STOP remains in place')
  time.sleep(2)
 step=C.read(OUT/'status.json')['step'];checkpoint=OUT/f'step_{step:06d}.pt'
 assert checkpoint.with_suffix('.complete.json').exists()
 import torch
 torch.set_num_threads(8);C.activate();saved=C.native('full','checkpoint').checked(checkpoint)
 assert saved['step']==step and len(saved['rank_rng'])==31 and saved['identity']==C.read(OUT/'run.json')
 cl=C.read(OUT/'cluster.json')
 def check(index):
  node=cl['nodes'][index];offset=sum(len(n['gpus'])for n in cl['nodes'][:index]);ranks=list(range(offset,offset+len(node['gpus'])))
  code='out='+repr(str(OUT))+'\nranks='+repr(ranks)+'\n'+'''from pathlib import Path
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
 with concurrent.futures.ThreadPoolExecutor(4)as pool:workers=list(pool.map(check,range(4)))
 assert all(not n['alive']for n in workers)
 monitor=C.read(W/'HOURLY_CONTROLLER.json')
 if C.alive(monitor):
  cmd=Path('/proc',str(monitor['pid']),'cmdline').read_bytes().replace(b'\0',b' ').decode();assert str(W/'watch_hourly.py')in cmd;os.kill(monitor['pid'],signal.SIGTERM)
 C.write(W/'STOP_MONITOR',dict(at=time.time(),reason='Training stopped by user'))
 result=dict(success=True,at=time.time(),step=step,checkpoint=str(checkpoint),checkpoint_sha256=C.read(checkpoint.with_suffix('.complete.json'))['sha256'],model_adam_31rng_checked=True,all_owned_workers_exited=True,workers=workers,successor_launched=False,foreign_jobs_not_signaled=True,gate6500_cancelled=True)
 C.write(W/'STOP_NOW_ACCEPTED.json',result);C.write(W/'STATE.json',dict(stage='stopped_by_user',step=step,at=time.time()));C.write(W/'HOURLY_LATEST.json',dict(stage='stopped_by_user',step=step,at=time.time(),monitor_stopped=True))
 print(result,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  C.write(W/'STOP_NOW_ERROR.json',dict(at=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
