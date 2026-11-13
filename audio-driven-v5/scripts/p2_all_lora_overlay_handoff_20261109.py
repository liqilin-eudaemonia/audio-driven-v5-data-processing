"""One-time resource-policy switch after5500 latent commits; never interrupt render."""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_testsets_eval.protocol import read,save,sha
import p2_all_lora_eval_watch9h_20261109 as M

WATCH=M.WORK
WORK=WATCH/'resource_overlay_r3'
RUN=ROOT/'experiments/p2_ltx_all_lora5500_seamless200_sft2752_gpu7_20261109'
CHAIN=M.CHAIN
QUEUE=ROOT/'scripts/p2_all_lora_followup_queue_20261109.py'
OBSERVER=ROOT/'scripts/p2_all_lora_eval_watch9h_20261109.py'
OVERLAY=ROOT/'scripts/p2_all_lora_followups_gpu7_overlay_20261109.py'


def alive(pid,script):return M.process_identity(pid,script)['live']


def launch(script,out):
 cmd=['env','PYTHONPATH=','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1',
      '/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'if script==QUEUE else'/private/software/conda/bin/python','-u',str(script)]
 with out.open('ab',buffering=0)as log:return subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def stop_owned_controller(directory,script):
 f=directory/'CONTROLLER.json'
 if not f.exists():return None
 c=read(f);pid=c['pid']
 if alive(pid,script):os.kill(pid,signal.SIGTERM)
 return pid


def main():
 WORK.mkdir(exist_ok=True)
 with(WORK/'handoff.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=read(WORK/'PREPARED_NOT_ACTIVATED.json')
  for path,digest in plan['code'].items():assert sha(path)==digest,'Candidate changed: '+path
  shared=read(RUN/'shared_items.json');arm=read(RUN/'protocol.json')['arms'][0]['name']
  assert len(shared)==200 and len({x['key']for x in shared})==200
  last=sorted(shared,key=lambda x:(-x['geometry']['native_frames'],x['key']))[-1]
  marker=RUN/'latents'/arm/last['key']/'complete.json'
  save(WORK/'WAITING.json',dict(pid=os.getpid(),at=time.time(),last_marker=str(marker),only_after_render200=True))
  while not marker.exists():
   if(WORK/'STOP').exists()or(WATCH/'STOP').exists()or(CHAIN/'STOP').exists()or(RUN/'STOP').exists():return
   if time.time()>=read(WATCH/'CONFIG.json')['end_at']:return
   time.sleep(1)
  assert all((RUN/'latents'/arm/x['key']/'complete.json').exists()for x in shared)
  resource=M.remote();foreign=sum(float(a['used_mib'])for a in resource['apps']if not a.get('owned')and str(a.get('used_mib','')).replace('.','',1).isdigit())
  if foreign<=40*1024:
   save(WORK/'NOT_NEEDED.json',dict(at=time.time(),foreign_mib=foreign,reason='foreign_reservation_released_original_policy_retained'));return
  current=read(CHAIN/'status.json')
  assert current.get('step')==5500 and current.get('stage')=='evaluation'
  if any((p/'STOP').exists()for p in [WORK,WATCH,CHAIN,RUN]):return
  folder=WORK/('activation_'+str(time.time_ns()));folder.mkdir()
  metadata={str(RUN/'latents'/arm/x['key']/'metadata.json'):sha(RUN/'latents'/arm/x['key']/'metadata.json')for x in shared}
  save(folder/'BEFORE.json',dict(at=time.time(),generated=200,resource=resource,metadata_sha256=metadata))
  for p in [QUEUE,OBSERVER,CHAIN/'PLAN.json',WATCH/'CONFIG.json',WATCH/'STATE.json']:
   target=folder/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
  observer_stopped=False;restarted_observer=None
  try:
   stop_owned_controller(WATCH,OBSERVER);observer_stopped=True
   stop_owned_controller(CHAIN,QUEUE)
   for i in range(15):
    view=M.remote();owned=[a for a in view['apps']if a.get('owned')and a.get('work')==str(RUN)]
    if M.lock_free(CHAIN/'queue.lock')and M.lock_free(RUN/'controller.lock')and not owned:break
    time.sleep(3)
   else:raise RuntimeError('Cannot confirm own controller/worker cleanup; no source switch performed')
   assert all(sha(path)==digest for path,digest in metadata.items())
   assert all(sha(path)==digest for path,digest in read(RUN/'protocol.json')['execution_code'].items())
   for target,candidate in [(QUEUE,WORK/'candidate_queue.py'),(OBSERVER,WORK/'candidate_watcher.py')]:
    temp=target.with_name(target.name+'.overlay_pending');temp.write_bytes(candidate.read_bytes());temp.replace(target)
   p=read(CHAIN/'PLAN.json');p['code'][str(QUEUE)]=sha(QUEUE);p['code'][str(OVERLAY)]=sha(OVERLAY)
   p.update(resource_policy='shared_gpu7_bounded_v1',resource_policy_activated_at=time.time());save(CHAIN/'PLAN.json',p)
   c=read(WATCH/'CONFIG.json');c['code'][str(OBSERVER)]=sha(OBSERVER);c['revision']='phase-same-sample-overlay-owner-v3';save(WATCH/'CONFIG.json',c)
   # Keep every original checkpoint, sampler and condition fingerprint frozen.
   for step in (5500,7000,7500):
    w=ROOT/f'experiments/p2_ltx_all_lora{step}_seamless200_sft2752_gpu7_20261109'
    assert all(sha(f)==digest for f,digest in read(w/'protocol.json')['execution_code'].items())
   for pth in [CHAIN/'FAILURE.json',RUN/'FAILURE.json']:
    if pth.exists():pth.replace(folder/(pth.parent.name+'_FAILURE.json'))
   process=launch(QUEUE,CHAIN/'resource_overlay_controller.log')
   for i in range(60):
    f=CHAIN/'CONTROLLER.json'
    if f.exists()and read(f)['pid']==process.pid and alive(process.pid,QUEUE):break
    if process.poll()is not None:raise RuntimeError('New shared-resource queue exited; inspect log')
    time.sleep(.5)
   else:raise RuntimeError('New queue ownership not confirmed')
   restarted_observer=launch(OBSERVER,WATCH/'controller_r3.log')
   accepted=dict(success=True,at=time.time(),generated_latents_preserved=200,no_render_interrupted=True,
       queue_pid=process.pid,watcher_pid=restarted_observer.pid,only_physical_gpu=7,
       foreign_apps_not_terminated=True,bounded_memory_policy=True,full_gpu_acceptance_pending=True,
       evidence=str(folder),original_window_end=c['end_at'])
   save(WORK/'ACTIVATED.json',accepted);save(folder/'AFTER.json',accepted);print(json.dumps(accepted),flush=True)
  finally:
   if observer_stopped and restarted_observer is None:
    existing=read(WATCH/'CONTROLLER.json')
    if not alive(existing['pid'],OBSERVER):launch(OBSERVER,WATCH/'controller_handoff_recovery.log')


if __name__=='__main__':
 try:main()
 except Exception as exc:save(WORK/'FAILURE.json',dict(at=time.time(),error=repr(exc),traceback=traceback.format_exc()));raise
