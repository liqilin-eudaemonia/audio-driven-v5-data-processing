"""Explicitly authorized handoff: finish all12 ablations before GLARE GPU work."""
import fcntl,json,os,signal,subprocess,time,traceback
from pathlib import Path
import run as trial
BASE=trial.BASE;UPSTREAM=trial.ROOT/'experiments/p2_ablation12_test64_qwen7500_24gpu_20261113'
read,save,sha=trial.read,trial.save,trial.sha

def dependency_ready(path=UPSTREAM):
 if (path/'STOP').exists():return False,'upstream_paused'
 if path==UPSTREAM and (BASE/'ALLOW_TAIL_OVERLAP.json').exists():
  state=read(path/'status.json')
  if state.get('generated')==768 and all(j['stage']=='metric' for j in state.get('active',[])):
   return True,'generated768_overlap_remaining_scores_on_separate_gpus'
 if not (path/'COMPLETE.json').exists():return False,'waiting_for_all12_ablation_scores'
 complete=read(path/'COMPLETE.json')
 if not (complete.get('success') is True and complete.get('outputs')==768 and complete.get('combinations')==12):raise ValueError('Unexpected upstream completion scope')
 # COMPLETE is written just before exiting; also require controller to release its lock.
 with (path/'controller.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False,'waiting_for_upstream_controller_exit'
 return True,'ready'

def main():
 signal.signal(signal.SIGTERM,lambda *_:exit(143))
 child=None
 with (BASE/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);trial.verify()
  save(BASE/'QUEUE_CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),upstream=str(UPSTREAM)))
  contract=read(BASE/'QUEUE_PREPARED.json')
  try:
   while not (BASE/'STOP').exists():
    assert sha(UPSTREAM/'PLAN.json')==contract['upstream_plan_sha256'] and sha(UPSTREAM/'PREPARED.json')==contract['upstream_prepared_sha256']
    ready,reason=dependency_ready()
    status=read(UPSTREAM/'status.json') if (UPSTREAM/'status.json').exists() else {}
    save(BASE/'QUEUE_STATE.json',dict(at=time.time(),state=reason,upstream_generated=status.get('generated'),upstream_total=768,upstream_completed_combinations=len(status.get('completed_combinations',[])),next_step=3935,next_samples=256,next_seed=137,new_gpu_workers_started=False))
    if ready:break
    time.sleep(30)
   if (BASE/'STOP').exists():return
   trial.verify()
   complete=read(UPSTREAM/'COMPLETE.json') if (UPSTREAM/'COMPLETE.json').exists() else None
   save(BASE/'HANDOFF_ACCEPTED.json',dict(at=time.time(),upstream_complete=complete,upstream_controller_lock_released=reason=='ready',handoff_reason=reason,upstream_state=read(UPSTREAM/'status.json'),step=3935,seed=137))
   if not (BASE/'COMPLETE.json').exists():
    with (BASE/'controller.log').open('ab',buffering=0) as log:
     child=subprocess.Popen([str(trial.m.GENPY),'-u',str(BASE/'run_two_nodes.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(BASE/'QUEUE_STATE.json',dict(at=time.time(),state='running_glare3935',pid=child.pid,new_gpu_workers_started=True))
    code=child.wait()
    if code:raise RuntimeError(f'GLARE controller exited {code}; see controller.log')
   if not (BASE/'COMPLETE.json').exists():
    save(BASE/'QUEUE_STATE.json',dict(at=time.time(),state='paused',completed=False));return
   # The result-page request persists: publish only after full256 scoring/merge completes.
   with (BASE/'page_update.log').open('ab') as log:
    subprocess.run([str(trial.m.PYTHON) if hasattr(trial.m,'PYTHON') else '/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python',str(trial.ROOT/'scripts/render_glare_checkpoint_comparison.py')],cwd=trial.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   save(BASE/'QUEUE_COMPLETE.json',dict(success=True,at=time.time(),step=3935,samples=256,seed=137,page_updated=True))
   save(BASE/'QUEUE_STATE.json',dict(at=time.time(),state='complete',completed=True,page_updated=True))
  except BaseException:
   save(BASE/'QUEUE_FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
  finally:
   if child is not None and child.poll() is None:
    child.terminate()
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:pass
if __name__=='__main__':main()
