"""User-authorized strict recovery of the remaining full-LoRA no-Whisper arm."""
from pathlib import Path
import importlib.util,os,sys,threading,time,traceback,fcntl
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
entry=ROOT/'experiments/p2_ablation28_guard9h_20261111/recover.py'
spec=importlib.util.spec_from_file_location('ablation_recovery',entry);recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
Q=recovery.Q
from p2_listen.contracts import atomic_json,read

if __name__=='__main__':
 with(W/'controller.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  atomic_json(W/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),arm='no_whisper',resume=str(Q.W/'no_whisper/step_001760.pt'),target=6643,world=28,batch=28,save_every=500))
  done=threading.Event();threading.Thread(target=Q.monitor,args=(done,),daemon=True).start()
  try:
   recovery.run()
   atomic_json(W/'COMPLETE.json',dict(success=True,at=time.time(),arm='no_whisper',step=6643))
  except BaseException as error:
   atomic_json(W/'FAILURE.json',dict(at=time.time(),error=repr(error),traceback=traceback.format_exc()));raise
  finally:done.set()
