"""User requests current AV+Whisper6000 full save and stop; never launch another arm."""
from pathlib import Path
import fcntl,importlib.util,json,os,time,traceback
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sequence_stop_helpers',HERE/'sequence_v2.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
from p2_listen.contracts import read,sha,atomic_json
PARENT=helpers.PARENT

def state(stage,**extra):
 value=dict(stage=stage,at=time.time(),controller_pid=os.getpid(),target_step=6000,launch_next=False,**extra)
 atomic_json(HERE/'STOP6000_STATE.json',value);print(json.dumps(value),flush=True)

def main():
 with (HERE/'stop6000.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  atomic_json(HERE/'STOP6000_CONTROLLER.json',dict(pid=os.getpid(),start_ticks=Path('/proc',str(os.getpid()),'stat').read_text().split()[21],at=time.time()))
  # QUEUE_STOP intentionally cancels all training launches, not this save-only watcher.
  def cancelled():
   if (HERE/'STOP6000_WATCH_CANCEL').exists():raise RuntimeError('Explicit save-only watcher cancellation')
  helpers.cancelled=cancelled
  run=read(PARENT/'run.json')
  if any(sha(p)!=h for p,h in run['source'].items()):raise ValueError('Current training source changed')
  controller=read(PARENT/'CONTROLLER.json')
  state('armed_waiting_current6000',current_step=read(PARENT/'status.json')['step'])
  while not helpers.boundary(read(PARENT/'status.json')['step']):
   cancelled()
   if (PARENT/'STOP').exists():raise RuntimeError('Existing early training STOP preserved')
   if not helpers.alive(controller):raise RuntimeError('Training controller exited before6000')
   time.sleep(.05)
  if not (PARENT/'STOP').exists():atomic_json(PARENT/'STOP',dict(owner='user-save6000-stop-for-unified-evaluation',target_step=6000,at=time.time()))
  state('waiting_complete_save_and_exit')
  helpers.await_exit(PARENT,controller)
  receipt=helpers.verify_checkpoint(PARENT,'parent')
  atomic_json(HERE/'STOP6000_FINISHED.json',dict(success=True,at=time.time(),receipt=receipt,launch_next=False))
  state('saved_verified_stopped',receipt=receipt)

if __name__=='__main__':
 try:main()
 except Exception:
  atomic_json(HERE/'STOP6000_ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()))
  state('blocked');raise
