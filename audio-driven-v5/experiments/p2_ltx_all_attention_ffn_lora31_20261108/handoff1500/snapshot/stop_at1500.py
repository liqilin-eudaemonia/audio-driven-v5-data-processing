from pathlib import Path
import json,time,os,fcntl
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
OLD=ROOT/'experiments/p2_current_component_ablations31_20261108/no_event'
HERE=Path(__file__).resolve().parent

def put(path,value):
 tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)
with (HERE/'stop.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 put(HERE/'STOP_CONTROLLER.json',dict(pid=os.getpid(),at=time.time()))
 try:
  while True:
   if (HERE/'CANCEL').exists():raise RuntimeError('User cancelled handoff')
   step=json.loads((OLD/'status.json').read_text())['step']
   if step>1500:raise RuntimeError('Passed boundary; preserve training and require recovery decision')
   if step==1500:
    if (OLD/'STOP').exists():raise RuntimeError('Unexpected existing STOP')
    put(OLD/'STOP',dict(owner='user-switch-to-all-lora-after1500',target_step=1500,at=time.time()))
    put(HERE/'STOP_ARMED.json',dict(success=True,step=1500,at=time.time(),waiting_for_complete_checkpoint=True))
    break
   if (OLD/'STOP').exists():raise RuntimeError('Current run stopped before1500')
   time.sleep(.05)
 except Exception as e:
  put(HERE/'STOP_ERROR.json',dict(error=repr(e),at=time.time()));raise
