import json,time,os,fcntl
from pathlib import Path
p=Path(__file__).resolve().parent
def write(name,data):
 f=p/name;t=f.with_name(f.name+'.tmp');t.write_text(json.dumps(data,indent=2)+'\n');os.replace(t,f)
with (p/'stop_at_8000.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 st=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
 write('STOP_AT_8000_ARMED.json',dict(pid=os.getpid(),start_ticks=st[19],target_step=8000,at=time.time(),reason='User requested stop after complete step8000 for evaluation',reserved_gpu='node8:7'))
 while True:
  s=json.loads((p/'status.json').read_text())
  if s['step']>=8000:
   (p/'STOP').touch()
   write('STOP_AT_8000_TRIGGERED.json',dict(observed_step=s['step'],target_step=8000,at=time.time(),exact_trigger=s['step']==8000))
   break
  if (p/'STOP').exists() or (p/'queue/STOP').exists():
   write('STOP_AT_8000_EARLY_STOP.json',dict(step=s['step'],at=time.time()));raise SystemExit(0)
  time.sleep(.2)
 while True:
  state=json.loads((p/'queue/state.json').read_text())
  marker=p/'step_008000.complete.json'
  if state.get('stage')=='stopped' and marker.exists():
   write('STOP_AT_8000_COMPLETE.json',dict(state=state,checkpoint=json.loads(marker.read_text()),at=time.time()));break
  time.sleep(2)
