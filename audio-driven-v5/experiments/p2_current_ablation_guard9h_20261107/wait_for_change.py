"""Read-only wait on a live guardian and real training progress."""
from pathlib import Path
import json,time
p=Path(__file__).resolve().parent
base=p.parent;feature=base/'p2_current500h_av_feature31_20261107';whisper=base/'p2_current500h_av_whisper31_20261107'
def read(f):return json.loads(f.read_text())
def alive(r):
 try:
  s=Path('/proc',str(r['pid']),'stat').read_text().rsplit(')',1)[1].split();return s[0]!='Z'and s[19]==r['start_ticks']
 except OSError:return False
out=whisper if(whisper/'run.json').exists()else feature
markers={f.name for f in out.glob('step_*.complete.json')};reports={f.name for f in(p/'reports').glob('hour_*.json')};started=time.time()
while True:
 record=read(p/'GUARDIAN.json');state=read(p/'status.json');event=None
 if not alive(record):event={'event':'guardian_not_alive','record':record}
 elif time.time()-state['at']>60:event={'event':'observer_stale','state':state}
 elif state['flags']:event={'event':'anomaly','state':state}
 elif out==feature and(whisper/'run.json').exists():event={'event':'handoff_started'}
 else:
  new_markers={f.name for f in out.glob('step_*.complete.json')}-markers
  new_reports={f.name for f in(p/'reports').glob('hour_*.json')}-reports
  if new_markers:event={'event':'new_checkpoint','files':sorted(new_markers),'work':str(out)}
  elif new_reports:event={'event':'hourly_report','files':sorted(new_reports)}
  elif time.time()>=read(p/'protocol.json')['deadline']:event={'event':'deadline'}
  elif time.time()-started>=1800:event={'event':'wait_interval_finished'}
 if event:
  event.update(at=time.time(),step=state['step']);print(json.dumps(event),flush=True);break
 v=dict(event='verified_wait',at=time.time(),guardian_pid=record['pid'],active=state['active'],step=state['step'],seconds=state['mean_last50_seconds'],flags=state['flags'],handoff_alive=state['handoff_controller_alive'])
 with(p/'verified_wait.jsonl').open('a')as f:f.write(json.dumps(v)+'\n')
 print(json.dumps(v),flush=True);time.sleep(50)
