"""Kill only the queue leader during healthy E0; verify standby takeover and restore redundancy."""
from queue_common import *
state=read(QUEUE/'state.json')
assert state['stage']=='E0' and state['phase']=='waiting_predecessor' and not state.get('recovery')
old=state['leader'];rank=old['rank'];pid=old['pid'];before=metric_progress(BASELINE)
code='import os,signal;from pathlib import Path;'+f'p={pid};tokens=Path(f"/proc/{{p}}/cmdline").read_bytes().split(b"\\0");assert {str(OUT/"queue_watchdog.py").encode()!r} in tokens;os.kill(p,signal.SIGTERM)'
cmd=[str(PYTHON),'-c',code]
if rank:cmd=['ssh','-i',KEY,'-o','BatchMode=yes','root@'+HOSTS[rank],shlex.join(cmd)]
subprocess.run(cmd,check=True,timeout=20);start=time.monotonic()
while time.monotonic()-start<60:
 state=read(QUEUE/'state.json')
 if (state['leader']['pid']!=pid or state['leader']['rank']!=rank) and state.get('last_error') is None and state.get('progress',{}).get('time',0)>=before['time']:break
 time.sleep(1)
else:raise RuntimeError('Standby did not take over')
assert state['stage']=='E0' and state['phase']=='waiting_predecessor' and not state.get('recovery')
command=[str(PYTHON),str(OUT/'queue_watchdog.py'),'--node-rank',str(rank),'--detach']
if rank:command=['ssh','-i',KEY,'-o','BatchMode=yes','root@'+HOSTS[rank],shlex.join(command)]
r=subprocess.run(command,capture_output=True,text=True,check=True,timeout=40)
watchdogs=[read(QUEUE/f'watchdog{n}.json') for n in range(3)]
assert sum(w['role']=='leader' for w in watchdogs)==1
assert all(time.time()-w['time']<30 for w in watchdogs)
assert read(REFERENCE/'status.json')['state']=='training'
report=dict(status='passed',previous_leader=old,new_leader=state['leader'],takeover_seconds=time.monotonic()-start,
 restored_watchdogs=watchdogs,e0_progress_before=before,e0_progress_after=metric_progress(BASELINE),training_interrupted=False,checked_at=now())
write(OUT/'queue_failover_validation.json',report)
print(json.dumps(report,indent=2))
