"""Wait on verified-live watchdog handles; report hourly milestones or faults."""
from pathlib import Path
import time,sys,subprocess,os
import common as C
from agent_errors import pending_error
W=C.WORK;last=-1
C.atomic_json(W/'AGENT_WAITER.json',C.spawn_record(type('P',(),{'pid':os.getpid()})()))
while True:
 for file in ['SUPERVISOR.json','GUARDIAN.json']:
  if file=='SUPERVISOR.json'and time.time()>=C.read(W/'protocol.json')['deadline']:continue
  if file=='GUARDIAN.json'and(W/'WINDOW_FINISHED.json').exists():continue
  if not C.alive(C.read(W/file)):print('WATCHDOG_HANDLE_EXITED',file,flush=True);sys.exit(2)
 error=pending_error(W)
 if error:
  print('NEEDS_AGENT',error,flush=True);sys.exit(3)
 reports=list((W/'reports').glob('hour_*.json'));slot=max((int(f.stem.split('_')[-1])for f in reports),default=-1)
 if slot>last:
  last=slot;r=C.read(W/'reports'/f'hour_{slot:02d}.json');print('HOUR',slot,'step',r['health']['step'],'healthy28',r['cluster']['healthy28'],'flags',r['health']['flags'],flush=True)
 if(W/'WINDOW_FINISHED.json').exists():
  subprocess.run([C.C.PY,'-u',str(W/'final_audit.py')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'));sys.exit(0)
 time.sleep(30)
