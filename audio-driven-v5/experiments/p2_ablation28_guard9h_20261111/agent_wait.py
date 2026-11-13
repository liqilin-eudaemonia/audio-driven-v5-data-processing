"""Wait on exact live watcher PID and heartbeat; announce audits and failures."""
from pathlib import Path
import sys,time,json,os,subprocess
G=Path(__file__).resolve().parent;sys.path.insert(0,str(G));import watch as M
p=M.read(G/'protocol.json');owner=M.record(M.read(G/'CONTROLLER.json')['pid']);seen=set();started=time.time()
M.atomic_json(G/'AGENT_WAITER.json',dict(pid=os.getpid(),watcher=owner,started=started))
while True:
 if (G/'WINDOW_FINISHED.json').exists():
  print('Window ended; running complete final audit',flush=True)
  result=subprocess.run([M.Q.PY,'-u',str(G/'final_audit.py')],env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
  sys.exit(result.returncode)
 if not M.alive(owner):raise RuntimeError('Confirmed watcher process disappeared before window end')
 heartbeat=M.read(G/'HEARTBEAT.json');age=time.time()-heartbeat['at']
 if age>600:raise RuntimeError('Watcher heartbeat stale; inspect before any restart')
 for f in [*G.glob('hour*.json'),*G.glob('*ALERT*.json')]:
  key=(f.name,f.stat().st_mtime)
  if key not in seen:
   seen.add(key);print(json.dumps(dict(at=time.time(),file=f.name,watcher_live=True,heartbeat_age=age)),flush=True)
 M.atomic_json(G/'VERIFIED_WAIT.json',dict(at=time.time(),watcher_live=True,watcher=owner,step=heartbeat['step'],arm=heartbeat['arm'],heartbeat_age=age,deadline=p['deadline']))
 time.sleep(60)
