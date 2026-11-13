from pathlib import Path
import os,signal,json,time
W=Path(__file__).resolve().parent
seen=set()
while not(W/'RESUMED_ACCEPTED.json').exists()and not(W/'DRIVER_ERROR.json').exists():
 try:
  state=json.loads((W/'STATE.json').read_text());case=state.get('case','');key=(case,state['at'])
  if state.get('stage')=='measuring'and case.startswith('smoke8')and time.time()-state['at']>150 and key not in seen:
   pid=state['pid'];children=Path('/proc',str(pid),'task',str(pid),'children').read_text().split()
   for child in children:
    p=Path('/proc')/child
    try:cmd=(p/'cmdline').read_bytes()
    except FileNotFoundError:continue
    if b'ssh'in cmd and b'172.29.104.237'in cmd and str(W).encode()in cmd and b'smoke31.py'in cmd:
     os.kill(int(child),signal.SIGTERM);seen.add(key)
     (W/case/'WALL_TIMEOUT.json').write_text(json.dumps(dict(at=time.time(),phase=case,limit_seconds=150,terminated_own_ssh=int(child)),indent=2));break
 except (FileNotFoundError,ProcessLookupError):pass
 time.sleep(5)
