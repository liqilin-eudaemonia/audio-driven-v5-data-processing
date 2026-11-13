import json,os,signal,time,sys
from pathlib import Path
RUN=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_features_full_crop384_20261025');node=sys.argv[1];folder=RUN/'nodes'/node
state=json.loads((folder/'status.json').read_text());pid=state['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert str(RUN).encode() in cmd and cmd[cmd.index(b'--role')+1]==b'node' and cmd[cmd.index(b'--node')+1]==node.encode() and b'--watchdog' not in cmd and os.getpgid(pid)==pid
workers=[]
for name,record in state['children'].items():
 p=record['pid'];path=Path(f'/proc/{p}/cmdline')
 if not path.exists():continue
 c=path.read_bytes().split(b'\0')
 if not c:continue
 assert str(RUN).encode() in c and any(b'run_p2_full_features.py' in x for x in c) and c[c.index(b'--worker')+1]==name.encode() and os.getpgid(p)==p
 assert record['role'] in ['features','media']
 if record['role']=='features':assert record['gpu'] in ([4,5,6,7] if node=='node8' else list(range(8)))
 workers.append(dict(worker=name,pid=p,role=record['role']))
expected=8 if node=='node8' else 16;assert sum(x['role']=='features' for x in workers)==expected
record=dict(node=node,utc=time.time(),supervisor_pid=pid,workers=workers)
(RUN/f'frozen_config_runtime/restart-{node}.json').write_text(json.dumps(record,indent=2))
os.killpg(pid,signal.SIGTERM)
for w in workers:
 try:os.killpg(w['pid'],signal.SIGTERM)
 except ProcessLookupError:pass
print(json.dumps(dict(node=node,own_workers_restarted=len(workers),own_supervisor_restarted=pid,existing_watchdog_will_resume=True)),flush=True)
