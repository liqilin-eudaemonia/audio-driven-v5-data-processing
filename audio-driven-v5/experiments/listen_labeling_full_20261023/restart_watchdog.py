import json,os,signal,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;project=root.parents[1];old=json.loads((root/'watchdog/process.json').read_text());pid=old['pid'];proc=Path('/proc')/str(pid)/'cmdline'
if proc.exists() and proc.read_bytes():
 if b'scripts/watch_listen_labeling_full.py' not in proc.read_bytes():raise RuntimeError('Watchdog process identity mismatch')
 os.kill(pid,signal.SIGTERM)
 for _ in range(50):
  if not proc.exists() or not proc.read_bytes():break
  time.sleep(.1)
(root/'watchdog'/'previous_process.json').write_text(json.dumps(old,indent=2)+'\n')
cmd=['/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python',str(project/'scripts/watch_listen_labeling_full.py'),'--root',str(root),'--hours',str(max(0,(old['deadline']-time.time())/3600))]
p=subprocess.Popen(cmd,cwd=project,stdin=subprocess.DEVNULL,stdout=open(root/'watchdog.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('New watchdog',p.pid)
