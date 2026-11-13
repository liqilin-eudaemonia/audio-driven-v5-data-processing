"""Stop only this experiment's recorded server processes, verifying command identity."""
import json,os,signal,sys,time
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
report=out/'cleanup.json'
records=json.loads(report.read_text()) if report.exists() else {}
for name in sys.argv[1:]:
 if name not in {'joy','minicpm','qwen3-omni'}:raise ValueError(name)
 record=json.loads((out/(name+'.process.json')).read_text());pid=record['pid'];proc=Path('/proc')/str(pid)
 if proc.exists():
  cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if str(root/'scripts/serve_listen_labeling_models.py') not in cmd:raise RuntimeError('PID ownership mismatch; refusing to stop')
  os.kill(pid,signal.SIGTERM)
  for _ in range(50):
   if not proc.exists() or ') Z ' in (proc/'stat').read_text():break
   time.sleep(.1)
 alive=proc.exists() and ') Z ' not in (proc/'stat').read_text()
 records[name]=dict(pid=pid,running=alive,checked_at=datetime.now(timezone.utc).isoformat())
 report.write_text(json.dumps(records,indent=2)+'\n');print(name,records[name])
