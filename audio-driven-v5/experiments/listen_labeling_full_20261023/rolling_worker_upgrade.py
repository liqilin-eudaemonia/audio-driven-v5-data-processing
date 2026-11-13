import json,os,signal
from pathlib import Path
root=Path(__file__).resolve().parent/'production';parent=json.loads((root/'process.json').read_text())['pid'];state=json.loads((root/'supervisor_heartbeat.json').read_text())
for rank in map(str,range(8)):
 record=state['workers'].get(rank)
 if not record:continue
 pid=record['pid'];proc=Path('/proc')/str(pid)
 if not proc.exists():continue
 status=(proc/'status').read_text();ppid=int(next(l.split()[1] for l in status.splitlines() if l.startswith('PPid:')))
 cmd=(proc/'cmdline').read_bytes()
 if ppid!=parent or b'multiprocessing.spawn' not in cmd:raise RuntimeError('Worker ownership mismatch')
 os.kill(pid,signal.SIGTERM);print('Upgrade worker',rank,pid)
