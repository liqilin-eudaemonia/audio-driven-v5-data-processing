"""Finite two-window production measurement, not a recurring monitor."""
import json,time,sys,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;deadline=time.time()+1200
while True:
 states=[json.loads((W/'workers'/f'{n}-gpu-{g}.json').read_text()) for n in ['node17','node8','node4','node12'] for g in range(8)]
 ready=[s for s in states if '+roi-rgb-v1' in s['execution_revision'] and s['completed_clips']>=3 and time.time()-s['heartbeat']<60]
 if any(s.get('last_error') for s in ready):raise RuntimeError('New worker error')
 if len(ready)==32:break
 if time.time()>deadline:raise TimeoutError('32 GPU deployment incomplete')
 time.sleep(10)
time.sleep(60)
(D/'measured_config.json').write_bytes((W/'config.json').read_bytes())
for label in ['D1_after_roi','D2_after_roi']:
 subprocess.run([sys.executable,str(D/'run_window.py'),label,'180'],check=True)
 subprocess.run([sys.executable,str(D/'exact_work.py'),label],check=True)
print('TWO FULL CLUSTER WINDOWS COMPLETE',flush=True)
