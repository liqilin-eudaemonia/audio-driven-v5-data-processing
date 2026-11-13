"""One bounded rolling deployment; never restarts unrelated processes."""
import sys,time,json,subprocess,concurrent.futures
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1];sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import read,write,locked
nodes=['node17','node8','node4','node12'];marker='+roi-rgb-v1'
assert read(D/'roi_parity.json')['complete']
s=read(W/'workers/node12-gpu-7.json');assert marker in s['execution_revision'] and s['completed_clips']>=5 and not s.get('last_error')
with locked(W/'config-edit.lock'):
 c=read(W/'config.json');write(D/'before_full_roi_config.json',c)
 for p in c['input_optimization_profiles'].values():p.update(roi_crop=True,shared_raw_decode=True)
 # This already-completed one-shot benchmark must not repeat on every restart.
 c.pop('startup_input_speed_benchmarks',None)
 write(W/'config.json',c)
records=[]
for gs in [(0,1,2,3),(4,5,6,7)]:
 def launch(n):
  p=subprocess.run([sys.executable,str(D/'roll_roi.py'),n,*map(str,gs)],capture_output=True,text=True,timeout=75)
  if p.returncode:raise RuntimeError(p.stderr)
  return json.loads(p.stdout)
 with concurrent.futures.ThreadPoolExecutor(4) as pool:actions=list(pool.map(launch,nodes))
 rec=dict(gpus=gs,actions=actions,start=time.time(),stage='waiting');records.append(rec);write(D/'rollout.json',records)
 deadline=time.time()+900
 while True:
  states=[read(W/'workers'/f'{n}-gpu-{g}.json') for n in nodes for g in gs]
  failures=[(x['node'],x['gpu'],x.get('last_error')) for x in states if marker in x.get('execution_revision','') and (x.get('last_error') or x['failed_clips'])]
  if failures:raise RuntimeError(str(failures))
  ready=[x for x in states if marker in x.get('execution_revision','') and x['completed_clips']>=3 and time.time()-x['heartbeat']<60]
  rec.update(ready=len(ready),heartbeat=time.time());write(D/'rollout.json',records)
  if len(ready)==16:break
  if time.time()>deadline:raise TimeoutError('Production recovery deadline exceeded')
  time.sleep(10)
 rec.update(stage='productive',end=time.time());write(D/'rollout.json',records);print(json.dumps(rec),flush=True)
print('ALL 32 ROI WORKERS PRODUCTIVE',flush=True)
