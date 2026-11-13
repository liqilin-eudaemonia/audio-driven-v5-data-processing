import json,os,subprocess,time,hashlib
from pathlib import Path
r=Path('/private/liqi_workspace/project_code/audio-driven-v5');o=r/'experiments/p2_j_teacher500h_node8_20261102'
from datetime import datetime,timezone
start=time.time()
while not (o/'pilot_v3/benchmark-gpu0.json').exists():
 if time.time()-start>600:raise TimeoutError('Pilot verification has not passed; production not started')
 time.sleep(5)
reports=[json.loads((o/'pilot_v3'/f'benchmark-gpu{i}.json').read_text()) for i in range(8)]
assert all(x['success'] and x['samples']==32 for x in reports)
proof=reports[0];assert proof['full25fps_reference_equal']==32 and proof['guided_cache_teacher_read_ok']==32
assert all(row[mod]['exact'] for row in proof['batch_vs_single'] for mod in ('video','audio'))
assert '4 passed' in (o/'tests.log').read_text()
protocol=json.loads((o/'production/protocol.json').read_text())
assert protocol['samples']==285638
for path,h in protocol['code'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
for _ in range(30):
 csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
 values=[int(x.split(',')[1]) for x in csv.splitlines()]
 if len(values)==8 and max(values)<2048:break
 time.sleep(2)
else:raise RuntimeError('Node8 GPUs are still occupied; no processes were killed')
cmd=['/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python',str(r/'scripts/p2_j_teacher.py'),'node','--work',str(o/'production'),'--batch','8','--producers','4','--prefetch','16','--per-gpu','2']
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYAV_LOGGING='off')
with (o/'production/node.log').open('a') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=r,env=env,start_new_session=True)
record=dict(state='production_launched',pid=p.pid,time=time.time(),utc=datetime.now(timezone.utc).isoformat(),command=cmd,nodes=['172.29.104.233'],physical_gpus=8,workers=16,batch_scheduling=8,cpu_producers_per_worker=4,prefetch_per_worker=16,unchanged_source_cache=True)
(o/'production/launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
