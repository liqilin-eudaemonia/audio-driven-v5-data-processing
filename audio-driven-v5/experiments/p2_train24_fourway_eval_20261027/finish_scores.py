import sys,os,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from p2_glare_eval.common import read,atomic_json
arms=['E0','E1','E0F','E1F'];PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python';MP=read(ROOT/'configs/p2_omni_runtime.lock.json')['paths']['metrics_python']
while not all((OUT/arm/'omni'/f'shard-{i}.json').exists() and read(OUT/arm/'omni'/f'shard-{i}.json').get('finished') for arm in arms for i in range(6)):time.sleep(10)
with (OUT/'omni_aggregate.log').open('w') as log:subprocess.run([MP,str(OUT/'aggregate_omni.py')],stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
print('omni aggregate complete',flush=True)
while not all((OUT/a/'glare/results.json').exists() for a in arms):time.sleep(10)
sets=[{(r['sample_id'],r['sample_seed']) for r in read(OUT/a/'glare/detected_pairs.json')} for a in arms];shared=set.intersection(*sets)
ids=sorted({sid for sid,s in shared if (sid,137) in shared and (sid,251) in shared});keys=[(sid,s) for sid in ids for s in [137,251]]
atomic_json(OUT/'glare_common_scope.json',dict(sample_ids=ids,keys=keys,samples_per_seed=len(ids),planned_samples=24,selection='intersection of successful independent detections across all four arms and both seeds; post-hoc supported subset, not full-set result',full_set_status={a:{k:read(OUT/a/'glare/results.json')[k] for k in ['status','valid_pairs','failed_pairs']} for a in arms}))
print('glare common clips',len(ids),flush=True)
if ids:
 while not all((OUT/a/'glare_common.json').exists() for a in arms):time.sleep(10)
with (OUT/'compare.log').open('w') as log:subprocess.run([MP,str(OUT/'compare_fourway.py')],stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
atomic_json(OUT/'scoring_finished.json',dict(finished=True,time=time.time()));print('comparison complete',flush=True)
