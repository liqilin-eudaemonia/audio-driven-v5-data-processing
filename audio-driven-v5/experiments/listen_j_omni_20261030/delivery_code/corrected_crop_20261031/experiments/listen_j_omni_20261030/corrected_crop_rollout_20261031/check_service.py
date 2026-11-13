import sys,time,json
from pathlib import Path
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write,result_path
from listen_benchmark.j_omni_crop import _measure,_services
W=R/'experiments/listen_j_omni_20261030';D=W/'corrected_crop_rollout_20261031';C=read(W/'config.json');S=C['bounded_crop_input'];cache=Path(S['nod_cache']);rows=[x for x in read(D/'replay.json')['samples'] if x['crop']['coordinates_changed']];out=[]
for item in rows:
 sid=item['sample_id'];req=result_path(cache,'requests',sid);target=D/'service_replay'/f'{sid}.json';tick=time.time()
 _measure(dict(row=str(req),output=str(target),resources=C['nodes']['node12']['resource_work']),S,C,C['nodes']['node12']['resource_work'],cache)
 before=read(result_path(cache,'nod',sid));after=read(target)
 for k in ['source','media','pose','scores','events','status','parameters','model_sha256']:assert before[k]==after[k],k
 out.append(dict(sample_id=sid,seconds=time.time()-tick,passed=True));print(out[-1],flush=True)
assert len(_services)==1
for service in _services.values():service['process'].stdin.close();service['process'].wait(timeout=30)
write(D/'service_replay.json',dict(passed=True,one_reused_service=True,samples=out))
