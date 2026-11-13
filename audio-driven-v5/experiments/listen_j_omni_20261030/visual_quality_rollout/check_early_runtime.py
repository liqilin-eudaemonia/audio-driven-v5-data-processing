import json,sys,time
from pathlib import Path
r=Path('/private/liqi_workspace/project_code/audio-driven-v5');w=r/'experiments/listen_j_omni_20261030';sys.path.insert(0,str(r/'src'))
from listen_benchmark.j_annotation import read,write
from listen_benchmark.j_visual_quality_schedule import ensure_nod_quality
job=w/'jobs/naturalistic_V02_S3104_I00000483.json';config=read(w/'config.json')['visual_quality_gate'];tick=time.time();out=ensure_nod_quality(config,job)
reference=read(w/'visual_quality/measure/completed'/job.name)
assert out=={d['sample_id']:d for d in reference['decisions']}
result=dict(passed=True,elapsed_s=time.time()-tick,clips=len(out),runtime=sys.executable,reference=str(w/'visual_quality/measure/completed'/job.name))
write(w/'visual_quality_rollout/early_runtime_parity.json',result);print(json.dumps(result))
