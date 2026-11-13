import json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path.insert(0,str(ROOT/'src'))
from p3_listen.common import config,resolve,read,write
cfg=config(ROOT/'configs/p3_listen_300h.json');root=resolve(cfg['output'])
sys.path.insert(1,str(resolve(cfg['reuse_root'])/'src'))
from p3_listen import features
manifest=read(root/'windows.json')
available=[]
for row in manifest['windows'][:1500]:
 if (features.paths(root,row)['marker']).exists():available.append(row)
available.sort(key=lambda r:r['frames'])
rows=[available[round(i*(len(available)-1)/11)] for i in range(12)]
features.selection=lambda cfg,args:(manifest,rows)
original_seal=features.seal;records=[];last=time.monotonic()
def seal(folder,part,expected,files):
 global last
 original_seal(folder,part,expected,files)
 now=time.monotonic();record=dict(sample_id=Path(folder).name,part=part,seconds=now-last);last=now;records.append(record)
 print(json.dumps(record),flush=True)
 write(root/'cluster_checks/cache_timing_partial.json',dict(records=records,rows=[dict(sample_id=r['sample_id'],frames=r['frames']) for r in rows]))
features.seal=seal
for part in ('native','vae'):
 last=time.monotonic();features.cache(cfg,SimpleNamespace(part=part))
write(root/'cluster_checks/cache_timing.json',dict(records=records,rows=[dict(sample_id=r['sample_id'],frames=r['frames']) for r in rows],device=os.environ.get('CUDA_VISIBLE_DEVICES'),hostname=__import__('socket').gethostname()))
print('BENCHMARK_COMPLETED',flush=True)
