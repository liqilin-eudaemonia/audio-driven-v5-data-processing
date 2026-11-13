import json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from p3_listen.common import config,resolve,read,write
cfg=config(ROOT/'configs/p3_listen_300h.json');root=resolve(cfg['output']);sys.path.insert(1,str(resolve(cfg['reuse_root'])/'src'))
import torch
from p3_listen import features
from listen_stream_cache import Encoders,prefetched
ids={r['sample_id'] for r in read(root/'cluster_checks/cache_timing.json')['rows']}
rows=[r for r in read(root/'windows.json')['windows'] if r['sample_id'] in ids]
enc=Encoders(cfg,root);checks=[]
def compare(a,b,path):
 if isinstance(a,dict):
  if a.keys()!=b.keys():raise ValueError(path+' keys differ')
  for k in a:compare(a[k],b[k],path+'/'+k)
 elif isinstance(a,torch.Tensor):
  if not torch.equal(a,b):
   if not a.is_floating_point():raise ValueError(path+' integer tensor differs')
   diff=(a.float()-b.float()).abs();torch.testing.assert_close(a,b,rtol=0,atol=0)
 else:
  if a!=b:raise ValueError(path+' values differ')
with ThreadPoolExecutor(max_workers=2) as pool:
 begin=time.monotonic();last=begin
 for item in prefetched(pool,lambda r:enc.prepare(r,force=True),rows):
  result=enc.compute(item,save=False);torch.cuda.synchronize();now=time.monotonic()
  p=features.cache_dir(root,item['row'])
  for key,name in [('layout','layout.pt'),('targets','targets.pt'),('av','av.pt'),('reference','reference.pt')]:
   compare(result[key],torch.load(p/name,map_location='cpu',weights_only=True),key)
  checks.append(dict(sample_id=item['row']['sample_id'],frames=item['row']['frames'],seconds=now-last,bitwise_equal=True))
  last=time.monotonic();print(json.dumps(checks[-1]),flush=True)
write(root/'cluster_checks/stream_equivalence.json',dict(checks=checks,seconds=time.monotonic()-begin,all_bitwise_equal=True))
print('PASS',flush=True)
