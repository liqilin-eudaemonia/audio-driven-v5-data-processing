from pathlib import Path
import sys,json,torch
p=Path(__file__).resolve().parent;r=p.parents[1];sys.path.insert(0,str(r/'src'))
from p2_listen.training import checked_checkpoint

torch.set_num_threads(4)
def compare(a,b):
 if isinstance(a,torch.Tensor):
  return dict(tensors=1,unequal=0 if torch.equal(a,b) else 1,max_abs_error=float((a.double()-b.double()).abs().max()))
 if isinstance(a,dict):
  assert a.keys()==b.keys();rr=[compare(v,b[k]) for k,v in a.items()]
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b);rr=[compare(x,y) for x,y in zip(a,b)]
 else:
  assert a==b,(a,b);return dict(tensors=0,unequal=0,max_abs_error=0.)
 return dict(tensors=sum(x['tensors'] for x in rr),unequal=sum(x['unequal'] for x in rr),max_abs_error=max((x['max_abs_error'] for x in rr),default=0.))
result={}
for arm in ['E0','E1']:
 a=checked_checkpoint(p/f'correctness_1gpu/runs/s0_20261026/{arm}/step_000007.pt')
 b=checked_checkpoint(p/f'optimized_1gpu/runs/s0_20261026/{arm}/step_000007.pt')
 for key in ['seed','arm','observations','targets','labels','statistics','K_train','initialization','assets','numerics']:
  assert a['identity'][key]==b['identity'][key],key
 ca=dict(a['identity']['config']);cb=dict(b['identity']['config']);ca.pop('output');cb.pop('output');cb['training']=dict(cb['training']);cb['training'].pop('performance');assert ca==cb
 result[arm]={key:compare(a[key],b[key]) for key in ['model','optimizer','rank_rng']}
 del a,b
result['passed']=all(v['unequal']==0 for arm in ['E0','E1'] for v in result[arm].values())
(p/'optimization_equivalence.json').write_text(json.dumps(result,indent=2));print(result)
assert result['passed']
