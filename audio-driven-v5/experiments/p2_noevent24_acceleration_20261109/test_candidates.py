from pathlib import Path
import sys,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(Path(__file__).resolve().parent)]
from p2_whisper_av.execution import accumulator
import adapt_runtime as R

def test_norm_reporting_preserves_fp64_value_and_gradients():
 torch.manual_seed(12);ps=[torch.nn.Parameter(torch.randn(53,37))for _ in range(20)]
 for p in ps:p.grad=torch.randn_like(p)
 before=[p.grad.clone()for p in ps]
 old=sum(float(p.grad.double().square().sum())for p in ps)**.5
 actual=R.norms([dict(name='main',params=ps)])['main']
 assert abs(actual-old)/old<1e-12
 assert all(torch.equal(p.grad,g)for p,g in zip(ps,before))

def test_reused_accumulator_preserves_all_updates_and_unused_state():
 for micros in [1,2]:
  a=[torch.nn.Parameter(torch.zeros(7)),torch.nn.Parameter(torch.zeros(3))];b=[torch.nn.Parameter(p.detach().clone())for p in a]
  settings=dict(gradient_reduction='streaming_fp64',gradient_bucket_mb=1);cached=accumulator(a,micros,settings)
  for step in range(8):
   fresh=accumulator(b,micros,settings)
   for micro in range(micros):
    for i,(p,q)in enumerate(zip(a,b)):
     p.grad=torch.full_like(p,(step+micro+1)/10)if i==0 or step%2==0 else None
     q.grad=None if p.grad is None else p.grad.clone()
    cached.add();fresh.add()
   cached.finish();fresh.finish()
   for p,q in zip(a,b):assert(p.grad is None and q.grad is None)or torch.equal(p.grad,q.grad)
