from pathlib import Path
import sys,torch
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path[:0]=[str(W),str(ROOT/'src'),str(ROOT/'scripts')]
import batch_runtime as R
from p2_listen.contracts import read
from p2_ltx_events.heads import objective

def test_entire_remaining_stream_and_tail_are_exact():
 c=read(W.parent/'p2_ltx_all_lora_noevent_batch24_20261109/config.json');s=R.Schedule(c);p=s.plan;flat=[]
 for step in range(p['start_step']+1,p['new_end_step']+1):
  groups=[s.indices(s.size,step,24,s.seed,r,24)for r in range(24)];combined=[v for group in groups for v in group]
  assert len(combined)==s.count(step)and all(len(g)<=1 for g in groups)
  flat.extend(combined)
 assert flat==s.pairs[p['prefix_pairs']:]
 assert s.indices(s.size,p['start_step']+1,24,s.seed)==s.pairs[p['prefix_pairs']:p['prefix_pairs']+24]
 assert 1<=s.count(p['new_end_step'])<=24
 # Recovery at any later step uses absolute plan offset, not a new epoch.
 mid=p['start_step']+37;a,b=s.bounds(mid);assert a==p['prefix_pairs']+36*24

def test_objective_and_reporting_use_effective_batch():
 c=read(W.parent/'p2_ltx_all_lora_noevent_batch24_20261109/config.json')
 for n in [24,7,1]:
  x=torch.tensor(2.,requires_grad=True);v=dict(av=x,features=dict(video=x*3,audio=x*5),event_sums={})
  loss,_,_=objective(v,c,{},n,1);expected=(x+.025*x*3+.025*x*5)/n
  assert torch.equal(loss,expected)
  loss.backward();assert torch.allclose(x.grad,torch.tensor(1.2/n))

def test_training_only_requested_changes():
 s=(W/'training.py').read_text();assert 'objective(values,c,counts,effective_batch,1)'in s
 assert 'if not chosen:acc.add()'in s and 'accumulator(model.parameters(),1,execution)'in s
 assert '/t[\'global_batch\']'not in s
