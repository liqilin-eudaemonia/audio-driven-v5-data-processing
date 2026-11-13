from pathlib import Path
import sys,torch
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path[:0]=[str(W),str(ROOT/'src'),str(ROOT/'scripts')]
import runtime28 as R
from p2_listen.contracts import read

def test_every_remaining_sample_and_noise_ordinal_preserved():
 c=read(W.parent/'p2_ltx_all_lora_full28_batch28_20261109/config.json');s=R.Schedule(c);p=s.plan;flat=[]
 for step in range(p['start_step']+1,p['new_end_step']+1):
  groups=[s.indices(s.size,step,28,s.seed,r,28)for r in range(28)]
  assert all(len(g)<=1 for g in groups)
  pairs=[v for g in groups for v in g];assert len(pairs)==s.count(step);flat.extend(pairs)
 assert flat==s.pairs[199268:]and len(flat)==84971 and s.count(9463)==19
 assert s.bounds(7000)[0]==199268+(7000-6428-1)*28

def test_empty_rank_event_counts_and_effective_batch_objective():
 m=R.load();assert torch.equal(m._event_counts([],['clip/x'],'cpu',1)['clip/x'],torch.tensor(0.))
 samples=[dict(events={'clip/x':dict(known=torch.tensor([True,False]))})]
 assert float(m._event_counts(samples,['clip/x'],'cpu',1)['clip/x'])==1
 from p2_ltx_events.heads import objective
 for n in [28,19]:
  p=torch.tensor(2.,requires_grad=True);v=dict(av=p,features=dict(video=3*p,audio=5*p),event_sums={'clip/x':p*4})
  c=dict(loss_weights=dict(av=1.,event=.05,video_feature=.025,audio_feature=.025),hidden_event_supervision=dict(clip_weight=1.,temporal_weight=0.))
  total,_,_=objective(v,c,{'clip/x':torch.tensor(2.)},n,1)
  assert torch.allclose(total,1.2*p/n+.05*(p*4/2))
