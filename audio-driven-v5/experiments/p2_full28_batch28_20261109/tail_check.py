from pathlib import Path
import sys,os,torch
import torch.distributed as dist
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path[:0]=[str(W),str(ROOT/'src'),str(ROOT/'scripts')]
import runtime28 as R
from p2_whisper_av.execution import accumulator
from p2_ltx_events.heads import objective
from p2_listen.contracts import atomic_json
rank=int(os.environ['RANK']);torch.set_num_threads(1);dist.init_process_group('gloo');m=R.load()
p=torch.nn.Parameter(torch.tensor([.3,.7]));unused=torch.nn.Parameter(torch.tensor([1.]));ref=torch.nn.Parameter(p.detach().clone())
a=torch.optim.AdamW([p,unused],lr=.001,weight_decay=0.);b=torch.optim.AdamW([ref],lr=.001,weight_decay=0.)
c=dict(loss_weights=dict(av=1.,event=.05,video_feature=.025,audio_feature=.025),hidden_event_supervision=dict(clip_weight=1.,temporal_weight=0.))
for step in range(2):
 a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
 samples=[dict(events={'clip/x':dict(known=torch.tensor([rank==0]))})]if rank<2 else[]
 counts=m._event_counts(samples,['clip/x'],'cpu',3);assert float(counts['clip/x'])==1
 acc=accumulator([p,unused],1,dict(gradient_reduction='streaming_fp64',gradient_bucket_mb=1))
 if samples:
  av=(p*(rank+1)).square().sum();event=torch.nn.functional.softplus(-p.sum())if rank==0 else p.sum()*0
  v=dict(av=av,features=dict(video=av*3,audio=av*5),event_sums={'clip/x':event})
  objective(v,c,counts,2,1)[0].backward()
 acc.add();acc.finish(3)
 expected=1.2*(ref.square().sum()+(ref*2).square().sum())/2+.05*torch.nn.functional.softplus(-ref.sum());expected.backward()
 assert torch.allclose(p.grad,ref.grad,atol=1e-6,rtol=1e-6)and unused.grad is None
 a.step();b.step();assert torch.allclose(p,ref,atol=1e-7,rtol=0)
if rank==0:atomic_json(W/'TAIL_EVENT_ACCEPTED.json',dict(success=True,world=3,empty_rank=True,valid_samples=2,known_event_samples=1,updates=2,av_feature_sample_mean_and_event_global_known_correct=True))
dist.destroy_process_group()
