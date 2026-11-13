from pathlib import Path
import sys,os,torch
import torch.distributed as dist
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.execution import accumulator
from p2_listen.contracts import atomic_json
rank=int(os.environ['RANK']);torch.set_num_threads(1);dist.init_process_group('gloo')
p=torch.nn.Parameter(torch.tensor([.3,.7]));unused=torch.nn.Parameter(torch.tensor([1.]));ref=torch.nn.Parameter(p.detach().clone())
a=torch.optim.AdamW([p,unused],lr=.001,weight_decay=0.);b=torch.optim.AdamW([ref],lr=.001,weight_decay=0.)
for k in range(2):
 a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True);acc=accumulator([p,unused],1,dict(gradient_reduction='streaming_fp64',gradient_bucket_mb=1))
 if rank<2:((p*(rank+1)).square().sum()/2).backward()
 acc.add();acc.finish(3)
 (((ref.square().sum())+(ref*2).square().sum())/2).backward()
 assert torch.allclose(p.grad,ref.grad,atol=1e-7,rtol=1e-7)and unused.grad is None
 a.step();b.step();assert torch.equal(p,ref)
if rank==0:atomic_json(W/'TAIL_DISTRIBUTED_ACCEPTED.json',dict(success=True,world=3,valid_samples=2,empty_rank=True,updates=2,adam_exact=True))
dist.destroy_process_group()
