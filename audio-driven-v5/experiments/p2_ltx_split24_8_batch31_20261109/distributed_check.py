"""Real CPU process-group check of unequal local counts and mixed reducers."""
from pathlib import Path
import argparse,json,os,sys,tempfile
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(WORK),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from runtime import HostFP64Accumulator,partition
from p2_whisper_av.execution import accumulator

def check(rank,world,rendezvous,out):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=world)
    try:
        p=torch.nn.Parameter(torch.linspace(-.2,.3,140003));q=torch.nn.Parameter(torch.tensor([.2,-.3]));unused=torch.nn.Parameter(torch.ones(3))
        reference=[torch.zeros_like(p,dtype=torch.float64),torch.zeros_like(q,dtype=torch.float64)]
        def grads(i):return [torch.sin(p.detach()+(i+1)/31)/31,torch.tensor([i/1000,-i/2000])if i%3==0 else None]
        for i in range(31):
            gp,gq=grads(i);reference[0]+=gp.double()
            if gq is not None:reference[1]+=gq.double()
        local=partition(list(range(31)),rank,world)
        execution=dict(gradient_reduction='streaming_fp64',gradient_bucket_mb=1)
        acc=HostFP64Accumulator([p,q,unused],1)if world==8 else accumulator([p,q,unused],len(local),execution)
        for i in local:p.grad,q.grad=grads(i);acc.add()
        acc.finish(world)
        assert torch.equal(p.grad,reference[0].float())and torch.equal(q.grad,reference[1].float())and unused.grad is None
        a=[torch.nn.Parameter(p.detach().clone()),torch.nn.Parameter(q.detach().clone())]
        a[0].grad=reference[0].float();a[1].grad=reference[1].float()
        one=torch.optim.AdamW([p,q],lr=1e-4);other=torch.optim.AdamW(a,lr=1e-4);one.step();other.step()
        assert torch.equal(p,a[0])and torch.equal(q,a[1])
        if rank==0:Path(out).write_text(json.dumps(dict(success=True,world=world,batch=31,actual_gloo_processes=world,gradient_and_adam_equal=True,multiple_buckets=True,unused_parameters_preserved=True)))
    finally:dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--world',type=int,choices=[8,24],required=True);a=p.parse_args()
    with tempfile.TemporaryDirectory()as tmp:mp.spawn(check,args=(a.world,str(Path(tmp)/'rdzv'),str(WORK/f'GLOO_{a.world}_ACCEPTED.json')),nprocs=a.world,join=True)
