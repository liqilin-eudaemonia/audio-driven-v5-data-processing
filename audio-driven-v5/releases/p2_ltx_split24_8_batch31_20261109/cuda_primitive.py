"""Small8-GPU NCCL check of host64 accumulation, without loading another LTX."""
from pathlib import Path
import json,os,sys
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(WORK),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist
from runtime import HostFP64Accumulator,partition

def main():
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert world==8
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(local).total_memory,local)
    torch.set_num_threads(1);dist.init_process_group('nccl')
    try:
        p=torch.nn.Parameter(torch.linspace(-.2,.3,140003,device=device));unused=torch.nn.Parameter(torch.ones(3,device=device))
        expected=torch.zeros_like(p,dtype=torch.float64)
        for i in range(31):expected.add_((torch.sin(p.detach()+(i+1)/31)/31).double())
        acc=HostFP64Accumulator([p,unused],1)
        for i in partition(list(range(31)),rank,world):
            p.grad=torch.sin(p.detach()+(i+1)/31)/31;acc.add()
        acc.finish(world)
        if not torch.equal(p.grad,expected.float())or unused.grad is not None:raise ValueError('CUDA accumulation mismatch')
        reference=torch.nn.Parameter(p.detach().clone());reference.grad=expected.float()
        one=torch.optim.AdamW([p],lr=1e-4);two=torch.optim.AdamW([reference],lr=1e-4);one.step();two.step()
        if not torch.equal(p,reference):raise ValueError('CUDA Adam mismatch')
        peak=torch.tensor(torch.cuda.max_memory_reserved(device)/2**30,device=device)
        dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        if rank==0:(WORK/'CUDA8_PRIMITIVE_ACCEPTED.json').write_text(json.dumps(dict(success=True,world=8,global_batch=31,actual_nccl=True,gradient_and_adam_equal=True,max_rank_peak_gib=float(peak))))
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
