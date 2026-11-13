"""Isolated execution candidates; production checkpoint-bound modules stay intact."""
import torch
import torch.distributed as dist
from p2_whisper_av.execution import StreamingFP64Reduction


def group_norms(groups):
    result={}
    for group in groups:
        sums=[p.grad.double().square().sum() for p in group['params'] if p.grad is not None]
        # Same per-tensor reduction and Python addition order, one host transfer per group.
        values=torch.stack(sums).cpu().tolist() if sums else []
        result[group['name']]=sum(values)**.5
    return result


class PipelinedReduction(StreamingFP64Reduction):
    @torch.no_grad()
    def finish(self,world=1):
        if not self.added or world<1 or world>1 and (not dist.is_initialized() or dist.get_world_size()!=world):
            raise ValueError('Invalid reduction state/world')
        present=torch.tensor([p.grad is not None for p in self.parameters],dtype=torch.int32,device=self.device)
        if world>1:dist.all_reduce(present,op=dist.ReduceOp.MAX)
        for p,used in zip(self.parameters,present.tolist()):
            if used and p.grad is None:p.grad=torch.zeros_like(p)
        total=sum(p.numel() for p in self.parameters);index=offset=0;pending=[]
        buffers=[torch.empty(min(total,self.bucket_elements),dtype=torch.float64,device=self.device) for _ in range(2)]
        def drain():
            work,buffer,segments=pending.pop(0)
            if work is not None:work.wait()
            for grad,start,size in segments:
                if grad is not None:grad.copy_(buffer[start:start+size])
        for number,begin in enumerate(range(0,total,self.bucket_elements)):
            count=min(self.bucket_elements,total-begin);buffer=buffers[number%2][:count];buffer.zero_();segments=[];filled=0
            while filled<count:
                p=self.parameters[index];size=min(p.numel()-offset,count-filled)
                grad=None if p.grad is None else p.grad.view(-1)[offset:offset+size]
                if grad is not None:buffer[filled:filled+size].copy_(grad)
                segments.append((grad,filled,size));filled+=size;offset+=size
                if offset==p.numel():index+=1;offset=0
            work=dist.all_reduce(buffer,op=dist.ReduceOp.SUM,async_op=True) if world>1 else None
            pending.append((work,buffer,segments))
            if len(pending)==2:drain()
        while pending:drain()
        self.added=False
