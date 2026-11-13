"""Bound memory for the full Whisper residuals without changing AV/Adam arithmetic."""
import torch
import torch.distributed as dist


def settings(c):
    value=dict(gradient_reduction='streaming_fp64',gradient_bucket_mb=32,
               optimizer_state_cpu_offload=True,longest_sample_check=True)
    value.update(c.get('whisper_training_execution',{}))
    if (set(value)!={'gradient_reduction','gradient_bucket_mb','optimizer_state_cpu_offload','longest_sample_check'}
            or value['gradient_reduction'] not in ('streaming_fp64','full_fp64')
            or type(value['gradient_bucket_mb']) is not int or value['gradient_bucket_mb']<1
            or any(type(value[k]) is not bool for k in ('optimizer_state_cpu_offload','longest_sample_check'))):
        raise ValueError('Invalid Whisper training execution settings')
    return value


@torch.no_grad()
def move_adam_moments(optimizer, *, to_parameters=False):
    """Keep step scalars on their original device; move FP32 moment tensors only."""
    moved=0
    for parameter,state in optimizer.state.items():
        destination=parameter.device if to_parameters else torch.device('cpu')
        for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            value=state.get(name)
            if isinstance(value,torch.Tensor) and value.device!=destination:
                moved+=value.numel()*value.element_size()
                state[name]=value.to(device=destination,non_blocking=False)
    return moved


class StreamingFP64Reduction:
    """Same flat 32MiB FP64 SUM buckets, one local microbatch, no full FP64 buffer."""
    def __init__(self,parameters,bucket_mb=32):
        self.parameters=[p for p in parameters if p.requires_grad]
        if not self.parameters or type(bucket_mb) is not int or bucket_mb<1:
            raise ValueError('Invalid streaming gradient parameters / bucket')
        self.device=self.parameters[0].device
        if any(p.device!=self.device for p in self.parameters):
            raise ValueError('Gradient parameters must share one device')
        self.bucket_elements=bucket_mb*1024*1024//8
        self.added=False

    def add(self):
        if self.added:
            raise ValueError('Streaming FP64 path supports one microbatch per rank')
        if any(p.grad is not None and (p.grad.is_sparse or not p.grad.is_contiguous()) for p in self.parameters):
            raise ValueError('Streaming gradients require contiguous dense gradients')
        self.added=True

    @torch.no_grad()
    def finish(self,world=1):
        if not self.added or world<1 or world>1 and (not dist.is_initialized() or dist.get_world_size()!=world):
            raise ValueError('Gradient reduction world / add state differs')
        present=torch.tensor([p.grad is not None for p in self.parameters],dtype=torch.int32,device=self.device)
        if world>1:dist.all_reduce(present,op=dist.ReduceOp.MAX)
        for p,used in zip(self.parameters,present.tolist()):
            if used and p.grad is None:p.grad=torch.zeros_like(p)
        total=sum(p.numel() for p in self.parameters);index=offset=0
        for begin in range(0,total,self.bucket_elements):
            count=min(self.bucket_elements,total-begin)
            buffer=torch.zeros(count,dtype=torch.float64,device=self.device);segments=[];filled=0
            while filled<count:
                p=self.parameters[index];size=min(p.numel()-offset,count-filled)
                grad=None if p.grad is None else p.grad.view(-1)[offset:offset+size]
                if grad is not None:buffer[filled:filled+size].copy_(grad)
                segments.append((grad,filled,size));filled+=size;offset+=size
                if offset==p.numel():index+=1;offset=0
            if world>1:dist.all_reduce(buffer,op=dist.ReduceOp.SUM)
            for grad,start,size in segments:
                if grad is not None:grad.copy_(buffer[start:start+size])
        self.added=False


def accumulator(parameters, micro_count, execution):
    if micro_count==1 and execution['gradient_reduction']=='streaming_fp64':
        return StreamingFP64Reduction(parameters,execution['gradient_bucket_mb'])
    from p2_listen.performance import BucketGradientAccumulator
    return BucketGradientAccumulator(parameters,execution['gradient_bucket_mb'])
