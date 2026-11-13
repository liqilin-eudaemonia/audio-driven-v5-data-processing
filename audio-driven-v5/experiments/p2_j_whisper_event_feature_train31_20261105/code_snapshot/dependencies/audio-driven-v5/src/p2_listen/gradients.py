"""Global-batch gradients with stable accumulation across rank partitionings."""
import torch
import torch.distributed as dist


class GradientAccumulator:
    """Accumulate individual backwards in FP64, sum ranks, round once to params.

    Every microbatch must use the same GLOBAL loss normalization on every rank.
    DDP automatic synchronization must be disabled; this class sums, not averages.
    Unused parameters remain None, including on ranks with no local contribution.
    """
    def __init__(self, parameters):
        self.parameters = [p for p in parameters if p.requires_grad]
        if not self.parameters:
            raise ValueError("No trainable parameters to accumulate")
        self.buffers = [None] * len(self.parameters)

    def add(self):
        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                if p.grad.is_sparse:
                    raise ValueError("P2 gradient accumulation requires dense gradients")
                if self.buffers[i] is None:
                    self.buffers[i] = p.grad.detach().to(torch.float64).clone()
                else:
                    self.buffers[i].add_(p.grad.detach())
                p.grad = None

    def finish(self, world=1):
        if world < 1 or (world > 1 and (not dist.is_initialized() or dist.get_world_size() != world)):
            raise ValueError("Gradient world size differs from initialized process group")
        present = torch.tensor([b is not None for b in self.buffers], device=self.parameters[0].device, dtype=torch.int32)
        if world > 1:
            dist.all_reduce(present, op=dist.ReduceOp.MAX)
        for i, (p, used) in enumerate(zip(self.parameters, present.tolist())):
            if used:
                buffer = self.buffers[i]
                if buffer is None:
                    buffer = torch.zeros_like(p, dtype=torch.float64)
                if world > 1:
                    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
                p.grad = buffer.to(p.dtype)
            else:
                p.grad = None
            self.buffers[i] = None
