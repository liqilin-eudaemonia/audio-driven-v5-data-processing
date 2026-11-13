# Copyright (c) 2026 Omni-LTX Team
"""Distributed communications primitives for sequence parallelism with full autograd support."""

from typing import Any, Optional
import torch
import torch.distributed as dist
from torch import Tensor


def _all_to_all_tensor_list(
    input_list: list[Tensor],
    group: Optional[dist.ProcessGroup],
) -> list[Tensor]:
    """Execute dist.all_to_all on a list of tensors, with gloo/CPU fallback."""
    output_list = [torch.empty_like(t) for t in input_list]
    backend = dist.get_backend(group) if group is not None else None
    if backend != "gloo":
        dist.all_to_all(output_list, input_list, group=group)
    else:
        # Gloo does not implement native all_to_all. This path is only for CPU tests.
        world_size = len(input_list)
        rank = dist.get_rank(group) if group is not None else 0
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, [t.cpu() for t in input_list], group=group)
        for i in range(world_size):
            output_list[i].copy_(gathered[i][rank])
    return output_list


def _all_to_all(
    input_: Tensor,
    world_size: int,
    group: Optional[dist.ProcessGroup],
    scatter_dim: int,
    gather_dim: int,
) -> Tensor:
    """Core all-to-all communication splitting across scatter_dim and concatenating across gather_dim."""
    if world_size <= 1:
        return input_

    if input_.shape[scatter_dim] % world_size != 0:
        raise ValueError(
            f"All-to-all scatter dimension {scatter_dim} with size "
            f"{input_.shape[scatter_dim]} must be divisible by world_size={world_size}"
        )

    # Split input into world_size chunks along scatter_dim
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, dim=scatter_dim)]
    output_list = _all_to_all_tensor_list(input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class SeqAllToAll4D(torch.autograd.Function):
    """Autograd-aware 4D All-To-All communication for Ulysses attention.

    Forward:
        Transforms tensor from (B, S/P, H, D) -> (B, S, H/P, D) [scatter_dim=2, gather_dim=1]
        or (B, S, H/P, D) -> (B, S/P, H, D) [scatter_dim=1, gather_dim=2]
    Backward:
        Performs the exact adjoint All-To-All with reversed scatter and gather dimensions.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input_: Tensor,
        group: Optional[dist.ProcessGroup],
        scatter_dim: int,
        gather_dim: int,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(group) if group is not None else 1

        return _all_to_all(input_, ctx.world_size, group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        # The backward gradient pass reverses scatter and gather dimensions
        grad_input = _all_to_all(
            grad_output.contiguous(),
            ctx.world_size,
            ctx.group,
            scatter_dim=ctx.gather_dim,
            gather_dim=ctx.scatter_dim,
        )
        return grad_input, None, None, None


def all_to_all_4d(
    input_: Tensor,
    group: Optional[dist.ProcessGroup],
    scatter_dim: int = 2,
    gather_dim: int = 1,
) -> Tensor:
    """Convenience functional wrapper for SeqAllToAll4D."""
    world_size = dist.get_world_size(group) if group is not None else 1
    if world_size <= 1:
        return input_
    return SeqAllToAll4D.apply(input_, group, scatter_dim, gather_dim)


class AllGatherDim(torch.autograd.Function):
    """Autograd-aware AllGather along an arbitrary tensor dimension."""

    @staticmethod
    def forward(
        ctx: Any,
        input_: Tensor,
        dim: int,
        group: Optional[dist.ProcessGroup],
    ) -> Tensor:
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group) if group is not None else 1
        rank = dist.get_rank(group) if group is not None else 0

        ctx.world_size = world_size
        ctx.rank = rank

        if world_size <= 1:
            return input_

        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_.contiguous(), group=group)
        output = torch.cat(tensor_list, dim=dim).contiguous()
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        if ctx.world_size <= 1:
            return grad_output, None, None

        # The all-gather output is consumed independently on every rank. Its adjoint is
        # reduce-scatter: sum each rank's contribution to this input partition.
        chunks = torch.tensor_split(grad_output, ctx.world_size, dim=ctx.dim)
        grad_input = chunks[ctx.rank].contiguous()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_input, None, None


def all_gather_dim(
    input_: Tensor,
    dim: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> Tensor:
    """Convenience functional wrapper for AllGatherDim."""
    world_size = dist.get_world_size(group) if group is not None else 1
    if world_size <= 1:
        return input_
    return AllGatherDim.apply(input_, dim, group)


def broadcast_tensor(
    tensor: Tensor,
    *,
    group: Optional[dist.ProcessGroup],
    group_src: int = 0,
) -> Tensor:
    """Broadcast a stochastic tensor in-place so all CP ranks use identical values."""
    if group is not None and dist.get_world_size(group) > 1:
        src = dist.get_global_rank(group, group_src)
        if tensor.is_contiguous():
            dist.broadcast(tensor, src=src, group=group)
        else:
            contiguous = tensor.contiguous()
            dist.broadcast(contiguous, src=src, group=group)
            tensor.copy_(contiguous)
    return tensor
