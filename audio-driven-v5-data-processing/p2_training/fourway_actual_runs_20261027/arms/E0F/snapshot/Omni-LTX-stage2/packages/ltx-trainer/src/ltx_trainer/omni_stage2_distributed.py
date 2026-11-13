"""Joint Stage-2 gradients and resumable data-parallel sampling."""

from __future__ import annotations

from collections.abc import Iterable, Sized

import torch
from torch import Tensor

from ltx_trainer.cp_sampler import ContextParallelDistributedSampler


def joint_clip_coefficient(
    sharded_norm: Tensor, replicated_parameters: Iterable[Tensor], max_norm: float
) -> tuple[Tensor, Tensor]:
    """Combine a global FSDP norm with already-averaged Omni grads, counted once."""
    squared = sharded_norm.float().square()
    for parameter in replicated_parameters:
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().square().sum().to(squared.device)
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite joint LTX/Omni gradient norm")
    return norm, (max_norm / (norm + 1e-6)).clamp(max=1.0)


def resumed_dp_sampler(
    dataset: Sized,
    *,
    world_size: int,
    rank: int,
    batch_size: int,
    accumulation: int,
    completed_steps: int,
    seed: int,
    sp_size: int = 1,
) -> ContextParallelDistributedSampler:
    """Build the deterministic DP/CP sampler at an optimizer-step resume point."""
    sampler = ContextParallelDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        sp_size=sp_size,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    batches = sampler.num_samples // batch_size
    if batches < 1:
        raise ValueError("Dataset cannot supply one full batch per distributed rank")
    epoch, offset = divmod(completed_steps * accumulation, batches)
    sampler.set_epoch(epoch, start_index=offset * batch_size)
    return sampler
