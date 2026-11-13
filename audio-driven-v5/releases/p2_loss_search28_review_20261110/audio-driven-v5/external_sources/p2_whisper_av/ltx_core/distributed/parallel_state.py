# Copyright (c) 2026 Omni-LTX Team
"""Process group topology management for Sequence / Context Parallelism (CP)."""

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Optional
import torch.distributed as dist

_SP_GROUP: Optional[dist.ProcessGroup] = None
_SP_RANK: int = 0
_SP_WORLD_SIZE: int = 1

_DP_GROUP: Optional[dist.ProcessGroup] = None
_DP_RANK: int = 0
_DP_WORLD_SIZE: int = 1

_SP_INITIALIZED: bool = False
_SP_RUNTIME_ENABLED: bool = True


def init_sequence_parallel(sp_size: int) -> None:
    """Initialize sequence parallelism (Context Parallelism) process groups.

    Args:
        sp_size: Degree of sequence parallelism (e.g., 2, 4, 8). Must divide world size.
    """
    global _SP_GROUP, _SP_RANK, _SP_WORLD_SIZE
    global _DP_GROUP, _DP_RANK, _DP_WORLD_SIZE
    global _SP_INITIALIZED, _SP_RUNTIME_ENABLED

    if sp_size < 1:
        raise ValueError(f"sp_size must be >= 1, got {sp_size}")
    if _SP_INITIALIZED:
        if sp_size != _SP_WORLD_SIZE:
            raise RuntimeError(
                f"Sequence parallelism is already initialized with sp_size={_SP_WORLD_SIZE}, "
                f"cannot reinitialize with sp_size={sp_size}"
            )
        return

    if not dist.is_initialized():
        _SP_RANK = 0
        _SP_WORLD_SIZE = 1
        _DP_RANK = 0
        _DP_WORLD_SIZE = 1
        _SP_INITIALIZED = True
        _SP_RUNTIME_ENABLED = True
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if sp_size <= 1:
        _SP_GROUP = None
        _SP_RANK = 0
        _SP_WORLD_SIZE = 1
        _DP_GROUP = dist.group.WORLD
        _DP_RANK = rank
        _DP_WORLD_SIZE = world_size
        _SP_INITIALIZED = True
        _SP_RUNTIME_ENABLED = True
        return

    if world_size % sp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by sp_size ({sp_size})"
        )

    dp_size = world_size // sp_size

    # Build SP groups: ranks [i*sp_size, (i+1)*sp_size) share the same data sample
    # Consecutive ranks within the same node communicate via ultra-fast NVLink!
    for i in range(dp_size):
        sp_ranks = list(range(i * sp_size, (i + 1) * sp_size))
        group = dist.new_group(sp_ranks)
        if rank in sp_ranks:
            _SP_GROUP = group
            _SP_RANK = rank % sp_size
            _SP_WORLD_SIZE = sp_size

    # Build DP groups: ranks [j, j+sp_size, j+2*sp_size, ...]
    for j in range(sp_size):
        dp_ranks = list(range(j, world_size, sp_size))
        group = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            _DP_GROUP = group
            _DP_RANK = rank // sp_size
            _DP_WORLD_SIZE = dp_size

    _SP_INITIALIZED = True
    _SP_RUNTIME_ENABLED = True


def is_sp_enabled() -> bool:
    return _SP_RUNTIME_ENABLED and _SP_WORLD_SIZE > 1 and _SP_GROUP is not None


@contextmanager
def suspend_sequence_parallel() -> Iterator[None]:
    """Temporarily bypass CP collectives while retaining the process-group topology."""
    global _SP_RUNTIME_ENABLED
    previous = _SP_RUNTIME_ENABLED
    _SP_RUNTIME_ENABLED = False
    try:
        yield
    finally:
        _SP_RUNTIME_ENABLED = previous


def get_sp_group() -> Optional[dist.ProcessGroup]:
    return _SP_GROUP


def get_sp_rank() -> int:
    return _SP_RANK


def get_sp_world_size() -> int:
    return _SP_WORLD_SIZE


def get_dp_group() -> Optional[dist.ProcessGroup]:
    return _DP_GROUP


def get_dp_rank() -> int:
    return _DP_RANK


def get_dp_world_size() -> int:
    return _DP_WORLD_SIZE


def reset_sequence_parallel() -> None:
    global _SP_GROUP, _SP_RANK, _SP_WORLD_SIZE
    global _DP_GROUP, _DP_RANK, _DP_WORLD_SIZE
    global _SP_INITIALIZED, _SP_RUNTIME_ENABLED

    _SP_GROUP = None
    _SP_RANK = 0
    _SP_WORLD_SIZE = 1
    _DP_GROUP = None
    _DP_RANK = 0
    _DP_WORLD_SIZE = 1
    _SP_INITIALIZED = False
    _SP_RUNTIME_ENABLED = True
