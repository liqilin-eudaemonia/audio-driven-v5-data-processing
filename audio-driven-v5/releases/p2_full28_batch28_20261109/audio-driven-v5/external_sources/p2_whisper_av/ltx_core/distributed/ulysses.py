# Copyright (c) 2026 Omni-LTX Team
"""Ulysses Attention implementation for sequence parallelism."""

from typing import Callable, Optional
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F

from ltx_core.distributed.comm import SeqAllToAll4D


def apply_ulysses_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    attention_fn: Callable,
    mask: Optional[Tensor] = None,
    sp_group: Optional[dist.ProcessGroup] = None,
) -> Tensor:
    """Execute Ulysses sequence-parallel attention.

    Transposes sequence tokens across ranks in the SP group to gather full context length
    while distributing attention heads across ranks, computes attention locally per head partition,
    and transposes back to the original sequence-sharded partition.

    Args:
        q: Query tensor of shape (B, S_local, H, D) or (B, S_local, H * D)
        k: Key tensor of shape (B, S_local, H, D) or (B, S_local, H * D)
        v: Value tensor of shape (B, S_local, H, D) or (B, S_local, H * D)
        heads: Total number of attention heads across all ranks.
        attention_fn: Concrete AttentionCallable (PytorchAttention, XFormers, etc.)
        mask: Optional attention mask (typically None for full video self-attention)
        sp_group: Process group for sequence parallelism.

    Returns:
        Output tensor of shape (B, S_local, H * D)
    """
    sp_size = dist.get_world_size(sp_group) if sp_group is not None else 1
    b, s_local = q.shape[0], q.shape[1]

    # Normalize 3D inputs [B, S_local, H*D] to 4D [B, S_local, H, D]
    if q.ndim == 3:
        dim_head = q.shape[-1] // heads
        q = q.view(b, s_local, heads, dim_head)
        k = k.view(b, s_local, heads, dim_head)
        v = v.view(b, s_local, heads, dim_head)
    else:
        dim_head = q.shape[-1]

    if sp_size <= 1:
        # Fallback to local execution
        q_flat = q.reshape(b, s_local, heads * dim_head)
        k_flat = k.reshape(b, s_local, heads * dim_head)
        v_flat = v.reshape(b, s_local, heads * dim_head)
        return attention_fn(q_flat, k_flat, v_flat, heads, mask)

    if heads % sp_size != 0:
        raise ValueError(
            f"Total attention heads ({heads}) must be divisible by sp_size ({sp_size})"
        )
    heads_per_rank = heads // sp_size

    # 1. All-To-All: (B, S_local, H, D) -> (B, S_full, H_local, D)
    # Scatter heads (dim 2), Gather sequence (dim 1)
    q_sp = SeqAllToAll4D.apply(q, sp_group, 2, 1)
    k_sp = SeqAllToAll4D.apply(k, sp_group, 2, 1)
    v_sp = SeqAllToAll4D.apply(v, sp_group, 2, 1)

    s_full = q_sp.shape[1]

    # 2. Local Attention on head partition across full sequence length. CP padding
    # uses a compact, right-padded key mask. Cropping Q/K/V before attention keeps
    # padded tokens completely outside the softmax and preserves unpadded semantics.
    valid_len = s_full
    if mask is not None:
        if mask.ndim != 4 or mask.shape[-2] != 1 or mask.shape[-1] != s_full:
            raise NotImplementedError(
                "Ulysses supports only a compact right-padding mask of shape "
                f"(B, 1, 1, S_full), got {tuple(mask.shape)}"
            )
        valid_tokens = mask[:, 0, 0, :] == 0
        valid_counts = valid_tokens.sum(dim=-1)
        if not (valid_counts == valid_counts[0]).all():
            raise ValueError("All samples in a CP batch must have the same padded sequence length")
        valid_len = int(valid_counts[0].item())
        expected = (
            F.pad(
                valid_tokens.new_ones((b, valid_len)),
                (0, s_full - valid_len),
                value=False,
            )
        )
        if not valid_tokens.equal(expected):
            raise ValueError("Ulysses compact mask must describe contiguous right padding")

    q_flat = q_sp[:, :valid_len].reshape(b, valid_len, heads_per_rank * dim_head)
    k_flat = k_sp[:, :valid_len].reshape(b, valid_len, heads_per_rank * dim_head)
    v_flat = v_sp[:, :valid_len].reshape(b, valid_len, heads_per_rank * dim_head)

    out_sp = attention_fn(q_flat, k_flat, v_flat, heads_per_rank, None)
    if valid_len < s_full:
        out_sp = F.pad(out_sp, (0, 0, 0, s_full - valid_len))
    out_sp = out_sp.view(b, s_full, heads_per_rank, dim_head)

    # 3. All-To-All: (B, S_full, H_local, D) -> (B, S_local, H, D)
    # Scatter sequence (dim 1), Gather heads (dim 2)
    out = SeqAllToAll4D.apply(out_sp, sp_group, 1, 2)

    # 4. Final reshape to [B, S_local, H * D]
    return out.reshape(b, s_local, heads * dim_head)
