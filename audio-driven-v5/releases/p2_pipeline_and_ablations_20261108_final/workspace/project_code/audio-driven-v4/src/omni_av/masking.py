import math

import torch


def hybrid_attention(prefix_valid, target_valid, dtype):
    """Additive [B,1,L,L]: causal prefix, target bidirectional, no future leakage."""
    b, p = prefix_valid.shape
    t = target_valid.shape[1]
    valid = torch.cat([prefix_valid, target_valid], dim=1).bool()
    allow = torch.zeros((b, p + t, p + t), dtype=torch.bool, device=valid.device)
    allow[:, :p, :p] = torch.ones(p, p, dtype=torch.bool, device=valid.device).tril()
    allow[:, p:, :] = True
    allow &= valid[:, None, :]
    # Invalid queries only attend themselves, avoiding all-masked softmax rows.
    allow &= valid[:, :, None]
    allow |= torch.diag_embed(~valid)
    result = torch.zeros(allow.shape, dtype=dtype, device=valid.device)
    return result.masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def sample_target_mask(valid, times, modalities, min_ratio=0.7, all_probability=0.2):
    if valid.dtype != torch.bool or not 0 < min_ratio <= 1 or not 0 <= all_probability <= 1:
        raise ValueError("Invalid mask specification")
    mask = torch.zeros_like(valid)
    for i in range(valid.shape[0]):
        if torch.rand((), device=valid.device) < all_probability:
            mask[i] = valid[i]
            continue
        ratio = min_ratio + (1 - min_ratio) * torch.rand((), device=valid.device).item()
        for mod in (0, 1):
            indices = torch.where(valid[i] & (modalities[i] == mod))[0]
            if indices.numel():
                n = max(1, math.ceil(indices.numel() * ratio))
                order = torch.randperm(indices.numel(), device=valid.device)
                mask[i, indices[order[:n]]] = True
        # Joint temporal block across both modalities, in addition to token masks.
        if torch.rand((), device=valid.device) < 0.5:
            ts = times[i, valid[i]]
            span = (ts.max() - ts.min()).clamp_min(0.25) * 0.25
            start = ts.min() + torch.rand((), device=valid.device) * (ts.max() - ts.min())
            mask[i] |= valid[i] & (times[i] >= start) & (times[i] <= start + span)
    return mask


def fill_schedule(count, steps):
    if count < 0 or steps < 1:
        raise ValueError("Invalid planning schedule")
    filled = 0
    for k in range(steps):
        remaining = 0 if k == steps - 1 else int(count * math.cos(math.pi * (k + 1) / (2 * steps)))
        end = count - remaining
        yield filled, end
        filled = end
