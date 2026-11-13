import torch
from torch import nn
from torch.nn import functional as F
from .contracts import EVENTS, VISUAL_TYPES, AUDIO_TYPES
from .layout import pool_by_time


class EventHeads(nn.Module):
    def __init__(self, video_dim, audio_dim, width=128):
        super().__init__()
        self.heads = nn.ModuleDict({
            mod: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(), nn.Linear(width, count))
            for mod, dim, count in (("video", video_dim, len(VISUAL_TYPES)), ("audio", audio_dim, len(AUDIO_TYPES)))})

    def forward(self, u, layout, mode):
        if mode not in ("probe_only", "joint_event"):
            raise ValueError("Unknown event gradient mode")
        result = {}
        for mod, names in (("video", VISUAL_TYPES), ("audio", AUDIO_TYPES)):
            values = getattr(u, mod)
            # The small auxiliary head is precision-sensitive. In particular,
            # spatial scatter sums in BF16 can vary with GPU execution order.
            # Keep its reduction and MLP in FP32 even inside the AV autocast;
            # casting preserves the joint gradient back into the original U.
            with torch.autocast(values.device.type, enabled=False):
                x, _ = pool_by_time(values.float(), layout[mod + "_groups"], getattr(u, mod + "_valid"),
                                    layout[mod + "_intervals"].shape[1])
                if mode == "probe_only":
                    x = x.detach()
                logits = self.heads[mod](x)
            result.update({name: logits[..., i] for i, name in enumerate(names)})
        return result


def event_sums(logits, labels, pos_weights):
    if any(set(x) != set(EVENTS) for x in (logits, labels, pos_weights)):
        raise ValueError("Exactly seven event keys required")
    sums, counts = {}, {}
    for name in EVENTS:
        y, known = labels[name]["y"], labels[name]["known"]
        if logits[name].shape != y.shape or known.shape != y.shape:
            raise ValueError("Event target shape mismatch")
        if known.dtype != torch.bool or not torch.isfinite(y).all() or ((y < 0) | (y > 1)).any():
            raise ValueError("Invalid event occupancy/mask")
        raw = F.binary_cross_entropy_with_logits(logits[name].float(), y.float(), reduction="none",
                 pos_weight=torch.as_tensor(pos_weights[name], device=y.device, dtype=torch.float32))
        sums[name] = raw.masked_fill(~known, 0).sum()
        counts[name] = known.sum()
    return sums, counts


def normalized_event_loss(sums, global_counts, world_size=1):
    active = sum(float(global_counts[k]) > 0 for k in EVENTS)
    loss = sum(v * 0 for v in sums.values())
    if active:
        for name in EVENTS:
            if float(global_counts[name]) > 0:
                loss = loss + world_size * sums[name] / global_counts[name] / active
    return loss
