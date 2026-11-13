import torch
from torch import nn
from torch.nn import functional as F
from .contracts import EVENTS
from .layout import pool_by_time


class EventHeads(nn.Module):
    def __init__(self, video_dim, audio_dim, width=128):
        super().__init__()
        self.heads = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1))
                                   for k, d in zip(EVENTS, (audio_dim, video_dim))})

    def forward(self, u, layout, mode):
        if mode not in ("probe_only", "joint_event"):
            raise ValueError("Unknown event gradient mode")
        result = {}
        for name, mod in zip(EVENTS, ("audio", "video")):
            x, _ = pool_by_time(getattr(u, mod), layout[mod + "_groups"], getattr(u, mod + "_valid"),
                                layout[mod + "_intervals"].shape[1])
            if mode == "probe_only":
                x = x.detach()
            result[name] = self.heads[name](x).squeeze(-1)
        return result


def event_sums(logits, labels, pos_weights):
    sums, counts = {}, {}
    for name in EVENTS:
        y, known = labels[name]["y"], labels[name]["known"]
        if logits[name].shape != y.shape or known.shape != y.shape:
            raise ValueError("Event target shape mismatch")
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
