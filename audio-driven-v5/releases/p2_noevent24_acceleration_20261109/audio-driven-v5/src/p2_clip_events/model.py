"""Read behavior presence from the same U that conditions audiovisual generation."""
import torch
from torch import nn
from torch.nn import functional as F


def classes(spec):
    groups = spec['classes']
    return tuple(groups['video']) + tuple(groups['audio'])


def validate_spec(spec):
    if 'min_positive' in spec and (type(spec['min_positive']) is not int or spec['min_positive'] < 1):
        raise ValueError('min_positive must be a positive integer')
    if spec['mode'] != 'clip_multilabel' or set(spec['classes']) != {'video', 'audio'}:
        raise ValueError('Expected clip_multilabel with video/audio classes')
    names = classes(spec)
    if not names or len(names) != len(set(names)) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('Clip classes must be nonempty unique names')
    return spec


class ClipEventHeads(nn.Module):
    def __init__(self, video_dim, audio_dim, spec, width=128):
        super().__init__()
        self.spec = validate_spec(spec)
        self.heads = nn.ModuleDict({
            mod: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(),
                               nn.Linear(width, len(spec['classes'][mod])))
            for mod, dim in [('video', video_dim), ('audio', audio_dim)] if spec['classes'][mod]})

    def forward(self, u, layout=None, mode='joint_event'):
        if mode not in ('joint_event', 'probe_only'):
            raise ValueError('Unknown event gradient mode')
        result = {}
        for mod, head in self.heads.items():
            x, valid = getattr(u, mod), getattr(u, mod + '_valid')
            if valid.dtype != torch.bool or x.shape[:2] != valid.shape or not valid.any(1).all():
                raise ValueError('Clip pooling requires valid condition tokens')
            with torch.autocast(x.device.type, enabled=False):
                # masked_fill also prevents nonfinite padding from contaminating the mean.
                pooled = x.float().masked_fill(~valid[..., None], 0).sum(1) / valid.sum(1, keepdim=True)
                if mode == 'probe_only':
                    pooled = pooled.detach()
                scores = head(pooled)
            result.update({name: scores[:, i] for i, name in enumerate(self.spec['classes'][mod])})
        return result


def event_sums(logits, labels, pos_weights):
    names = tuple(logits)
    if not names or set(names) != set(labels) or set(names) != set(pos_weights):
        raise ValueError('Clip label/class mismatch')
    z = torch.stack([logits[k] for k in names], -1).float()
    y = torch.stack([labels[k]['y'] for k in names], -1).float()
    known = torch.stack([labels[k]['known'] for k in names], -1)
    weight = torch.stack([labels[k]['weight'] for k in names], -1).float()
    if z.ndim != 2 or z.shape != y.shape or known.shape != z.shape or weight.shape != z.shape:
        raise ValueError('Clip targets must have shape [batch, classes], without time bins')
    if (known.dtype != torch.bool or not torch.isfinite(y).all() or not torch.isfinite(weight).all()
            or ((y != 0) & (y != 1)).any() or (weight < 0).any()):
        raise ValueError('Invalid clip labels/masks/weights')
    pos = torch.as_tensor([pos_weights[k] for k in names], device=z.device, dtype=torch.float32)
    if not torch.isfinite(pos).all() or (pos <= 0).any():
        raise ValueError('Invalid positive weights')
    weight = weight.masked_fill(~known, 0)
    losses = F.binary_cross_entropy_with_logits(z, y, pos_weight=pos, reduction='none') * weight
    # Normalize by known examples, not weight mass: otherwise a batch containing
    # only weak negatives would cancel the intended absence downweighting.
    sums, counts = losses.sum(0), known.sum(0)
    return ({k: sums[i] for i, k in enumerate(names)}, {k: counts[i] for i, k in enumerate(names)})


def normalized_loss(sums, counts, world_size=1):
    keys = tuple(sums)
    values = torch.stack([sums[k] for k in keys])
    denominators = torch.stack([torch.as_tensor(counts[k], device=values.device) for k in keys])
    active = denominators > 0
    return (values * world_size / denominators.clamp_min(1e-12) * active).sum() / active.sum().clamp_min(1)


def mask_classes(labels, enabled):
    return {k: dict(v, known=v['known'] if k in enabled else torch.zeros_like(v['known']))
            for k, v in labels.items()}
