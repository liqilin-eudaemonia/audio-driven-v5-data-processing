"""Equivalent deterministic kernels for reproducible native training."""
import torch.nn.functional as F


def token_cross_entropy(logits, target, **kwargs):
    if logits.ndim == 3 and target.ndim == 2:
        classes = logits.shape[1]
        logits = logits.transpose(1, 2).reshape(-1, classes)
        target = target.reshape(-1)
    return F.cross_entropy(logits, target, **kwargs)


def linear_audio_interpolate(value, *args, **kwargs):
    if value.requires_grad and value.is_cuda and kwargs.get('mode') == 'linear':
        return F.interpolate(value.cpu(), *args, **kwargs).to(value.device)
    return F.interpolate(value, *args, **kwargs)
