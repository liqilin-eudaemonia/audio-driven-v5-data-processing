"""Conditional feature flow matching, sigma=1 noise -> sigma=0 target."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def time_embedding(sigma, width):
    half = width // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=sigma.device) / max(half - 1, 1))
    phase = sigma.float()[:, None] * 1000 * freq[None]
    return F.pad(torch.cat([phase.cos(), phase.sin()], -1), (0, width - 2 * half))


class AdaBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, 3 * width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width))

    def forward(self, x, c):
        shift, scale, gate = self.modulation(F.silu(c)).chunk(3, -1)
        return x + gate.tanh() * self.ff(self.norm(x) * (1 + scale) + shift)


class FeatureFlow(nn.Module):
    def __init__(self, dimension, condition_dim, width=512, depth=3):
        super().__init__()
        self.dimension, self.width = dimension, width
        self.input = nn.Linear(dimension, width)
        self.condition = nn.Sequential(nn.LayerNorm(condition_dim), nn.Linear(condition_dim, width))
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([AdaBlock(width) for _ in range(depth)])
        self.output = nn.Linear(width, dimension)

    def forward(self, x, sigma, condition):
        c = self.condition(condition) + self.time(time_embedding(sigma, self.width).to(condition.dtype))
        h = self.input(x)
        for block in self.blocks:
            h = block(h, c)
        return self.output(F.silu(h))

    def loss(self, targets, conditions, repeats=1):
        if not targets.numel():
            return conditions.sum() * 0
        targets = targets.detach().float().repeat_interleave(repeats, 0)
        conditions = conditions.repeat_interleave(repeats, 0)
        noise = torch.randn_like(targets)
        sigma = torch.rand(targets.shape[0], device=targets.device)
        x = targets * (1 - sigma[:, None]) + noise * sigma[:, None]
        pred = self(x.to(conditions.dtype), sigma, conditions).float()
        return (pred - (noise - targets)).square().mean()

    @torch.no_grad()
    def sample(self, conditions, steps=5, generator=None):
        if steps < 1:
            raise ValueError("steps must be positive")
        x = torch.randn((conditions.shape[0], self.dimension), device=conditions.device,
                        dtype=torch.float32, generator=generator)
        for k in range(steps):
            sigma = torch.full((x.shape[0],), 1 - k / steps, device=x.device)
            velocity = self(x.to(conditions.dtype), sigma, conditions).float()
            x = x - velocity / steps
        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite feature flow sample")
        return x.to(conditions.dtype)
