# Portions adapted from Bernini, Copyright (c) 2026 Bytedance Ltd. and/or its affiliate.
# Licensed under the Apache License, Version 2.0. See THIRD_PARTY_NOTICES.md.
"""Bernini's SimpleMLPAdaLN topology; explicit self-contained FM training."""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        value = torch.cat([torch.cos(args), torch.sin(args)], -1)
        return torch.cat([value, torch.zeros_like(value[:, :1])], -1) if dim % 2 else value

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size).to(t.dtype))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels))

    def forward(self, x, y):
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, -1)
        return x + gate * self.mlp(modulate(self.in_ln(x), shift, scale))


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, -1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks,
                 grad_checkpointing=False):
        super().__init__()
        self.grad_checkpointing = grad_checkpointing
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for index in (0, 2):
            nn.init.normal_(self.time_embed.mlp[index].weight, std=.02)
        for block in self.res_blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        for module in (self.final_layer.adaLN_modulation[-1], self.final_layer.linear):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        y = self.time_embed(t) + self.cond_embed(c)
        for block in self.res_blocks:
            x = checkpoint(block, x, y, use_reentrant=False) if self.grad_checkpointing and self.training else block(x, y)
        return self.final_layer(x, y)


class EmbeddingDecoder(nn.Module):
    def __init__(self, target_dim, condition_dim, width, depth, repeats=4):
        super().__init__()
        self.repeats = repeats
        self.net = SimpleMLPAdaLN(target_dim, width, target_dim, condition_dim, depth, True)

    def loss(self, targets, condition, valid, generator=None):
        if targets.shape[:2] != condition.shape[:2] or valid.shape != targets.shape[:2]:
            raise ValueError("Target/condition correspondence mismatch")
        if not valid.any():
            raise ValueError("Empty feature supervision")
        z = targets[valid].detach().float().repeat_interleave(self.repeats, 0)
        c = condition[valid].repeat_interleave(self.repeats, 0)
        noise = torch.randn(z.shape, device=z.device, generator=generator)
        sigma = torch.rand(z.shape[0], device=z.device, generator=generator)
        x = z * (1 - sigma[:, None]) + noise * sigma[:, None]
        pred = self.net(x.to(c.dtype), (1000 * sigma).to(c.dtype), c).float()
        return (pred - (noise - z)).square().mean()
