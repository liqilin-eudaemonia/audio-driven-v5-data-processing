"""Separate feature connectors and Bernini AdaLN decoders; no renderer in phase 1."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from omni_av.lora import inject_planning_lora
from omni_av.masking import fill_schedule, hybrid_attention
from omni_av_p2.connector import RMSNorm
from omni_av_p2.decoder import EmbeddingDecoder


@dataclass(frozen=True)
class Config:
    omni_path: str = "/public/2_CKPT/3_PreTRAINED/Qwen2.5-Omni-3B"
    hidden_size: int = 2048
    vision_max_pixels: int = 224 * 224
    vision_fps: float = 2.0
    decoder_width: int = 512
    decoder_depth: int = 16
    diffusion_batch_mul: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    beta_alpha: float = 12.0
    beta_beta: float = .9
    all_mask_probability: float = .2
    time_ticks_per_second: int = 25
    seed: int = 20261008
    steps: int = 200
    heads_lr: float = 1e-4
    lora_lr: float = 1e-4
    train_samples: int = 16
    dev_samples: int = 8
    max_duration: float = 8.0
    eval_seeds: tuple = (137, 251)
    planning_steps: int = 25
    denoising_steps: int = 5

    def __post_init__(self):
        for name in ("hidden_size", "decoder_width", "decoder_depth", "diffusion_batch_mul",
                     "lora_rank", "steps", "train_samples", "dev_samples", "planning_steps",
                     "denoising_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.beta_alpha <= 0 or self.beta_beta <= 0 or not 0 <= self.all_mask_probability <= 1:
            raise ValueError("Invalid masking distribution")


@dataclass
class Layout:
    """Inference input deliberately has no target-feature field."""
    prefix: torch.Tensor
    prefix_positions: torch.Tensor
    prefix_valid: torch.Tensor
    coordinates: torch.Tensor
    modalities: torch.Tensor
    valid: torch.Tensor

    def to(self, device):
        return Layout(**{k: v.to(device) for k, v in vars(self).items()})

    def validate(self):
        b, p, _ = self.prefix.shape
        n = self.valid.shape[1]
        if self.prefix_positions.shape != (3, b, p) or self.prefix_valid.shape != (b, p):
            raise ValueError("Prefix shape mismatch")
        if self.coordinates.shape != (b, n, 3) or self.modalities.shape != (b, n):
            raise ValueError("Target layout shape mismatch")
        for mask in (self.prefix_valid, self.valid):
            if mask.dtype != torch.bool or not mask.any(-1).all():
                raise ValueError("Invalid/empty layout mask")
        if not torch.isfinite(self.prefix).all() or not torch.isfinite(self.coordinates).all():
            raise ValueError("Nonfinite layout")
        if (self.coordinates < 0).any() or ((self.modalities < 0) | (self.modalities > 1)).any():
            raise ValueError("Invalid coordinates/modality")


def content_mask(layout, cfg, seed, all_mask=False):
    """CPU RNG independent of model parameter count and device execution order."""
    rng = np.random.default_rng(seed)
    if all_mask or rng.random() < cfg.all_mask_probability:
        return layout.valid.clone()
    mask = torch.zeros_like(layout.valid)
    for b in range(len(mask)):
        ratio = rng.beta(cfg.beta_alpha, cfg.beta_beta)
        for mod in (0, 1):
            ids = torch.where(layout.valid[b] & (layout.modalities[b] == mod))[0]
            if len(ids):
                chosen = rng.permutation(len(ids))[:max(1, math.ceil(ratio * len(ids)))]
                mask[b, ids[torch.as_tensor(chosen, device=ids.device)]] = True
    return mask


def feature_connector(dim):
    # Exact Bernini MLPConnector.for_vit topology (not its shorter for_gen branch).
    return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim),
                         RMSNorm(dim), nn.Linear(dim, dim))


class Planner(nn.Module):
    def __init__(self, thinker, cfg, arm):
        super().__init__()
        if arm not in {"frozen", "lora"}:
            raise ValueError("Unknown comparison arm")
        self.thinker, self.cfg, self.arm = thinker, cfg, arm
        thinker.requires_grad_(False)
        if arm == "lora":
            inject_planning_lora(thinker, cfg.lora_rank, cfg.lora_alpha)
        # Isolated RNG: both arms have exactly identical common head initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            self.mask_token = nn.Parameter(torch.zeros(2, cfg.hidden_size))
            self.modality_embedding = nn.Parameter(torch.randn(2, cfg.hidden_size) * .01)
            self.connectors = nn.ModuleList([feature_connector(cfg.hidden_size) for _ in range(2)])
            self.decoders = nn.ModuleList([EmbeddingDecoder(cfg.hidden_size, cfg.hidden_size,
                cfg.decoder_width, cfg.decoder_depth, cfg.diffusion_batch_mul) for _ in range(2)])

    def train(self, mode=True):
        super().train(mode)
        # Frozen perception and no stochastic dropout: paired arms receive identical randomness.
        for name in ("visual", "audio_tower"):
            module = getattr(self.thinker, name, None)
            if module is not None:
                module.eval()
        return self

    def hidden(self, layout, masked, values):
        layout.validate()
        if masked.shape != layout.valid.shape or (masked & ~layout.valid).any():
            raise ValueError("Invalid content mask")
        expected = (*layout.valid.shape, layout.prefix.shape[-1])
        if values.shape != expected:
            raise ValueError("Feature/layout mismatch")
        mod = layout.modalities
        target = torch.where(masked[..., None], self.mask_token[mod].to(values.dtype), values)
        target = (target + self.modality_embedding[mod].to(values.dtype)) * layout.valid[..., None]
        embeds = torch.cat([layout.prefix.detach(), target], 1)
        coords = layout.coordinates.clone()
        coords[..., 0] = (coords[..., 0] * self.cfg.time_ticks_per_second).round()
        offset = layout.prefix_positions.amax(dim=(0, 2)) + 1
        positions = torch.cat([layout.prefix_positions, coords.long().permute(2, 0, 1)
                               + offset[None, :, None]], -1)
        attention = hybrid_attention(layout.prefix_valid, layout.valid, embeds.dtype)
        out = self.thinker.model(inputs_embeds=embeds, attention_mask=attention,
            position_ids=positions, use_cache=False, output_hidden_states=True, return_dict=True)
        return out.hidden_states[-2][:, layout.prefix.shape[1]:]

    def loss(self, layout, targets, seed, all_mask=False):
        if targets.requires_grad or not torch.isfinite(targets).all():
            raise ValueError("Teacher must be frozen and finite")
        masked = content_mask(layout, self.cfg, seed, all_mask)
        h = self.hidden(layout, masked, targets.detach())
        losses = {}
        for mod, name in enumerate(("video", "audio")):
            selected = masked & layout.valid & (layout.modalities == mod)
            if not selected.any():
                raise ValueError("Both modalities require masked supervision")
            c = self.connectors[mod](h[selected].float())[None]
            z = targets[selected][None]
            generator = torch.Generator(device=targets.device).manual_seed(seed + mod + 100)
            losses[name] = self.decoders[mod].loss(z, c, torch.ones(z.shape[:2],
                dtype=torch.bool, device=z.device), generator)
        return losses

    @torch.no_grad()
    def sample(self, layout, seed, steps=None):
        """No GT accepted; position-specific noise reused across one/25-round evaluation."""
        if self.training:
            raise ValueError("Sampling requires eval mode")
        if layout.prefix.shape[0] != 1:
            raise ValueError("Pilot sampler supports batch size 1")
        steps = self.cfg.planning_steps if steps is None else steps
        masked = layout.valid.clone()
        values = torch.zeros((*masked.shape, self.cfg.hidden_size), device=masked.device,
                             dtype=layout.prefix.dtype)
        gen = torch.Generator(device=values.device).manual_seed(seed)
        initial_noise = torch.randn(values.shape, device=values.device, generator=gen)
        ids = torch.where(masked[0])[0]
        order = ids[torch.randperm(len(ids), device=ids.device, generator=gen)]
        remaining = []
        for begin, end in fill_schedule(len(ids), steps):
            if end > begin:
                h = self.hidden(layout, masked, values)
                selected = torch.zeros_like(masked)
                selected[0, order[begin:end]] = True
                for mod in (0, 1):
                    take = selected & (layout.modalities == mod)
                    if not take.any():
                        continue
                    c = self.connectors[mod](h[take].float())
                    x = initial_noise[take].clone()
                    for k in range(self.cfg.denoising_steps):
                        sigma = 1 - k / self.cfg.denoising_steps
                        t = torch.full((len(x),), 1000 * sigma, device=x.device, dtype=c.dtype)
                        x = x - self.decoders[mod].net(x.to(c.dtype), t, c).float() / self.cfg.denoising_steps
                    values[take] = x.to(values.dtype)
                masked[selected] = False
            remaining.append(int(masked.sum()))
        if masked.any() or not torch.isfinite(values).all():
            raise RuntimeError("Invalid planning result")
        return values, remaining

    def trainable_state(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}
