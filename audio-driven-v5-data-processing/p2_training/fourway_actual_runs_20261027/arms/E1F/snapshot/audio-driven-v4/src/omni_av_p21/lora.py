"""Only native video/audio TEXT cross-attention projections receive adapters."""
import math

import torch
from torch import nn

PROJECTIONS = ("to_q", "to_k", "to_v", "to_out.0")
STREAMS = ("attn2", "audio_attn2")


class LowRankUpdate(nn.Module):
    def __init__(self, base, config):
        super().__init__()
        self.scale = config.alpha / config.rank
        # FP32 master parameters and Adam state; autocast controls forward dtype.
        self.a = nn.Linear(base.in_features, config.rank, bias=False,
                           device=base.weight.device, dtype=torch.float32)
        self.b = nn.Linear(config.rank, base.out_features, bias=False,
                           device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.b(self.a(x.to(self.a.weight.dtype))) * self.scale


class AdaptedLinear(nn.Module):
    def __init__(self, base, update):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.update = update

    def forward(self, x):
        value = self.base(x)
        return value + self.update(x).to(value.dtype)


def inject_text_lora(renderer, config):
    """Validate the entire target set before mutating; refuse repeated injection."""
    if any(isinstance(m, AdaptedLinear) for m in renderer.modules()):
        raise ValueError("Text LoRA is already installed")
    targets = []
    for index, block in enumerate(renderer.transformer_blocks):
        for stream in STREAMS:
            attention = getattr(block, stream, None)
            if attention is None:
                raise ValueError(f"Missing native text attention {index}.{stream}")
            for projection in PROJECTIONS:
                base = attention.get_submodule(projection)
                if not isinstance(base, nn.Linear):
                    raise ValueError("Expected an unmodified text attention Linear")
                targets.append((f"transformer_blocks.{index}.{stream}.{projection}", base))
    if not targets:
        raise ValueError("No native text attention targets")
    renderer.requires_grad_(False)
    updates = nn.ModuleList()
    for name, base in targets:
        update = LowRankUpdate(base, config)
        owner, leaf = name.rsplit(".", 1)
        setattr(renderer.get_submodule(owner), leaf, AdaptedLinear(base, update))
        updates.append(update)
    return updates, [name for name, _ in targets]
