"""Planning-only LoRA; disabled native mode is exactly the frozen base operation."""
import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class PlanningLinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.enabled = True
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        delta = F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
        return out + delta * self.scale


def inject_planning_lora(thinker, rank=16, alpha=32):
    thinker.requires_grad_(False)
    last_prefix = f"model.layers.{len(thinker.model.layers) - 1}."
    names = [name for name, mod in thinker.named_modules()
             if name.startswith("model.layers.") and isinstance(mod, nn.Linear)
             and not name.startswith(last_prefix)
             and name.endswith(tuple(f"self_attn.{x}_proj" for x in ("q", "k", "v", "o")))]
    if len(names) != (len(thinker.model.layers) - 1) * 4:
        raise ValueError("Unexpected Qwen2.5-Omni attention layout")
    for name in names:
        parent, leaf = name.rsplit(".", 1)
        owner = thinker.get_submodule(parent)
        setattr(owner, leaf, PlanningLinear(getattr(owner, leaf), rank, alpha))
    return names


@contextmanager
def native_reply_mode(model):
    """No training/backward is allowed inside this inference-only context.

    Restore each module's training flag and each adapter's enable flag even on failure.
    Planning mode stays enabled outside the context, including checkpoint recomputation.
    """
    flags = [(mod, mod.training) for mod in model.modules()]
    adapters = [(mod, mod.enabled) for mod in model.modules() if isinstance(mod, PlanningLinear)]
    try:
        model.eval()
        for mod, _ in adapters:
            mod.enabled = False
        with torch.no_grad():
            yield
    finally:
        for mod, enabled in adapters:
            mod.enabled = enabled
        for mod, training in flags:
            mod.training = training
