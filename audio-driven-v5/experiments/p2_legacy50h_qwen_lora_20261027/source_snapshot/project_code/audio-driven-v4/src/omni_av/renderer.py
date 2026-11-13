"""Explicit LTX omni_context integration, without edits to the LTX checkout."""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F


class PlanAttention(nn.Module):
    def __init__(self, query_dim, context_dim, width=256, heads=4):
        super().__init__()
        self.heads, self.width = heads, width
        self.q_norm = nn.LayerNorm(query_dim)
        self.kv_norm = nn.LayerNorm(context_dim)
        self.q = nn.Linear(query_dim, width, bias=False)
        self.k = nn.Linear(context_dim, width, bias=False)
        self.v = nn.Linear(context_dim, width, bias=False)
        self.out = nn.Linear(width, query_dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x, context, mask=None, **kwargs):
        b, n, _ = x.shape
        context = self.kv_norm(context)
        q = self.q(self.q_norm(x)).view(b, n, self.heads, -1).transpose(1, 2)
        k = self.k(context).view(b, -1, self.heads, self.width // self.heads).transpose(1, 2)
        v = self.v(context).view(b, -1, self.heads, self.width // self.heads).transpose(1, 2)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :].bool()
            elif mask.ndim == 3:
                mask = mask[:, None]
            if mask.ndim != 4:
                raise ValueError("Unsupported planning attention mask")
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0)
        return self.out(attended.transpose(1, 2).reshape(b, n, self.width))


class PlanningConnector(nn.Module):
    def __init__(self, hidden_size, video_dim=512, audio_dim=512):
        super().__init__()
        self.video = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, video_dim), nn.SiLU())
        self.audio = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, audio_dim), nn.SiLU())

    def forward(self, hidden, valid):
        if valid.shape != hidden.shape[:2] or not valid.any(-1).all():
            raise ValueError("Empty or invalid renderer condition")
        # Retain all selected text/target positions; no random global compression.
        return self.video(hidden), self.audio(hidden), valid


def install_plan_attention(model, context_dim=512, width=256):
    """Use existing explicit Modality.omni_context transport, no context-capturing hooks."""
    model.requires_grad_(False)
    device = next(model.parameters()).device
    for block in model.transformer_blocks:
        if hasattr(block, "omni_attn2") or hasattr(block, "audio_omni_attn2"):
            raise ValueError("Model already has a planning branch")
        for base_name, name in (("attn2", "omni_attn2"), ("audio_attn2", "audio_omni_attn2")):
            base = getattr(block, base_name, None)
            if base is not None:
                branch = PlanAttention(base.to_q.in_features, context_dim, width).to(device)
                setattr(block, name, branch)


def attach_plan(prepared, connector, hidden, valid):
    video, audio, mask = connector(hidden, valid)
    return replace(prepared,
        video=replace(prepared.video, omni_context=video, omni_context_mask=mask),
        audio=replace(prepared.audio, omni_context=audio, omni_context_mask=mask)
              if prepared.audio is not None else None)


def configure_renderer_lora(model, rank=8, alpha=8):
    from peft import LoraConfig, get_peft_model
    targets = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)
               and name.rsplit(".", 1)[-1] in {"to_q", "to_k", "to_v"}
               and "omni_attn2" not in name]
    if not targets:
        raise ValueError("No native LTX LoRA targets")
    wrapped = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha,
                            target_modules=targets, lora_dropout=0.0, bias="none"))
    for name, param in wrapped.named_parameters():
        if "omni_attn2" in name:
            param.requires_grad_(True)
    return wrapped
