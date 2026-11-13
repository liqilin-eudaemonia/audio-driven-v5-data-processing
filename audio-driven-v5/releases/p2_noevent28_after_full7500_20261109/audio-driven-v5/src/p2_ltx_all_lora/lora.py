"""Extend native text LoRA to every native block attention and FFN Linear."""
from pathlib import Path
import torch
from torch import nn

ATTENTION = ('attn1', 'attn2', 'audio_attn1', 'audio_attn2',
             'audio_to_video_attn', 'video_to_audio_attn')
FFN = ('ff', 'audio_ff')
PROJECTIONS = ('to_q', 'to_k', 'to_v', 'to_out.0')
WHISPER = ('qwen_attn2', 'audio_qwen_attn2')


def targets(renderer):
    """Explicit module boundary: never adapt Whisper or any Linear outside blocks."""
    from omni_av_p21.lora import AdaptedLinear
    result = []
    for i, block in enumerate(renderer.transformer_blocks):
        for name in ATTENTION:
            attention = getattr(block, name)
            leaves = list(PROJECTIONS)
            if getattr(attention, 'to_gate_logits', None) is not None:
                leaves.append('to_gate_logits')
            for leaf in leaves:
                module = attention.get_submodule(leaf)
                if not isinstance(module, (nn.Linear, AdaptedLinear)):
                    raise ValueError('Unexpected attention projection: ' + name + '.' + leaf)
                result.append((f'transformer_blocks.{i}.{name}.{leaf}', module))
        for name in FFN:
            linear = [(leaf, m) for leaf, m in getattr(block, name).named_modules()
                      if isinstance(m, nn.Linear)]
            adapted = [(leaf, m) for leaf, m in getattr(block, name).named_modules()
                       if isinstance(m, AdaptedLinear)]
            # An already-adapted FFN is never an acceptable new installation.
            if adapted or not linear:
                raise ValueError('Missing or already adapted FFN: ' + name)
            for leaf, module in linear:
                result.append((f'transformer_blocks.{i}.{name}.{leaf}', module))
    if not result:
        raise ValueError('Empty native LTX target set')
    return result


def extend(model):
    """Keep the existing text-LoRA checkpoint prefix and full Whisper parameters."""
    from omni_av_p21.config import LoRAConfig
    from omni_av_p21.lora import AdaptedLinear, LowRankUpdate
    if getattr(model, '_all_lora_installed', False):
        raise ValueError('All-attention/FFN LoRA already installed')
    renderer = model.renderer
    plan = targets(renderer)
    existing = [(name, m) for name, m in plan if isinstance(m, AdaptedLinear)]
    if [name for name, _ in existing] != model.text_targets:
        raise ValueError('Expected exactly the existing native text LoRA prefix')
    if any(id(m.update) != id(a) for (_, m), a in zip(existing, model.text_lora, strict=True)):
        raise ValueError('Existing adapters are not owned by the model')
    additions = [(name, m) for name, m in plan if isinstance(m, nn.Linear)]
    cfg = LoRAConfig(**{k: v for k, v in model.spec['text_lora'].items() if k != 'enabled'})
    device = next(renderer.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    # Preserve baseline modules and the following sample/noise random stream.
    with torch.random.fork_rng(devices=devices):
        for name, base in additions:
            owner, leaf = name.rsplit('.', 1)
            update = LowRankUpdate(base, cfg)
            setattr(renderer.get_submodule(owner), leaf, AdaptedLinear(base, update))
            model.text_lora.append(update)
            model.text_targets.append(name)
    model._all_lora_installed = True
    model.ltx_lora_targets = list(model.text_targets)
    model.ltx_lora_report = dict(
        blocks=len(renderer.transformer_blocks), projections=len(model.text_targets),
        existing_text_projections=len(existing), added_projections=len(additions),
        rank=cfg.rank, alpha=cfg.alpha,
        parameters=sum(p.numel() for p in model.text_lora.parameters()),
        whisper_full_parameters=sum(p.numel() for p in model.whisper_branches.parameters()),
        attention=list(ATTENTION), ffn=list(FFN), whisper_excluded=list(WHISPER))
    model.assert_ownership()
    return model.ltx_lora_report

