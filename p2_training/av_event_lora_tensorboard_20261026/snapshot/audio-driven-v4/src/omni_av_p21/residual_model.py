"""Feature-only warmup, then full residual branches, then original attention LoRA."""
from dataclasses import replace
import torch
from torch import nn
from omni_av_p2.model import Extractor
from omni_av_p2.ltx import av_loss
from .lora import AdaptedLinear, LowRankUpdate, PROJECTIONS
from .residual_config import STAGES, VERSION

STREAMS = ('attn1', 'attn2', 'audio_attn1', 'audio_attn2',
           'audio_to_video_attn', 'video_to_audio_attn')
QWEN_STREAMS = ('qwen_attn2', 'audio_qwen_attn2')


def attach_hidden(prepared, u):
    """Text tensors are unchanged; U uses the existing independent Qwen slots."""
    values = {}
    for name in ('video', 'audio'):
        modality = getattr(prepared, name)
        if modality.qwen_context is not None:
            raise ValueError('Qwen conditions already supplied')
        values[name] = replace(modality, qwen_context=getattr(u, name).to(modality.context.dtype),
                               qwen_context_mask=getattr(u, name + '_valid'))
    return replace(prepared, **values)


def inject_attention_lora(renderer, config):
    if any(isinstance(m, AdaptedLinear) for m in renderer.modules()):
        raise ValueError('LoRA already installed')
    targets = []
    for i, block in enumerate(renderer.transformer_blocks):
        for stream in STREAMS:
            attention = getattr(block, stream, None)
            if attention is None:
                raise ValueError(f'Missing AV attention: {i}.{stream}')
            for projection in PROJECTIONS:
                base = attention.get_submodule(projection)
                if not isinstance(base, nn.Linear):
                    raise ValueError('LoRA requires original Linear projections')
                targets.append((f'transformer_blocks.{i}.{stream}.{projection}', base))
    updates = nn.ModuleList()
    for name, base in targets:
        update = LowRankUpdate(base, config)
        owner, leaf = name.rsplit('.', 1)
        setattr(renderer.get_submodule(owner), leaf, AdaptedLinear(base, update))
        updates.append(update)
    return updates, [name for name, _ in targets]


class ResidualTrainingModel(nn.Module):
    def __init__(self, cfg, lora, residual, stage, renderer=None, strategy=None,
                 extractor=None, auxiliary=True):
        super().__init__()
        if stage not in STAGES:
            raise ValueError('Expected stage 1, 2 or 3')
        if stage == 1 and (renderer is not None or not auxiliary):
            raise ValueError('Stage 1 is feature-only and must not load LTX')
        if stage > 1 and (renderer is None or strategy is None):
            raise ValueError('Stage 2/3 require the native renderer and strategy')
        self.cfg, self.lora, self.residual, self.stage = cfg, lora, residual, stage
        self.extractor = extractor if extractor is not None else Extractor(cfg, auxiliary=auxiliary)
        self.qwen = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.targets, self.qwen_targets = [], []
        object.__setattr__(self, 'renderer', renderer)
        object.__setattr__(self, 'strategy', strategy)
        if renderer is not None:
            renderer.requires_grad_(False)
            if any(hasattr(b, s) for b in renderer.transformer_blocks for s in QWEN_STREAMS):
                raise ValueError('Expected a fresh base without Qwen branches')
            renderer.enable_qwen_conditioning(cfg.video_dim, cfg.audio_dim)
            for i, block in enumerate(renderer.transformer_blocks):
                for stream in QWEN_STREAMS:
                    module = getattr(block, stream).float().requires_grad_(True)
                    self.qwen.append(module)
                    self.qwen_targets.append(f'transformer_blocks.{i}.{stream}')
            if stage == 3:
                self.adapters, self.targets = inject_attention_lora(renderer, lora)
        self.assert_parameter_ownership()

    def architecture(self):
        return dict(version=VERSION, model=self.cfg.to_dict(), lora=self.lora.to_dict(),
                    residual=self.residual.to_dict(), stage=self.stage, targets=self.targets,
                    qwen_targets=self.qwen_targets)

    def assert_parameter_ownership(self):
        expected = {id(p) for p in self.qwen.parameters()} | {id(p) for p in self.adapters.parameters()}
        actual = {id(p) for p in self.renderer.parameters() if p.requires_grad} if self.renderer else set()
        if actual != expected:
            raise RuntimeError('Unexpected renderer trainables: only full Qwen and stage3 attention LoRA')
        registered = {id(p) for p in self.parameters()}
        if registered != expected | {id(p) for p in self.extractor.parameters()}:
            raise RuntimeError('DDP parameter ownership mismatch')
        if any(not p.requires_grad for p in self.parameters()):
            raise RuntimeError('This DDP root must register only active trainables')

    def train(self, mode=True):
        super().train(mode)
        if self.renderer is not None:
            self.renderer.train(mode)
        return self

    def forward(self, condition, targets, prepared=None, generator=None):
        if targets is None:
            raise ValueError('Training requires feature targets in all three stages')
        u, losses = self.extractor(condition, targets, generator)
        lv, la = losses['video'], losses['audio']
        lav = lv.new_zeros(())
        vv, aa = lav, lav
        if self.stage == 1:
            if prepared is not None:
                raise ValueError('Stage 1 must not use AV inputs or AV loss')
        else:
            lav, parts = av_loss(self.renderer, self.strategy, attach_hidden(prepared, u))
            vv, aa = parts['video_loss'], parts['audio_loss']
        loss = self.cfg.feature_video_weight * lv + self.cfg.feature_audio_weight * la
        if self.stage > 1:
            loss = loss + self.cfg.av_weight * lav
        return loss, lav, lv, la, vv, aa

    def optimizer_groups(self):
        groups = [dict(params=list(self.extractor.parameters()), lr=self.cfg.learning_rate, name='extractor')]
        if self.stage >= 2:
            groups.append(dict(params=list(self.qwen.parameters()), lr=self.residual.learning_rate, name='qwen_full'))
        if self.stage == 3:
            groups.append(dict(params=list(self.adapters.parameters()), lr=self.lora.learning_rate, name='dit_attention_lora'))
        return groups
