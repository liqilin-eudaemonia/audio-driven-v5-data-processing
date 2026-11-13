"""One DDP root owns all trainables; frozen DiT still propagates AV gradients."""
import torch
from torch import nn
from omni_av_p2.connector import SharedCondition, bernini_projection
from omni_av_p2.ltx import append_conditions, av_loss
from omni_av_p21.config import LoRAConfig
from omni_av_p21.lora import inject_text_lora


class GenerationConnector(nn.Module):
    def __init__(self, hidden_size=2048, video_dim=4096, audio_dim=2048):
        super().__init__()
        self.video = bernini_projection(hidden_size, video_dim, zero_init=True)
        self.audio = bernini_projection(hidden_size, audio_dim, zero_init=True)

    def forward(self, data):
        values = {}
        for mod in ('video', 'audio'):
            h = data[mod].detach()  # C0/A freeze planner; stage B requires online graph.
            if h.ndim != 3 or h.shape[0] != 1 or not h.shape[1] or not torch.isfinite(h).all():
                raise ValueError('Expected finite unpadded single-sample readout')
            values[mod] = getattr(self, mod)(h.float())
            values[mod + '_valid'] = torch.ones(h.shape[:2], device=h.device, dtype=torch.bool)
        return SharedCondition(**values)


class RenderTraining(nn.Module):
    def __init__(self, renderer, strategy, stage, connector=None, lora=None):
        super().__init__()
        if stage not in ('C0', 'A'):
            raise ValueError('Only C0/A implemented; B requires live planner gradients')
        self.stage = stage
        self.connector = connector if connector is not None else GenerationConnector()
        renderer.requires_grad_(False)
        self.adapters, self.targets = (inject_text_lora(renderer, lora or LoRAConfig())
                                        if stage == 'A' else (nn.ModuleList(), []))
        object.__setattr__(self, 'renderer', renderer)
        object.__setattr__(self, 'strategy', strategy)
        self.assert_ownership()

    def assert_ownership(self):
        expected = {id(p) for p in self.adapters.parameters()}
        if {id(p) for p in self.renderer.parameters() if p.requires_grad} != expected:
            raise RuntimeError('Unexpected DiT trainables')
        if {id(p) for p in self.parameters()} != expected | {id(p) for p in self.connector.parameters()}:
            raise RuntimeError('DDP trainable ownership mismatch')

    def train(self, mode=True):
        super().train(mode)
        self.renderer.train(mode)
        return self

    def forward(self, hidden, prepared):
        loss, parts = av_loss(self.renderer, self.strategy,
                              append_conditions(prepared, self.connector(hidden)))
        return loss, parts['video_loss'], parts['audio_loss']

    def optimizer_groups(self, connector_lr, dit_lr):
        groups = [dict(params=list(self.connector.parameters()), lr=connector_lr, name='connector')]
        if self.stage == 'A':
            groups.append(dict(params=list(self.adapters.parameters()), lr=dit_lr, name='dit_text_lora'))
        return groups
