"""Preserve native Qwen/Gemma attention; add two independent Whisper residuals."""
from dataclasses import replace

import torch
from torch import nn

from p2_matrix16.model import MatrixModel
from .config import validate


class QwenWhisperAV(MatrixModel):
    def __init__(self, c, cfg, renderer):
        from .bootstrap import activate_ltx
        activate_ltx()
        validate(c)
        # Original Connector/Event/LoRA initialization order remains unchanged.
        super().__init__(c, cfg, renderer)
        self.event_heads.requires_grad_(False)
        w = c['whisper']
        for block in renderer.transformer_blocks:
            if any(hasattr(block, n) for n in ('qwen_attn2', 'audio_qwen_attn2')):
                raise ValueError('Independent Whisper context is already occupied')
        device = next(renderer.parameters()).device
        # The baseline renderer is kept outside model.state_dict(); register only
        # the new trainables, as done for the original text LoRA.
        # Fork both RNGs so constructing this extra branch does not alter baseline
        # Connector/LoRA RNG or subsequent per-sample diffusion noise.
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=cuda_devices):
            renderer.enable_qwen_conditioning(w['context_dim'], enable_video=True, enable_audio=True)
            self.whisper_branches = nn.ModuleList([
                getattr(block, n) for block in renderer.transformer_blocks
                for n in ('qwen_attn2', 'audio_qwen_attn2')])
            self.whisper_branches.float().requires_grad_(True)
            from ltx_core.model.transformer.attention import PytorchAttention
            for branch in self.whisper_branches:
                branch.attention_function = PytorchAttention()
            self.whisper_adapter = nn.Sequential(nn.LayerNorm(w['input_dim']),
                nn.Linear(w['input_dim'], w['context_dim'])).to(device=device, dtype=torch.float32)
        audio_pre = renderer.audio_args_preprocessor
        audio_pre = getattr(audio_pre, 'simple_preprocessor', audio_pre)
        self.time_max_pos = audio_pre.max_pos[0]
        if self.time_max_pos <= 0:
            raise ValueError('Invalid shared LTX audio time scale')
        for name in ('video_args_preprocessor', 'audio_args_preprocessor'):
            pre = getattr(renderer, name)
            getattr(pre, 'simple_preprocessor', pre).qwen_time_max_pos = self.time_max_pos
        self.assert_ownership()

    def assert_ownership(self):
        if not hasattr(self, 'whisper_branches'):
            return super().assert_ownership()
        ids = lambda m: {id(p) for p in m.parameters() if p.requires_grad}
        expected = ids(self.text_lora) | ids(self.whisper_branches)
        if ids(self.renderer) != expected:
            raise ValueError('Frozen LTX base or unowned renderer parameter became trainable')
        owned = ids(self.connector) | expected | ids(self.whisper_adapter)
        if ids(self) != owned or ids(self.event_heads) or len(self.decoders) or len(self.qwen_lora):
            raise ValueError('AV-only trainable ownership mismatch')

    def groups(self):
        groups = [dict(name='connector', params=list(self.connector.parameters()),
                       lr=self.spec['training']['connector_lr']),
                  dict(name='text_lora', params=list(self.text_lora.parameters()),
                       lr=self.spec['text_lora']['learning_rate']),
                  dict(name='whisper', params=list(self.whisper_adapter.parameters()) +
                       list(self.whisper_branches.parameters()), lr=self.spec['whisper']['learning_rate'])]
        parameters = [p for g in groups for p in g['params']]
        if (len(parameters) != len({id(p) for p in parameters}) or
                {id(p) for p in parameters} != {id(p) for p in self.parameters() if p.requires_grad}):
            raise ValueError('Duplicate or missing optimizer parameter')
        return groups

    def condition(self, prepared, speaker):
        from ltx_core.distributed.parallel_state import is_sp_enabled
        if is_sp_enabled():
            raise ValueError('Whisper W1 supports DDP only; sequence parallelism is not audited')
        if speaker is None:
            raise ValueError('Whisper condition is required; do not silently fall back to baseline')
        speaker.validate()
        if speaker.hidden.shape[-1] != self.spec['whisper']['input_dim']:
            raise ValueError('Whisper dimension differs')
        if not speaker.valid.any(-1).all():
            # A silent waveform still has valid Whisper tokens. An empty mask is
            # missing evidence, not silence, and must never produce a masked softmax.
            raise ValueError('Whisper input must contain valid observation tokens for every sample')
        hidden = self.whisper_adapter(speaker.hidden.float())
        result = {}
        for name in ('video', 'audio'):
            mod = getattr(prepared, name)
            if mod is None or mod.latent.shape[0] != hidden.shape[0] or mod.qwen_context is not None:
                raise ValueError('Whisper / renderer batch mismatch or occupied residual context')
            result[name] = replace(mod, qwen_context=hidden.to(mod.latent.dtype),
                qwen_context_mask=speaker.valid,
                qwen_time_positions=speaker.bounds if self.spec['whisper']['temporal_rope'] else None)
        return replace(prepared, **result)

    def prepare(self, prepared, u, speaker):
        """Shared training/inference connection, including both CFG paths."""
        from omni_av_p2.ltx import append_conditions
        return self.condition(append_conditions(prepared, u), speaker)

    def forward(self, bundle, strategy, sampler):
        from omni_av.ltx_runtime import with_text
        from p2_listen.runtime import av_loss
        if bundle.get('condition') is None or bundle.get('teacher') is not None:
            raise ValueError('Whisper AV requires frozen Qwen H and forbids Feature teacher targets')
        u = self.connector(bundle['condition'])
        prepared = strategy.prepare_training_inputs(with_text(bundle['av']['av'], bundle['text']['positive']), sampler)
        prepared = self.prepare(prepared, u, bundle['whisper'])
        loss, components = av_loss(self.renderer, prepared, bundle['av']['support'], return_components=True)
        return dict(av=loss, av_components=components)
