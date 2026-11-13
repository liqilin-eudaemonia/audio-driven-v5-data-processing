"""Same U supervises Event/Feature; independent speaker branch conditions LTX."""
import torch
from p2_whisper_av.model import QwenWhisperAV
from p2_matrix16.model import MatrixModel
from .config import validate,pure_av_view
from . import VERSION


class QwenWhisperEventFeature(QwenWhisperAV):
    def __init__(self,c,cfg,renderer):
        validate(c)
        # Original constructor preserves Connector/text LoRA/Whisper values and
        # RNG. Auxiliary decoder initialization is also forked like MatrixModel.
        super().__init__(pure_av_view(c),cfg,renderer)
        from omni_av_p2.decoder import EmbeddingDecoder
        with torch.random.fork_rng(devices=[]):
            for mod,dim in [('video',cfg.video_dim),('audio',cfg.audio_dim)]:
                self.decoders[mod]=EmbeddingDecoder(cfg.target_dim,dim,cfg.decoder_width,
                    cfg.decoder_depth,cfg.diffusion_batch_mul)
        self.decoders.to(next(renderer.parameters()).device)
        self.event_heads.requires_grad_(True)
        self.spec=c
        self.assert_ownership()

    def assert_ownership(self):
        if (not hasattr(self,'whisper_branches') or self.spec.get('loss')!='av_event_feature'):
            return QwenWhisperAV.assert_ownership(self)
        ids=lambda m:{id(p) for p in m.parameters() if p.requires_grad}
        renderer_owned=ids(self.text_lora)|ids(self.whisper_branches)
        if ids(self.renderer)!=renderer_owned:
            raise ValueError('Frozen renderer or unowned adapter became trainable')
        owned=renderer_owned|ids(self.connector)|ids(self.whisper_adapter)|ids(self.event_heads)|ids(self.decoders)
        if (ids(self)!=owned or len(self.qwen_lora) or set(self.decoders)!= {'video','audio'}
                or not all(p.requires_grad for p in self.event_heads.parameters())
                or not all(p.requires_grad for p in self.decoders.parameters())):
            raise ValueError('Combined loss/Whisper trainable ownership mismatch')

    def groups(self):
        t=self.spec['training']
        groups=[dict(name=name,params=list(module.parameters()),lr=lr) for name,module,lr in [
            ('connector',self.connector,t['connector_lr']),('events',self.event_heads,t['event_lr']),
            ('features',self.decoders,t['feature_lr']),('text_lora',self.text_lora,self.spec['text_lora']['learning_rate']),
            ('whisper',torch.nn.ModuleList([self.whisper_adapter,self.whisper_branches]),self.spec['whisper']['learning_rate'])]]
        params=[p for g in groups for p in g['params']]
        if len(params)!=len({id(p) for p in params}) or {id(p) for p in params}!={id(p) for p in self.parameters() if p.requires_grad}:
            raise ValueError('Missing, frozen or duplicate optimizer parameters')
        return groups

    def forward(self,bundle,pos_weights,generator,strategy,sampler):
        from omni_av.ltx_runtime import with_text
        from p2_listen.runtime import av_loss
        if bundle.get('teacher') is None or bundle.get('whisper') is None or bundle.get('events') is None:
            raise ValueError('Combined training requires teacher, Event targets and speaker Whisper')
        def av_callback(u,pack,text):
            prepared=strategy.prepare_training_inputs(with_text(pack['av'],text),sampler)
            prepared=self.prepare(prepared,u,bundle['whisper'])
            return av_loss(self.renderer,prepared,pack['support'],return_components=True)
        # One Connector evaluation; native masked teacher losses and Event
        # normalization remain unchanged. Whisper learns through AV gradients.
        return MatrixModel.forward(self,bundle,pos_weights,generator,av_callback)
