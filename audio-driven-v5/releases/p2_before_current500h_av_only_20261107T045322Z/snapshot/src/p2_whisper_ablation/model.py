import torch
from p2_whisper_av.model import QwenWhisperAV
from p2_matrix16.model import MatrixModel
from p2_matrix16.config import event_enabled,feature_enabled
from .config import validate,pure_av_view


class WhisperAblation(QwenWhisperAV):
    def __init__(self,c,cfg,renderer):
        validate(c);super().__init__(pure_av_view(c),cfg,renderer)
        if feature_enabled(c):
            from omni_av_p2.decoder import EmbeddingDecoder
            with torch.random.fork_rng(devices=[]):
                for mod,dim in [('video',cfg.video_dim),('audio',cfg.audio_dim)]:
                    self.decoders[mod]=EmbeddingDecoder(cfg.target_dim,dim,cfg.decoder_width,cfg.decoder_depth,cfg.diffusion_batch_mul)
            self.decoders.to(next(renderer.parameters()).device)
        self.event_heads.requires_grad_(event_enabled(c));self.spec=c;self.assert_ownership()

    def assert_ownership(self):
        if not hasattr(self,'whisper_branches') or self.spec.get('loss')=='av':return QwenWhisperAV.assert_ownership(self)
        ids=lambda m:{id(p) for p in m.parameters() if p.requires_grad}
        renderer=ids(self.text_lora)|ids(self.whisper_branches)
        if ids(self.renderer)!=renderer:raise ValueError('Frozen renderer trainability changed')
        expected=renderer|ids(self.connector)|ids(self.whisper_adapter)|ids(self.event_heads)|ids(self.decoders)
        if ids(self)!=expected or len(self.qwen_lora):raise ValueError('Parameter ownership differs')
        if bool(ids(self.event_heads))!=event_enabled(self.spec):raise ValueError('Disabled Event head is trainable')
        if set(self.decoders)!=({'video','audio'} if feature_enabled(self.spec) else set()):raise ValueError('Disabled Feature branch exists')

    def groups(self):
        c=self.spec;t=c['training']
        candidates=[('connector',self.connector,t['connector_lr']),('text_lora',self.text_lora,c['text_lora']['learning_rate']),
                    ('whisper',torch.nn.ModuleList([self.whisper_adapter,self.whisper_branches]),c['whisper']['learning_rate'])]
        if event_enabled(c):candidates.append(('events',self.event_heads,t['event_lr']))
        if feature_enabled(c):candidates.append(('features',self.decoders,t['feature_lr']))
        groups=[dict(name=n,params=list(m.parameters()),lr=lr) for n,m,lr in candidates]
        params=[p for g in groups for p in g['params']]
        if len(params)!=len({id(p) for p in params}) or {id(p) for p in params}!={id(p) for p in self.parameters() if p.requires_grad}:raise ValueError('Duplicate/missing optimizer parameters')
        return groups

    def forward(self,bundle,pos_weights,generator,strategy,sampler):
        from omni_av.ltx_runtime import with_text
        from p2_listen.runtime import av_loss
        if bundle.get('whisper') is None or (bundle.get('teacher') is not None)!=feature_enabled(self.spec):raise ValueError('Wrong source or target branch')
        def av_callback(u,pack,text):
            prepared=strategy.prepare_training_inputs(with_text(pack['av'],text),sampler)
            return av_loss(self.renderer,self.prepare(prepared,u,bundle['whisper']),pack['support'],return_components=True)
        return MatrixModel.forward(self,bundle,pos_weights,generator,av_callback)
