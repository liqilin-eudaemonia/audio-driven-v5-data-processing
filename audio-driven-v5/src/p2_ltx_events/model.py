"""Shared U/Feature targets, with Event gradients through the AV backbone."""
import torch
from p2_whisper_event_feature.model import QwenWhisperEventFeature
from .config import validate
from .heads import HiddenEventHeads,event_sums
from .layout import check_prepared
from .renderer import forward_with_hidden


class LTXHiddenEvents(QwenWhisperEventFeature):
    def __init__(self,c,cfg,renderer):
        validate(c)
        super().__init__(c,cfg,renderer)
        with torch.random.fork_rng(devices=[]):
            self.event_heads=HiddenEventHeads(renderer.inner_dim,renderer.audio_inner_dim,c)
        self.event_heads.to(next(renderer.parameters()).device)
        self.assert_ownership()

    def forward(self,bundle,pos_weights,generator,strategy,sampler):
        from omni_av.ltx_runtime import with_text
        from p2_listen.runtime import masked_mse
        if any(bundle.get(k) is None for k in ('teacher','whisper','events','hidden_event_layout','condition')):
            raise ValueError('Missing frozen cache or hidden Event target/layout')
        u=self.connector(bundle['condition'])
        features={}
        for mod in ('video','audio'):
            teacher=bundle['teacher'];mask=teacher[mod+'_valid'] & getattr(u,mod+'_valid')
            features[mod]=(self.decoders[mod].loss(teacher[mod],getattr(u,mod),mask,generator) if mask.any()
                else sum(p.sum()*0 for p in self.decoders[mod].parameters())+getattr(u,mod).sum()*0)
        pack=bundle['av']
        prepared=strategy.prepare_training_inputs(with_text(pack['av'],bundle['text']['positive']),sampler)
        prepared=self.prepare(prepared,u,bundle['whisper'])
        check_prepared(bundle['hidden_event_layout'],prepared)
        (pv,pa),hidden=forward_with_hidden(self.renderer,prepared.video,prepared.audio)
        masks={mod:getattr(prepared,mod+'_loss_mask') & pack['support'][mod] for mod in ('video','audio')}
        vl=masked_mse(pv,prepared.video_targets,masks['video'])
        al=masked_mse(pa,prepared.audio_targets,masks['audio'])
        logits=self.event_heads(hidden,bundle['hidden_event_layout'],masks)
        sums,counts=event_sums(logits,bundle['events'],pos_weights)
        return dict(av=vl+al,av_components={'video':vl.detach(),'audio':al.detach()},features=features,
            event_sums=sums,event_counts=counts,u=u)
