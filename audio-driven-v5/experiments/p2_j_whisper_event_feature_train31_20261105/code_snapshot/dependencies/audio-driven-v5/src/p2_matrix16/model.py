"""Shared initialization and explicit loss/adapter ownership for all sixteen arms."""
from types import MethodType,SimpleNamespace
import torch
from torch import nn
from .config import feature_enabled,event_enabled,connector_config


def differentiable_memory(self,data):
    if not torch.equal(data.layer_ids,self.layer_ids):raise ValueError('Layer IDs differ')
    values=torch.stack([self.projection(norm(data.hidden[:,i])) for i,norm in enumerate(self.norms)],dim=1)
    if self.mode=='single':return values[:,0],data.hidden_valid
    if self.mode=='weighted_sum':return (values*self.layer_logits.softmax(0)[None,:,None,None]).sum(1),data.hidden_valid
    values=values+self.layer_embedding[None,:,None,:].to(values)
    b,s,n,d=values.shape
    return values.reshape(b,s*n,d),data.hidden_valid[:,None].expand(b,s,n).reshape(b,s*n)


class MatrixModel(nn.Module):
    def __init__(self,c,cfg,renderer=None,qwen_runtime=None,processor=None):
        super().__init__()
        from omni_av_p2.connector import FusionConnector
        from p2_listen.events import EventHeads
        self.spec=c
        self.connector=FusionConnector(connector_config(c,cfg))
        self.connector.memory.forward=MethodType(differentiable_memory,self.connector.memory)
        if 'event_supervision' in c:
            from p2_clip_events.model import ClipEventHeads
            self.event_heads=ClipEventHeads(cfg.video_dim,cfg.audio_dim,c['event_supervision'])
        else:
            self.event_heads=EventHeads(cfg.video_dim,cfg.audio_dim,128)
        if c.get('event_head_training') is False:self.event_heads.requires_grad_(False)
        self.text_lora=nn.ModuleList();self.qwen_lora=nn.ModuleList();self.decoders=nn.ModuleDict()
        self.text_targets=[];self.qwen_targets=[]
        if renderer is not None:
            from omni_av_p21.lora import inject_text_lora
            from omni_av_p21.config import LoRAConfig
            self.text_lora,self.text_targets=inject_text_lora(renderer,LoRAConfig(**{k:v for k,v in c['text_lora'].items() if k!='enabled'}))
            object.__setattr__(self,'renderer',renderer)
        if feature_enabled(c):
            from omni_av_p2.decoder import EmbeddingDecoder
            with torch.random.fork_rng(devices=[]):
                for mod,dim in [('video',cfg.video_dim),('audio',cfg.audio_dim)]:
                    self.decoders[mod]=EmbeddingDecoder(cfg.target_dim,dim,cfg.decoder_width,cfg.decoder_depth,cfg.diffusion_batch_mul)
        if c['stage']==2:
            if qwen_runtime is None:raise ValueError('Stage2 requires live Qwen, never frozen hidden')
            from .qwen import inject_qwen_lora
            self.qwen_lora,self.qwen_targets=inject_qwen_lora(qwen_runtime,c['qwen_lora'])
            object.__setattr__(self,'qwen_runtime',qwen_runtime)
            object.__setattr__(self,'processor',processor)
        self.assert_ownership()

    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'renderer'):self.renderer.train(mode)
        if hasattr(self,'qwen_runtime'):
            # Activate non-reentrant checkpointing in the text transformer only.
            self.qwen_runtime.thinker.eval()
            self.qwen_runtime.thinker.model.train(mode)
        return self

    def assert_ownership(self):
        params=list(self.parameters())
        if len({id(p) for p in params})!=len(params):raise ValueError('Duplicate parameter ownership')
        for owner,adapters in [(getattr(self,'renderer',None),self.text_lora),
                               (getattr(getattr(self,'qwen_runtime',None),'thinker',None),self.qwen_lora)]:
            if owner is not None and {id(p) for p in owner.parameters() if p.requires_grad}!={id(p) for p in adapters.parameters()}:
                raise ValueError('Frozen backbone or unowned adapter parameter')

    def groups(self):
        t=self.spec['training'];groups=[dict(name='connector',params=list(self.connector.parameters()),lr=t['connector_lr']),
                                      dict(name='events',params=list(self.event_heads.parameters()),lr=t['event_lr'])]
        for name,module,lr in [('features',self.decoders,t['feature_lr']),('text_lora',self.text_lora,self.spec['text_lora']['learning_rate']),('qwen_lora',self.qwen_lora,self.spec['qwen_lora']['learning_rate'])]:
            if len(module):groups.append(dict(name=name,params=list(module.parameters()),lr=lr))
        if {id(p) for g in groups for p in g['params']}!={id(p) for p in self.parameters()}:raise ValueError('Optimizer parameters incomplete')
        return groups

    def forward(self,bundle,pos_weights,generator,av_callback):
        from p2_listen.events import event_sums
        if 'event_supervision' in self.spec:
            from p2_clip_events.model import event_sums
        x=bundle.get('condition')
        if self.spec['stage']==2:
            if x is not None:raise ValueError('Stage2 forbids cached hidden')
            from .qwen import online_condition
            x=online_condition(self.qwen_runtime,self.processor,bundle['observation'],bundle['plan'])
            for mod in ('video','audio'):
                for key in ('_coordinates','_valid'):
                    if not torch.equal(getattr(x,mod+key),bundle['layout'][mod+key]):
                        raise ValueError('Online Qwen/event layout differs')
        elif x is None:raise ValueError('Stage1 requires frozen conditions')
        u=self.connector(x)
        mode='joint_event' if event_enabled(self.spec) else 'probe_only'
        if self.spec.get('event_head_training') is False:
            zero=(u.video.sum()+u.audio.sum())*0
            sums={k:zero for k in bundle['events']}
            counts={k:zero.detach() for k in bundle['events']}
        else:
            logits=self.event_heads(u,bundle['layout'],mode)
            sums,counts=event_sums(logits,bundle['events'],pos_weights)
        teacher=bundle.get('teacher');losses={}
        if feature_enabled(self.spec):
            if teacher is None:raise ValueError('New postprocessed teacher required')
            for mod in ('video','audio'):
                mask=teacher[mod+'_valid'] & getattr(u,mod+'_valid')
                losses[mod]=self.decoders[mod].loss(teacher[mod],getattr(u,mod),mask,generator) if mask.any() else sum(p.sum()*0 for p in self.decoders[mod].parameters())+getattr(u,mod).sum()*0
        elif teacher is not None:raise ValueError('Disabled Feature branch must not load teacher')
        av,components=av_callback(u,bundle['av'],bundle['text']['positive'])
        return dict(av=av,av_components=components,features=losses,event_sums=sums,event_counts=counts,u=u)
