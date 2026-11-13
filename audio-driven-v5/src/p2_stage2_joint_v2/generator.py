"""Existing native components, with disabled branches absent from the optimizer."""
import torch
from torch import nn
from p2_matrix16.model import MatrixModel
from p2_whisper_av.model import QwenWhisperAV
from p2_ltx_events.heads import HiddenEventHeads,event_sums,objective
from p2_ltx_events.renderer import forward_with_hidden
from p2_ltx_events.layout import check_prepared
from .config import flags,native_whisper_view

class Model(QwenWhisperAV):
 def _initialize_native(self,c,cfg,renderer):
  self._building=True;self._components=flags(c)
  if self._components['whisper']:
   QwenWhisperAV.__init__(self,native_whisper_view(c),cfg,renderer)
   if self._components['feature']:
    from omni_av_p2.decoder import EmbeddingDecoder
    with torch.random.fork_rng(devices=[]):
     for mod,dim in [('video',cfg.video_dim),('audio',cfg.audio_dim)]:
      self.decoders[mod]=EmbeddingDecoder(cfg.target_dim,dim,cfg.decoder_width,cfg.decoder_depth,cfg.diffusion_batch_mul)
    self.decoders.to(next(renderer.parameters()).device)
  else:MatrixModel.__init__(self,c,cfg,renderer)
  self.spec=c
  if self._components['event']:
   with torch.random.fork_rng(devices=[]):self.event_heads=HiddenEventHeads(renderer.inner_dim,renderer.audio_inner_dim,c)
   self.event_heads.to(next(renderer.parameters()).device)
  else:self.event_heads=nn.ModuleDict()
  self._building=False;self.assert_ownership()

 def __init__(self,c,cfg,renderer):
  self._initialize_native(c,cfg,renderer)
  if c.get('ltx_lora_scope') is not None:
   from p2_ltx_all_lora_components.lora import extend
   extend(self)

 def assert_ownership(self):
  if self._building:
   return QwenWhisperAV.assert_ownership(self)if self._components['whisper']else MatrixModel.assert_ownership(self)
  ids=lambda m:{id(p)for p in m.parameters()if p.requires_grad}
  f=self._components;owned=ids(self.connector)|ids(self.text_lora)|ids(self.event_heads)|ids(self.decoders)
  renderer_owned=ids(self.text_lora)
  if f['whisper']:
   owned|=ids(self.whisper_adapter)|ids(self.whisper_branches);renderer_owned|=ids(self.whisper_branches)
  elif hasattr(self,'whisper_adapter')or any(hasattr(b,n)for b in self.renderer.transformer_blocks for n in ('qwen_attn2','audio_qwen_attn2')):raise ValueError('Disabled Whisper exists')
  if ids(self)!=owned or ids(self.renderer)!=renderer_owned or len(self.qwen_lora):raise ValueError('Frozen/duplicate/unowned parameters')
  if set(self.decoders)!=({'video','audio'}if f['feature']else set())or bool(ids(self.event_heads))!=f['event']:raise ValueError('Disabled auxiliary parameters exist')

 def groups(self):
  c=self.spec;t=c['training'];modules=[('connector',self.connector,t['connector_lr']),('text_lora',self.text_lora,c['text_lora']['learning_rate'])]
  if self._components['whisper']:modules.append(('whisper',nn.ModuleList([self.whisper_adapter,self.whisper_branches]),c['whisper']['learning_rate']))
  if self._components['feature']:modules.append(('features',self.decoders,t['feature_lr']))
  if self._components['event']:modules.append(('events',self.event_heads,t['event_lr']))
  groups=[dict(name=('ltx_lora'if n=='text_lora'and self.spec.get('ltx_lora_scope') is not None else n),params=list(m.parameters()),lr=lr)for n,m,lr in modules];ps=[p for g in groups for p in g['params']]
  if len(ps)!=len({id(p)for p in ps})or {id(p)for p in ps}!={id(p)for p in self.parameters()if p.requires_grad}:raise ValueError('Optimizer ownership mismatch')
  return groups

 def prepare(self,prepared,u,speaker=None):
  from omni_av_p2.ltx import append_conditions
  prepared=append_conditions(prepared,u)
  if self._components['whisper']:return self.condition(prepared,speaker)
  if speaker is not None:raise ValueError('Disabled Whisper supplied')
  return prepared

 def forward(self,bundle,pos_weights,generator,strategy,sampler):
  from omni_av.ltx_runtime import with_text
  from p2_listen.runtime import masked_mse,av_loss
  f=self._components
  if (bundle.get('teacher')is not None)!=f['feature']or(bundle.get('whisper')is not None)!=f['whisper']:raise ValueError('Wrong active inputs')
  if ('events'in bundle)!=f['event']or('hidden_event_layout'in bundle)!=f['event']:raise ValueError('Wrong Event inputs')
  u=self.connector(bundle['condition']);features={}
  if f['feature']:
   teacher=bundle['teacher']
   for mod in ('video','audio'):
    mask=teacher[mod+'_valid']&getattr(u,mod+'_valid')
    features[mod]=self.decoders[mod].loss(teacher[mod],getattr(u,mod),mask,generator)if mask.any()else sum(p.sum()*0 for p in self.decoders[mod].parameters())+getattr(u,mod).sum()*0
  pack=bundle['av'];prepared=strategy.prepare_training_inputs(with_text(pack['av'],bundle['text']['positive']),sampler)
  prepared=self.prepare(prepared,u,bundle.get('whisper'));sums={};counts={}
  if f['event']:
   check_prepared(bundle['hidden_event_layout'],prepared)
   (pv,pa),hidden=forward_with_hidden(self.renderer,prepared.video,prepared.audio)
   masks={m:getattr(prepared,m+'_loss_mask')&pack['support'][m]for m in ('video','audio')}
   vl=masked_mse(pv,prepared.video_targets,masks['video']);al=masked_mse(pa,prepared.audio_targets,masks['audio'])
   av=vl+al;components={'video':vl.detach(),'audio':al.detach()}
   sums,counts=event_sums(self.event_heads(hidden,bundle['hidden_event_layout'],masks),bundle['events'],pos_weights)
  else:av,components=av_loss(self.renderer,prepared,pack['support'],return_components=True)
  return dict(av=av,av_components=components,features=features,event_sums=sums,event_counts=counts,u=u)
