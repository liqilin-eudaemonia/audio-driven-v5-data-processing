"""Native AV+Feature forward: shared U, frozen Qwen, no event head execution."""
from p2_matrix16.model import MatrixModel

class Model(MatrixModel):
 def assert_ownership(self):
  super().assert_ownership()
  ids=lambda m:{id(p)for p in m.parameters()if p.requires_grad}
  if (ids(self)!=(ids(self.connector)|ids(self.text_lora)|ids(self.decoders))
      or ids(self.event_heads) or set(self.decoders)!={'video','audio'} or len(self.qwen_lora)):
   raise ValueError('Only Connector, text LoRA and feature decoders may train')
  if hasattr(self,'renderer') and any(hasattr(b,n)for b in self.renderer.transformer_blocks for n in ('qwen_attn2','audio_qwen_attn2')):
   raise ValueError('Whisper conditioning must be absent')

 def groups(self):
  t=self.spec['training']
  groups=[dict(name=n,params=list(m.parameters()),lr=lr)for n,m,lr in
   [('connector',self.connector,t['connector_lr']),('text_lora',self.text_lora,self.spec['text_lora']['learning_rate']),
    ('features',self.decoders,t['feature_lr'])]]
  ps=[p for g in groups for p in g['params']]
  if len(ps)!=len({id(p)for p in ps})or {id(p)for p in ps}!={id(p)for p in self.parameters()if p.requires_grad}:
   raise ValueError('Invalid Feature optimizer ownership')
  return groups

 def forward(self,bundle,pos_weights,generator,av_callback):
  if bundle.get('teacher') is None or any(k in bundle for k in ('events','whisper','hidden_event_layout')):
   raise ValueError('Require teacher only, no Event or Whisper')
  return super().forward(dict(bundle,events={}),{},generator,av_callback)

def av_objective(values,c,counts,micro_count,world):
 if set(values['features'])!={'video','audio'} or values['event_sums']:
  raise ValueError('Require two Feature losses and no Event')
 total=values['av']+sum(c['loss_weights'][m+'_feature']*v for m,v in values['features'].items())
 return total/(micro_count*world),total,total.new_zeros(())
