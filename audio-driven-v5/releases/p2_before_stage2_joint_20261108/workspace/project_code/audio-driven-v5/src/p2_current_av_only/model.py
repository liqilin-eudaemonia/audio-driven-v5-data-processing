"""Use the original AV-only forward and exact initialization order."""
from p2_j_av_only import AVOnlyModel,av_objective


class Model(AVOnlyModel):
 def assert_ownership(self):
  super().assert_ownership()
  ids=lambda m:{id(p)for p in m.parameters()if p.requires_grad}
  if ids(self)!=(ids(self.connector)|ids(self.text_lora)) or ids(self.event_heads) or len(self.decoders) or len(self.qwen_lora):
   raise ValueError('Only Connector and text LoRA may train')
  if hasattr(self,'renderer'):
   if any(hasattr(b,n)for b in getattr(self.renderer,'transformer_blocks',[])for n in ('qwen_attn2','audio_qwen_attn2')):
    raise ValueError('Whisper conditioning must be absent')

 def groups(self):
  groups=[dict(name='connector',params=list(self.connector.parameters()),lr=self.spec['training']['connector_lr']),
          dict(name='text_lora',params=list(self.text_lora.parameters()),lr=self.spec['text_lora']['learning_rate'])]
  parameters=[p for g in groups for p in g['params']]
  if len(parameters)!=len({id(p)for p in parameters}) or {id(p)for p in parameters}!={id(p)for p in self.parameters()if p.requires_grad}:
   raise ValueError('Missing/duplicate/frozen AV optimizer parameters')
  return groups
