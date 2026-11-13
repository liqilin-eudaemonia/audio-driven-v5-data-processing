"""Remove exactly one branch; adapt every native attention/FFN projection."""
from p2_current_component_ablation.model import Model as Original
from .lora import extend
from . import ARMS
class Model(Original):
 def __init__(self,c,cfg,renderer):
  if c.get('component_arm') not in ARMS:raise ValueError('Expected no_feature or no_whisper')
  super().__init__(c,cfg,renderer);extend(self)
 def groups(self):
  groups=super().groups()
  for group in groups:
   if group['name']=='text_lora':group['name']='ltx_lora'
  return groups
