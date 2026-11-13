"""Existing AV/Whisper/Feature forward with expanded native LTX adapters."""
from p2_current_component_ablation.model import Model as Original
from p2_ltx_all_lora.lora import extend
class Model(Original):
 def __init__(self,c,cfg,renderer):
  if c.get('component_arm')!='no_event' or c['event_head_training'] or c['loss_weights']['event']!=0:
   raise ValueError('This independent arm has no Event supervision')
  super().__init__(c,cfg,renderer);extend(self)
 def groups(self):
  groups=super().groups()
  for group in groups:
   if group['name']=='text_lora':group['name']='ltx_lora'
  return groups
