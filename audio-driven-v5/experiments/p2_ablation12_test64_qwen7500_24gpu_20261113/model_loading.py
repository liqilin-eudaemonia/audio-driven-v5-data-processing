"""Reuse each trained component's own native loader and exact state dictionary."""
import copy,importlib,time
from pathlib import Path
from unittest.mock import patch
import run as a

def load_native(combo,record,device):
 import torch
 component=combo['component'];rec=next(r for r in a.read(a.WORK/'CHECKPOINTS.json') if r['key']==combo['key'])
 verified=a.read(a.WORK/'integrity'/f'{combo["key"]}.json')
 assert verified['sha256']==rec['sha256'] and a.base.stamp(rec['path'])==rec['stamp']
 v=torch.load(rec['path'],map_location='cpu',weights_only=True,mmap=True)
 info=a.read(a.WORK/'PLAN.json')['identities'][component];identity=info['identity']
 assert v['identity']==identity and v['step']==combo['ltx'] and v['schema']==rec['schema']
 assert all(torch.isfinite(t).all() for t in v['model'].values())
 package='p2_ltx_all_lora_noevent' if component=='no_event' else 'p2_ltx_all_lora_components'
 inference=importlib.import_module(package+'.inference');checkpoint=importlib.import_module(package+'.checkpoint');conf=importlib.import_module(package+'.config')
 checked_validate=conf.validate
 def normalize(c):
  assert c==identity['experiment']
  return checked_validate(copy.deepcopy(info['normalized_config']))
 # The batch28 continuation modifies schedule metadata only. Use the same normalization
 # as training, after verifying the complete frozen training source in prepare().
 assert all(a.sha(k)==h for k,h in identity['source'].items())
 with patch.object(checkpoint,'checked',lambda _:v),patch.object(conf,'source_identity',lambda:identity['source']),patch.object(conf,'validate',normalize):
  result=inference.load_model(rec['path'],device)
 model=result[0];assert len(model.text_targets)==1632 and list(model.state_dict())==list(v['model'])
 flags=model._components;assert flags==a.read(a.child(combo['key'])/'protocol.json')['component_flags']
 assert hasattr(model,'whisper_adapter')==(component!='no_whisper')
 assert bool(model.event_heads)==(component!='no_event')
 assert set(model.decoders)==(set() if component=='no_feature' else {'video','audio'})
 a.save(a.WORK/'model_acceptance'/f'{combo["key"]}_{record["rank"]}.json',dict(success=True,**record,loaded_at=time.time(),strict_tensors=len(v['model']),targets=1632,original_sha256=rec['sha256'],component=component,flags=flags,whisper_present=hasattr(model,'whisper_adapter')))
 return result
