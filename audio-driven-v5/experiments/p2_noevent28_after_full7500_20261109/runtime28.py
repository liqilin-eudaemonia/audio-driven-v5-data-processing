"""Explicit batch28 continuation preserving the entire original pair sequence."""
from pathlib import Path
import sys,copy,math,importlib.util
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=W.parent/'p2_ltx_split24_8_batch31_20261109'
sys.path[:0]=[str(S),str(ROOT/'src'),str(ROOT/'scripts')]
import runtime as parent
from p2_listen.contracts import read,sha

def normalized(c):
 v=copy.deepcopy(c);p=v.pop('batch28_continuation');v['training']['global_batch']=31;v['training']['steps']=p['original_steps'];return v

def validate(c):
 from p2_ltx_all_lora_noevent.config import validate as native
 p=read(W/'PLAN.json');assert c['batch28_continuation']==p
 v=normalized(c);native(v)
 assert c['training']['global_batch']==28 and c['training']['steps']==p['new_end_step']
 assert p['prefix_pairs']==p['start_step']*31 and p['new_end_step']==p['start_step']+math.ceil((p['total_pairs']-p['prefix_pairs'])/28)
 assert sha(p['parent_config'])==p['parent_config_sha256']
 parent=read(p['parent_config']);parent['output']=v['output'];assert parent==v
 return c

def source():
 result=parent.source_identity('noevent')
 for n in ['runtime28.py','worker.py','training.py','canary.py','PLAN.json']:result[str(W/n)]=sha(W/n)
 return result

class Schedule:
 def __init__(self,c):
  from p2_ltx_all_lora_noevent.schedule import Schedule as Native
  validate(c);base=Native(normalized(c));self.pairs=base.pairs;self.size=base.size;self.seed=base.seed;self.plan=c['batch28_continuation'];self.batch=28
  assert len(self.pairs)==self.plan['total_pairs']
 def bounds(self,step):
  p=self.plan
  if not p['start_step']<step<=p['new_end_step']:raise ValueError('Outside continuation')
  begin=p['prefix_pairs']+(step-p['start_step']-1)*28;return begin,min(begin+28,len(self.pairs))
 def count(self,step):
  a,b=self.bounds(step);return b-a
 def indices(self,size,step,batch,seed,rank=0,world=1):
  assert(size,batch,seed)==(self.size,28,self.seed)and world in (1,28)and 0<=rank<world
  a,b=self.bounds(step);return self.pairs[a:b][rank::world]

def model(c,cfg,renderer):
 from p2_ltx_all_lora_noevent.model import Model
 m=Model(normalized(c),cfg,renderer);m.spec=c;return m

def load():
 spec=importlib.util.spec_from_file_location('p2_ltx_all_lora_noevent._batch28_training',W/'training.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.validate=validate;m.source_identity=source;m.Schedule=Schedule;m.QwenWhisperEventFeature=model
 from p2_ltx_all_lora_noevent.data import Dataset,data_identity
 m.Dataset=lambda c:Dataset(normalized(c));m.data_identity=lambda c:data_identity(normalized(c))
 spec=importlib.util.spec_from_file_location('batch28_canary',W/'canary.py');check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)
 m.split_canary=lambda **kw:check.longest_canary(training=m,arm='noevent',**kw)
 return m
