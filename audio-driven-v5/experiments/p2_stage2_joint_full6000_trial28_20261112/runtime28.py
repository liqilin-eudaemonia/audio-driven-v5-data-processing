"""Sealed28-rank execution adapter; native joint losses and model remain unchanged."""
from pathlib import Path
import sys,copy,inspect
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,digest,atomic_json
import p2_stage2_joint_v2.config as C
ORIGINAL_VALIDATE=C.validate;ORIGINAL_SOURCE=C.source_identity
RUN=W/'run'
def validate(c,verify_payloads=False):
 if c['global_batch']!=28:raise ValueError('Require approved batch28')
 v=copy.deepcopy(c);v['global_batch']=31;ORIGINAL_VALIDATE(v,verify_payloads=verify_payloads)
 return c
def source():
 v=ORIGINAL_SOURCE()
 for p in [W/'runtime28.py',W/'train_worker.py']:v[str(p)]=sha(p)
 return v
class Schedule:
 def __init__(self,c):
  from p2_current_av_feature.schedule import Schedule as Parent
  old=Parent(c['generator']);self.size=old.size;self.seed=old.seed;self.batch=28
  self.pairs=old.pairs[:c['steps']*28]
 def indices(self,size,step,batch,seed,rank=0,world=1):
  if(size,batch,seed)!=(self.size,28,self.seed)or world not in(1,28)or not 0<=rank<world or not 1<=step<=300:raise ValueError('Invalid joint28 schedule')
  return self.pairs[(step-1)*28:step*28][rank::world]
def activate():
 C.validate=validate;C.source_identity=source
 import p2_stage2_joint_v2.training as T
 T.validate=validate;T.source_identity=source;T.Schedule=Schedule
 if not getattr(T,'_batch28',False):
  code=inspect.getsource(T.train).replace("if 31%world:","if c['global_batch']%world:").replace("divide global batch31","divide global batch28").replace("step,31,c['generator']","step,c['global_batch'],c['generator']")
  ns=dict(T.__dict__);exec(compile(code,str(W/'runtime28.py'),'exec'),ns);T.train=ns['train'];T._batch28=True
 return T
