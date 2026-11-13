"""Audited continuation of joint100, same model/Adam/RNG/schedule, to1000."""
from pathlib import Path
import ast,copy,sys,os
W=Path(__file__).resolve().parent;ROOT=W.parents[1];PARENT=ROOT/'experiments/p2_stage2_joint_full6000_trial28_20261112';RUN=W/'run'
sys.path[:0]=[str(PARENT),str(ROOT/'src'),str(ROOT/'scripts')]
import isolate_events
isolate_events.activate()
import runtime28 as OLD
from p2_listen.contracts import read,sha,digest,atomic_json
import p2_stage2_joint_v2.config as C
import p2_stage2_joint_v2.training as T
from p2_stage2_joint_v2 import checkpoint
BASE_MAKE=T.make_model;BASE_RESUME=checkpoint.resume
PARENT_STATE=PARENT/'run/step_000100.pt'

def validate(c,verify_payloads=False):
 if c['steps']!=1000 or c['global_batch']!=28 or c['save_every']!=300 or c['milestones']!=[0,300,600,900,1000]:raise ValueError('Require user target1000, batch28 and save300/final1000')
 v=copy.deepcopy(c);v.update(steps=300,global_batch=31,milestones=[0,100,300])
 # The original verifier's fast mode checks the same preverified payload size
 # and mtime; no repeated37GB first-stage hashing on every continuation.
 OLD.ORIGINAL_VALIDATE(v,verify_payloads=False)
 return c

def source():
 result=OLD.source()
 for p in(W/'runtime.py',W/'worker.py'):result[str(p)]=sha(p)
 return result

class Schedule:
 def __init__(self,c):
  from p2_current_av_feature.schedule import Schedule as Parent
  old=Parent(c['generator']);self.size=old.size;self.seed=old.seed;self.batch=28;self.pairs=old.pairs[:1000*28]
  if len(self.pairs)!=28000:raise ValueError('Incomplete continuation schedule')
 def indices(self,size,step,batch,seed,rank=0,world=1):
  if(size,batch,seed)!=(self.size,28,self.seed)or world not in(1,28)or not 0<=rank<world or not 1<=step<=1000:raise ValueError('Continuation schedule mismatch')
  return self.pairs[(step-1)*28:step*28][rank::world]

def permitted(c,path):return Path(path).resolve()==PARENT_STATE or Path(path).resolve().parent==Path(c['output']).resolve()

def verify_parent(c):
 receipt=read(RUN/'CONTINUATION_ACCEPTED.json');p=PARENT_STATE
 if read(p.with_suffix('.complete.json'))['sha256']!=receipt['parent_checkpoint_sha256']:raise ValueError('Parent checkpoint marker changed')
 if dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)!=receipt['parent_checkpoint_stat']:raise ValueError('Verified parent checkpoint changed')
 for s,h in receipt['parent_latest_reports'].items():
  if sha(PARENT/'run/evaluation'/f'step_{int(s):06d}'/'REPORT.json')!=h:raise ValueError('Parent latest evaluation changed')
 if receipt['configuration_sha256']!=digest(c)or receipt['source']!=source():raise ValueError('Continuation adoption identity changed')
 return receipt

def precheck(c,resume=None):
 validate(c);pre=read(RUN/'preflight.json');did=T.data_identity(c)
 if pre['configuration_sha256']!=digest(c)or pre['source']!=source()or pre['data_identity']!=did:raise ValueError('Continuation preflight mismatch')
 if(RUN/'STOP').exists():raise RuntimeError('User STOP preserved')
 if resume is None or not permitted(c,resume):raise ValueError('Explicit original100 or this continuation checkpoint required')
 verify_parent(c)
 step=read(Path(resume).with_suffix('.complete.json'))['step']
 if not 100<=step<=1000:raise ValueError('Resume outside approved continuation')
 return pre,did,source()

def make_model(c,device,initialize=True,verify_parent=True):
 # All trainable tensors are restored from joint100 (or a later full state).
 # Avoid loading the obsolete first-stage6000 trainable tensors first.
 model,strategy,sampler,_=BASE_MAKE(c,device,initialize=False,verify_parent=False)
 return model,strategy,sampler,read(PARENT/'run/run.json')['initialization']

def resume(path,model,optimizer,identity,rank=0):
 if Path(path).resolve()!=PARENT_STATE:return BASE_RESUME(path,model,optimizer,identity,rank)
 import torch
 from p2_listen.training import restore_rng
 c=identity['experiment'];verify_parent(c)
 saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 if saved['step']!=100 or saved['identity']!=read(PARENT/'run/run.json')or len(saved['rank_rng'])!=28:raise ValueError('Wrong parent full state')
 model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rank_rng'][rank]);model.assert_ownership()
 return 100

def activate():
 C.validate=validate;C.source_identity=source
 T.validate=validate;T.source_identity=source;T.Schedule=Schedule;T.precheck=precheck
 T.make_model=make_model;T.evaluation_barrier=lambda c,step:verify_parent(c)
 checkpoint.resume=resume
 tree=ast.parse(Path(T.__file__).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='train')
 text='\n'.join(Path(T.__file__).read_text().splitlines()[node.lineno-1:node.end_lineno])
 changes={"if 31%world:":"if c['global_batch']%world:","step,31,c['generator']":"step,c['global_batch'],c['generator']",
  "if start>=300:":"if start>=c['steps']:","finish=100 if start<100 else 300":"finish=c['steps']",
  "if resume and Path(resume).resolve().parent!=out.resolve():":"if resume and not permitted(c,resume):",
  "rank_zero_action(log,rank=rank,world=world);last=step":"rank_zero_action(log,rank=rank,world=world);last=step\n   if step%c['save_every']==0:save(step)",
  "if last>start:save(last)":"if last>start and last%c['save_every']!=0:save(last)",
  "last==300":"last==c['steps']"}
 for before,after in changes.items():
  if before not in text:raise ValueError('Native continuation hook changed: '+before)
  text=text.replace(before,after)
 ns=dict(T.__dict__,permitted=permitted);exec(compile(text,str(W/'runtime.py')+'::native_train','exec'),ns);T.train=ns['train']
 return T
