"""Finite real-model benchmark, production optimizer/loss on admitted debug caches."""
import argparse,copy,json,os,sys,time,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
import torch.distributed as dist
from p2_listen.numerics import configure_training_numerics
from p2_listen.config import load,extractor_config
from p2_listen.contracts import atomic_json,EVENTS
from p2_listen.dataset import load_observations,load_targets
from p2_listen.training_data import load_training_sample
from p2_listen.features import load_text
from p2_listen.annotations import class_statistics,mask_classes
from p2_listen.model import build_model
from p2_listen.runtime import av_loss
from p2_listen.training import optimization_step,indices
from p2_listen import training as train_module

class BucketAccumulator:
    def __init__(self,parameters):
        self.parameters=[p for p in parameters if p.requires_grad]
        self.sizes=[p.numel() for p in self.parameters]
        self.buffer=torch.zeros(sum(self.sizes),dtype=torch.float64,device=self.parameters[0].device)
        self.views=list(self.buffer.split(self.sizes));self.present=[False]*len(self.parameters)
    def add(self):
        dst=[];src=[]
        for i,p in enumerate(self.parameters):
            if p.grad is not None:
                dst.append(self.views[i]);src.append(p.grad.detach().reshape(-1));self.present[i]=True
        if dst:torch._foreach_add_(dst,src)
        for p in self.parameters:p.grad=None
    def finish(self,world=1):
        present=torch.tensor(self.present,dtype=torch.int32,device=self.buffer.device)
        if world>1:
            dist.all_reduce(present,op=dist.ReduceOp.MAX)
            # Bounded buckets, no division: objectives already normalize globally.
            for chunk in self.buffer.split(4*1024*1024):dist.all_reduce(chunk,op=dist.ReduceOp.SUM)
        rounded=self.buffer.to(self.parameters[0].dtype)
        for p,used,view in zip(self.parameters,present.tolist(),rounded.split(self.sizes)):
            p.grad=view.view_as(p) if used else None

p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--batch',type=int,default=24);p.add_argument('--steps',type=int,default=6)
p.add_argument('--profiles',default='baseline,bucket,no_checkpoint');p.add_argument('--sample',default='mixed',choices=['mixed','longest'])
p.add_argument('--real-long-cache',type=Path)
a=p.parse_args();numerics=configure_training_numerics();torch.set_num_threads(2)
rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
assert a.batch%world==0
torch.cuda.set_device(local);device=f'cuda:{local}'
if world>1:dist.init_process_group('nccl')
c=load(a.config);c['training']['global_batch']=a.batch;cfg=extractor_config(c)
from omni_av_p2.ltx import load_renderer,append_conditions
from omni_av.ltx_runtime import with_text
out=a.output;out.mkdir(parents=True,exist_ok=True)
rows=load_observations(c,'train');targets=load_targets(c,rows)
print('loading_samples',rank,flush=True)
if a.real_long_cache:
 from p2_listen.cache import load_part
 from p2_listen.contracts import input_identity
 from omni_av_p2.connector import ConditionInputs
 obs=json.loads((Path(__file__).parent/'long_observation.json').read_text())
 rows=[obs];payloads={};provenance={}
 for part in ['input','vae','events']:
  folder=a.real_long_cache/'cache'/part/obs['sample_id']
  marker=json.loads((folder/'complete.json').read_text())
  assert marker['contract']['input']==input_identity(obs)
  payloads[part]=load_part(folder,marker['contract']);provenance[part]=marker
 x=ConditionInputs(**payloads['input']['condition']);x.validate()
 cpu_samples=[(x,payloads['input']['layout'],None,payloads['vae'],payloads['events'])]
 atomic_json(out/f'long_cache_provenance_rank{rank}.json',dict(observation=obs,cache_contracts=provenance,performance_only=True))
else:
 cpu_samples=[load_training_sample(c,r,targets[r['sample_id']]) for r in rows]
stats=class_statistics([s[4] for s in cpu_samples],c['events']['pos_weight_cap'],c['events']['K_train'])
effective=[k for k in EVENTS if stats[k]['enabled']];weights={k:stats[k]['pos_weight'] for k in EVENTS}
for i,s in enumerate(cpu_samples):cpu_samples[i]=(*s[:4],mask_classes(s[4],effective))
longest=max(range(len(rows)),key=lambda i:rows[i]['requested_duration_s'])
print('loading_renderer',rank,flush=True);t=time.monotonic()
renderer,strategy,sampler=load_renderer(cfg,device)
torch.manual_seed(20261026);raw=build_model(c,renderer).to(device);raw.train()
initial={k:v.detach().clone() for k,v in raw.state_dict().items()}
text=load_text(c,device)['positive'];renderer_load_seconds=time.monotonic()-t
model=raw if world==1 else torch.nn.parallel.DistributedDataParallel(raw,device_ids=[local],broadcast_buffers=False)
base_accumulator=train_module.GradientAccumulator
profiles=[]
import types
original_process=renderer._process_transformer_blocks
def selective_process(self,video,audio,perturbations):
 for idx,block in enumerate(self.transformer_blocks):
  selected=(idx%2==0) if self.benchmark_fraction=='half' else (idx%4!=3)
  if selected and self.training:
   video,audio=torch.utils.checkpoint.checkpoint(block,video,audio,perturbations,use_reentrant=False)
  else:
   video,audio=block(video=video,audio=audio,perturbations=perturbations)
 return video,audio
def move(s):
 from p2_listen.cache import tree_to
 from omni_av_p2.connector import ConditionInputs
 x=ConditionInputs(**{k:v.to(device) for k,v in vars(s[0]).items()})
 return x,tree_to(s[1],device),None,tree_to(s[3],device),tree_to(s[4],device)
def callback(u,pack):
 prepared=strategy.prepare_training_inputs(with_text(pack['av'],text),sampler)
 return av_loss(renderer,append_conditions(prepared,u),pack['support'])
try:
 for profile in a.profiles.split(','):
  raw.load_state_dict(initial);renderer.set_gradient_checkpointing(profile!='no_checkpoint')
  renderer.benchmark_fraction=profile
  renderer._process_transformer_blocks=types.MethodType(selective_process,renderer) if profile in ('half','threequarters') else original_process
  train_module.GradientAccumulator=base_accumulator if profile=='baseline' else BucketAccumulator
  opt=torch.optim.AdamW(raw.optimizer_groups(c['training']),weight_decay=0)
  records=[];torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  for step in range(1,a.steps+1):
   chosen=indices(len(rows),step,a.batch,20261026,rank,world)
   if a.sample=='longest':chosen=[(longest,n) for i,n in chosen]
   torch.cuda.synchronize()
   if world>1:dist.barrier()
   begin=time.monotonic();t=begin
   samples=[move(cpu_samples[i]) for i,n in chosen]
   torch.cuda.synchronize();input_seconds=time.monotonic()-t
   r=optimization_step(model,opt,samples,callback,mode='joint_event',pos_weights=weights,event_weight=c['events']['weight'],
                       seed=20261026,ordinals=[n for i,n in chosen],config=c['training'],world=world)
   torch.cuda.synchronize();elapsed=time.monotonic()-begin
   measurement=torch.tensor([elapsed,sum(rows[i]['requested_duration_s'] for i,n in chosen)],device=device,dtype=torch.float64)
   if world>1:
    dist.all_reduce(measurement[:1],op=dist.ReduceOp.MAX);dist.all_reduce(measurement[1:],op=dist.ReduceOp.SUM)
   record=dict(profile=profile,step=step,seconds=float(measurement[0]),local_seconds=elapsed,input_seconds=input_seconds,
               samples=a.batch,samples_per_second=a.batch/float(measurement[0]),media_seconds=float(measurement[1]),
               media_seconds_per_second=float(measurement[1]/measurement[0]),
               peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
               av=r['av'],event=r['event'],grad_norm=r['grad_norm'])
   records.append(record)
   with (out/f'rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   if rank==0:print(record,flush=True)
   del samples
  # Same initial state, sample order, seeds, losses and steps across profiles.
  if profile=='baseline':reference={k:v.detach().clone() for k,v in raw.state_dict().items()}
  max_error=max((float((v-reference[k]).abs().max()) for k,v in raw.state_dict().items()),default=0.) if 'reference' in locals() else None
  result=dict(profile=profile,records=records,parameter_error_vs_baseline=max_error)
  profiles.append(result);atomic_json(out/f'profile_{profile}_rank{rank}.json',result)
  del opt
 atomic_json(out/f'summary_rank{rank}.json',dict(host=socket.gethostname(),rank=rank,world=world,batch=a.batch,steps=a.steps,
  sample=a.sample,renderer_load_seconds=renderer_load_seconds,profiles=profiles,numerics=numerics))
finally:
 train_module.GradientAccumulator=base_accumulator
 if world>1:dist.destroy_process_group()
