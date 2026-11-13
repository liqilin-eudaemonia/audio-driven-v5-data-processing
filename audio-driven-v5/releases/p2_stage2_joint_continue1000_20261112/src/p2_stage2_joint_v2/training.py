"""Online joint trial with checkpoint0/100/300 and mandatory evaluation barriers."""
from pathlib import Path
import copy,json,os,time
import torch
import torch.distributed as dist
from p2_listen.contracts import read,sha,digest,atomic_json
from . import VERSION,checkpoint
from .config import validate,source_identity
from .data import Dataset,data_config,data_identity,to_device
from .model import JointModel,objective
from .schedule import Schedule


def make_model(c,device,initialize=True,verify_parent=True):
 from omni_av_p2.ltx import load_renderer
 from p2_listen.config import extractor_config
 from p2_listen.performance import configure_renderer
 from .generator import Model
 from .qwen import load_online
 from .initialization import initialize_generator
 cfg=extractor_config(data_config(c['generator']))
 renderer,strategy,sampler=load_renderer(cfg,device)
 configure_renderer(renderer,'all')
 torch.manual_seed(c['generator']['training']['seed'])
 gen=Model(c['generator'],cfg,renderer).to(device)
 initialized=initialize_generator(gen,c,verify_checksum=verify_parent)if initialize else None
 qwen,peft=load_online(cfg,Path(c['assets']['adapter_config']['path']).parent,device)
 model=JointModel(gen,qwen,c['qwen_lr'],c['generation_lr_scale']).to(device)
 return model,strategy,sampler,initialized


def preflight(c):
 from p2_ltx_events.data import statistics
 from p2_listen.features import weights_identity
 validate(c);dataset=Dataset(c);schedule=Schedule(c);g=c['generator']
 selected=read(c['calibration_manifest'])
 if sha(c['calibration_manifest'])!=c['calibration_manifest_sha256']:raise ValueError('Calibration selection changed')
 if len(selected['samples'])!=c['calibration']['samples']or len({r['sample_id']for r in selected['samples']})!=c['calibration']['samples']:raise ValueError('Duplicate/incomplete calibration membership')
 for r in selected['samples']:
  if dataset.ids[r['index']]!=r['sample_id']or r['sample_id']not in dataset.planning_ids or dataset.planning_hashes[r['sample_id']]!=r['caption_sha256']:raise ValueError('Calibration is not the approved current training subset')
 canaries=read(Path(g['j_cache'])/'protocol.json')['canary_ids']
 positions={sid:i for i,sid in enumerate(dataset.ids)};records=[]
 for sid in canaries:
  v=dataset.sample(positions[sid]);assert v['condition']is None
  if v['generation_prompt']!=v['text']['positive']['prompt']:raise ValueError('Guidance/caption mismatch')
  records.append(dict(sample_id=sid,source_split=v['source_split'],planning_valid=v['planning_valid'],planning_tokens=v['planning_tokens'],online_inputs=True,frozen_hidden_loaded=False))
 statistics=statistics(g,dataset.labels.manifest)if g['components']['event']else{}
 result=dict(success=True,schema=VERSION,configuration_sha256=digest(c),source=source_identity(),
  data_identity=data_identity(c),samples=len(dataset.rows),canary_reads=records,statistics=statistics,
  schedule_sha256=digest(schedule.pairs),trial_samples=len(schedule.pairs),training_started=False,
  assets={kind:weights_identity(data_config(g),kind)for kind in ('omni','ltx','gemma')})
 atomic_json(Path(c['output'])/'preflight.json',result);return result


def global_counts(samples,names,device,world):
 values=[sum(float(s['events'][k]['known'].sum())for s in samples)for k in names]
 values.append(sum(s['planning_tokens']for s in samples))
 totals=torch.tensor(values,device=device,dtype=torch.float64)
 if world>1:dist.all_reduce(totals)
 return {k:totals[i]for i,k in enumerate(names)},float(totals[-1])


def update(model,optimizer,c,samples,ordinals,strategy,sampler,pos,effective,step,device,world=1,audit=False):
 from p2_clip_events.model import mask_classes
 from p2_whisper_av.execution import accumulator
 g=c['generator'];names=list(pos)
 if names:samples=[dict(s,events=mask_classes(s['events'],effective))for s in samples]
 counts,tokens=global_counts(samples,names,device,world)
 optimizer.zero_grad(set_to_none=True)
 acc=accumulator(model.parameters(),len(samples),g['execution'])
 record=dict(av=0.,av_video=0.,av_audio=0.,feature_video=0.,feature_audio=0.,event=0.,planning_sum=0.)
 objective_config=dict(g,planning_weight=c['planning_weight'])
 audit_record=[]
 for sample,ordinal in zip(samples,ordinals,strict=True):
  sample=to_device(sample,device)
  seed=g['training']['seed'];torch.manual_seed(seed+ordinal)
  rng=torch.Generator(device=device).manual_seed(seed+1000000+ordinal)
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
   v=model(sample,pos,rng,strategy,sampler)
   if v['planning_tokens']!=sample['planning_tokens']:raise ValueError('Planning token denominator changed')
   loss,parts,event=objective(v,objective_config,counts,tokens,len(samples),world)
  if audit:
   from p2_ltx_events.heads import objective as generation_objective
   params=list(model.qwen.adapters)
   generation=generation_objective(v,g,counts,len(samples),world)[0]
   norms={}
   for name,term in [('generation',generation),('weighted_planning',c['planning_weight']*parts['planning'])]:
    gradients=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True)
    norms[name]=sum(float(x.detach().double().square().sum())for x in gradients if x is not None)**.5
    del gradients
   if norms['generation']<=0 or not all(torch.isfinite(torch.tensor(x))for x in norms.values()):raise ValueError('Missing/invalid generation-to-Qwen gradient')
   if sample['planning_tokens']and norms['weighted_planning']<=0:raise ValueError('Missing planning-to-Qwen gradient')
   audit_record.append(dict(norms,scope='local_preclip_canary_only; use CALIBRATION.json for global comparison'))
   del generation
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint objective')
  loss.backward();acc.add()
  record['av']+=float(v['av'].detach())/len(samples)
  record['event']+=float(event.detach());record['planning_sum']+=float(v['planning_sum'].detach())
  for mod in ('video','audio'):
   record['av_'+mod]+=float(v['av_components'][mod])/len(samples)
   record['feature_'+mod]+=float(v['features'].get(mod,torch.tensor(0.)).detach())/len(samples)
  del sample,v,loss,event,parts
 acc.finish(world)
 norms={group['name']:sum(float(p.grad.double().square().sum())for p in group['params']if p.grad is not None)**.5 for group in optimizer.param_groups}
 for special in (False,True):
  params=[p for group in optimizer.param_groups if (group['name']=='events')==special for p in group['params']]
  if params:torch.nn.utils.clip_grad_norm_(params,g['training']['clip_norm'],error_if_nonfinite=True)
 optimizer.step();model.assert_ownership()
 keys=list(record);scalars=torch.tensor([record[k]for k in keys],device=device,dtype=torch.float64)
 if world>1:dist.all_reduce(scalars)
 record={k:float(v)/(world if k not in ('event','planning_sum')else 1)for k,v in zip(keys,scalars)}
 record['planning_ce']=record['planning_sum']/max(tokens,1);record['planning_tokens']=tokens
 weights=g['loss_weights']
 record['weighted_total']=record['av']+weights['event']*record['event']+sum(weights[m+'_feature']*record['feature_'+m]for m in ('video','audio'))+c['planning_weight']*record['planning_ce']
 record.update(step=step,grad_norm=norms)
 if audit:record['qwen_objective_gradient_probe']=audit_record
 return record


def evaluation_barrier(c,step):
 from .policy import require_evaluation
 # Both baseline and100-step decisions bind exact checkpoint/validation/policy.
 require_evaluation(c,0)
 if step>=100:require_evaluation(c,100)


def precheck(c,resume=None):
 """Reject missing evaluations/source drift before reserving GPUs or loading22B."""
 validate(c);out=Path(c['output']);pre=read(out/'preflight.json');did=data_identity(c);source=source_identity()
 if (out/'STOP').exists():raise RuntimeError('Explicit Stage2 STOP preserved')
 if pre['configuration_sha256']!=digest(c)or pre['source']!=source or pre['data_identity']!=did:raise ValueError('Exact preflight required')
 if (out/'run.json').exists()and resume is None:raise ValueError('Existing joint run requires explicit resume')
 if resume is not None:
  from .calibration import verify_receipt
  verify_receipt(c)
  if Path(resume).resolve().parent!=out.resolve():raise ValueError('Resume only this joint experiment')
  marker=read(Path(resume).with_suffix('.complete.json'))
  if not 0<=marker['step']<=300:raise ValueError('Resume outside short trial budget')
  evaluation_barrier(c,marker['step'])
 return pre,did,source


def train(c,resume=None):
 from p2_listen.numerics import configure_training_numerics
 from p2_listen.monitoring import rank_zero_action,truncate_jsonl
 from p2_listen.training import rng_state
 from p2_whisper_event_feature.execution import CPUInputStream
 from p2_current_component_ablation.training import canary_snapshot,restore_canary
 out=Path(c['output']);pre,did,source=precheck(c,resume)
 if not torch.cuda.is_available():raise RuntimeError('Joint training requires GPU; no CPU fallback')
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
 if 31%world:raise ValueError('World must divide global batch31')
 torch.cuda.set_device(local);device=f'cuda:{local}'
 limit=float(os.environ.get('P2_TRAIN_MEMORY_GIB',134));torch.cuda.set_per_process_memory_fraction(limit*2**30/torch.cuda.get_device_properties(local).total_memory,local)
 if world>1:dist.init_process_group('nccl')
 stream=None;writer=None
 try:
  numerics=configure_training_numerics();dataset=Dataset(c);schedule=Schedule(c)
  rank_zero_action(lambda:validate(c,verify_payloads=True),rank=rank,world=world)
  model,strategy,sampler,initialized=make_model(c,device,verify_parent=False);model.train()
  stats=pre['statistics'];pos={k:v['pos_weight']for k,v in stats.items()};effective=[k for k,v in stats.items()if v['enabled']]
  from .calibration import run as calibrate,verify_receipt
  if resume is None:
   report=calibrate(model,c,dataset,strategy,sampler,pos,effective,device,rank,world)
   rank_zero_action(lambda:atomic_json(out/'CALIBRATION.json',report),rank=rank,world=world)
   if not report['passed']:
    state=dict(step=0,stage='awaiting_calibration_review',complete=False)
    rank_zero_action(lambda:atomic_json(out/'status.json',state),rank=rank,world=world)
    return state
  verify_receipt(c)
  identity=dict(schema=VERSION,experiment=c,data_identity=did,source=source,world=world,numerics=numerics,
   initialization=initialized,qwen_parameter_names=model.qwen.parameter_names,calibration_sha256=sha(out/'CALIBRATION.json'))
  if world>1:
   identities=[None]*world;dist.all_gather_object(identities,digest(identity))
   if len(set(identities))!=1:raise ValueError('Ranks disagree on joint identity')
  def register():
   if (out/'run.json').exists()and resume is None:raise ValueError('Existing joint run requires explicit resume')
   if resume and Path(resume).resolve().parent!=out.resolve():raise ValueError('Resume only this joint experiment')
   atomic_json(out/'run.json',identity,immutable=True)
  rank_zero_action(register,rank=rank,world=world)
  optimizer=torch.optim.AdamW(model.groups(),weight_decay=0.);base_lr={g['name']:g['lr']for g in optimizer.param_groups}
  start=checkpoint.resume(resume,model,optimizer,identity,rank)if resume else 0
  def save(step):
   states=[rng_state()]
   if world>1:states=[None]*world;dist.all_gather_object(states,rng_state())
   rank_zero_action(lambda:checkpoint.save(out/f'step_{step:06d}.pt',model,optimizer,step,identity,states),rank=rank,world=world)
  if resume is None:
   # Actual longest input, full optimizer states, then pristine parameters/Adam/RNG.
   selected=max(range(len(dataset.rows)),key=lambda i:float(dataset.rows[i]['duration_s']))
   eligible=max((i for i,sid in enumerate(dataset.ids)if sid in dataset.planning_ids),key=lambda i:float(dataset.rows[i]['duration_s']))
   snapshot=canary_snapshot(model);original=copy.deepcopy(optimizer.state_dict())
   with torch.random.fork_rng(devices=[local]):
    torch.cuda.reset_peak_memory_stats(device);began=time.monotonic()
    probes=[]
    for i,index in enumerate((selected,eligible)):
     sample=dataset.sample(index)
     record=update(model,optimizer,c,[sample],[1000000+i],strategy,sampler,pos,effective,i+1,device,world,audit=True)
     probes.append(dict(sample_id=dataset.ids[index],gradient_probe=record['qwen_objective_gradient_probe']))
     if any(not torch.isfinite(torch.tensor(v))for v in record['grad_norm'].values()):raise FloatingPointError('Invalid canary gradients')
    peak=torch.tensor(torch.cuda.max_memory_reserved(device)/2**30,device=device)
    if world>1:dist.all_reduce(peak,op=dist.ReduceOp.MAX)
    restore_canary(model,snapshot);optimizer.load_state_dict(original);optimizer.zero_grad(set_to_none=True)
   del snapshot,original,sample;torch.cuda.empty_cache()
   report=dict(success=True,real_updates=2,restored=True,sample_id=dataset.ids[selected],peak_reserved_gib=float(peak),seconds=time.monotonic()-began,probes=probes)
   rank_zero_action(lambda:atomic_json(out/'LONGEST_SAMPLE_CHECK.json',report),rank=rank,world=world)
   save(0)
   rank_zero_action(lambda:atomic_json(out/'status.json',dict(step=0,stage='awaiting_baseline_evaluation',complete=False)),rank=rank,world=world)
   return dict(step=0,stage='awaiting_baseline_evaluation')
  rank_zero_action(lambda:evaluation_barrier(c,start),rank=rank,world=world)
  if start>=300:return dict(step=start,stage='trial_finished')
  finish=100 if start<100 else 300
  def open_log():
   nonlocal writer
   from torch.utils.tensorboard import SummaryWriter
   truncate_jsonl(out/'metrics.jsonl',start)
   writer=SummaryWriter(str(out/'tensorboard'),purge_step=start+1)
  rank_zero_action(open_log,rank=rank,world=world)
  requests=((step,schedule.indices(len(dataset.rows),step,31,c['generator']['training']['seed'],rank,world))for step in range(start+1,finish+1))
  stream=CPUInputStream(dataset,requests,2);last=start
  for step in range(start+1,finish+1):
   stop=torch.tensor(int((out/'STOP').exists()),device=device)
   if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
   if stop.item():break
   began=time.monotonic();inputs=next(stream)
   if inputs['step']!=step:raise ValueError('Online prefetch changed schedule')
   for g in optimizer.param_groups:g['lr']=base_lr[g['name']]*min(1.,step/c['warmup_steps'])
   record=update(model,optimizer,c,inputs['samples'],[o for i,o in inputs['chosen']],strategy,sampler,pos,effective,step,device,world)
   record.update(seconds=time.monotonic()-began,input_read_seconds=inputs['read_seconds'],peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
   def log():
    with (out/'metrics.jsonl').open('a')as f:f.write(json.dumps(record)+'\n')
    if writer:
     for k in ('av','event','feature_video','feature_audio','planning_ce','weighted_total'):writer.add_scalar('loss/'+k,record[k],step)
     for k,v in record['grad_norm'].items():writer.add_scalar('grad_norm/'+k,v,step)
     writer.add_scalar('performance/step_seconds',record['seconds'],step)
    atomic_json(out/'status.json',dict(step=step,stage='training',complete=False))
   rank_zero_action(log,rank=rank,world=world);last=step
  if last>start:save(last)
  state=dict(step=last,stage='trial_finished'if last==300 else'awaiting_evaluation'if last==100 else'stopped',complete=last==300)
  rank_zero_action(lambda:atomic_json(out/'status.json',state),rank=rank,world=world)
  return state
 finally:
  if stream:stream.close()
  if writer:writer.close()
  if world>1 and dist.is_initialized():dist.destroy_process_group()
