"""Original AV arithmetic with bounded CPU prefetch and strict fresh-run identity."""
from pathlib import Path
import json,os,time
import torch
import torch.distributed as dist
from p2_listen.contracts import read,digest,atomic_json,sha
from p2_whisper_event_feature.execution import CPUInputStream
from p2_whisper_av.execution import accumulator
from . import VERSION,checkpoint
from .config import validate,source_identity
from .data import Dataset,data_config,data_identity,to_device
from .model import Model,av_objective
from .schedule import Schedule


def preflight(c):
 from .config import REFERENCE
 from p2_j_current_qwen_hidden import Reader as HReader
 from p2_j_cache.data import caption_prompt
 validate(c);did=data_identity(c);schedule=Schedule(c);dataset=Dataset(c)
 previous=read(REFERENCE/'preflight.json');old=previous['data_identity'];index=dataset.index
 if (old['current_cache_index']!=sha(c['current_cache_index']) or old['current_cache_receipts']!=index['caches']
     or previous['samples']!=len(dataset.rows)):
  raise ValueError('Existing full-data preflight does not cover this exact selection/cache set')
 # Deterministic32 real cached reads: train/dev, restored media, old/new AV,
 # short/long inputs. The existing full receipt covers every remaining member.
 q=read(Path(c['j_cache'])/'protocol.json');ids=q['canary_ids'];positions={r['sample_id']:i for i,r in enumerate(dataset.rows)}
 records=[]
 for sid in ids:
  row=dataset.reader.source.row(sid);sample=dataset.sample(positions[sid])
  if sample['text']['positive']['prompt']!=caption_prompt(row['caption']):raise ValueError('Not the full latest caption')
  if set(sample)&{'events','whisper','hidden_event_layout'} or sample['teacher'] is not None:
   raise ValueError('AV-only read a forbidden supervision/condition')
  # training_admitted is the retained historical train-only flag. The frozen
  # final membership and source split above explicitly admit selected dev rows.
  if row['split']not in ('train','dev'):raise ValueError('Invalid current member')
  records.append(dict(sample_id=sid,split=row['split'],duration_s=row['duration_s'],caption_sha256=row['caption_file_sha256'],
   caption_chars=len(row['caption']),av_disposition=dataset.reader.plan(sid)['disposition'],
   restored='orientation_restoration'in row,video_shape=list(sample['av']['av']['latents']['latents'].shape),
   audio_shape=list(sample['av']['av']['audio_latents']['latents'].shape)))
  del sample
 if {r['split']for r in records}!={'train','dev'}:raise ValueError('Canary reads must cover train and dev')
 result=dict(schema=VERSION,configuration_sha256=digest(c),data_identity=did,source=source_identity(),samples=len(dataset.rows),
  experiment_mode=['guided','av',1],objective='video_av+audio_av only; Qwen conditions; no Whisper/Event/Feature',
  reused_full_preflight=dict(path=str(REFERENCE/'preflight.json'),sha256=sha(REFERENCE/'preflight.json'),
   reason='Same current cache index, final train+dev membership and identical active cache reader code/protocols; all caches already admitted'),
  canary_reads=records,schedule_pairs=len(schedule.pairs),tail_padding=len(schedule.pairs)-schedule.size,training_started=False)
 atomic_json(Path(c['output'])/'preflight.json',result)
 return result


def train(c,resume_path=None,stop_after=None):
 from p2_listen.config import extractor_config
 from p2_listen.numerics import configure_training_numerics
 from p2_listen.performance import configure_renderer
 from p2_listen.features import weights_identity
 from p2_listen.training import rng_state
 from p2_listen.monitoring import rank_zero_action,truncate_jsonl
 validate(c);out=Path(c['output']);t=c['training'];pre=read(out/'preflight.json');did=data_identity(c);source=source_identity()
 if (pre['schema']!=VERSION or pre['configuration_sha256']!=digest(c) or pre['data_identity']!=did or pre['source']!=source):
  raise ValueError('Run preflight for this exact current AV-only source/config/data')
 if resume_path is not None and Path(resume_path).resolve().parent!=out.resolve():raise ValueError('Resume only this independent AV-only run')
 if stop_after is not None and (type(stop_after)is not int or stop_after<1):raise ValueError('Invalid stop-after')
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
 if t['global_batch']%world:raise ValueError('World must divide global batch')
 if not torch.cuda.is_available():raise RuntimeError('Real LTX training requires GPU')
 d=data_config(c);cfg=extractor_config(d);dataset=Dataset(c);schedule=Schedule(c);execution=c['execution']
 from omni_av_p2.ltx import load_renderer,append_conditions
 from omni_av.ltx_runtime import with_text
 from p2_listen.runtime import av_loss
 numerics=configure_training_numerics();torch.cuda.set_device(local);device=f'cuda:{local}';writer=None;stream=None
 if os.environ.get('P2_TRAIN_MEMORY_GIB'):
  torch.cuda.set_per_process_memory_fraction(float(os.environ['P2_TRAIN_MEMORY_GIB'])*2**30/torch.cuda.get_device_properties(local).total_memory,local)
 if world>1:dist.init_process_group('nccl')
 try:
  renderer,strategy,sampler=load_renderer(cfg,device);configure_renderer(renderer,t['gradient_checkpointing'])
  torch.manual_seed(t['seed']);model=Model(c,cfg,renderer).to(device);model.train()
  identity=dict(schema=VERSION,experiment=c,data_identity=did,source=source,world=world,numerics=numerics,
   assets={k:weights_identity(d,k)for k in ('omni','ltx','gemma')},text_targets=model.text_targets,
   qwen_targets=[],initialization='pretrained_only',execution=execution)
  if world>1:
   identities=[None]*world;dist.all_gather_object(identities,digest(identity))
   if len(set(identities))!=1:raise ValueError('Ranks disagree on AV-only identity')
  def register():
   if (out/'run.json').exists()and resume_path is None:raise ValueError('Existing run requires strict resume')
   atomic_json(out/'run.json',identity,immutable=True)
  rank_zero_action(register,rank=rank,world=world)
  groups=model.groups();base_lr={g['name']:g['lr']for g in groups};optimizer=torch.optim.AdamW(groups,weight_decay=0.)
  start=checkpoint.resume(resume_path,model,optimizer,identity,rank)if resume_path else 0
  def av_callback(u,pack,text):
   prepared=strategy.prepare_training_inputs(with_text(pack['av'],text),sampler)
   return av_loss(renderer,append_conditions(prepared,u),pack['support'],return_components=True)
  def save(step):
   states=[rng_state()]
   if world>1:states=[None]*world;dist.all_gather_object(states,rng_state())
   rank_zero_action(lambda:checkpoint.save(out/f'step_{step:06d}.pt',model,optimizer,step,identity,states),rank=rank,world=world)
  if execution['longest_sample_check']and resume_path is None:
   index=max(range(schedule.size),key=lambda i:schedule.durations[i])
   with torch.random.fork_rng(devices=[local]):
    sample=dataset.sample(index,device);torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast('cuda',dtype=torch.bfloat16):v=model(sample,{},None,av_callback)
    if not torch.isfinite(v['av']):raise FloatingPointError('Nonfinite longest sample')
    v['av'].backward();optimizer.zero_grad(set_to_none=True)
    atomic_json(out/'longest_probe'/f'rank{rank:03d}.json',dict(sample_id=sample['sample_id'],duration_s=schedule.durations[index],
     peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,optimizer_updated=False))
    del sample,v
   torch.cuda.empty_cache()
  def open_log():
   nonlocal writer
   if resume_path:truncate_jsonl(out/'metrics.jsonl',start)
   if t['tensorboard']:
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(out/'tensorboard'),purge_step=start+1 if resume_path else None)
  rank_zero_action(open_log,rank=rank,world=world)
  if resume_path is None:save(0)
  last=start;requests=((step,schedule.indices(schedule.size,step,t['global_batch'],t['seed'],rank,world))for step in range(start+1,t['steps']+1))
  stream=CPUInputStream(dataset,requests,execution['prefetch_batches'])
  for step in range(start+1,t['steps']+1):
   stop=torch.tensor(int((out/'STOP').exists()or stop_after is not None and step>stop_after),device=device)
   if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
   if stop.item():break
   began=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
   for g in optimizer.param_groups:g['lr']=base_lr[g['name']]*min(1.,step/t['warmup_steps'])
   wait=time.monotonic();inputs=next(stream);wait=time.monotonic()-wait
   if inputs['step']!=step:raise ValueError('CPU prefetch changed sample/noise schedule')
   chosen=inputs['chosen'];optimizer.zero_grad(set_to_none=True);acc=accumulator(model.parameters(),len(chosen),execution)
   record=dict(av=0.,av_video=0.,av_audio=0.)
   for cached,(_,ordinal)in zip(inputs['samples'],chosen):
    sample=to_device(cached,device);torch.manual_seed(t['seed']+ordinal)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     values=model(sample,{},None,av_callback);loss,_,_=av_objective(values,c,{},len(chosen),world)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pure AV loss')
    loss.backward();acc.add();record['av']+=float(values['av'].detach())/len(chosen)
    for mod in ('video','audio'):record['av_'+mod]+=float(values['av_components'][mod])/len(chosen)
    del sample,values,loss
   acc.finish(world)
   norms={g['name']:sum(float(p.grad.double().square().sum())for p in g['params']if p.grad is not None)**.5 for g in optimizer.param_groups}
   norms['main']=float(torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],t['clip_norm'],error_if_nonfinite=True))
   optimizer.step();model.assert_ownership();last=step
   keys=('av','av_video','av_audio');numbers=torch.tensor([record[k]for k in keys],device=device,dtype=torch.float64)
   if world>1:dist.all_reduce(numbers)
   record.update(zip(keys,(numbers/world).tolist()));record.update(weighted_total=record['av'],step=step,seconds=time.monotonic()-began,
    input_wait_seconds=wait,input_read_seconds=inputs['read_seconds'],grad_norm=norms,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
   def log():
    with (out/'metrics.jsonl').open('a')as f:f.write(json.dumps(record)+'\n')
    if writer:
     for k in (*keys,'weighted_total'):writer.add_scalar('loss/'+k,record[k],step)
     for k,v in norms.items():writer.add_scalar('grad_norm/'+k,v,step)
     writer.add_scalar('performance/step_seconds',record['seconds'],step)
    atomic_json(out/'status.json',dict(step=step,budget=t['steps'],complete=False))
   rank_zero_action(log,rank=rank,world=world)
   if step%t['save_every']==0:save(step)
   del acc,inputs
  if last%t['save_every']or last==start:save(last)
  rank_zero_action(lambda:atomic_json(out/'status.json',dict(step=last,budget=t['steps'],complete=last==t['steps'])),rank=rank,world=world)
  return dict(step=last,budget=t['steps'],complete=last==t['steps'],output=str(out))
 finally:
  if stream:stream.close()
  if writer:writer.close()
  if world>1 and dist.is_initialized():dist.destroy_process_group()
