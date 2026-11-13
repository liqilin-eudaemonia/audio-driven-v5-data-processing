"""Native joint training execution, with only the selected component branches."""
import copy,csv,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from p2_listen.contracts import read,digest,atomic_json,sha
from p2_whisper_av.execution import accumulator,move_adam_moments
from p2_whisper_event_feature.execution import CPUInputStream
from p2_ltx_events.heads import objective,names as event_names
from . import VERSION
from .config import validate,source_identity,flags,CACHE_REFERENCE as REFERENCE
from .data import Dataset,data_identity,to_device
from .model import Model as QwenWhisperEventFeature
from . import checkpoint

def check_preflight(c,pre,source,did):
 if pre['schema']!=VERSION or pre['configuration_sha256']!=digest(c)or pre['source']!=source or pre['data_identity']!=did:
  raise ValueError('Run exact current preflight before training')

def preflight(c):
 from .schedule import Schedule
 from p2_j_cache.data import caption_prompt
 from p2_ltx_events.data import statistics
 validate(c);did=data_identity(c);dataset=Dataset(c);schedule=Schedule(c);index=dataset.index
 previous=read(REFERENCE/'preflight.json');old=previous['data_identity']
 full=read(Path(c['full_reference']).parent/'preflight.json')
 if previous['samples']!=index['samples']or any(old['active_caches'][k]!=index['caches'][k]for k in old['active_caches']):raise ValueError('Current cache reuse receipt changed')
 if full['samples']!=index['samples']or full['data_identity']['current_cache_index']!=sha(c['current_cache_index'])or full['data_identity']['current_cache_receipts']!=index['caches']:raise ValueError('Full current all-cache preflight changed')
 ids=read(Path(c['j_cache'])/'protocol.json')['canary_ids'];positions={r['sample_id']:i for i,r in enumerate(dataset.rows)};records=[];f=flags(c)
 for sid in ids:
  row=dataset.reader.source.row(sid);v=dataset.sample(positions[sid])
  if v['text']['positive']['prompt']!=caption_prompt(row['caption']):raise ValueError('Not the complete latest caption')
  if (v.get('teacher')is not None)!=f['feature']or(v.get('whisper')is not None)!=f['whisper']or('events'in v)!=f['event']:raise ValueError('Wrong active cache inputs')
  records.append(dict(sample_id=sid,split=row['split'],duration_s=row['duration_s'],caption_sha256=row['caption_file_sha256'],caption_chars=len(row['caption']),teacher=f['feature'],whisper=f['whisper'],event_classes=len(v.get('events',{}))))
  del v
 if {r['split']for r in records}!={'train','dev'}:raise ValueError('Both source splits must be read')
 stats=statistics(c,dataset.labels.manifest)if f['event']else{}
 result=dict(schema=VERSION,configuration_sha256=digest(c),source=source_identity(),data_identity=did,statistics=stats,experiment_mode=[c['text_condition'],c['loss'],c['stage']],samples=len(dataset.rows),canary_reads=records,schedule_pairs=len(schedule.pairs),tail_padding=len(schedule.pairs)-schedule.size,training_started=False,reused_full_preflight=dict(path=str(Path(c['full_reference']).parent/'preflight.json'),sha256=sha(Path(c['full_reference']).parent/'preflight.json')))
 atomic_json(Path(c['output'])/'preflight.json',result);return result

def _event_counts(samples,names,device,world):
    if not names:return {}
    counts=torch.tensor([sum(float(s['events'][k]['known'].float().sum()) for s in samples) for k in names],dtype=torch.float32,device=device)
    if world>1:dist.all_reduce(counts)
    return {k:counts[i] for i,k in enumerate(names)}


def canary_snapshot(model):
    # The optimizer cannot change frozen base weights. Snapshot trainables and
    # all buffers, including aliases, without copying the full frozen LTX per rank.
    names={name for name,p in model.named_parameters(remove_duplicate=False) if p.requires_grad}
    names.update(name for name,_ in model.named_buffers(remove_duplicate=False))
    return {name:value.detach().to(device='cpu',copy=True)
            for name,value in model.state_dict(keep_vars=True).items() if name in names}


def restore_canary(model,snapshot):
    result=model.load_state_dict(snapshot,strict=False)
    if result.unexpected_keys or set(result.missing_keys)!=set(model.state_dict())-set(snapshot):
        raise ValueError('Canary snapshot keys differ')


def longest_check(model,dataset,c,optimizer,execution,strategy,sampler,pos,effective,names,device,local,rank,world):
    """Allocate real Adam/auxiliary states, measure peak, then restore pristine init."""
    from p2_clip_events.model import mask_classes
    chosen=[None]
    if rank==0:
        with Path(c['sample_list']).open() as f:
            index,row=max(enumerate(csv.DictReader(f)),key=lambda x:float(x[1]['duration_s']))
        chosen[0]=dict(index=index,sample_id=row['sample_id'],duration_s=float(row['duration_s']))
    if world>1:dist.broadcast_object_list(chosen,src=0)
    original=canary_snapshot(model)
    original_optimizer=copy.deepcopy(optimizer.state_dict())
    with torch.random.fork_rng(devices=[local]):
        torch.cuda.reset_peak_memory_stats(device)
        for update in range(2):
            sample=dataset.sample(chosen[0]['index'],device)
            if names:sample['events']=mask_classes(sample['events'],effective)
            counts=_event_counts([sample],names,device,world)
            optimizer.zero_grad(set_to_none=True);acc=accumulator(model.parameters(),1,execution)
            generator=torch.Generator(device=device).manual_seed(c['training']['seed']+1000000+update)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                values=model(sample,pos,generator,strategy,sampler)
                loss,_,_=objective(values,c,counts,1,world)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite longest-sample objective')
            loss.backward();acc.add();acc.finish(world)
            main=[p for g in optimizer.param_groups if g['name']!='events' for p in g['params']]
            torch.nn.utils.clip_grad_norm_(main,c['training']['clip_norm'],error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(model.event_heads.parameters(),c['training']['clip_norm'],error_if_nonfinite=True)
            if execution['optimizer_state_cpu_offload']:move_adam_moments(optimizer,to_parameters=True)
            optimizer.step()
            if execution['optimizer_state_cpu_offload']:move_adam_moments(optimizer)
            del values,loss,sample,acc
        peak=torch.tensor(torch.cuda.max_memory_reserved(device)/2**30,device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        restore_canary(model,original);optimizer.load_state_dict(original_optimizer);optimizer.zero_grad(set_to_none=True)
    del original,original_optimizer;torch.cuda.empty_cache()
    model.assert_ownership()
    return dict(success=True,sample=chosen[0],max_rank_peak_reserved_gib=float(peak),
        real_adam_and_auxiliary_states_tested=True,updates_restored=True,rng_restored=True)


def train(c,resume_path=None,stop_after=None):
    from .data import data_config
    from p2_clip_events.model import classes,mask_classes
    from p2_listen.config import extractor_config
    from p2_listen.numerics import configure_training_numerics
    from p2_listen.performance import configure_renderer
    from p2_listen.training import rng_state
    from p2_listen.features import weights_identity
    from p2_listen.monitoring import rank_zero_action,truncate_jsonl
    # Schedule is injected by the separately hashed split runtime.
    validate(c)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    t=c['training'];out=Path(c['output']);execution=c['execution']
    if c.get('resume_execution') and resume_path is None:
        raise ValueError('World31 continuation requires a complete migrated checkpoint')
    if t['global_batch']!=28 or world!=28:raise ValueError('Require batch28/world28')
    if stop_after is not None and (type(stop_after) is not int or stop_after<1):raise ValueError('Invalid stop-after')
    pre=read(out/'preflight.json');did=data_identity(c);source=source_identity()
    check_preflight(c,pre,source,did)
    if not torch.cuda.is_available():raise RuntimeError('Real training requires GPU; CPU tests use a tiny native model')
    d=data_config(c);cfg=extractor_config(d);dataset=Dataset(c);schedule=Schedule(c)
    names=event_names(c);stats=pre['statistics'];effective=[k for k in names if stats[k]['enabled']];pos={k:stats[k]['pos_weight'] for k in names}
    numerics=configure_training_numerics();torch.cuda.set_device(local);device=f'cuda:{local}';writer=None;input_stream=None
    if os.environ.get('P2_TRAIN_MEMORY_GIB'):
        torch.cuda.set_per_process_memory_fraction(float(os.environ['P2_TRAIN_MEMORY_GIB'])*2**30/torch.cuda.get_device_properties(local).total_memory,local)
    if world>1:dist.init_process_group('nccl')
    try:
        from omni_av_p2.ltx import load_renderer
        renderer,strategy,sampler=load_renderer(cfg,device);configure_renderer(renderer,t['gradient_checkpointing'])
        torch.manual_seed(t['seed']);model=QwenWhisperEventFeature(c,cfg,renderer).to(device)
        model.train()
        identity=dict(schema=VERSION,experiment=c,data_identity=did,source=source,world=world,numerics=numerics,
            assets={k:weights_identity(d,k) for k in ['omni','ltx','gemma']},text_targets=model.text_targets,
            qwen_targets=model.qwen_targets,statistics=stats,execution=execution,whisper_time_max_pos=getattr(model,'time_max_pos',None),initialization='pretrained_only',ltx_lora_report=model.ltx_lora_report)
        if world>1:
            identities=[None]*world;dist.all_gather_object(identities,digest(identity))
            if len(set(identities))!=1:raise ValueError('Ranks disagree on combined experiment identity')
        def register():
            if (out/'run.json').exists() and resume_path is None:raise ValueError('Existing run requires explicit strict resume')
            if resume_path is not None and Path(resume_path).resolve().parent!=out.resolve():raise ValueError('Resume only this new experiment')
            atomic_json(out/'run.json',identity,immutable=True)
        rank_zero_action(register,rank=rank,world=world)
        groups=model.groups();base_lr={g['name']:g['lr'] for g in groups};optimizer=torch.optim.AdamW(groups,weight_decay=0.)
        start=checkpoint.resume(resume_path,model,optimizer,identity,rank) if resume_path else 0
        if execution['optimizer_state_cpu_offload']:raise ValueError('Split requires existing GPU Adam')
        split_canary(c=c,model=model,optimizer=optimizer,dataset=dataset,strategy=strategy,sampler=sampler,
            pos=pos,effective=effective,names=names,device=device,local=local,rank=rank,world=world,
            resume_path=resume_path,identity=identity,start=start)
        def save(step):
            states=[rng_state()]
            if world>1:states=[None]*world;dist.all_gather_object(states,rng_state())
            rank_zero_action(lambda:checkpoint.save(out/f'step_{step:06d}.pt',model,optimizer,step,identity,states),rank=rank,world=world)
        if execution['longest_sample_check'] and resume_path is None:
            report=longest_check(model,dataset,c,optimizer,execution,strategy,sampler,pos,effective,names,device,local,rank,world)
            rank_zero_action(lambda:atomic_json(out/'LONGEST_SAMPLE_CHECK.json',report),rank=rank,world=world)
        def open_log():
            nonlocal writer
            if resume_path:truncate_jsonl(out/'metrics.jsonl',start)
            if t['tensorboard']:
                from torch.utils.tensorboard import SummaryWriter
                writer=SummaryWriter(str(out/'tensorboard'),purge_step=start+1 if resume_path else None)
        rank_zero_action(open_log,rank=rank,world=world)
        model.train()
        if resume_path is None:save(0)
        last=start
        requests=((step,schedule.indices(len(dataset.rows),step,t['global_batch'],t['seed'],rank,world))
                  for step in range(start+1,t['steps']+1))
        input_stream=CPUInputStream(dataset,requests,execution['prefetch_batches'])
        for step in range(start+1,t['steps']+1):
            stop=torch.tensor(int((out/'STOP').exists() or stop_after is not None and step>stop_after),device=device)
            if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if stop.item():break
            begin=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
            for group in optimizer.param_groups:group['lr']=base_lr[group['name']]*min(1.,step/t['warmup_steps'])
            wait_started=time.monotonic();inputs=next(input_stream)
            if inputs['step']!=step:raise ValueError('Prefetch changed training schedule')
            wait_seconds=time.monotonic()-wait_started
            chosen=inputs['chosen'];samples=inputs['samples']
            if names:
                for sample in samples:sample['events']=mask_classes(sample['events'],effective)
            counts=_event_counts(samples,names,device,world)
            effective_batch=schedule.count(step)
            optimizer.zero_grad(set_to_none=True);acc=accumulator(model.parameters(),1,execution)
            record=dict(av=0.,av_video=0.,av_audio=0.,event=0.,event_clip=0.,event_temporal=0.,feature_video=0.,feature_audio=0.)
            totals={k:0. for k in names}
            for sample,(_,ordinal) in zip(samples,chosen):
                sample=to_device(sample,device)
                torch.manual_seed(t['seed']+ordinal);generator=torch.Generator(device=device).manual_seed(t['seed']+1000000+ordinal)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    values=model(sample,pos,generator,strategy,sampler)
                    loss,event_parts,event=objective(values,c,counts,effective_batch,1)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite combined objective')
                loss.backward();acc.add()
                record['av']+=float(values['av'].detach())/effective_batch;record['event']+=float(event.detach())
                for task in ('clip','temporal'):record['event_'+task]+=float(event_parts[task].detach())
                for mod in ['video','audio']:
                    record['av_'+mod]+=float(values['av_components'][mod])/effective_batch
                    record['feature_'+mod]+=(float(values['features'][mod].detach()) if mod in values['features']else 0.)/effective_batch
                for k in names:totals[k]+=float(values['event_sums'][k].detach())
                del values,loss,event,event_parts,sample
            if not chosen:acc.add()
            acc.finish(world)
            norms={g['name']:sum(float(p.grad.double().square().sum()) for p in g['params'] if p.grad is not None)**.5 for g in optimizer.param_groups}
            main=[p for g in optimizer.param_groups if g['name']!='events' for p in g['params']]
            norms['main']=float(torch.nn.utils.clip_grad_norm_(main,t['clip_norm'],error_if_nonfinite=True))
            torch.nn.utils.clip_grad_norm_(model.event_heads.parameters(),t['clip_norm'],error_if_nonfinite=True)
            if execution['optimizer_state_cpu_offload']:move_adam_moments(optimizer,to_parameters=True)
            optimizer.step()
            if execution['optimizer_state_cpu_offload']:move_adam_moments(optimizer)
            model.assert_ownership();last=step
            keys=['av','av_video','av_audio','event','event_clip','event_temporal','feature_video','feature_audio']
            scalars=torch.tensor([record[k] for k in keys]+[totals[k] for k in names],device=device,dtype=torch.float64)
            if world>1:dist.all_reduce(scalars)
            for k,v in zip(keys,scalars.tolist()):record[k]=v
            record['weighted_total']=record['av']+c['loss_weights']['event']*record['event']+sum(c['loss_weights'][m+'_feature']*record['feature_'+m]for m in ('video','audio'))
            record['per_event']={k:dict(loss=v/max(float(counts[k]),1e-12),known=float(counts[k])) for k,v in zip(names,scalars.tolist()[len(keys):])}
            record.update(step=step,effective_batch=effective_batch,consumed_pairs=min(len(schedule.pairs),schedule.plan['prefix_pairs']+(step-schedule.plan['start_step'])*28),seconds=time.monotonic()-begin,input_wait_seconds=wait_seconds,
                          input_read_seconds=inputs['read_seconds'],grad_norm=norms,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
            def log():
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                if writer:
                    for k in keys+['weighted_total']:writer.add_scalar('loss/'+k,record[k],step)
                    for k,v in norms.items():writer.add_scalar('grad_norm/'+k,v,step)
                    writer.add_scalar('performance/step_seconds',record['seconds'],step)
                atomic_json(out/'status.json',dict(step=step,budget=t['steps'],complete=False))
            rank_zero_action(log,rank=rank,world=world)
            if step%t['save_every']==0:save(step)
            del samples,acc,inputs
        if last%t['save_every']or last==start:save(last)
        rank_zero_action(lambda:atomic_json(out/'status.json',dict(step=last,budget=t['steps'],complete=last==t['steps'])),rank=rank,world=world)
        return dict(step=last,complete=last==t['steps'],output=str(out))
    finally:
        if input_stream:input_stream.close()
        if writer:writer.close()
        if world>1 and dist.is_initialized():dist.destroy_process_group()
