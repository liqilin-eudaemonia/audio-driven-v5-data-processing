"""Existing normalized auxiliary objective plus the proven Whisper execution path."""
import copy,csv,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from p2_listen.contracts import read,digest,atomic_json
from p2_whisper_av.execution import accumulator,move_adam_moments
from .execution import settings,CPUInputStream
from .heads import objective,names as event_names
from . import VERSION
from .config import validate,validate_baseline,source_identity
from .data import Dataset,data_identity,to_device
from .model import LTXHiddenEvents as QwenWhisperEventFeature
from . import checkpoint

_context=None


def _fork_init():
    from p2_clip_events.labels import LabelReader
    _,dataset=_context
    if dataset.spec.get('current_cache_index'):
        dataset.reopen_after_fork();return
    dataset.labels.conn.close()
    dataset.labels=LabelReader(dataset.spec['clip_labels'],verify=False)
    from .labels import Reader
    if dataset.temporal is not None:
        dataset.temporal.conn.close()
        dataset.temporal=Reader(dataset.spec['temporal_labels'],verify=False)


def _preflight_row(row):
    from p2_j_cache.data import sample
    from p2_j_cache.runtime import cache_identity
    c,dataset=_context;sid=row['sample_id']
    if c.get('current_cache_index'):return dataset.preflight_row(sid)
    r=dataset.reader;item=sample(r.work,r.manifest,sid)
    if dataset.labels.row(sid)['source_digest']!=r.manifest['entries'][sid]:raise ValueError('Stale Event source')
    if dataset.temporal is not None and dataset.temporal.row(sid)['source_digest']!=r.manifest['entries'][sid]:raise ValueError('Stale temporal source')
    for part in ['vae','text_guided','input_plan']:
        folder=r.work/'cache'/part/sid;marker=read(folder/'complete.json')
        expected=cache_identity(r.manifest,item,part,r.assets[part])
        if marker['identity']!=digest(expected) or marker['contract']!=expected or not (folder/'payload.pt').stat().st_size:
            raise ValueError('Invalid frozen base cache: '+sid+'/'+part)
    for reader,folder,expected in [
        (dataset.whisper,dataset.whisper.folder(sid),dataset.whisper.identity(item)),
        (dataset.teacher_reader,dataset.teacher_reader.work/'cache'/sid,dataset.teacher_reader.identity(dataset.teacher_reader.item(sid)))]:
        marker=read(folder/'complete.json')
        if marker['identity']!=digest(expected) or marker['contract']!=expected or not (folder/'payload.pt').stat().st_size:
            raise ValueError('Invalid Whisper/teacher cache: '+str(folder))


def preflight(c,processes=8):
    global _context
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from .schedule import Schedule
    from p2_clip_events.backend import selected_statistics
    validate(c);validate_baseline(c,read(c['hidden_event_parent_config']))
    if type(processes) is not int or processes<1:raise ValueError('Expected positive bounded process count')
    if torch.cuda.is_initialized():raise RuntimeError('CPU metadata preflight must precede CUDA')
    dataset=Dataset(c);Schedule(c)
    if c.get('current_cache_index'):
        from .current import checked_index
        checked_index(c['current_cache_index'])
    else:
        dataset.reader.part(None,'text');complete=read(dataset.reader.work/'COMPLETE.json')
        if not complete['success'] or complete['samples']!=len(dataset.rows):raise ValueError('Full base cache not accepted')
    start=time.monotonic();_context=c,dataset
    try:
        if processes==1:
            for row in dataset.rows:_preflight_row(row)
        else:
            with ProcessPoolExecutor(max_workers=processes,mp_context=multiprocessing.get_context('fork'),initializer=_fork_init) as pool:
                for offset in range(0,len(dataset.rows),4096):
                    for _ in pool.map(_preflight_row,dataset.rows[offset:offset+4096],chunksize=16):pass
                    atomic_json(Path(c['output'])/'preflight_progress.json',dict(checked=min(offset+4096,len(dataset.rows)),total=len(dataset.rows)))
    finally:_context=None
    from .data import statistics
    stats=statistics(c,dataset.labels.manifest)
    if not any(v['enabled'] for v in stats.values()):raise ValueError('No usable Event classes')
    value=dict(schema=VERSION,configuration_sha256=digest(c),source=source_identity(),data_identity=data_identity(c),statistics=stats,
        experiment_mode=[c['text_condition'],c['loss'],c['stage']],samples=len(dataset.rows),
        processes=processes,elapsed_s=time.monotonic()-start,training_started=False)
    atomic_json(Path(c['output'])/'preflight.json',value)
    return value


def check_preflight(c,pre,source,did):
    if (pre.get('configuration_sha256')!=digest(c) or pre['schema']!=VERSION
            or pre['source']!=source or pre['data_identity']!=did
            or pre['experiment_mode']!=[c['text_condition'],c['loss'],c['stage']]):
        raise ValueError('Run exact new preflight before training')


def _event_counts(samples,names,device,world):
    counts=torch.stack([sum(s['events'][k]['known'].float().sum() for s in samples) for k in names]).to(device)
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
        from .data import sample_list
        with sample_list(c).open() as f:
            index,row=max(enumerate(csv.DictReader(f)),key=lambda x:float(x[1]['duration_s']))
        chosen[0]=dict(index=index,sample_id=row['sample_id'],duration_s=float(row['duration_s']))
    if world>1:dist.broadcast_object_list(chosen,src=0)
    original=canary_snapshot(model)
    original_optimizer=copy.deepcopy(optimizer.state_dict())
    with torch.random.fork_rng(devices=[local]):
        torch.cuda.reset_peak_memory_stats(device)
        for update in range(2):
            sample=dataset.sample(chosen[0]['index'],device)
            sample['events']=mask_classes(sample['events'],effective)
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
    from .schedule import Schedule
    validate(c);validate_baseline(c,read(c['hidden_event_parent_config']))
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    t=c['training'];out=Path(c['output']);execution=settings(c)
    if t['global_batch']%world:raise ValueError('World must divide fixed global batch')
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
            qwen_targets=model.qwen_targets,statistics=stats,execution=execution,whisper_time_max_pos=model.time_max_pos)
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
        if execution['optimizer_state_cpu_offload']:move_adam_moments(optimizer)
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
            for sample in samples:sample['events']=mask_classes(sample['events'],effective)
            counts=_event_counts(samples,names,device,world)
            optimizer.zero_grad(set_to_none=True);acc=accumulator(model.parameters(),len(chosen),execution)
            record=dict(av=0.,av_video=0.,av_audio=0.,event=0.,event_clip=0.,event_temporal=0.,feature_video=0.,feature_audio=0.)
            totals={k:0. for k in names}
            for sample,(_,ordinal) in zip(samples,chosen):
                sample=to_device(sample,device)
                torch.manual_seed(t['seed']+ordinal);generator=torch.Generator(device=device).manual_seed(t['seed']+1000000+ordinal)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    values=model(sample,pos,generator,strategy,sampler)
                    loss,event_parts,event=objective(values,c,counts,len(chosen),world)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite combined objective')
                loss.backward();acc.add()
                record['av']+=float(values['av'].detach())/len(chosen);record['event']+=float(event.detach())
                for task in ('clip','temporal'):record['event_'+task]+=float(event_parts[task].detach())
                for mod in ['video','audio']:
                    record['av_'+mod]+=float(values['av_components'][mod])/len(chosen)
                    record['feature_'+mod]+=float(values['features'][mod].detach())/len(chosen)
                for k in names:totals[k]+=float(values['event_sums'][k].detach())
                del values,loss,event,event_parts,sample
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
            for k,v in zip(keys,scalars.tolist()):record[k]=v if k.startswith('event') else v/world
            record['weighted_total']=record['av']+.05*record['event']+.025*(record['feature_video']+record['feature_audio'])
            record['per_event']={k:dict(loss=v/max(float(counts[k]),1e-12),known=float(counts[k])) for k,v in zip(names,scalars.tolist()[len(keys):])}
            record.update(step=step,seconds=time.monotonic()-begin,input_wait_seconds=wait_seconds,
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
        if last%t['save_every'] or last==start:save(last)
        rank_zero_action(lambda:atomic_json(out/'status.json',dict(step=last,budget=t['steps'],complete=last==t['steps'])),rank=rank,world=world)
        return dict(step=last,complete=last==t['steps'],output=str(out))
    finally:
        if input_stream:input_stream.close()
        if writer:writer.close()
        if world>1 and dist.is_initialized():dist.destroy_process_group()
