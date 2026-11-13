"""Per-sample text and online LoRA; explicit all-reduce owns all adapters."""
from pathlib import Path
import json,os,time
import torch
import torch.distributed as dist
from p2_listen.contracts import EVENTS,read,sha,digest,atomic_json
from p2_listen.cache import load_part
from .config import validate,data_config,event_enabled,feature_enabled,source_identity


def data_identity(c):
    from p2_plan_cache.protocol import open_protocol
    d=data_config(c);root=Path(d['output']);p=open_protocol(c['plan_cache'])
    if p['config']!=d:raise ValueError('Plan/data mismatch')
    value=dict(dataset_config=sha(c['data_config']),dataset_complete=sha(root/'COMPLETE.json'),
               observations=sha(root/'manifests/observations.json'),targets=sha(root/'manifests/targets.json'),
               plan_protocol=digest(p),teacher_schema='p2-postprocessed-feature-teacher-v1' if feature_enabled(c) else None)
    return value


def preflight(c):
    """One reusable event-statistics/marker check; never load all hidden or AV payloads."""
    from p2_listen.annotations import class_statistics
    from p2_listen import features
    from .data import Dataset
    from .teacher import identity as teacher_identity
    dataset=Dataset(c);labels=[]
    for index,o in enumerate(dataset.rows):
        sid=o['sample_id'];label=read(Path(dataset.data['output'])/'labels/canonical'/(sid+'.json'))
        labels.append(load_part(features.folder(dataset.data,'events',sid),features.identity(dataset.data,'events',o,label=label)))
        parts=[]
        if c['text_condition']=='guided':parts+=['text']
        if c['stage']==1:parts+=['input_plan'] if c['text_condition']=='guided' else []
        for part in parts:
            if not (Path(c['plan_cache'])/'cache'/part/sid/'complete.json').exists():raise FileNotFoundError(f'Missing {part}: {sid}')
        if feature_enabled(c):
            root=Path(c['teacher_cache']);ident=teacher_identity(dataset.data,o,dataset.targets[sid],root)
            marker=read(root/'cache'/sid/'complete.json')
            if marker['identity']!=digest(ident):raise ValueError('Stale postprocessed teacher')
        if (index+1)%1000==0:print(json.dumps(dict(stage='preflight',samples=index+1,total=len(dataset.rows))),flush=True)
    stats=class_statistics(labels,dataset.data['events']['pos_weight_cap'],dataset.data['events']['K_train'])
    result=dict(schema='p2-matrix-preflight-v1',data_identity=data_identity(c),experiment_mode=[c['text_condition'],c['loss'],c['stage']],
                samples=len(dataset.rows),statistics=stats,source=source_identity())
    atomic_json(Path(c['output'])/'preflight.json',result)
    return result


def objective(values,c,counts,micro_count,world):
    from p2_listen.events import normalized_event_loss
    if 'event_supervision' in c:
        from p2_clip_events.model import normalized_loss as normalized_event_loss
    base=values['av']*c['loss_weights']['av']
    for mod,v in values['features'].items():base=base+c['loss_weights'][mod+'_feature']*v
    ev=normalized_event_loss(values['event_sums'],counts)
    return base/(micro_count*world)+c['loss_weights']['event']*ev,base,ev


def gradient_audit(model,values,c,counts):
    from p2_listen.events import normalized_event_loss
    if 'event_supervision' in c:
        from p2_clip_events.model import normalized_loss as normalized_event_loss
    objs=dict(av=values['av'],event=normalized_event_loss(values['event_sums'],counts))
    if values['features']:objs['feature']=sum(values['features'].values())
    result={}
    for group,module in [('connector',model.connector),('qwen_lora',model.qwen_lora),('text_lora',model.text_lora)]:
        params=list(module.parameters());result[group]={}
        for name,value in objs.items():
            grads=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True) if params else []
            norm=sum(float(g.detach().double().square().sum()) for g in grads if g is not None)**.5
            result[group][name]=norm
    for group in ('connector','qwen_lora'):
        if group=='qwen_lora' and c['stage']==1:continue
        if not event_enabled(c) and result[group]['event']!=0:raise RuntimeError('Probe gradient leaked into generation path')
    for name in ('event','feature'):
        if result['text_lora'].get(name,0)!=0:raise RuntimeError('Auxiliary loss reached downstream LTX LoRA')
    # Zero-initialized connector can legitimately block upstream gradients at step1.
    # Report exact norms; CPU tests and initialized-parent S0 assert positive paths.
    return result


def train(c,*,resume_path=None,parent=None,steps=None,global_batch=None,stop_after=None,audit=False,backend=None):
    import copy
    c=copy.deepcopy(c)
    if steps is not None:c['training']['steps']=steps
    if global_batch is not None:c['training']['global_batch']=global_batch
    validate(c,formal=True)
    if resume_path and parent:raise ValueError('Choose strict resume or Stage1 initialization, not both')
    if c['stage']==2 and not (parent or resume_path):raise ValueError('Stage2 requires matching Stage1 checkpoint')
    if c['stage']==1 and parent:raise ValueError('Stage1 does not accept a parent')
    from p2_listen.config import extractor_config
    from p2_listen.numerics import configure_training_numerics
    from p2_listen.training import indices,rng_state
    from p2_listen.performance import BucketGradientAccumulator
    from p2_listen.annotations import mask_classes
    event_names=EVENTS
    if 'event_supervision' in c:
        from p2_clip_events.model import classes,mask_classes
        event_names=classes(c['event_supervision'])
    from p2_listen.monitoring import rank_zero_action,truncate_jsonl
    from .data import Dataset
    from .model import MatrixModel
    from . import checkpoint
    get_data=backend.data_config if backend is not None else data_config
    get_identity=backend.data_identity if backend is not None else data_identity
    get_source=backend.source_identity if backend is not None else source_identity
    if backend is not None:Dataset=backend.Dataset
    initialize=backend.initialize_stage2 if backend is not None else checkpoint.initialize_stage2
    numerics=configure_training_numerics();d=get_data(c);cfg=extractor_config(d)
    dataset=Dataset(c);out=Path(c['output']);t=c['training']
    pre=read(out/'preflight.json')
    did=get_identity(c)
    if pre['data_identity']!=did or pre['source']!=get_source() or pre['experiment_mode']!=[c['text_condition'],c['loss'],c['stage']]:raise ValueError('Run preflight for this exact experiment/source first')
    stats=pre['statistics'];effective=[k for k in event_names if stats[k]['enabled']];pos={k:stats[k]['pos_weight'] for k in event_names}
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if t['global_batch']%world:raise ValueError('World size must divide global batch')
    if not torch.cuda.is_available():raise RuntimeError('Real LTX training requires GPU')
    if audit and world!=1:raise ValueError('Gradient-path audit is single-rank only')
    from torch.utils.tensorboard import SummaryWriter
    torch.cuda.set_device(local);device=f'cuda:{local}';writer=None
    if os.environ.get('P2_TRAIN_MEMORY_GIB'):
        torch.cuda.set_per_process_memory_fraction(float(os.environ['P2_TRAIN_MEMORY_GIB'])*2**30/torch.cuda.get_device_properties(local).total_memory,local)
    if world>1:dist.init_process_group('nccl')
    try:
        from omni_av_p2.ltx import load_renderer,append_conditions
        from omni_av.ltx_runtime import with_text
        from p2_listen.runtime import av_loss
        from p2_listen.performance import configure_renderer
        from p2_listen.features import weights_identity
        renderer,strategy,sampler=load_renderer(cfg,device)
        configure_renderer(renderer,t['gradient_checkpointing'])
        runtime=processor=None
        if c['stage']==2:
            from omni_av_p2.data import load_omni
            runtime,processor=load_omni(cfg,device)
        torch.manual_seed(t['seed'])
        model=MatrixModel(c,cfg,renderer,runtime,processor).to(device)
        parent_meta=None
        if parent:parent_meta=initialize(parent,model,c,did)
        if resume_path:
            old=checkpoint.checked(resume_path)
            parent_meta=old['identity']['parent']
        identity=dict(experiment=c,data_identity=did,source=get_source(),world=world,numerics=numerics,
                      assets={k:weights_identity(d,k) for k in ('omni','ltx','gemma')},
                      parent=parent_meta,qwen_targets=model.qwen_targets,text_targets=model.text_targets,statistics=stats)
        if world>1:
            identities=[None]*world;dist.all_gather_object(identities,digest(identity))
            if len(set(identities))!=1:raise ValueError('Ranks disagree on run identity')
        def register():
            if (out/'run.json').exists() and not resume_path:raise ValueError('Run exists; resume explicitly')
            if resume_path and Path(resume_path).resolve().parent!=out.resolve():raise ValueError('Resume must belong to run')
            atomic_json(out/'run.json',identity,immutable=True)
        rank_zero_action(register,rank=rank,world=world)
        groups=model.groups();base_lr={g['name']:g['lr'] for g in groups}
        optimizer=torch.optim.AdamW(groups,weight_decay=0.)
        start=checkpoint.resume(resume_path,model,optimizer,identity,rank) if resume_path else 0
        def open_writer():
            nonlocal writer
            if resume_path:truncate_jsonl(out/'metrics.jsonl',start)
            if t['tensorboard']:writer=SummaryWriter(str(out/'tensorboard'),purge_step=start+1 if resume_path else None)
        rank_zero_action(open_writer,rank=rank,world=world)
        model.train();renderer.train()
        def av_callback(u,pack,text):
            prepared=strategy.prepare_training_inputs(with_text(pack['av'],text),sampler)
            return av_loss(renderer,append_conditions(prepared,u),pack['support'],return_components=True)
        def save(step):
            states=[rng_state()]
            if world>1:states=[None]*world;dist.all_gather_object(states,rng_state())
            rank_zero_action(lambda:checkpoint.save(out/f'step_{step:06d}.pt',model,optimizer,step,identity,states),rank=rank,world=world)
        if not resume_path:save(0)
        last=start
        for step in range(start+1,t['steps']+1):
            stop=torch.tensor(int((out/'STOP').exists() or stop_after is not None and step>stop_after),device=device)
            if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if stop.item():break
            begin=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
            for g in optimizer.param_groups:g['lr']=base_lr[g['name']]*min(1.,step/t['warmup_steps'])
            chosen=indices(len(dataset.rows),step,t['global_batch'],t['seed'],rank,world)
            samples=[dataset.sample(i,device) for i,_ in chosen]
            for s in samples:s['events']=mask_classes(s['events'],effective)
            count_values=torch.stack([sum(s['events'][k]['known'].float().sum() for s in samples) for k in event_names])
            if world>1:dist.all_reduce(count_values)
            counts={k:count_values[i] for i,k in enumerate(event_names)}
            optimizer.zero_grad(set_to_none=True);acc=BucketGradientAccumulator(model.parameters(),32)
            record=dict(av=0.,av_video=0.,av_audio=0.,event=0.,feature_video=0.,feature_audio=0.)
            event_totals={k:0. for k in event_names}
            for s,(_,ordinal) in zip(samples,chosen):
                torch.manual_seed(t['seed']+ordinal)
                gen=torch.Generator(device=device).manual_seed(t['seed']+1000000+ordinal)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    values=model(s,pos,gen,av_callback)
                    loss,base,event=objective(values,c,counts,len(samples),world)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
                if audit:record['gradient_paths']=gradient_audit(model,values,c,counts)
                loss.backward();acc.add()
                record['av']+=float(values['av'].detach())/len(samples)
                for mod in ('video','audio'):record['av_'+mod]+=float(values['av_components'][mod])/len(samples)
                record['event']+=float(event.detach())
                for k in event_names:event_totals[k]+=float(values['event_sums'][k].detach())
                for mod,v in values['features'].items():record['feature_'+mod]+=float(v.detach())/len(samples)
                del values,loss
            acc.finish(world)
            norms={g['name']:sum(float(p.grad.double().square().sum()) for p in g['params'] if p.grad is not None)**.5 for g in optimizer.param_groups}
            main=[p for g in optimizer.param_groups if g['name']!='events' for p in g['params']]
            norms['main']=float(torch.nn.utils.clip_grad_norm_(main,t['clip_norm'],error_if_nonfinite=True))
            torch.nn.utils.clip_grad_norm_(model.event_heads.parameters(),t['clip_norm'],error_if_nonfinite=True)
            optimizer.step();model.assert_ownership();last=step
            keys=('av','av_video','av_audio','event','feature_video','feature_audio')
            scalars=torch.tensor([record[k] for k in keys]+[event_totals[k] for k in event_names],device=device,dtype=torch.float64)
            if world>1:dist.all_reduce(scalars)
            for k,v in zip(keys,scalars.tolist()):record[k]=v if k=='event' else v/world
            record['per_event']={k:dict(loss=v/max(float(counts[k]),1e-12),known=float(counts[k])) for k,v in zip(event_names,scalars.tolist()[len(keys):])}
            record['weighted_total']=(c['loss_weights']['av']*record['av']+c['loss_weights']['event']*record['event']
                +c['loss_weights']['video_feature']*record['feature_video']+c['loss_weights']['audio_feature']*record['feature_audio'])
            record.update(step=step,seconds=time.monotonic()-begin,grad_norm=norms,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
            def log():
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                if writer:
                    for k in (*keys,'weighted_total'):writer.add_scalar('loss/'+k,record[k],step)
                    for k,v in record['per_event'].items():
                        writer.add_scalar('event/'+k+'/loss',v['loss'],step)
                        writer.add_scalar('event/'+k+'/known',v['known'],step)
                    for k,v in norms.items():writer.add_scalar('grad_norm/'+k,v,step)
                    for g in optimizer.param_groups:writer.add_scalar('lr/'+g['name'],g['lr'],step)
                    writer.add_scalar('performance/step_seconds',record['seconds'],step)
                    writer.add_scalar('memory/peak_reserved_gib',record['peak_reserved_gib'],step)
                atomic_json(out/'status.json',dict(step=step,budget=t['steps'],complete=False))
            rank_zero_action(log,rank=rank,world=world)
            if step%100==0:save(step)
            del samples,acc
        save(last)
        rank_zero_action(lambda:atomic_json(out/'status.json',dict(step=last,budget=t['steps'],complete=last==t['steps'])),rank=rank,world=world)
        return dict(success=True,step=last,output=str(out))
    finally:
        if writer:writer.close()
        if world>1 and dist.is_initialized():dist.destroy_process_group()
