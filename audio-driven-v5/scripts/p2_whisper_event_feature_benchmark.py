"""Matched 32-rank combined-loss execution benchmark; zero formal updates."""
import argparse,csv,hashlib,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()


def tensor_digest(state):
    h=hashlib.sha256()
    for name,tensor in state:
        h.update(name.encode())
        if tensor is None:h.update(b'NONE');continue
        value=tensor.detach().cpu().contiguous()
        h.update(str((tuple(value.shape),value.dtype)).encode());h.update(value.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    import torch
    import torch.distributed as dist
    from p2_whisper_event_feature.config import load,validate_baseline,source_identity
    from p2_whisper_event_feature.data import Dataset,data_identity,to_device
    from p2_whisper_event_feature.model import QwenWhisperEventFeature
    from p2_whisper_event_feature.execution import CPUInputStream,settings
    from p2_whisper_event_feature.training import longest_check,_event_counts
    from p2_whisper_av.execution import accumulator
    from p2_matrix16.training import objective
    from p2_clip_events.backend import data_config,selected_statistics
    from p2_clip_events.model import classes,mask_classes
    from p2_listen.config import extractor_config
    from p2_listen.numerics import configure_training_numerics
    from p2_listen.performance import configure_renderer
    from p2_listen.contracts import read,atomic_json,digest,sha
    from p2_j_duration_execution import Schedule
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train'])
    p.add_argument('--config',required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--steps',type=int,default=16);p.add_argument('--start-step',type=int,default=3)
    a=p.parse_args();c=load(a.config);validate_baseline(c,read(c['baseline_config']))
    if a.steps<4 or a.start_step<1:raise ValueError('Expected at least4 matched timing batches')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=32 or c['training']['global_batch']!=32:raise ValueError('Expected32 ranks/global32')
    torch.set_num_threads(2);numerics=configure_training_numerics();torch.cuda.set_device(local);device=f'cuda:{local}'
    if os.environ.get('P2_TRAIN_MEMORY_GIB'):
        torch.cuda.set_per_process_memory_fraction(float(os.environ['P2_TRAIN_MEMORY_GIB'])*2**30/torch.cuda.get_device_properties(local).total_memory,local)
    dist.init_process_group('nccl');a.work.mkdir(exist_ok=True,parents=True)
    def reduce(values,op=dist.ReduceOp.MAX):
        t=torch.tensor(values,device=device,dtype=torch.float64);dist.all_reduce(t,op=op);return t.tolist()
    def progress(**kw):
        if rank==0:atomic_json(a.work/'progress.json',dict(at=time.time(),**kw))
    try:
        progress(stage='reading_frozen_caches')
        dataset=Dataset(c);schedule=Schedule(c);cfg=extractor_config(data_config(c));names=classes(c['event_supervision'])
        stats=selected_statistics(c,dataset.labels.manifest);effective=[k for k in names if stats[k]['enabled']]
        pos={k:stats[k]['pos_weight'] for k in names};execution=settings(c)
        identity=dict(source=source_identity(),data=data_identity(c),experiment=c,numerics=numerics)
        ids=[None]*world;dist.all_gather_object(ids,digest(identity))
        if len(set(ids))!=1:raise ValueError('32 ranks disagree on sources/data/config')
        progress(stage='loading_renderer')
        from omni_av_p2.ltx import load_renderer
        renderer,strategy,sampler=load_renderer(cfg,device);configure_renderer(renderer,'all')
        torch.manual_seed(c['training']['seed']);model=QwenWhisperEventFeature(c,cfg,renderer).to(device);model.train()
        optimizer=torch.optim.AdamW(model.groups(),weight_decay=0.)
        progress(stage='longest_sample_full_update_check')
        canary=longest_check(model,dataset,c,optimizer,execution,strategy,sampler,pos,effective,names,device,local,rank,world)
        if rank==0:atomic_json(a.work/'LONGEST_SAMPLE_CHECK.json',canary)
        parameters=[p for p in model.parameters() if p.requires_grad]
        reference_state=tensor_digest(model.state_dict().items()) if rank==0 else None
        main_parameters=[p for g in optimizer.param_groups if g['name']!='events' for p in g['params']]
        reference_gradient=None;reference_losses=None;results=[]
        candidates=[('all256_serial',256,0),('all256_prefetch2',256,2),('all512_prefetch2',512,2)]
        for name,bucket,depth in candidates:
            progress(stage='benchmark',arm=name,offset=-2)
            optimizer.zero_grad(set_to_none=True);optimizer.state.clear();torch.cuda.empty_cache()
            optimizer=torch.optim.AdamW(model.groups(),lr=0.,weight_decay=0.)
            for group in optimizer.param_groups:group['lr']=0.
            actual=dict(execution,gradient_bucket_mb=bucket,prefetch_batches=depth)
            requests=((offset,schedule.indices(len(dataset.rows),a.start_step+max(0,offset),32,c['training']['seed'],rank,world))
                      for offset in range(-1,a.steps))
            times=[];losses=[];waits=[];reads=[];fingerprint=None;norms=None;failed=None
            torch.cuda.reset_peak_memory_stats(device)
            with CPUInputStream(dataset,requests,depth) as stream:
                for offset in range(-1,a.steps):
                    optimizer.zero_grad(set_to_none=True);dist.barrier();torch.cuda.synchronize(device);begin=time.monotonic()
                    wait_begin=time.monotonic();inputs=next(stream);wait=time.monotonic()-wait_begin
                    if inputs['step']!=offset:raise ValueError('Prefetch reordered the fixed benchmark')
                    samples=inputs['samples']
                    for value in samples:value['events']=mask_classes(value['events'],effective)
                    counts=_event_counts(samples,names,device,world)
                    error=None;value=None;loss=None;values=None
                    try:
                        value=to_device(samples[0],device);ordinal=inputs['chosen'][0][1]
                        torch.manual_seed(c['training']['seed']+ordinal)
                        generator=torch.Generator(device=device).manual_seed(c['training']['seed']+1000000+ordinal)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            values=model(value,pos,generator,strategy,sampler)
                            loss,base,event=objective(values,c,counts,1,world)
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite combined objective')
                        loss.backward()
                        local_losses=[float(values['av'].detach())/world,float(event.detach()),
                                      float(values['features']['video'].detach())/world,float(values['features']['audio'].detach())/world]
                    except torch.cuda.OutOfMemoryError as exc:error=str(exc)[:300]
                    if reduce([int(error is None)],dist.ReduceOp.MIN)[0]!=1:
                        failed=error or 'Peer rank OOM';del values,loss,value;optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();break
                    acc=accumulator(parameters,1,actual);acc.add();acc.finish(world)
                    torch.nn.utils.clip_grad_norm_(main_parameters,c['training']['clip_norm'],error_if_nonfinite=True)
                    torch.nn.utils.clip_grad_norm_(model.event_heads.parameters(),c['training']['clip_norm'],error_if_nonfinite=True)
                    optimizer.step();torch.cuda.synchronize(device)
                    elapsed=reduce([time.monotonic()-begin])[0]
                    if offset>=0:
                        times.append(elapsed);waits.append(reduce([wait])[0]);reads.append(reduce([inputs['read_seconds']])[0])
                        losses.append(reduce(local_losses,dist.ReduceOp.SUM))
                    if offset==0:
                        # The all-reduced gradient is identical across ranks;
                        # hash all of rank0's trainable gradients, not a sample.
                        fingerprint=tensor_digest((n,p.grad) for n,p in model.named_parameters()) if rank==0 else None
                        norms={g['name']:sum(float(p.grad.double().square().sum()) for p in g['params'] if p.grad is not None)**.5
                               for g in optimizer.param_groups}
                    progress(stage='benchmark',arm=name,offset=offset,seconds=elapsed)
                    del values,loss,base,event,value,samples,inputs,acc
            peak=reduce([torch.cuda.max_memory_reserved(device)/2**30])[0]
            row=dict(name=name,bucket_mb=bucket,prefetch_batches=depth,safe=failed is None and peak<=128,
                     peak_reserved_gib=peak,error=failed,seconds=times,input_wait_seconds=waits,input_read_seconds=reads)
            if times:row['mean_seconds']=sum(times)/len(times)
            if failed is None:
                if rank==0:
                    if reference_gradient is None:reference_gradient=fingerprint
                    if reference_losses is None:reference_losses=losses
                    row.update(gradient_bitwise_equal=fingerprint==reference_gradient,losses_bitwise_equal=losses==reference_losses,
                               fixed_weights_unchanged=tensor_digest(model.state_dict().items())==reference_state,losses=losses,grad_norms=norms)
                flags=[None];flags[0]=row if rank==0 else None;dist.broadcast_object_list(flags,src=0);row=flags[0]
            results.append(row)
            if rank==0:atomic_json(a.work/'partial_results.json',results)
            optimizer.zero_grad(set_to_none=True);optimizer.state.clear();torch.cuda.empty_cache()
            if name=='all256_serial' and not row['safe']:raise RuntimeError('Combined baseline unsafe; inspect canary/peak/error')
        eligible=[r for r in results if r['safe'] and r.get('gradient_bitwise_equal') and r.get('losses_bitwise_equal') and r.get('fixed_weights_unchanged')]
        selected=min(eligible,key=lambda r:r['mean_seconds']);baseline=results[0]
        report=dict(success=True,world=world,measured_steps=a.steps,matched_samples=a.steps*32,
                    results=results,selected=selected,relative_speedup=baseline['mean_seconds']/selected['mean_seconds'],
                    formal_updates=0,method='Same sample/noise ordinals and weights; zero LR Adam; warmup excluded; global32 and objective fixed',
                    identity=identity,script_sha256=sha(__file__),longest_sample=canary)
        if rank==0:atomic_json(a.work/'RESULT.json',report)
        progress(stage='complete',selected=selected['name'])
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
