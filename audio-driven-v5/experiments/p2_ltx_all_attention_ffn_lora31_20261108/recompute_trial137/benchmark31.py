"""31-rank recomputation test: two longest real updates then matched ABBA timing."""
import argparse,csv,hashlib,os,sys,time
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
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
    from p2_ltx_all_lora.config import load,source_identity
    from p2_ltx_all_lora.data import Dataset,data_identity,to_device,data_config
    from p2_ltx_all_lora.model import Model as QwenWhisperEventFeature
    from p2_whisper_event_feature.execution import CPUInputStream
    from p2_ltx_all_lora.training import _event_counts
    from p2_whisper_av.execution import accumulator
    from p2_ltx_events.heads import objective,names as event_names
    from p2_clip_events.model import classes,mask_classes
    from p2_listen.config import extractor_config
    from p2_listen.numerics import configure_training_numerics
    from p2_listen.performance import configure_renderer
    from p2_listen.contracts import read,atomic_json,digest,sha
    from p2_ltx_all_lora.schedule import Schedule
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train'])
    p.add_argument('--config',required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--steps',type=int,default=8);p.add_argument('--start-step',type=int,default=3)
    a=p.parse_args();c=load(a.config)
    if a.steps<4 or a.start_step<1:raise ValueError('Expected at least4 matched timing batches')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=31 or c['training']['global_batch']!=31:raise ValueError('Expected30 ranks/global31')
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
        dataset=Dataset(c);schedule=Schedule(c);cfg=extractor_config(data_config(c));names=event_names(c)
        stats=read(Path(c['output'])/'preflight.json')['statistics'];effective=[k for k in names if stats[k]['enabled']]
        pos={k:stats[k]['pos_weight'] for k in names};execution=c['execution']
        identity=dict(checkpoint=dict(path=str(a.checkpoint.resolve()),receipt=read(a.checkpoint.with_suffix('.complete.json'))),trial_sha256=sha(__file__),policy_support_sha256=sha(ROOT/'src/p2_listen/performance.py'),source=source_identity(),data=data_identity(c),experiment=c,numerics=numerics)
        ids=[None]*world;dist.all_gather_object(ids,digest(identity))
        if len(set(ids))!=1:raise ValueError('32 ranks disagree on sources/data/config')
        progress(stage='loading_renderer')
        from omni_av_p2.ltx import load_renderer
        renderer,strategy,sampler=load_renderer(cfg,device);configure_renderer(renderer,'all')
        torch.manual_seed(c['training']['seed']);model=QwenWhisperEventFeature(c,cfg,renderer).to(device);model.train()
        optimizer=torch.optim.AdamW(model.groups(),weight_decay=0.)
        progress(stage='longest_sample_full_update_check')
        canary=dict(method='Two real longest-sample Adam updates per policy; 137GiB process limit')
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
        if saved['identity']!=read(Path(c['output'])/'run.json'):raise ValueError('Checkpoint/run identity differs')
        model.load_state_dict(saved['model'],strict=True)
        if rank==0:atomic_json(a.work/'LONGEST_SAMPLE_CHECK.json',canary)
        parameters=[p for p in model.parameters() if p.requires_grad]
        reference_state=tensor_digest(model.state_dict().items()) if rank==0 else None
        main_parameters=[p for g in optimizer.param_groups if g['name']!='events' for p in g['params']]
        reference_gradient=None;reference_losses=None;results=[]
        candidates=[('baseline',512,2),('three_quarters',512,2),('three_quarters_repeat',512,2),('baseline_repeat',512,2)]
        with Path(c['sample_list']).open() as f:longest_index,longest_row=max(enumerate(csv.DictReader(f)),key=lambda x:float(x[1]['duration_s']))
        reference_update=None;reference_optimizer=None;reference_diagnostics=None
        for name,bucket,depth in candidates:
            if name=='three_quarters_repeat' and any(r['name']=='three_quarters' and not r['safe'] for r in results):
                continue
            progress(stage='benchmark',arm=name,offset=-3)
            configure_renderer(renderer,'three_quarters' if name.startswith('three_quarters') else 'all')
            optimizer.zero_grad(set_to_none=True);optimizer.state.clear();torch.cuda.empty_cache()
            model.load_state_dict(saved['model'],strict=True)
            optimizer=torch.optim.AdamW(model.groups(),lr=0.,weight_decay=0.)
            optimizer.load_state_dict(dict(saved['optimizer'],state={k:{n:(v.clone() if n=='step' and torch.is_tensor(v) else v) for n,v in state.items()} for k,state in saved['optimizer']['state'].items()}))
            for group in optimizer.param_groups:group['lr']=0.
            actual=dict(execution,gradient_bucket_mb=bucket,prefetch_batches=depth)
            requests=((offset,([(longest_index,10000000+offset)] if offset<-1 else schedule.indices(len(dataset.rows),a.start_step+max(0,offset),31,c['training']['seed'],rank,world)))
                      for offset in range(-3,a.steps))
            canary_peak=None;canary_updates=0
            times=[];losses=[];diagnostics=[];waits=[];reads=[];fingerprint=None;norms=None;failed=None
            torch.cuda.reset_peak_memory_stats(device)
            with CPUInputStream(dataset,requests,depth) as stream:
                for offset in range(-3,a.steps):
                    optimizer.zero_grad(set_to_none=True);dist.barrier();torch.cuda.synchronize(device);begin=time.monotonic()
                    wait_begin=time.monotonic();inputs=next(stream);wait=time.monotonic()-wait_begin
                    if inputs['step']!=offset:raise ValueError('Prefetch reordered the fixed benchmark')
                    samples=inputs['samples']
                    for value in samples:value['events']=mask_classes(value['events'],effective)
                    counts=_event_counts(samples,names,device,world)
                    error=None;value=None;loss=None;values=None;base=None;event=None;acc=None
                    try:
                        acc=accumulator(parameters,1,actual)
                        value=to_device(samples[0],device);ordinal=inputs['chosen'][0][1]
                        torch.manual_seed(c['training']['seed']+ordinal)
                        generator=torch.Generator(device=device).manual_seed(c['training']['seed']+1000000+ordinal)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            values=model(value,pos,generator,strategy,sampler)
                            loss,base,event=objective(values,c,counts,1,world)
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite combined objective')
                        loss.backward()
                        local_losses=[float(values['av'].detach())/world,float(event.detach()),
                                      float(values['features'].get('video',torch.tensor(0.)).detach())/world,float(values['features'].get('audio',torch.tensor(0.)).detach())/world]
                    except torch.cuda.OutOfMemoryError as exc:error=str(exc)[:300]
                    if reduce([int(error is None)],dist.ReduceOp.MIN)[0]!=1:
                        failed=error or 'Peer rank OOM';values=loss=value=base=event=acc=None;optimizer.zero_grad(set_to_none=True);__import__('gc').collect();torch.cuda.empty_cache();break
                    acc.add();acc.finish(world)
                    diagnostic={g['name']:sum(float(p.grad.double().square().sum()) for p in g['params'] if p.grad is not None)**.5 for g in optimizer.param_groups}
                    torch.nn.utils.clip_grad_norm_(main_parameters,c['training']['clip_norm'],error_if_nonfinite=True)
                    torch.nn.utils.clip_grad_norm_(model.event_heads.parameters(),c['training']['clip_norm'],error_if_nonfinite=True)
                    for g in optimizer.param_groups:g['lr']=next(x['lr'] for x in model.groups() if x['name']==g['name']) if offset<-1 or offset==a.steps-1 else 0.
                    optimizer.step();torch.cuda.synchronize(device)
                    if offset<-1:canary_updates+=1
                    elapsed=reduce([time.monotonic()-begin])[0]
                    if offset>=0:
                        diagnostics.append(diagnostic);times.append(elapsed);waits.append(reduce([wait])[0]);reads.append(reduce([inputs['read_seconds']])[0])
                        losses.append(reduce(local_losses,dist.ReduceOp.SUM))
                    if offset==0:
                        # The all-reduced gradient is identical across ranks;
                        # hash all of rank0's trainable gradients, not a sample.
                        fingerprint=tensor_digest((n,p.grad) for n,p in model.named_parameters()) if rank==0 else None
                        norms={g['name']:sum(float(p.grad.double().square().sum()) for p in g['params'] if p.grad is not None)**.5
                               for g in optimizer.param_groups}
                    progress(stage='benchmark',arm=name,offset=offset,seconds=elapsed)
                    if offset==-2:
                        canary_peak=reduce([torch.cuda.max_memory_reserved(device)/2**30])[0]
                        if canary_peak>137:
                            failed='Longest two-update peak exceeds137GiB';values=loss=base=event=value=None;optimizer.zero_grad(set_to_none=True);break
                    del values,loss,base,event,value,samples,inputs,acc
            peak=reduce([torch.cuda.max_memory_reserved(device)/2**30])[0]
            row=dict(name=name,gradient_checkpointing='three_quarters' if name.startswith('three_quarters') else 'all',bucket_mb=bucket,prefetch_batches=depth,safe=failed is None and peak<=137,
                     peak_reserved_gib=peak,error=failed,longest_sample_id=longest_row['sample_id'],longest_duration_s=float(longest_row['duration_s']),longest_peak_gib=canary_peak,longest_real_updates=canary_updates,seconds=times,input_wait_seconds=waits,input_read_seconds=reads)
            if times:row['mean_seconds']=sum(times)/len(times)
            if failed is None:
                if rank==0:
                    if reference_gradient is None:reference_gradient=fingerprint
                    if reference_losses is None:reference_losses=losses
                    updated=tensor_digest(model.state_dict().items())
                    if reference_update is None:reference_update=updated
                    optimizer_hash=tensor_digest((str(k)+'/'+n,v) for k,state in optimizer.state_dict()['state'].items() for n,v in state.items() if torch.is_tensor(v))
                    if reference_optimizer is None:reference_optimizer=optimizer_hash
                    if reference_diagnostics is None:reference_diagnostics=diagnostics
                    row.update(optimizer_bitwise_equal=optimizer_hash==reference_optimizer,diagnostics_bitwise_equal=diagnostics==reference_diagnostics,update_bitwise_equal=updated==reference_update,gradient_bitwise_equal=fingerprint==reference_gradient,losses_bitwise_equal=losses==reference_losses,
                               fixed_weights_unchanged=False,losses=losses,grad_norms=norms)
                flags=[None];flags[0]=row if rank==0 else None;dist.broadcast_object_list(flags,src=0);row=flags[0]
            results.append(row)
            if rank==0:atomic_json(a.work/'partial_results.json',results)
            optimizer.zero_grad(set_to_none=True);optimizer.state.clear();torch.cuda.empty_cache()
            if name=='baseline' and not row['safe']:raise RuntimeError('Combined baseline unsafe; inspect canary/peak/error')
        eligible=[r for r in results if r['safe'] and r.get('gradient_bitwise_equal') and r.get('losses_bitwise_equal') and r.get('update_bitwise_equal') and r.get('optimizer_bitwise_equal') and r.get('diagnostics_bitwise_equal')]
        selected=min(eligible,key=lambda r:r['mean_seconds']);baseline=results[0]
        report=dict(success=True,world=world,measured_steps=a.steps,matched_samples=a.steps*31,
                    results=results,selected=selected,relative_speedup=baseline['mean_seconds']/selected['mean_seconds'],
                    formal_updates=0,method='Same saved model/Adam; two actual longest-sample updates, one warmup, matched timing with last actual update; diagnostic included; global31; no formal updates',
                    identity=identity,script_sha256=sha(__file__),longest_sample=canary)
        if rank==0:atomic_json(a.work/'RESULT.json',report)
        progress(stage='complete',selected=selected['name'])
    finally:dist.destroy_process_group()


def node_main():
    import subprocess
    from p2_listen.contracts import read
    p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--config',required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--node-index',type=int,required=True);p.add_argument('--start-step',type=int,required=True);p.add_argument('--steps',type=int,default=8)
    a=p.parse_args();c=read(a.config);cluster=read(Path(c['output'])/'cluster.json');n=cluster['nodes'][a.node_index];off=sum(len(x['gpus']) for x in cluster['nodes'][:a.node_index]);entry=dict(gpus=n['gpus'],ranks=list(range(off,off+len(n['gpus']))));processes=[];streams=[]
    assert [x['name'] for x in cluster['nodes']]==['node12','node8','node4','node11']
    assert [x['gpus'] for x in cluster['nodes']]==[list(range(8))]*3+[list(range(1,8))]
    a.work.mkdir(parents=True,exist_ok=True)
    try:
        for local,rank in enumerate(entry['ranks']):
            env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(local),WORLD_SIZE='31',LOCAL_WORLD_SIZE=str(len(entry['gpus'])),MASTER_ADDR=cluster['nodes'][0]['host'],MASTER_PORT='29947')
            log=(a.work/f'rank{rank:02d}.log').open('a');streams.append(log)
            processes.append(subprocess.Popen([cluster['python'],__file__,'train','--config',a.config,'--work',str(a.work),'--checkpoint',a.checkpoint,'--start-step',str(a.start_step),'--steps',str(a.steps)],env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Benchmark rank failed')
            time.sleep(2)
        if any(p.returncode for p in processes):raise RuntimeError('Benchmark failed')
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in streams:f.close()


if __name__=='__main__':
    if sys.argv[1]=='node':node_main()
    else:main()

