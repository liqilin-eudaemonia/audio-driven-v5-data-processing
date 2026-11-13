"""32-rank, fixed-weight AV benchmark; never advances the formal training run."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    import torch
    import torch.distributed as dist
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_whisper_av.config import load,source_identity
    from p2_whisper_av.data import Dataset,data_identity
    from p2_whisper_av.model import QwenWhisperAV
    from p2_whisper_av.checkpoint import checked
    from p2_whisper_av.execution import StreamingFP64Reduction
    from p2_whisper_av.training import scaled_av
    from p2_listen.contracts import read,atomic_json,digest,sha
    from p2_listen.numerics import configure_training_numerics
    from p2_listen.config import extractor_config
    from p2_clip_events.backend import data_config
    from p2_listen.performance import configure_renderer
    from p2_j_duration_execution import Schedule
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train'])
    p.add_argument('--config',required=True);p.add_argument('--resume',required=True);p.add_argument('--work',required=True)
    p.add_argument('--check-inputs',action='store_true')
    a=p.parse_args();c=load(a.config);work=Path(a.work);work.mkdir(exist_ok=True,parents=True)
    cfg=extractor_config(data_config(c))
    from omni_av_p2.ltx import load_renderer
    if a.check_inputs:
        print('Benchmark bootstrap and renderer imports passed',flush=True);return
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert world==32 and c['training']['global_batch']==32
    torch.set_num_threads(2);numerics=configure_training_numerics();torch.cuda.set_device(local);device=f'cuda:{local}'
    torch.cuda.set_per_process_memory_fraction(134*2**30/torch.cuda.get_device_properties(local).total_memory,local)
    dist.init_process_group('nccl')
    try:
        identity=read(Path(c['output'])/'run.json')
        current=source_identity()
        changed={key for key in set(current)|set(identity['source']) if current.get(key)!=identity['source'].get(key)}
        # The only pre-probe source addition is administrative handoff validation.
        if (not changed<={str(ROOT/'src/p2_whisper_av/migration.py')}
                or identity['numerics']!=numerics or identity['data_identity']!=data_identity(c)):
            raise ValueError('Benchmark source, numerical policy or data differs from its parent')
        dataset=Dataset(c);schedule=Schedule(c)
        renderer,strategy,sampler=load_renderer(cfg,device);torch.manual_seed(c['training']['seed'])
        model=QwenWhisperAV(c,cfg,renderer).to(device);model.train()
        saved=checked(a.resume)
        if saved['identity']!=identity:raise ValueError('Benchmark checkpoint identity differs')
        model.load_state_dict(saved['model'],strict=True)
        optimizer=torch.optim.AdamW(model.groups(),weight_decay=0.)
        def reset_optimizer():
            # Adam updates its CPU step scalars in place; do not mutate mmap checkpoint tensors.
            state=dict(saved['optimizer']);state['state']={k:{n:(v.clone() if n=='step' and torch.is_tensor(v) else v)
                for n,v in value.items()} for k,value in state['state'].items()}
            optimizer.load_state_dict(state)
            for group in optimizer.param_groups:group['lr']=0.;assert group['weight_decay']==0
        reset_optimizer()
        with (Path(c['j_cache'])/'sample_list.csv').open() as stream:
            longest=max(enumerate(csv.DictReader(stream)),key=lambda pair:float(pair[1]['duration_s']))
        start=saved['step'];parameters=[v for v in model.parameters() if v.requires_grad]
        def reduce_values(values,op=dist.ReduceOp.MAX):
            values=torch.tensor(values,dtype=torch.float64,device=device);dist.all_reduce(values,op=op);return values.tolist()
        def forward_backward(index,ordinal):
            # An OOM returns after releasing the exception frame and activation graph.
            try:
                sample=dataset.sample(index,device);torch.manual_seed(c['training']['seed']+ordinal)
                with torch.autocast('cuda',dtype=torch.bfloat16):values=model(sample,strategy,sampler);loss=scaled_av(values['av'],1,world)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite benchmark loss')
                loss.backward();return float(values['av'].detach()),None
            except torch.cuda.OutOfMemoryError as exc:return None,str(exc)[:400]
        def grads_sha():
            h=hashlib.sha256()
            for parameter in parameters:
                if parameter.grad is None:h.update(b'NONE');continue
                value=parameter.grad.detach().cpu().contiguous()
                h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
            return h.hexdigest()
        def apply_update():
            try:
                torch.nn.utils.clip_grad_norm_(parameters,c['training']['clip_norm'],error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize(device);return None
            except torch.cuda.OutOfMemoryError as exc:return str(exc)[:400]
        reference=None;results=[]
        for name,policy,bucket in [('all32','all',32),('all256','all',256),('three_quarters256','three_quarters',256)]:
            optimizer.zero_grad(set_to_none=True);configure_renderer(renderer,policy);reset_optimizer();torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            _,error=forward_backward(longest[0],longest[0]);health=reduce_values([int(error is None)],dist.ReduceOp.MIN)[0]
            if health:
                reducer=StreamingFP64Reduction(parameters,bucket);reducer.add();reducer.finish(world)
                error=apply_update();health=reduce_values([int(error is None)],dist.ReduceOp.MIN)[0]
            peak=reduce_values([torch.cuda.max_memory_reserved(device)/2**30])[0]
            safe=bool(health and peak<=128)
            if not safe:
                results.append(dict(name=name,policy=policy,bucket_mb=bucket,safe=False,peak_gib=peak,oom=not bool(health)))
                optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();continue
            optimizer.zero_grad(set_to_none=True);reset_optimizer();times=[];losses=[];fingerprint=None;failed=False
            for offset in range(-1,16):
                step=start+1+max(0,offset);index,ordinal=schedule.indices(len(dataset.rows),step,32,c['training']['seed'],rank,world)[0]
                optimizer.zero_grad(set_to_none=True);dist.barrier();torch.cuda.synchronize(device);began=time.monotonic()
                loss,error=forward_backward(index,ordinal)
                if reduce_values([int(error is None)],dist.ReduceOp.MIN)[0]!=1:
                    failed=True;break
                reducer=StreamingFP64Reduction(parameters,bucket);reducer.add();reducer.finish(world)
                error=apply_update()
                if reduce_values([int(error is None)],dist.ReduceOp.MIN)[0]!=1:
                    failed=True;break
                elapsed=reduce_values([time.monotonic()-began])[0]
                if offset>=0:times.append(elapsed);losses.append(reduce_values([loss],dist.ReduceOp.SUM)[0]/world)
                if offset==0:
                    fingerprint=grads_sha()
                    if reference is None:reference=fingerprint
                if rank==0:atomic_json(work/'progress.json',dict(arm=name,offset=offset,parent_step=start))
            peak=reduce_values([torch.cuda.max_memory_reserved(device)/2**30])[0]
            if failed:
                results.append(dict(name=name,policy=policy,bucket_mb=bucket,safe=False,peak_gib=peak,oom=True))
                optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();continue
            same=bool(reduce_values([int(fingerprint==reference)],dist.ReduceOp.MIN)[0])
            results.append(dict(name=name,policy=policy,bucket_mb=bucket,safe=peak<=128,peak_gib=peak,
                mean_seconds=sum(times)/len(times),times=times,losses=losses,gradient_bitwise_equal=same))
            if rank==0:atomic_json(work/'partial_results.json',results)
        baseline=results[0]
        if not baseline['safe'] or not baseline['gradient_bitwise_equal']:raise ValueError('Baseline benchmark invalid')
        for row in results:
            if row['safe']:row['losses_bitwise_equal']=row['losses']==baseline['losses']
        eligible=[x for x in results if x['safe'] and x['gradient_bitwise_equal'] and x['losses_bitwise_equal']]
        selected=min(eligible,key=lambda x:x['mean_seconds'])
        report=dict(success=True,world=world,parent_checkpoint=a.resume,parent_step=start,results=results,selected=selected,
            time_reduction=1-selected['mean_seconds']/baseline['mean_seconds'],adopt=selected['mean_seconds']<=baseline['mean_seconds']*.95,
            formal_updates=0,method='Same 512 sample/noise positions, fixed weights, zero LR Adam, resident optimizer, warmup excluded',
            source_identity=digest(identity['source']),source_changes={k:dict(before=identity['source'].get(k),after=current.get(k)) for k in changed},
            benchmark_script_sha256=sha(__file__))
        if rank==0:atomic_json(work/'RESULT.json',report)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
