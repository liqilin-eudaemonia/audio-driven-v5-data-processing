"""Production P/C0/A DDP with rank RNG, immutable data seal and strict resume."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sample_indices(size, seed, step, batch, world, rank):
    """One deterministic global permutation, then disjoint rank slices per update."""
    cache={};result=[]
    for micro in range(batch//world):
        ordinal=(step-1)*batch+micro*world+rank
        epoch,offset=divmod(ordinal,size)
        if epoch not in cache:
            seq=list(range(size));random.Random(seed+epoch).shuffle(seq);cache[epoch]=seq
        result.append((cache[epoch][offset],ordinal))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/p3_listen_300h.json')
    p.add_argument('--stage',choices=('P','C0','A'),default='P')
    p.add_argument('--output',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--s0',action='store_true');p.add_argument('--steps',type=int)
    p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    local=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    visible=os.environ.get('CUDA_VISIBLE_DEVICES',','.join(map(str,range(8)))).split(',')
    os.environ['CUDA_VISIBLE_DEVICES']=visible[local]
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from p3_listen.common import config,resolve,read,write,write_once,digest,sha,source_contract
    from p3_listen.training import build,identity,checked,restore_model
    from p3_listen.features import cpu,load_native,load_av,load_text
    cfg=config(a.config);sys.path.insert(1,str(resolve(cfg['reuse_root'])/'src'))
    from omni_av.ltx_runtime import with_text
    from omni_av_p2.checkpoint import capture_rng,restore_rng,atomic_save
    torch.set_num_threads(2);torch.cuda.set_device(0)
    dist.init_process_group('nccl',device_id=torch.device('cuda:0'),timeout=timedelta(minutes=15))
    try:
        root=resolve(cfg['output']);manifest=read(root/'windows.json')
        rows=[r for r in manifest['windows'] if r['split']=='train']
        dev=[r for r in manifest['windows'] if r['split']=='dev']
        if a.s0:
            ids=read(root/'s0_selection.json')['train'];lookup={r['sample_id']:r for r in rows}
            rows=[lookup[s] for s in ids]
        else:
            if world!=24:raise ValueError('Production requires the configured 24 ranks')
            ready=read(root/'data_ready.json')
            if ready['manifest_sha256']!=sha(root/'windows.json') or ready['config_sha256']!=digest(cfg):
                raise ValueError('Data seal mismatch')
            if manifest.get('unique_seconds',{}).get('train',0)<300*3600:
                raise ValueError('Fewer than 300 unique training hours')
            if a.steps is not None:raise ValueError('Production step budget belongs to config')
        t=cfg['training'];batch=t['batch']
        if a.s0 and batch<world:batch=world
        if batch%world:raise ValueError('Global batch must be divisible by world size')
        steps=(a.steps or 2) if a.s0 else min(t[a.stage]['steps'],math.ceil(len(rows)*t[a.stage]['epochs']/batch))
        if a.checkpoint and a.resume:raise ValueError('Choose transition or resume')
        saved=checked(a.resume or a.checkpoint) if a.resume or a.checkpoint else None
        if a.stage!='P' and saved is None:raise ValueError('Previous stage required')
        if a.stage=='P' and a.checkpoint:raise ValueError('P must start from base weights')
        initialization=saved['identity']['initialization'] if a.resume else sha(a.checkpoint) if a.checkpoint else None
        ident=identity(cfg,manifest,rows,a.stage,steps,a.s0,False,initialization)
        ident.update(world_size=world,global_batch=batch,distributed_script_sha256=sha(__file__))
        signature=digest(ident);signatures=[None]*world;dist.all_gather_object(signatures,signature)
        if len(set(signatures))!=1:raise ValueError('Nodes disagree on data, source or configuration')
        if saved:
            for key in ('config','manifest','prompts','native','renderer','code'):
                if saved['identity'][key]!=ident[key]:raise ValueError('Checkpoint mismatch: '+key)
            if saved['identity']['s0'] and not a.s0:raise ValueError('S0 cannot initialize formal training')
            if saved['identity']['oracle_fit']:raise ValueError('Oracle cannot initialize formal training')
            if a.resume and saved['identity']!=ident:raise ValueError('Resume identity differs')
        out=a.output or root/'runs'/a.stage
        if rank==0:
            if (out/'run.json').exists() and not a.resume:raise FileExistsError('Run exists; use resume')
            write_once(out/'run.json',ident)
        dist.barrier()
        # S0 has no full-corpus seal; check its small selection explicitly.
        if a.s0:
            for row in rows[rank::world]:
                load_native(root,row,cfg,targets=True)
                if a.stage!='P':load_av(root,row,cfg,'cpu')
        model,strategy,sampler=build(cfg,a.stage,steps)
        if saved:restore_model(model,saved,transition=not bool(a.resume))
        optimizer=torch.optim.AdamW(model.optimizer_groups(t['connector_lr'],t['dit_lr'],t['planner_lr'],t['heads_lr']),weight_decay=0.)
        start=0
        if a.resume:
            optimizer.load_state_dict(saved['optimizer']);start=saved['step']
        text=None if a.stage=='P' else load_text(root,cfg,'cuda:0')
        ddp=DDP(model,device_ids=[0],broadcast_buffers=False,find_unused_parameters=True)
        if a.resume:restore_rng(saved['rank_rng'][rank])
        del saved
        model.train()
        frozen=[(x,x._version) for m in (model.planner,model.renderer) for x in m.parameters() if not x.requires_grad]
        base_lr={g['name']:t[{'omni_lora':'planner_lr','feature_heads':'heads_lr','connector':'connector_lr','dit_text_lora':'dit_lr'}[g['name']]] for g in optimizer.param_groups}
        warmup=max(1,math.ceil(steps*t['warmup_fraction']));last=start
        last_checkpoint=start if a.resume else -1
        def parameter_hash():
            h=hashlib.sha256()
            for name,value in model.named_parameters():
                if value.requires_grad:
                    h.update(name.encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            hashes=[None]*world;dist.all_gather_object(hashes,h.hexdigest())
            if len(set(hashes))!=1:raise RuntimeError('DDP parameters diverged')
            return hashes[0]
        def checkpoint(step):
            nonlocal last_checkpoint
            path=out/f'step_{step:06d}.pt'
            if last_checkpoint==step:return path
            states=[None]*world;dist.all_gather_object(states,capture_rng())
            checksum=parameter_hash()
            if rank==0:
                atomic_save(path,dict(stage=model.stage,step=step,identity=ident,planner_names=model.planner_names,
                    model=cpu(model.state_dict()),optimizer=optimizer.state_dict(),rng=states[0],rank_rng=states))
                write(path.with_suffix('.complete.json'),dict(step=step,sha256=sha(path),world_size=world,parameters_sha256=checksum))
                write(out/'latest.json',dict(step=step,checkpoint=str(path)))
            dist.barrier();last_checkpoint=step;return path
        @torch.no_grad()
        def evaluate(step):
            state=capture_rng();model.eval();records=[]
            try:
                limit=t.get('dev_limit',240)
                for i,row in list(enumerate(dev[:limit]))[rank::world]:
                    seed=cfg['seed']+100000+i;torch.manual_seed(seed)
                    layout,target=load_native(root,row,cfg,True,'cuda:0')
                    prepared=None if a.stage=='P' else strategy.prepare_training_inputs(with_text(load_av(root,row,cfg,'cuda:0'),text),sampler)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        values=model(layout,target,prepared,seed,step,layout.valid.clone())
                    if not all(torch.isfinite(v).all() for v in values.values()):raise FloatingPointError('Nonfinite dev loss')
                    records.append(dict(sample_id=row['sample_id'],loss={k:float(v) for k,v in values.items()}))
                gathered=[None]*world;dist.all_gather_object(gathered,records)
                if rank==0:
                    flat=[r for chunk in gathered for r in chunk]
                    write(out/f'dev_{step:06d}.json',dict(rows=flat,note='Fixed development subset, all-mask numerical diagnostic; not generation quality'))
            finally:restore_rng(state);model.train()
        log=(out/'train.jsonl').open('a') if rank==0 else None
        try:
            for step in range(start+1,steps+1):
                stop=torch.tensor(int((out/'STOP').exists() or a.stop_after is not None and step>a.stop_after),device='cuda')
                dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if stop.item():break
                begin=time.monotonic();optimizer.zero_grad(set_to_none=True);sums={}
                for group in optimizer.param_groups:
                    factor=min(1.,step/warmup)
                    if a.stage=='A' and group['name'] in ('omni_lora','feature_heads'):
                        factor=max(0.,min(1.,(step-model.config.planner_warmup_steps)/warmup))
                    group['lr']=base_lr[group['name']]*factor
                schedule=sample_indices(len(rows),cfg['seed'],step,batch,world,rank)
                for micro,(index,ordinal) in enumerate(schedule):
                    row=rows[index];seed=cfg['seed']+ordinal;torch.manual_seed(seed)
                    layout,target=load_native(root,row,cfg,True,'cuda:0')
                    prepared=None if a.stage=='P' else strategy.prepare_training_inputs(with_text(load_av(root,row,cfg,'cuda:0'),text),sampler)
                    with (ddp.no_sync() if micro+1<len(schedule) else nullcontext()):
                        with torch.autocast('cuda',dtype=torch.bfloat16):values=ddp(layout,target,prepared,seed,step)
                        if not all(torch.isfinite(v).all() for v in values.values()):raise FloatingPointError('Nonfinite loss')
                        (values['loss']/len(schedule)).backward()
                    for k,v in values.items():sums[k]=sums.get(k,0.)+float(v.detach())/len(schedule)
                    del layout,target,prepared,values
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),t['grad_clip'],error_if_nonfinite=True)
                optimizer.step();model.assert_ownership()
                if any(x.grad is not None or x._version!=v for x,v in frozen):raise RuntimeError('Frozen parameters changed')
                reduced=torch.tensor(list(sums.values()),device='cuda');dist.all_reduce(reduced)
                elapsed=torch.tensor(time.monotonic()-begin,device='cuda');dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
                if rank==0:
                    record=dict(step=step,loss=dict(zip(sums,(reduced/world).tolist())),grad_norm=float(norm),
                        seconds=float(elapsed),world_size=world,global_batch=batch,examples_seen=step*batch,
                        effective_epochs=step*batch/len(rows),peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
                    log.write(json.dumps(record)+'\n');log.flush();write(out/'progress.json',record);print(record,flush=True)
                last=step
                if step%t['save_every']==0 or step==steps or step==a.stop_after:checkpoint(step)
                if not a.s0 and step%t['validate_every']==0:evaluate(step)
            final=checkpoint(last)
            if rank==0:write(out/'status.json',dict(step=last,target_steps=steps,stopped=last<steps,checkpoint=str(final),world_size=world,frozen_unchanged=True,s0=a.s0))
        finally:
            if log:log.close()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
