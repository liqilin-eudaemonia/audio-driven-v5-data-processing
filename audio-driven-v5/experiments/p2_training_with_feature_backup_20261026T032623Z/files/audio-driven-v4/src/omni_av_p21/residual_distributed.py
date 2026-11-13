"""One stage per torchrun transaction: fresh DDP groups at each explicit transition."""
from contextlib import nullcontext
from datetime import timedelta
from collections import OrderedDict
import json
import os
import random
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from omni_av.ltx_runtime import with_text
from omni_av_p2.checkpoint import atomic_json_new
from omni_av_p2.data import contract, read_sample
from omni_av_p2.ltx import load_renderer
from omni_av_p2.launch import agree, establish_output, source_signature
from omni_av_p2.model import Extractor
from omni_av_p2.performance import TrainingSamples, global_sample_index, sample_costs, balanced_sample_indices
from omni_av_p2.validation import check_indexes, distributed_mean
from .residual_config import read_config, STAGES
from .residual_model import ResidualTrainingModel
from . import residual_checkpoint as ckpt


class FeatureSamples:
    """Stage1 never reads target VAE latents or loads the LTX model."""
    def __init__(self, entries, cfg, device, capacity=2, split='train'):
        self.entries, self.cfg, self.device = entries, cfg, device
        self.capacity, self.split, self.cache = capacity, split, OrderedDict()

    def __getitem__(self, ordinal):
        i = ordinal % len(self.entries)
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        condition, targets, text, payload = read_sample(self.entries[i], self.cfg, self.device, self.split)
        result = condition, targets, text, payload, None
        if self.capacity:
            self.cache[i] = result
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return result


def grad_norm(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


@torch.no_grad()
def evaluate(raw, strategy, sampler, samples, rank, world, device, seed):
    py_rng, np_rng = random.getstate(), np.random.get_state()
    mode = raw.training
    sums, count = [0.] * 6, 0
    try:
        raw.eval()
        with torch.random.fork_rng(devices=[device.index]):
            for i in range(rank, len(samples.entries), world):
                random.seed(seed + i); np.random.seed(seed + i); torch.manual_seed(seed + i)
                condition, targets, text, _, av = samples[i]
                generator = torch.Generator(device=device).manual_seed(seed + i + 100000)
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    prepared = strategy.prepare_training_inputs(with_text(av, text), sampler) if raw.stage > 1 else None
                    values = raw(condition, targets, prepared, generator)
                if not all(torch.isfinite(x) for x in values):
                    raise FloatingPointError('Nonfinite validation loss')
                sums = [s + float(x) for s, x in zip(sums, values)]; count += 1
    finally:
        raw.train(mode); random.setstate(py_rng); np.random.set_state(np_rng)
    means, count = distributed_mean(sums, count, device)
    result = dict(zip(('loss', 'av', 'video_feature', 'audio_feature', 'av_video', 'av_audio'), means),
                  samples=count, stage=raw.stage, av_active=raw.stage > 1, seed=seed)
    if raw.stage == 1:
        for key in ('av', 'av_video', 'av_audio'): result[key] = None
    return result


def run(args):
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device); torch.set_num_threads(4)
    allocator=torch.cuda.memory._snapshot()['allocator_settings']
    if not allocator['expandable_segments']:
        raise RuntimeError('This PyTorch requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True')
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=15))
    ready = False
    try:
        establish_output(args.output, rank); ready = True
        if args.checkpoint_dir and args.checkpoint_dir != args.output:
            establish_output(args.checkpoint_dir, rank)
        result = train(args, rank, world, device)
        if rank == 0: atomic_json_new(args.output/'completed.json', result)
    except Exception:
        if ready: atomic_json_new(args.output/f'failed_rank{rank}.json', dict(traceback=traceback.format_exc()))
        raise
    finally:
        dist.destroy_process_group()


def train(args, rank, world, device):
    cfg, lora, residual = read_config(args.config)
    index = json.loads(args.cache.read_text())
    dev = json.loads(args.validation_cache.read_text()) if args.validation_cache else None
    check_indexes(index, dev, contract(cfg))
    identity = dict(contract=contract(cfg), cache_sha256=ckpt.sha(args.cache), cache=str(args.cache.resolve()),
        world=world, accumulation=args.accumulation, seed=args.seed, tiny=args.tiny,
        balance_workload=args.balance_workload, stage=args.stage,
        optimizer_sharded=residual.shard_optimizer and world > 1,
        gradient_communication=args.gradient_communication,
        branch_warmup_steps=args.branch_warmup_steps,
        validation_cache_sha256=ckpt.sha(args.validation_cache) if args.validation_cache else None,
        validation_seed=args.validation_seed)
    # Reject incompatible stages/checkpoints BEFORE allocating the 22B base or new branches.
    source_path = args.resume or args.init_previous
    source_bundle = ckpt.read(source_path) if source_path else None
    identity['stage_begin_step']=(source_bundle[0]['identity'].get('stage_begin_step',0) if args.resume
                                  else source_bundle[0]['step']) if source_bundle else 0
    agree(dict(identity=identity, source=source_signature(), config=cfg.to_dict(), lora=lora.to_dict(),
               residual=residual.to_dict(), args={k:str(v) for k,v in vars(args).items() if k not in ('worker','node_rank','gpus')}))
    if source_bundle is not None:
        saved, marker = source_bundle
        expected_stage = args.stage if args.resume else args.stage - 1
        if saved.get('deployment_only') or saved['stage'] != expected_stage:
            raise ValueError('Invalid checkpoint stage/deployment for training')
        for key, value in [('model',cfg.to_dict()),('lora',lora.to_dict()),('residual',residual.to_dict())]:
            if saved['architecture'][key] != value: raise ValueError('Checkpoint configuration changed')
        if args.resume and (saved['identity'] != identity or marker['world'] != world):
            raise ValueError('Strict resume topology or data changed')
        if any(saved['identity'][key] != identity[key] for key in ('cache_sha256','contract')):
            raise ValueError('Stage initialization data changed')
    extractor = Extractor(cfg).to(device)
    for name in ('video_decoder', 'audio_decoder'):
        getattr(extractor, name).net.grad_checkpointing = args.decoder_checkpointing
    renderer = strategy = sampler = None
    if args.stage > 1:
        renderer, strategy, sampler = load_renderer(cfg, device=device)
        from .residual_performance import configure_grouped_checkpointing
        configure_grouped_checkpointing(renderer,args.checkpoint_group_size)
    raw = ResidualTrainingModel(cfg, lora, residual, args.stage, renderer, strategy, extractor=extractor).to(device)
    if identity['optimizer_sharded']:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(raw.optimizer_groups(), optimizer_class=torch.optim.AdamW,
                                            lr=cfg.learning_rate, foreach=False)
    else:
        optimizer = torch.optim.AdamW(raw.optimizer_groups(), foreach=False)
    start = 0; initialization = 'fresh_stage1'
    if args.init_previous:
        saved = ckpt.initialize_previous(args.init_previous, raw, identity, saved_bundle=source_bundle)
        start = saved['step']; initialization = 'previous_stage_weights_fresh_optimizer'; del saved
    if args.resume:
        saved = ckpt.resume(args.resume, raw, optimizer, identity, rank, world, saved_bundle=source_bundle)
        start = saved['step']; initialization = 'strict_resume'; del saved
    del source_bundle
    raw.train()
    base_rates={group['name']:group['lr'] for group in raw.optimizer_groups()}
    model = DistributedDataParallel(raw, device_ids=[device.index], broadcast_buffers=False,
                                     gradient_as_bucket_view=True) if world > 1 else raw
    if world>1 and args.gradient_communication=='bf16':
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook
        model.register_comm_hook(state=None,hook=bf16_compress_hook)
    frozen = [(p, p._version) for p in renderer.parameters() if not p.requires_grad] if renderer else []
    def samples(entries, split):
        if args.stage == 1: return FeatureSamples(entries, cfg, device, args.cache_samples, split)
        return TrainingSamples(entries, cfg, args.data_root, device, args.tiny,
                               args.allow_terminal_trim, args.cache_samples, split)
    data = samples(index['entries'], 'train'); validation = samples(dev['entries'], 'dev') if dev else None
    costs = sample_costs(index['entries'], args.data_root) if args.balance_workload and args.stage > 1 else None
    checkpoints = args.checkpoint_dir or args.output
    if rank == 0:
        atomic_json_new(args.output/'run.json', dict(identity=identity, architecture=raw.architecture(),
            initialization=initialization, source_step=start, global_batch=world*args.accumulation,
            stage_name=STAGES[args.stage], train_samples=len(index['entries']),
            parameters={group['name']:sum(p.numel() for p in group['params']) for group in raw.optimizer_groups()},
            performance=dict(checkpoint_group_size=args.checkpoint_group_size,checkpoint_token_threshold=32768,
                decoder_checkpointing=args.decoder_checkpointing,cache_samples=args.cache_samples,
                allocator=torch.cuda.memory._snapshot()['allocator_settings']),
            original_dit_frozen=True, qwen_full_trainable=args.stage>1, ffn_frozen=True))
    writer = None
    if rank == 0 and args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(args.output/'tensorboard'))
    def validate(step):
        if validation is None: return
        result = evaluate(raw, strategy, sampler, validation, rank, world, device, args.validation_seed)
        if rank == 0:
            atomic_json_new(args.output/f'dev_step_{step:06d}.json',result)
            if writer:
                for k,v in result.items():
                    if k in ('loss','av','video_feature','audio_feature','av_video','av_audio') and v is not None:
                        writer.add_scalar('dev/'+k,v,step)
    validate(start)
    last = saved_step = start; durations = []; dist.barrier(); started = time.time()
    try:
        with (args.output/f'train_rank{rank}.jsonl').open('x') as log:
            for step in range(start+1, start+args.steps+1):
                stop = torch.tensor(int((args.output/'STOP').exists()),device=device)
                dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if stop.item(): break
                from .residual_performance import branch_learning_rate
                for group in optimizer.param_groups:
                    group['lr']=branch_learning_rate(base_rates[group['name']],group['name'],step,
                        identity['stage_begin_step'],args.branch_warmup_steps)
                # Keep DDP bucket gradient views across accumulated updates.
                torch.cuda.synchronize(); tick=time.time(); optimizer.zero_grad(set_to_none=False)
                totals=torch.zeros(6,device=device,dtype=torch.float64)
                schedule=balanced_sample_indices(step,world,args.accumulation,costs)[rank] if costs else None
                ids=[];timing=[];events=[]
                for micro in range(args.accumulation):
                    ordinal=schedule[micro] if schedule else global_sample_index(step,micro,rank,world,args.accumulation)
                    load_start=time.perf_counter()
                    condition,targets,text,payload,av=data[ordinal]; seed=args.seed+ordinal+1
                    item=dict(load_seconds=time.perf_counter()-load_start,sample_id=payload['sample_id'])
                    progress=args.output/f'progress_rank{rank}.json'
                    temporary=progress.with_suffix('.tmp')
                    temporary.write_text(json.dumps(dict(step=step,micro=micro,sample_id=payload['sample_id'],time=time.time())))
                    temporary.replace(progress)
                    torch.manual_seed(seed); generator=torch.Generator(device=device).manual_seed(seed+100000)
                    sync=model.no_sync() if world>1 and micro+1<args.accumulation else nullcontext()
                    clocks=[torch.cuda.Event(enable_timing=True) for _ in range(3)];clocks[0].record()
                    with sync:
                        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                            prepared=strategy.prepare_training_inputs(with_text(av,text),sampler) if args.stage>1 else None
                            values=model(condition,targets,prepared,generator)
                        if not all(torch.isfinite(v) for v in values): raise FloatingPointError('Nonfinite loss')
                        clocks[1].record()
                        (values[0]/args.accumulation).backward()
                        clocks[2].record()
                    timing.append(item);events.append(clocks)
                    totals+=torch.stack([v.detach() for v in values]).double()/args.accumulation
                    ids.append(payload['sample_id']); del prepared, values
                raw.assert_parameter_ownership()
                gradients={name:grad_norm(module.parameters()) for name,module in [('connector',raw.extractor.connector),('qwen',raw.qwen),('lora',raw.adapters)]}
                norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                if any(p.requires_grad or p.grad is not None or p._version!=v for p,v in frozen):
                    raise RuntimeError('Original frozen DiT weights changed')
                torch.cuda.synchronize(); duration=torch.tensor(time.time()-tick,device=device)
                for item,clocks in zip(timing,events):
                    item.update(forward_seconds=clocks[0].elapsed_time(clocks[1])/1000,
                                backward_seconds=clocks[1].elapsed_time(clocks[2])/1000)
                dist.all_reduce(duration,op=dist.ReduceOp.MAX);dist.all_reduce(totals);totals/=world
                row=dict(zip(('loss','av','video_feature','audio_feature','av_video','av_audio'),totals.tolist()),
                    step=step,stage=args.stage,av_active=args.stage>1,seconds=float(duration),grad_norm=float(norm),
                    gradients=gradients,sample_ids=ids,timing=timing,peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
                if args.stage==1:
                    for k in ('av','av_video','av_audio'):row[k]=None
                log.write(json.dumps(row)+'\n');log.flush()
                if rank==0:
                    print(json.dumps(row),flush=True)
                    if writer:
                        for k in ('loss','av','video_feature','audio_feature','av_video','av_audio','seconds','grad_norm'):
                            if row[k] is not None:writer.add_scalar('train/'+k,row[k],step)
                        for k,v in gradients.items():writer.add_scalar('grad/'+k,v,step)
                        for group in optimizer.param_groups:writer.add_scalar('lr/'+group['name'],group['lr'],step)
                        writer.flush()
                last=step;durations.append(float(duration))
                if step%args.save_every==0 or step==start+args.steps:
                    ckpt.save(checkpoints/f'stage{args.stage}_step_{step:06d}.pt',raw,optimizer,step,identity,rank,world);saved_step=step
                if step%args.validate_every==0 or step==start+args.steps:validate(step)
            if last!=saved_step:
                ckpt.save(checkpoints/f'stage{args.stage}_step_{last:06d}.pt',raw,optimizer,last,identity,rank,world)
        # Verify exact synchronization, including all full residual parameters.
        from .distributed import parameter_digest
        hashes=[None]*world;dist.all_gather_object(hashes,parameter_digest(raw))
        if len(set(hashes))!=1:raise RuntimeError('Rank model states differ')
        return dict(stage=args.stage,start_step=start,end_step=last,updates=last-start,
                    world=world,global_batch=world*args.accumulation,ranks_identical=True,
                    parameter_sha256=hashes[0],original_dit_frozen=True,
                    checkpoint=str(checkpoints/f'stage{args.stage}_step_{last:06d}.pt') if last>start else None,
                    stopped=(args.output/'STOP').exists(),seconds=time.time()-started,step_seconds=durations)
    finally:
        if writer:writer.close()
