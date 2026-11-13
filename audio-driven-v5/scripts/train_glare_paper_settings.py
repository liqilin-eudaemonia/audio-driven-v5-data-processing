#!/usr/bin/env python3
"""Train native GLARE from the sealed 500h cache registry. No auto-launch."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from baseline_training.glare_cached import Registry, CachedFLOAT, save_checkpoint, sha
from baseline_training.glare_execution_fast import FastCachedFLOAT
from baseline_training.training import build, seed, rng, restore_rng, set_train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, default=ROOT / 'experiments/glare_cache_28gpu_20261110/production_gpu_fer/TRAINING_CACHES_LATEST.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-data', type=int, default=0, help='Read this many real samples on CPU and exit')
    p.add_argument('--steps', type=int, required=True, help='Explicit total optimizer-update budget, preserved on resume')
    p.add_argument('--accumulate', type=int, default=256, help='Pairs per rank per update; effective batch256 on one GPU')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--eval-samples', type=int, default=0, help='0 means full dev')
    p.add_argument('--pilot-eval-step', type=int, default=64)
    p.add_argument('--pilot-eval-samples', type=int, default=256)
    p.add_argument('--early-stop-patience', type=int, default=3)
    p.add_argument('--precision', choices=['fp16','fp32'], default='fp16')
    p.add_argument('--resume', type=Path)
    p.add_argument('--seed', type=int, default=137)
    p.add_argument('--allow-original-resume', action='store_true', help='One-time execution-only migration; original source hashes and all semantic fields must match')
    p.add_argument('--stop-after', type=int, default=0, help='Diagnostic: save and stop after this many updates; scheduler budget unchanged')
    a = p.parse_args()
    stop_requested = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.__setitem__(0, True))
    assert a.steps > 0 and a.accumulate > 0 and a.workers >= 0
    assert a.save_every > 0 and a.eval_every > 0 and a.eval_samples >= 0
    train = Registry(a.registry, 'train')
    if a.check_data:
        ids = np.linspace(0, len(train) - 1, min(a.check_data, len(train)), dtype=int)
        for i in ids:
            v = train[int(i)]; print(json.dumps(dict(sample_id=v['sample_id'], frames=len(v['motion']))), flush=True)
        return
    dev = Registry(a.registry, 'dev')
    world = int(os.environ.get('WORLD_SIZE', '1')); rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0')); torch.cuda.set_device(local)
    device = f'cuda:{local}'
    if world > 1: torch.distributed.init_process_group('nccl')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed(a.seed)
    model, implementation = build('glare', device)
    model.fmt.initialize_weights()  # Same fresh FMT initialization as official training.
    wrapper = FastCachedFLOAT(model)
    # Same parameter groups as upstream train_multigpu.py (includes reaction/intensity heads).
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=a.precision=='fp16')
    from float.utils import WarmupCosineScheduler
    scheduler = WarmupCosineScheduler(optimizer, int(.1 * a.steps), a.steps, min_lr=a.lr / 10)
    source_files = [Path(__file__), ROOT / 'src/baseline_training/glare_cached.py',
                    ROOT / 'src/baseline_training/training.py', ROOT / 'src/baseline_training/features.py',
                    ROOT / 'src/baseline_training/glare_execution_fast.py']
    identity = dict(registry=train.identity, steps=a.steps, accumulate=a.accumulate, world=world,
                    lr=a.lr, weight_decay=a.weight_decay, seed=a.seed, precision=a.precision,
                    validation=dict(every=a.eval_every,samples=a.eval_samples,patience=a.early_stop_patience,relative_min_improvement=0.005),
                    source={str(f):sha(f) for f in source_files}, implementation=implementation)
    start = 0; epoch = 0; cursor = 0; best = float('inf'); resume_rng = None
    bad_evals = 0; successful_updates = 0
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        if ck['identity'] != identity:
            assert a.allow_original_resume, 'Resume contract changed'
            old = ck['identity']
            assert {k:v for k,v in old.items() if k!='source'} == {k:v for k,v in identity.items() if k!='source'}
            expected_old = [ROOT/'scripts/train_glare_cached.py', ROOT/'src/baseline_training/glare_cached.py',
                            ROOT/'src/baseline_training/training.py', ROOT/'src/baseline_training/features.py']
            assert old['source'] == {str(f):sha(f) for f in expected_old}, 'Original source changed'
            a.output.mkdir(parents=True, exist_ok=True)
            if rank == 0:
                (a.output/'execution_migration.json').write_text(json.dumps(dict(old=old,new=identity,
                    checkpoint=str(a.resume),step=ck['step'],reason='window cache and deferred loss synchronization only'),indent=2)+'\n')
        # Position table is nontrainable and variable-length; reconstructed per pair.
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        assert set(ck['trainable']) == expected
        model.load_state_dict(ck['trainable'], strict=False)
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        bad_evals=ck['bad_evals']; successful_updates=ck['successful_updates']
        scheduler.current_step = ck['scheduler_step']
        start, epoch, cursor, best = ck['step'], ck['epoch'], ck['cursor'], ck['best']
        resume_rng = ck['rng_by_rank'][rank]
    if world > 1:
        wrapper = torch.nn.parallel.DistributedDataParallel(wrapper, device_ids=[local],
                    broadcast_buffers=False, find_unused_parameters=True)
    # Each rank gets an equal-length shard; last incomplete world-sized batch
    # is dropped, at most world-1 pairs/epoch, rotated by epoch shuffle.
    def order(e):
        g = torch.Generator().manual_seed(a.seed + e)
        ids = torch.randperm(len(train), generator=g).tolist()
        return ids[:len(ids) // world * world][rank::world]
    def loader(e, offset):
        return iter(torch.utils.data.DataLoader(train, batch_size=None, sampler=order(e)[offset:],
                    num_workers=a.workers, generator=torch.Generator().manual_seed(a.seed + e),
                    **({'prefetch_factor':2} if a.workers else {})))
    it = loader(epoch, cursor)
    if resume_rng is not None: restore_rng(resume_rng)
    else: seed(a.seed + rank)
    a.output.mkdir(parents=True, exist_ok=True)
    if rank == 0: (a.output / 'protocol.json').write_text(json.dumps(identity, indent=2) + '\n')
    def checkpoint(step, name):
        states = [None] * world
        if world > 1: torch.distributed.all_gather_object(states, rng())
        else: states[0] = rng()
        if rank == 0:
            save_checkpoint(a.output / name, dict(identity=identity, step=step, epoch=epoch, cursor=cursor,
                trainable={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
                optimizer=optimizer.state_dict(), scheduler_step=scheduler.current_step,
                best=best, rng_by_rank=states,scaler=scaler.state_dict(),bad_evals=bad_evals,successful_updates=successful_updates))
    def evaluate(count):
        saved=rng();seed(a.seed+100000);model.eval()
        ids=np.linspace(0,len(dev)-1,min(count,len(dev)),dtype=int) if count else range(len(dev))
        raw=wrapper.module if world>1 else wrapper
        vals=[]
        with torch.no_grad():
            for i in ids:
                with torch.autocast('cuda',dtype=torch.float16,enabled=a.precision=='fp16'):
                    v=raw(dev[int(i)])['loss']
                vals.append(float(v))
        assert all(math.isfinite(v) for v in vals)
        restore_rng(saved);set_train(model,'glare')
        return sum(vals)/len(vals)
    if start==0:
        baseline=dict(step=0,pilot_dev_loss=evaluate(a.pilot_eval_samples),samples=a.pilot_eval_samples,at=time.time())
        if rank==0:
            (a.output/'initial_validation.json').write_text(json.dumps(baseline,indent=2)+'\n')
            print(json.dumps(baseline),flush=True)
    for step in range(start, a.steps):
        set_train(model, 'glare'); optimizer.zero_grad(set_to_none=True); total = 0.; pending_losses = []
        for micro in range(a.accumulate):
            try: value = next(it)
            except StopIteration:
                epoch += 1; cursor = 0; it = loader(epoch, cursor); value = next(it)
            cursor += 1
            ctx = wrapper.no_sync() if world > 1 and micro < a.accumulate - 1 else contextlib.nullcontext()
            with ctx:
                with torch.autocast('cuda', dtype=torch.float16, enabled=a.precision=='fp16'):
                    losses = wrapper(value); loss = losses['loss']
                scaler.scale(loss / a.accumulate).backward(); pending_losses.append(loss.detach())
        loss_values = torch.stack(pending_losses)
        assert bool(torch.isfinite(loss_values).all()), 'Nonfinite training loss'
        total = sum(loss_values.cpu().tolist()) / a.accumulate
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=a.precision=='fp32')
        previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        skipped=scaler.get_scale()<previous_scale
        if not skipped:
            successful_updates+=1; scheduler.step()
        entry = dict(step=step+1, loss=total, grad_norm=float(grad) if torch.isfinite(grad) else None, lr=optimizer.param_groups[0]['lr'], at=time.time(),
                     skipped_overflow=skipped,scaler_scale=scaler.get_scale(),successful_updates=successful_updates,consumed_pairs=(step+1)*a.accumulate*world)
        if step+1==a.pilot_eval_step:
            entry['pilot_dev_loss']=evaluate(a.pilot_eval_samples)
        should_stop=False
        if (step+1) % a.eval_every == 0 or step+1 == a.steps:
            score=evaluate(a.eval_samples); entry['dev_loss']=score
            improved=score < best*0.995
            if improved: bad_evals=0
            else: bad_evals+=1
            if score<best:
                best=score;checkpoint(step+1,'best.pt')
            entry['bad_evals']=bad_evals
            should_stop=a.early_stop_patience>0 and bad_evals>=a.early_stop_patience
        if rank == 0:
            with (a.output / 'metrics.jsonl').open('a') as f: f.write(json.dumps(entry)+'\n')
            print(json.dumps(entry), flush=True)
        stopping = should_stop or stop_requested[0] or (a.stop_after > 0 and step+1-start >= a.stop_after)
        if (step+1) % a.save_every == 0 or step+1 == a.steps or stopping: checkpoint(step+1, 'latest.pt')
        if stopping: break
    if world > 1: torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
