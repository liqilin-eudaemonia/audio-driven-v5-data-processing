import torch
import torch.distributed as dist
from pathlib import Path
import time

def longest_canary(*, training, arm, c, model, optimizer, dataset, strategy, sampler,
                   pos, effective, names, device, local, rank, world, resume_path, identity, start):
    """Two real worst-duration unequal-batch updates, then strict checkpoint restore."""
    import csv
    from p2_clip_events.model import mask_classes
    from p2_listen.contracts import atomic_json
    from p2_listen.monitoring import rank_zero_action
    from p2_ltx_events.heads import objective
    from p2_ltx_all_lora.data import to_device
    from p2_whisper_av.execution import move_adam_moments
    if resume_path is None:
        raise ValueError('The split is a continuation only')
    with Path(c['sample_list']).open() as f:
        index, row = max(enumerate(csv.DictReader(f)), key=lambda pair: float(pair[1]['duration_s']))
    assert world == 31
    chosen = [(index,9000000+rank)]
    sample = dataset.sample(index)
    if names:
        sample['events'] = mask_classes(sample['events'], effective)
    counts = training._event_counts([sample]*len(chosen), names, device, world)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for update in range(2):
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        acc = training.accumulator(model.parameters(), len(chosen), c['execution'])
        for _, ordinal in chosen:
            value = to_device(sample, device)
            torch.manual_seed(c['training']['seed']+ordinal+update*31)
            generator = torch.Generator(device=device).manual_seed(c['training']['seed']+1000000+ordinal+update*31)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                values = model(value, pos, generator, strategy, sampler)
                loss, _, _ = objective(values, c, counts, 31, 1)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite split canary loss')
            loss.backward(); acc.add()
            del values, loss, value
        acc.finish(world)
        main = [p for g in optimizer.param_groups if g['name']!='events' for p in g['params']]
        torch.nn.utils.clip_grad_norm_(main, c['training']['clip_norm'], error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(model.event_heads.parameters(), c['training']['clip_norm'], error_if_nonfinite=True)
        optimizer.step(); model.assert_ownership()
        torch.cuda.synchronize(device); times.append(time.monotonic()-begin)
        del acc
    peak = torch.tensor([torch.cuda.max_memory_reserved(device)/2**30, *times], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    del sample
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    torch.cuda.empty_cache()
    restored = training.checkpoint.resume(resume_path, model, optimizer, identity, rank)
    if restored != start:
        raise ValueError('Canary restore changed training step')
    model.assert_ownership()
    torch.cuda.empty_cache()
    if world > 1:
        dist.barrier()
    report = dict(success=True, arm=arm, world=world, global_batch=31, sample_id=row['sample_id'],
                  duration_s=float(row['duration_s']), updates=2, max_rank_peak_reserved_gib=float(peak[0]),
                  max_rank_update_seconds=peak[1:].tolist(), resumed_step=start,
                  model_optimizer_rng_restored_from_checkpoint=True)
    rank_zero_action(lambda: atomic_json(Path(c['output'])/'LONGEST_UNEVEN_CHECK.json', report), rank=rank, world=world)
    return report
