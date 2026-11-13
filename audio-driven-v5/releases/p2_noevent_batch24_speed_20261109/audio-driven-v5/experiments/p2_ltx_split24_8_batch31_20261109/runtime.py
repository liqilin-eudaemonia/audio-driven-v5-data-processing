"""Batch31 partitioning and bounded-device FP64 accumulation for the approved split."""
from pathlib import Path
import importlib, os, time
import torch
import torch.distributed as dist

WORK = Path(__file__).resolve().parent
PACKAGES = {'full': 'p2_ltx_all_lora', 'noevent': 'p2_ltx_all_lora_noevent'}


def partition(pairs, rank, world):
    if len(pairs) != 31 or world not in (8, 24) or not 0 <= rank < world:
        raise ValueError('Require batch31 on the approved8/24 ranks')
    q, extra = divmod(31, world)
    begin = rank*q + min(rank, extra)
    return pairs[begin:begin+q+(rank < extra)]


def source_identity(arm):
    from p2_listen.contracts import sha
    native = importlib.import_module(PACKAGES[arm]+'.config').source_identity()
    for name in ('runtime.py', 'worker.py', 'training_full.py', 'training_noevent.py'):
        p = WORK/name
        native[str(p)] = sha(p)
    return native


class HostFP64Accumulator:
    """Exact FP64 local sums on CPU; only a bounded staging bucket resides on GPU.

    Collective order matches native StreamingFP64Reduction/BucketGradientAccumulator.
    Every grad is consumed before it is released; unused parameters preserve None.
    """
    def __init__(self, parameters, bucket_mb=512):
        self.parameters = [p for p in parameters if p.requires_grad]
        if not self.parameters or type(bucket_mb) is not int or bucket_mb < 1:
            raise ValueError('Invalid parameters/bucket')
        self.device = self.parameters[0].device
        if any(p.device != self.device for p in self.parameters):
            raise ValueError('Mixed devices')
        self.sizes = [p.numel() for p in self.parameters]
        self.total = sum(self.sizes)
        self.bucket_elements = bucket_mb*1024*1024//8
        self.buffer = torch.zeros(self.total, dtype=torch.float64, device='cpu')
        count = min(self.total, self.bucket_elements)
        self.stage = torch.empty(count, dtype=torch.float64, device='cpu', pin_memory=self.device.type=='cuda')
        self.present = [False]*len(self.parameters)
        self.added = 0

    def segments(self):
        index = offset = 0
        for begin in range(0, self.total, self.bucket_elements):
            count = min(self.bucket_elements, self.total-begin)
            pieces, filled = [], 0
            while filled < count:
                size = min(self.sizes[index]-offset, count-filled)
                pieces.append((index, offset, filled, size))
                filled += size; offset += size
                if offset == self.sizes[index]:
                    index += 1; offset = 0
            yield begin, count, pieces

    @torch.no_grad()
    def add(self):
        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                if p.grad.is_sparse or not p.grad.is_contiguous():
                    raise ValueError('Require contiguous dense grads')
                self.present[i] = True
        for begin, count, pieces in self.segments():
            device_buffer = torch.zeros(count, dtype=torch.float64, device=self.device)
            for i, offset, dest, size in pieces:
                grad = self.parameters[i].grad
                if grad is not None:
                    device_buffer[dest:dest+size].copy_(grad.view(-1)[offset:offset+size])
            self.stage[:count].copy_(device_buffer, non_blocking=False)
            self.buffer[begin:begin+count].add_(self.stage[:count])
            del device_buffer
        for p in self.parameters:
            p.grad = None
        self.added += 1

    @torch.no_grad()
    def finish(self, world=1):
        if not self.added or world < 1 or (world > 1 and (not dist.is_initialized() or dist.get_world_size()!=world)):
            raise ValueError('Invalid reduction state/world')
        present = torch.tensor(self.present, dtype=torch.int32, device=self.device)
        if world > 1:
            dist.all_reduce(present, op=dist.ReduceOp.MAX)
        for p, used in zip(self.parameters, present.tolist()):
            p.grad = torch.empty_like(p) if used else None
        for begin, count, pieces in self.segments():
            self.stage[:count].copy_(self.buffer[begin:begin+count])
            device_buffer = self.stage[:count].to(self.device, copy=True, non_blocking=False)
            if world > 1:
                dist.all_reduce(device_buffer, op=dist.ReduceOp.SUM)
            for i, offset, src, size in pieces:
                grad = self.parameters[i].grad
                if grad is not None:
                    grad.view(-1)[offset:offset+size].copy_(device_buffer[src:src+size])
            del device_buffer
        self.added = 0


def load_training(arm):
    """Load a hashed, minimal native-loop derivative without editing the native tree."""
    import importlib.util
    package = PACKAGES[arm]
    spec = importlib.util.spec_from_file_location(package+'._split24_8_training', WORK/f'training_{arm}.py')
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    original = importlib.import_module(package+'.schedule').Schedule
    class Schedule(original):
        def indices(self, size, step, batch, seed, rank=0, world=1):
            if batch != 31:
                raise ValueError('Batch changed')
            return partition(super().indices(size, step, batch, seed, 0, 1), rank, world)
    training.Schedule = Schedule
    training.source_identity = lambda: source_identity(arm)
    if arm == 'full':
        training.accumulator = lambda parameters, micro_count, execution: HostFP64Accumulator(parameters, execution['gradient_bucket_mb'])
    training.split_canary = lambda **kw: longest_canary(training=training, arm=arm, **kw)
    return training


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
    chosen = partition([(index, 9000000+i) for i in range(31)], rank, world)
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
