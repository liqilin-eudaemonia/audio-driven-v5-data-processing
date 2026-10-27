"""P3 DDP: synchronize trainable parameters, keep frozen Omni local to each GPU."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time
import traceback

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from omni_av.data import write_json_new
from omni_av_p2.checkpoint import capture_rng, restore_rng
from omni_av_p2.launch import agree, establish_output
from .accelerated import FastPlanner, Resident, audit_gradients, execution_mode, optimizer_for, prepare_mask, resident
from .data import contract, load_native
from .diagnostics import restore
from .experiment import digest
from .model import Config, Layout

FORMAT = "p3_ddp_v1"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class TrainableDDP(nn.Module):
    """Register only trainable aliases; forward still differentiates through frozen Omni.

    Registering raw as a child would broadcast GBs of frozen parameters at startup.
    The optimizer and DDP aliases reference exactly the same Parameter objects.
    """
    def __init__(self, raw):
        super().__init__()
        object.__setattr__(self, "raw", raw)
        self.trainable = nn.ParameterList([p for p in raw.parameters() if p.requires_grad])

    def forward(self, data, masked, ids, seed):
        losses = (self.raw.prepared_batch_loss(data,masked,ids,seed) if isinstance(data,list)
                  else self.raw.prepared_loss(data, masked, ids, seed))
        return torch.stack([losses["video"], losses["audio"]])


class GlobalOrder:
    """No sampler padding: sample/noise ordinals independent of rank placement."""
    def __init__(self, count, seed):
        if count < 1:
            raise ValueError("Empty training split")
        self.count, self.seed, self.epochs = count, seed, OrderedDict()

    def __getitem__(self, ordinal):
        epoch, pos = divmod(ordinal, self.count)
        if epoch not in self.epochs:
            self.epochs[epoch] = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(self.count)
            if len(self.epochs) > 2:
                self.epochs.popitem(last=False)
        return int(self.epochs[epoch][pos])


def ordinal_for(step, micro, rank, world, accumulation):
    return (step - 1) * world * accumulation + micro * world + rank


class BatchAssignments:
    """Balance only inside one global batch; preserve examples and their noise seeds."""
    def __init__(self, order, costs, world, accumulation, balanced=False):
        self.order, self.costs = order, costs
        self.world, self.accumulation, self.balanced = world, accumulation, balanced
        self.cache = OrderedDict()

    def __getitem__(self, step):
        if step not in self.cache:
            rows = [[ordinal_for(step,m,r,self.world,self.accumulation) for r in range(self.world)]
                    for m in range(self.accumulation)]
            if self.balanced:
                jobs = sorted((j for row in rows for j in row),
                              key=lambda j: (-self.costs[self.order[j]], j))
                groups, loads = [[] for _ in range(self.world)], [0.]*self.world
                for j in jobs:
                    rank = min((r for r in range(self.world) if len(groups[r])<self.accumulation),
                               key=lambda r:(loads[r],r))
                    groups[rank].append(j)
                    loads[rank] += self.costs[self.order[j]]
                rows = [[groups[r][m] for r in range(self.world)] for m in range(self.accumulation)]
            self.cache[step] = rows
            if len(self.cache)>2:
                self.cache.popitem(last=False)
        return self.cache[step]


class Samples:
    """Bounded GPU cache; verify each feature file on first read, never cache hidden states."""
    def __init__(self, entries, split, device, capacity, synthetic=None, prefetch=False):
        self.entries, self.split, self.device = entries, split, device
        self.capacity, self.synthetic = capacity, synthetic
        self.cache, self.verified = OrderedDict(), set()
        self.pool = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self.pending = {}

    def __len__(self):
        return len(self.entries)

    def read_cpu(self, index):
        entry = self.entries[index]
        if self.synthetic is not None:
            row = self.synthetic[index]
        else:
            if index not in self.verified:
                if sha(entry["path"]) != entry["sha256"]:
                    raise ValueError("Feature cache changed")
                self.verified.add(index)
            row = torch.load(entry["path"], map_location="cpu", weights_only=True)
        if row["sample_id"] != entry["sample_id"] or row["split"] != self.split:
            raise ValueError("Feature identity/split mismatch")
        return resident(row, torch.device('cpu'))

    def schedule(self, indices):
        if self.pool:
            for index in indices:
                if index not in self.cache and index not in self.pending:
                    self.pending[index] = self.pool.submit(self.read_cpu, index)

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)

    def __getitem__(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        cpu = self.pending.pop(index).result() if index in self.pending else self.read_cpu(index)
        item = Resident(cpu.layout.to(self.device), cpu.targets.to(self.device), cpu.cpu_layout, cpu.sample_id)
        if self.capacity:
            self.cache[index] = item
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return item


def tiny_model(cfg, arm, device):
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTextConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerTextModel
    text = Qwen2_5OmniTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}, attention_dropout=0.)
    text._attn_implementation = "sdpa"
    thinker = nn.Module()
    thinker.model = Qwen2_5OmniThinkerTextModel(text).to(device)
    return FastPlanner(thinker, cfg, arm)


def tiny_rows(count, split):
    rows = []
    for i in range(count):
        generator = torch.Generator().manual_seed(3100 + i + (100 if split == "dev" else 0))
        p, n = 4 + i % 2, 6 + 2 * (i % 3)
        layout = Layout(torch.randn(1, p, 32, generator=generator),
            torch.arange(p)[None, None].expand(3, 1, -1), torch.ones(1, p, dtype=torch.bool),
            torch.tensor([[[j * .04, 0., 0.] for j in range(n)]]),
            torch.tensor([[0] * (n // 2) + [1] * (n // 2)]), torch.ones(1, n, dtype=torch.bool))
        rows.append(dict(layout=vars(layout), targets=torch.randn(1, n, 32, generator=generator),
                         sample_id=f"{split}_{i}", split=split))
    return rows


def validate_index(index):
    sets = []
    for split in ("train", "dev"):
        ids = [e["sample_id"] for e in index[split]]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Empty/duplicate cache split")
        sets.append(set(ids))
    if sets[0] & sets[1]:
        raise ValueError("Train/Dev overlap")


def master_action(action, rank):
    error = [None]
    if rank == 0:
        try:
            action()
        except Exception:
            error[0] = traceback.format_exc()
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def save_checkpoint(path, raw, optimizer, step, identity, rank, world):
    rngs = [None] * world
    dist.all_gather_object(rngs, capture_rng())
    def publish():
        # New filenames only. Completion sidecar is published only after flush/fsync.
        with path.open("xb") as stream:
            torch.save(dict(format=FORMAT, arm=raw.arm, config=asdict(raw.cfg), identity=identity,
                state=raw.trainable_state(), optimizer=optimizer.state_dict(), step=step, rank_rng=rngs), stream)
            stream.flush()
            os.fsync(stream.fileno())
        write_json_new(path.with_suffix(".complete.json"), dict(step=step, sha256=sha(path)))
    master_action(publish, rank)


def read_checkpoint(path, identity):
    marker = path.with_suffix(".complete.json")
    if not marker.is_file() or json.loads(marker.read_text())["sha256"] != sha(path):
        raise ValueError("Checkpoint incomplete or corrupted")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["format"] != FORMAT or state["identity"] != identity:
        raise ValueError("Resume contract mismatch (arm/cache/world/accumulation/seed/optimizer)")
    return state


@torch.no_grad()
def validate(raw, data, rank, world, device, use_amp, seed):
    raw.eval()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    records = []
    for i in range(rank, len(data), world):
        a, b = data[i], data[(i + 1) % len(data)]
        wrong = Resident(replace(a.layout, prefix=b.layout.prefix, prefix_valid=b.layout.prefix_valid,
                                 prefix_positions=b.layout.prefix_positions), a.targets, a.cpu_layout, a.sample_id)
        masked, ids = prepare_mask(a, raw.cfg, seed + i, all_mask=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            correct = raw.prepared_loss(a, masked, ids, seed + i)
            shuffled = raw.prepared_loss(wrong, masked, ids, seed + i)
        values = torch.stack([correct["video"], correct["audio"], shuffled["video"], shuffled["audio"]])
        totals[:4] += values.double()
        totals[4] += 1
        records.append(dict(sample_id=a.sample_id, seed=seed + i, values=values.cpu().tolist()))
        raw._geometry.clear()
    dist.all_reduce(totals)
    all_records = [None] * world
    dist.all_gather_object(all_records, records)
    values = totals.cpu().tolist()
    if int(values[4]) != len(data):
        raise RuntimeError("Dev partition count mismatch")
    return dict(count=int(values[4]), video=values[0] / values[4], audio=values[1] / values[4],
                shuffled_video=values[2] / values[4], shuffled_audio=values[3] / values[4],
                records=[r for rows in all_records for r in rows])


def run(args):
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    device = torch.device("cpu" if args.cpu else f"cuda:{local}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    dist.init_process_group("gloo" if args.cpu else "nccl", timeout=timedelta(seconds=args.timeout))
    ready = False
    try:
        establish_output(args.output, rank)
        ready = True
        result = train(args, rank, world, device)
        master_action(lambda: write_json_new(args.output / "completed.json", result), rank)
    except Exception:
        if ready:
            write_json_new(args.output / f"failed_rank{rank}.json", dict(traceback=traceback.format_exc()))
        raise
    finally:
        dist.destroy_process_group()


def train(args, rank, world, device):
    from .launch import signature
    items_per_rank = args.accumulation * args.microbatch
    if args.tiny:
        cfg = Config(hidden_size=32, decoder_width=32, decoder_depth=2, lora_rank=2, lora_alpha=4,
                     diffusion_batch_mul=2, seed=args.seed)
        synth = {s: tiny_rows(n, s) for s, n in (("train", 7), ("dev", 3))}
        index = {s: [dict(sample_id=r["sample_id"],prefix_tokens=r['layout']['prefix'].shape[1],
                         video_tokens=r['video_tokens'] if 'video_tokens' in r else r['targets'].shape[1]//2,
                         audio_tokens=r['audio_tokens'] if 'audio_tokens' in r else r['targets'].shape[1]//2)
                     for r in rows] for s, rows in synth.items()}
        cache_sha = "synthetic_v1"
    else:
        index = json.loads(args.cache.read_text())
        cfg = Config(**index["contract"]["config"])
        if index["contract"] != json.loads(json.dumps(contract(cfg))):
            raise ValueError("P3 feature contract mismatch")
        cache_sha, synth = sha(args.cache), {"train": None, "dev": None}
    validate_index(index)
    identity = dict(format=FORMAT, config=json.loads(json.dumps(asdict(cfg))), cache_sha256=cache_sha,
        arm=args.arm, world_size=world, accumulation=args.accumulation, seed=args.seed,
        tiny=args.tiny, fused=args.fused_adam, checkpointing=args.checkpointing,
        train_ids=[e["sample_id"] for e in index["train"]])
    if args.balance_workload:
        identity['balance_workload'] = 'global_batch_lpt_v1'
    if args.microbatch != 1:
        identity['microbatch'] = args.microbatch
    agree(dict(identity=identity, source=signature(), steps=args.total_steps, resume=str(args.resume),
               initialize_from=str(args.initialize_from), save_every=args.save_every,
               validate_every=args.validate_every, torch=str(torch.__version__)))
    torch.manual_seed(cfg.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    raw = tiny_model(cfg, args.arm, device) if args.tiny else FastPlanner(load_native(cfg, device)[0].thinker, cfg, args.arm)
    raw.connectors.to(device)
    raw.decoders.to(device)
    raw.mask_token.data = raw.mask_token.data.to(device)
    raw.modality_embedding.data = raw.modality_embedding.data.to(device)
    execution_mode(raw, not args.checkpointing)
    loaded = read_checkpoint(args.resume, identity) if args.resume else None
    if loaded:
        restore(raw, loaded)
    elif args.initialize_from:
        initial = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
        if json.loads(json.dumps(initial["config"])) != identity["config"]:
            raise ValueError("Initialization model configuration mismatch")
        restore(raw, initial)
    optimizer, parameters = optimizer_for(raw, loaded, fused=args.fused_adam)
    start = loaded["step"] if loaded else 0
    if start >= args.total_steps:
        raise ValueError("No remaining updates")
    wrapped = DDP(TrainableDDP(raw), device_ids=[device.index] if device.type == "cuda" else None,
                  broadcast_buffers=False, gradient_as_bucket_view=True)
    if loaded:
        restore_rng(loaded["rank_rng"][rank])
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    master_action(lambda: write_json_new(args.output / "run.json", dict(identity=identity,
        start_step=start, total_steps=args.total_steps, global_batch=world * items_per_rank,
        hosts=hosts, source=signature(), resume=str(args.resume), initialize_from=str(args.initialize_from),
        execution=dict(prefetch=args.prefetch,cache_samples=args.cache_samples,balance_workload=args.balance_workload))), rank)
    train_data = Samples(index["train"], "train", device, args.cache_samples, synth["train"], args.prefetch)
    dev_data = Samples(index["dev"], "dev", device, args.cache_samples, synth["dev"])
    order = GlobalOrder(len(train_data), args.seed)
    costs = [(e.get('prefix_tokens',1)+e.get('video_tokens',1)+e.get('audio_tokens',1))**2 for e in index['train']]
    assignments = BatchAssignments(order,costs,world,items_per_rank,args.balance_workload)
    reference = deepcopy(raw) if args.tiny else None
    ref_opt = optimizer_for(reference, dict(optimizer=optimizer.state_dict()), fused=args.fused_adam)[0] if reference else None
    serial_error = 0.
    frozen = [(p, p._version) for p in raw.parameters() if not p.requires_grad]
    validations, durations, last, saved = [], [], start, -1
    writer = None
    if rank == 0 and not args.tiny:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(args.output/'tensorboard'))
    def validation(step):
        result = validate(raw, dev_data, rank, world, device, not args.tiny, args.seed + 900000)
        result["step"] = step
        validations.append(result)
        master_action(lambda: write_json_new(args.output / f"dev_{step:06d}.json", result), rank)
        if writer:
            for key in ('video','audio','shuffled_video','shuffled_audio'):
                writer.add_scalar('dev/'+key,result[key],step)
            writer.flush()
        raw.train()
    validation(start)
    with (args.output / f"train_rank{rank}.jsonl").open("x") as log:
        for step in range(start + 1, args.total_steps + 1):
            stop = torch.tensor(int(rank == 0 and (args.output / "STOP").exists()), device=device)
            dist.broadcast(stop, 0)
            if stop.item():
                break
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            parts = torch.zeros(2, device=device)
            samples, seeds = [], []
            data_wait = 0.
            for ahead in (step,step+1):
                if ahead <= args.total_steps:
                    train_data.schedule([order[assignments[ahead][m][rank]] for m in range(items_per_rank)])
            for micro in range(args.accumulation):
                read_begin = time.perf_counter()
                batch, mask_batch, ids_batch, seed_batch = [], [], [], []
                for item in range(args.microbatch):
                    ordinal = assignments[step][micro*args.microbatch+item][rank]
                    data = train_data[order[ordinal]]
                    seed = args.seed + 1009 * ordinal
                    masked, ids = prepare_mask(data, cfg, seed)
                    batch.append(data);mask_batch.append(masked);ids_batch.append(ids);seed_batch.append(seed)
                    samples.append(data.sample_id);seeds.append(seed)
                data_wait += time.perf_counter()-read_begin
                with wrapped.no_sync() if micro + 1 < args.accumulation else nullcontext():
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.tiny):
                        loss = (wrapped(batch,mask_batch,ids_batch,seed_batch) if args.microbatch>1
                                else wrapped(data, masked, ids, seed))
                    (loss.sum() / args.accumulation).backward()
                parts += loss.detach() / args.accumulation
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if reference is not None:
                reference.train()
                ref_opt.zero_grad(set_to_none=True)
                for micro in range(items_per_rank):
                    for source in range(world):
                        ordinal = assignments[step][micro][source]
                        data = train_data[order[ordinal]]
                        seed = args.seed + 1009 * ordinal
                        loss = reference.prepared_loss(data, *prepare_mask(data, cfg, seed), seed)
                        (sum(loss.values()) / (world * items_per_rank)).backward()
                torch.nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                ref_opt.step()
                for p, q in zip(raw.parameters(), reference.parameters()):
                    delta = float((p - q).detach().abs().max())
                    serial_error = max(serial_error, delta)
                    torch.testing.assert_close(p, q, atol=3e-5, rtol=3e-5)
                reference._geometry.clear()
            if any(p.requires_grad or p.grad is not None or p._version != v for p, v in frozen):
                raise RuntimeError("Frozen base modified")
            audit = audit_gradients(raw) if step == start + 1 or step % args.save_every == 0 or step == args.total_steps else None
            raw._geometry.clear()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            local_seconds = time.perf_counter()-begin
            dist.all_reduce(parts)
            parts /= world
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = torch.tensor(time.perf_counter() - begin, device=device, dtype=torch.float64)
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            durations.append(float(seconds))
            record = dict(step=step, rank=rank, samples=samples, seeds=seeds, loss=parts.cpu().tolist(),
                          grad_norm=float(norm), seconds=durations[-1], gradient_audit=audit,
                          data_wait_seconds=data_wait,local_seconds=local_seconds,
                          peak_allocated_gb=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if rank == 0:
                print(json.dumps(record), flush=True)
                if writer:
                    writer.add_scalar('train/video_fm',record['loss'][0],step)
                    writer.add_scalar('train/audio_fm',record['loss'][1],step)
                    writer.add_scalar('train/grad_norm',record['grad_norm'],step)
                    writer.add_scalar('speed/update_seconds',durations[-1],step)
                    writer.add_scalar('speed/samples_per_second',world*items_per_rank/durations[-1],step)
                    writer.add_scalar('speed/rank0_data_wait_seconds',data_wait,step)
            last = step
            if step % args.save_every == 0 or step == args.total_steps:
                save_checkpoint(args.output / f"step_{step:06d}.pt", raw, optimizer, step, identity, rank, world)
                saved = step
            if step % args.validate_every == 0 or step == args.total_steps:
                validation(step)
    if saved != last:
        save_checkpoint(args.output / f"step_{last:06d}.pt", raw, optimizer, last, identity, rank, world)
    if validations[-1]["step"] != last:
        validation(last)
    hashes = [None] * world
    dist.all_gather_object(hashes, digest(raw.trainable_state()))
    if len(set(hashes)) != 1:
        raise RuntimeError("DDP trainable parameters diverged")
    train_data.close()
    if writer:
        writer.close()
    return dict(arm=args.arm, world_size=world, physical_nodes=len(set(hosts)), hosts=hosts,
        tiny=args.tiny, pretrained_omni=not args.tiny, backend=dist.get_backend(),
        start_step=start, end_step=last, ranks_identical=True, trainable_sha256=hashes[0],
        global_batch=world * items_per_rank, serial_reference_max_abs_error=serial_error if args.tiny else None,
        validation_count=validations[-1]["count"], final_validation=validations[-1],
        mean_update_seconds=sum(durations) / len(durations) if durations else None,
        stopped=last < args.total_steps)
