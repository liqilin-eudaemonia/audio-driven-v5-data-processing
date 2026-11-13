"""Pure AV training with the baseline duration order and explicit Whisper ownership."""
from pathlib import Path
import json
import os
import time

import torch
import torch.distributed as dist

from p2_listen.contracts import read, digest, atomic_json
from . import VERSION
from .config import validate, validate_baseline, source_identity
from .data import Dataset, data_identity
from .model import QwenWhisperAV
from . import checkpoint


def preflight(c, processes=16):
    from concurrent.futures import ThreadPoolExecutor
    from p2_j_cache.data import sample
    from p2_j_cache.runtime import cache_identity
    from p2_j_duration_execution import Schedule
    validate(c);validate_baseline(c,read(c['baseline_config']))
    if type(processes) is not int or processes < 1:
        raise ValueError('preflight processes must be a positive integer')
    dataset = Dataset(c);Schedule(c)
    reader=dataset.reader;w=dataset.whisper
    base_complete=read(reader.work/'COMPLETE.json')
    if base_complete.get('success') is not True or base_complete.get('samples') != len(dataset.rows):
        raise ValueError('Baseline full cache audit incomplete')
    reader.part(None,'text')
    def check(row):
        sid=row['sample_id'];item=sample(reader.work,reader.manifest,sid)
        for part in ('vae','text_guided','input_plan'):
            folder=reader.work/'cache'/part/sid;marker=read(folder/'complete.json')
            expected=cache_identity(reader.manifest,item,part,reader.assets[part])
            if (marker['identity'] != digest(expected) or marker['contract'] != expected
                    or not (folder/'payload.pt').stat().st_size):
                raise ValueError('Invalid baseline cache receipt: '+str(folder))
        marker=read(w.folder(sid)/'complete.json');expected=w.identity(item)
        if (marker['identity'] != digest(expected) or marker['contract'] != expected
                or not (w.folder(sid)/'payload.pt').stat().st_size):
            raise ValueError('Invalid speaker Whisper cache receipt: '+sid)
    # Bounded metadata work, avoiding loading all 500h of tensors into RAM.
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=processes) as pool:
        for offset in range(0,len(dataset.rows),4096):
            for _ in pool.map(check,dataset.rows[offset:offset+4096]):
                pass
            print(json.dumps(dict(stage='whisper_av_preflight',samples=min(offset+4096,len(dataset.rows)),
                                  total=len(dataset.rows))),flush=True)
    # The Whisper COMPLETE marker is written only by finalize(), which validates
    # every payload hash and the exact 20ms time grid.
    result = dict(schema=VERSION,data_identity=data_identity(c),source=source_identity(),
        experiment_mode=[c['text_condition'],c['loss'],c['stage']],samples=len(dataset.rows),
        processes=processes,elapsed_s=time.monotonic()-started,
        whisper_samples=len(dataset.rows),objective='video_av + audio_av; no Event or Feature',
        initialization='same pretrained LTX and Qwen Connector seed as AV baseline',
        baseline_config=c['baseline_config'],training_started=False)
    atomic_json(Path(c['output'])/'preflight.json',result)
    return result


def scaled_av(loss, micro_count, world):
    if type(micro_count) is not int or type(world) is not int or min(micro_count,world) < 1:
        raise ValueError('Invalid AV normalization geometry')
    return loss/(micro_count*world)


def train(c, *, resume_path=None, stop_after=None):
    from p2_clip_events.backend import data_config
    from p2_listen.config import extractor_config
    from p2_listen.numerics import configure_training_numerics
    from p2_listen.training import rng_state
    from p2_listen.performance import BucketGradientAccumulator, configure_renderer
    from p2_listen.monitoring import rank_zero_action, truncate_jsonl
    from p2_listen.features import weights_identity
    from p2_j_duration_execution import Schedule
    validate(c);validate_baseline(c,read(c['baseline_config']))
    d = data_config(c)
    from omni_av_p2.ltx import load_renderer
    out = Path(c['output']);t = c['training'];pre = read(out/'preflight.json')
    did = data_identity(c);sources = source_identity();numerics = configure_training_numerics()
    if (pre['data_identity'] != did or pre['source'] != sources or pre['schema'] != VERSION or
            pre['experiment_mode'] != [c['text_condition'],c['loss'],c['stage']]):
        raise ValueError('Run preflight for this exact data/source/config first')
    rank = int(os.environ.get('RANK',0));world = int(os.environ.get('WORLD_SIZE',1));local = int(os.environ.get('LOCAL_RANK',0))
    if world < 1 or t['global_batch'] % world:
        raise ValueError('World size must divide baseline global batch')
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('stop-after must be a positive step')
    if not torch.cuda.is_available():
        raise RuntimeError('Real LTX training requires GPU; CPU tests use a tiny native model')
    cfg = extractor_config(d);dataset = Dataset(c);schedule = Schedule(c)
    torch.cuda.set_device(local);device = f'cuda:{local}';writer = None
    if os.environ.get('P2_TRAIN_MEMORY_GIB'):
        torch.cuda.set_per_process_memory_fraction(float(os.environ['P2_TRAIN_MEMORY_GIB'])*2**30/
            torch.cuda.get_device_properties(local).total_memory,local)
    if world > 1:
        dist.init_process_group('nccl')
    try:
        renderer,strategy,sampler = load_renderer(cfg,device)
        configure_renderer(renderer,t['gradient_checkpointing'])
        torch.manual_seed(t['seed'])
        model = QwenWhisperAV(c,cfg,renderer).to(device)
        identity = dict(schema=VERSION,experiment=c,data_identity=did,source=sources,world=world,numerics=numerics,
            assets={k:weights_identity(d,k) for k in ('omni','ltx','gemma')},
            text_targets=model.text_targets,whisper_time_max_pos=model.time_max_pos)
        if world > 1:
            identities = [None]*world;dist.all_gather_object(identities,digest(identity))
            if len(set(identities)) != 1:
                raise ValueError('Ranks disagree on Whisper run identity')
        def register():
            if (out/'run.json').exists() and resume_path is None:
                raise ValueError('Run exists; resume its own complete checkpoint explicitly')
            if resume_path is not None and Path(resume_path).resolve().parent != out.resolve():
                raise ValueError('Resume must belong to this Whisper experiment')
            atomic_json(out/'run.json',identity,immutable=True)
        rank_zero_action(register,rank=rank,world=world)
        groups = model.groups();base_lr = {g['name']:g['lr'] for g in groups}
        optimizer = torch.optim.AdamW(groups,weight_decay=0.)
        start = checkpoint.resume(resume_path,model,optimizer,identity,rank) if resume_path else 0
        def open_writer():
            nonlocal writer
            if resume_path:
                truncate_jsonl(out/'metrics.jsonl',start)
            if t['tensorboard']:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(str(out/'tensorboard'),purge_step=start+1 if resume_path else None)
        rank_zero_action(open_writer,rank=rank,world=world)
        model.train()
        def save(step):
            states = [rng_state()]
            if world > 1:
                states = [None]*world;dist.all_gather_object(states,rng_state())
            rank_zero_action(lambda:checkpoint.save(out/f'step_{step:06d}.pt',model,optimizer,step,identity,states),
                             rank=rank,world=world)
        if resume_path is None:
            save(0)
        last = start
        for step in range(start+1,t['steps']+1):
            stop = torch.tensor(int((out/'STOP').exists() or stop_after is not None and step > stop_after),device=device)
            if world > 1:
                dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if stop.item():
                break
            begin = time.monotonic();torch.cuda.reset_peak_memory_stats(device)
            for group in optimizer.param_groups:
                group['lr'] = base_lr[group['name']]*min(1.,step/t['warmup_steps'])
            chosen = schedule.indices(len(dataset.rows),step,t['global_batch'],t['seed'],rank,world)
            optimizer.zero_grad(set_to_none=True)
            acc = BucketGradientAccumulator(model.parameters(),32)
            record = dict(av=0.,av_video=0.,av_audio=0.)
            # Load one microbatch at a time: large Whisper sequences must not
            # multiply GPU residency during gradient accumulation.
            for index,ordinal in chosen:
                sample = dataset.sample(index,device)
                torch.manual_seed(t['seed']+ordinal)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    values = model(sample,strategy,sampler)
                    loss = scaled_av(values['av'],len(chosen),world)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite AV training loss')
                loss.backward();acc.add()
                record['av'] += float(values['av'].detach())/len(chosen)
                for mod in ('video','audio'):
                    record['av_'+mod] += float(values['av_components'][mod])/len(chosen)
                del values,loss,sample
            acc.finish(world)
            norms = {g['name']:sum(float(p.grad.double().square().sum()) for p in g['params']
                if p.grad is not None)**.5 for g in optimizer.param_groups}
            trainables = [p for g in optimizer.param_groups for p in g['params']]
            norms['main'] = float(torch.nn.utils.clip_grad_norm_(trainables,t['clip_norm'],error_if_nonfinite=True))
            optimizer.step();model.assert_ownership();last = step
            keys = ('av','av_video','av_audio')
            numbers = torch.tensor([record[k] for k in keys],device=device,dtype=torch.float64)
            if world > 1:
                dist.all_reduce(numbers)
            record.update(zip(keys,(numbers/world).tolist()))
            record.update(weighted_total=record['av'],event=0.,feature_video=0.,feature_audio=0.,
                step=step,seconds=time.monotonic()-begin,grad_norm=norms,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
            def log():
                with (out/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                if writer:
                    for k in (*keys,'weighted_total'):
                        writer.add_scalar('loss/'+k,record[k],step)
                    for k,v in norms.items():
                        writer.add_scalar('grad_norm/'+k,v,step)
                    for group in optimizer.param_groups:
                        writer.add_scalar('lr/'+group['name'],group['lr'],step)
                    writer.add_scalar('performance/step_seconds',record['seconds'],step)
                    writer.add_scalar('memory/peak_reserved_gib',record['peak_reserved_gib'],step)
                atomic_json(out/'status.json',dict(step=step,budget=t['steps'],complete=False))
            rank_zero_action(log,rank=rank,world=world)
            if step % t['save_every'] == 0:
                save(step)
            del acc
        if last % t['save_every'] != 0 or last == start:
            save(last)
        rank_zero_action(lambda:atomic_json(out/'status.json',dict(step=last,budget=t['steps'],complete=last==t['steps'])),
                         rank=rank,world=world)
        return dict(step=last,complete=last==t['steps'],output=str(out))
    finally:
        if writer:
            writer.close()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
