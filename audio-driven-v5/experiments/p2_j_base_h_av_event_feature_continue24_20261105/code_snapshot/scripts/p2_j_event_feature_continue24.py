"""Explicit 28-to-24-rank continuation, preserving model/Adam and all remaining real samples."""
import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from p2_listen.contracts import read, sha, digest, atomic_json


def remaining_plan(pairs, size, prefix_step, old_batch, batch, durations, seed):
    pairs = [tuple(p) for p in pairs]
    cut = prefix_step * old_batch
    prefix = pairs[:cut]
    if any(o >= size for _, o in prefix):
        raise ValueError('Consumed prefix contains epoch padding')
    real = [p for p in pairs[cut:] if p[1] < size]
    if (len({o for _, o in prefix + real}) != size or
            {o for _, o in prefix + real} != set(range(size))):
        raise ValueError('Real sample/noise ordinals lost or duplicated')
    padding_count = (-len(real)) % batch
    available = [p for p in pairs[cut:] if p[1] >= size]
    if len(available) < padding_count:
        raise ValueError('Parent lacks deterministic padding for the last batch')
    remaining = real + available[:padding_count]
    aligned = []
    window = batch * 16
    for offset in range(0, len(remaining), window):
        block = sorted(remaining[offset:offset + window], key=lambda p: durations[p[0]])
        groups = [block[i:i + batch] for i in range(0, len(block), batch)]
        random.Random(seed + 104729 + cut + offset).shuffle(groups)
        aligned.extend(p for group in groups for p in group)
    return aligned, dict(consumed_real=len(prefix), remaining_real=len(real),
        final_padding=padding_count, removed_parent_padding=len(available) - padding_count,
        prefix_pairs_sha256=digest(prefix), remaining_steps=len(aligned) // batch,
        window=window, total_real=size)


class RemainingSchedule:
    def __init__(self, c):
        from p2_j_duration_execution import Schedule
        meta = c['continuation_24']; parent = read(meta['parent_config'])
        if sha(meta['parent_config']) != meta['parent_config_sha256']:
            raise ValueError('Parent configuration changed')
        path = Path(c['j_cache']) / 'sample_list.csv'
        if sha(path) != c['duration_sampling']['sample_list_sha256']:
            raise ValueError('Sample list changed')
        with path.open() as f:
            durations = [float(r['duration_s']) for r in csv.DictReader(f)]
        original = Schedule(parent)
        expected, proof = remaining_plan(original.pairs, len(durations), meta['prefix_step'],
            parent['training']['global_batch'], 24, durations, c['training']['seed'])
        p = Path(c['duration_sampling']['manifest'])
        if sha(p) != c['duration_sampling']['sha256']:
            raise ValueError('Continuation schedule changed')
        saved = read(p)
        if [tuple(p) for p in saved['remaining_pairs']] != expected or saved['proof'] != proof:
            raise ValueError('Continuation plan differs from deterministic parent remainder')
        if c['training']['steps'] != meta['prefix_step'] + proof['remaining_steps']:
            raise ValueError('Continuation budget differs')
        self.pairs = expected; self.size = len(durations)
        self.prefix = meta['prefix_step']; self.seed = c['training']['seed']

    def indices(self, size, step, batch, seed, rank=0, world=1):
        if (size, batch, seed, world) != (self.size, 24, self.seed, 24) or not 0 <= rank < 24:
            raise ValueError('Continuation sampler requires exactly 24 ranks, batch24')
        offset = (step - self.prefix - 1) * 24
        if offset < 0 or offset + rank >= len(self.pairs):
            raise ValueError('Continuation step outside remaining schedule')
        return [self.pairs[offset + rank]]


def install(c):
    from p2_j_duration_execution import install_backend
    from p2_listen import training
    backend = install_backend()
    if not getattr(backend, '_continue24_installed', False):
        original = backend.source_identity
        def source_identity():
            value = original(); value[str(SELF)] = sha(SELF); return value
        backend.source_identity = source_identity
        backend._continue24_installed = True
    training.indices = RemainingSchedule(c).indices
    return backend


def equal(a, b):
    import torch
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def prepare(out):
    import torch
    from p2_matrix16.checkpoint import checked
    from p2_listen.cache import atomic_torch
    from p2_j_duration_execution import install_backend, Schedule
    torch.set_num_threads(2)
    base_path = ROOT / 'configs/p2_j_base_h_av_event_feature_28gpu_20261103.json'
    base = read(base_path); old = Path(base['output']); parent_cp = old / 'step_005500.pt'
    saved = checked(parent_cp); backend = install_backend()
    if (saved['step'] != 5500 or saved['identity']['world'] != 28 or
            saved['identity']['experiment'] != base or len(saved['rank_rng']) != 28 or
            saved['identity']['source'] != backend.source_identity() or
            saved['identity']['data_identity'] != backend.data_identity(base)):
        raise ValueError('Parent checkpoint/data/source differs')
    with (Path(base['j_cache']) / 'sample_list.csv').open() as f:
        durations = [float(r['duration_s']) for r in csv.DictReader(f)]
    pairs, proof = remaining_plan(Schedule(base).pairs, len(durations), 5500, 28, 24, durations, base['training']['seed'])
    out.mkdir(exist_ok=False)
    plan = out / 'remaining.schedule.json'
    atomic_json(plan, dict(remaining_pairs=pairs, proof=proof), immutable=True)
    c = copy.deepcopy(base); c['output'] = str(out)
    c['training']['global_batch'] = 24
    c['training']['steps'] = 5500 + proof['remaining_steps']
    c['duration_sampling'].update(window=384, start_step=5500, manifest=str(plan), sha256=sha(plan))
    c['continuation_24'] = dict(kind='explicit-world28-to24-v1', prefix_step=5500,
        parent_config=str(base_path), parent_config_sha256=sha(base_path), parent_checkpoint=str(parent_cp),
        parent_checkpoint_sha256=sha(parent_cp), old_world=28, new_world=24,
        rng_parent_ranks=list(range(8)) + list(range(16, 28)) + list(range(8, 12)))
    normalized = copy.deepcopy(c); normalized.pop('continuation_24')
    normalized['output'] = base['output']; normalized['duration_sampling'] = base['duration_sampling']
    normalized['training']['global_batch'] = 28; normalized['training']['steps'] = base['training']['steps']
    if normalized != base: raise ValueError('Unapproved experimental change')
    from p2_matrix16.config import validate
    validate(c, formal=True)
    atomic_json(out / 'config.json', c, immutable=True)
    backend = install(c)
    identity = copy.deepcopy(saved['identity'])
    identity.update(experiment=c, source=backend.source_identity(), world=24)
    ranks = c['continuation_24']['rng_parent_ranks']
    target = out / parent_cp.name
    atomic_torch(target, dict(saved, identity=identity, rank_rng=[saved['rank_rng'][i] for i in ranks]))
    atomic_json(target.with_suffix('.complete.json'), dict(step=5500, sha256=sha(target)))
    verify = checked(target)
    if not all(equal(saved[k], verify[k]) for k in ('model', 'optimizer')):
        raise ValueError('Model/Adam changed during migration')
    if not equal(verify['rank_rng'], [saved['rank_rng'][i] for i in ranks]):
        raise ValueError('Mapped RNG states changed')
    atomic_json(out / 'run.json', identity, immutable=True)
    pre = read(old / 'preflight.json')
    if pre['source'] != saved['identity']['source'] or pre['data_identity'] != saved['identity']['data_identity']:
        raise ValueError('Parent full preflight differs')
    pre['source'] = identity['source']
    pre['reused_full_preflight'] = dict(path=str(old / 'preflight.json'), sha256=sha(old / 'preflight.json'),
        reason='Same data/teacher/events/model/loss; explicit batch24 and remaining schedule migration')
    atomic_json(out / 'preflight.json', pre, immutable=True)
    atomic_json(out / 'MIGRATION.json', dict(success=True, parent=str(parent_cp), parent_sha256=sha(parent_cp),
        model_bitwise_unchanged=True, optimizer_bitwise_unchanged=True, selected_rng_bitwise_unchanged=True,
        parent_rank_map=ranks, source_changes={str(SELF): sha(SELF)}, old_batch=28, new_batch=24,
        trajectory_identical=False, **proof), immutable=True)
    cluster = read(ROOT / 'configs/p2_j_base_h_av_event_feature_cluster28_20261103.json')
    cluster['nodes'] = [n for n in cluster['nodes'] if n['name'] != 'node8']
    for n in cluster['nodes']: n['gpus'] = list(range(8))
    cluster['master_port'] = 29849
    atomic_json(out / 'cluster.json', cluster, immutable=True)
    atomic_json(out / 'status.json', dict(step=5500, budget=c['training']['steps'], complete=False))
    print(json.dumps(dict(prepared=True, output=str(out), steps=c['training']['steps'], **proof)), flush=True)


def commands(cluster, config, resume):
    result = []
    for rank, node in enumerate(cluster['nodes']):
        result.append(['env', 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7', 'PYTHONUNBUFFERED=1',
            'TOKENIZERS_PARALLELISM=false', *[k + '=' + str(v) for k, v in cluster['network'].items()],
            cluster['python'], '-m', 'torch.distributed.run', '--max_restarts=0', '--nnodes=3',
            '--nproc_per_node=8', '--node_rank=' + str(rank), '--master_addr=' + cluster['nodes'][0]['host'],
            '--master_port=' + str(cluster['master_port']), str(SELF), 'train', '--config', str(config),
            '--resume', str(resume)])
    return result


def run(out):
    from p2_j_event_feature_launch import resources
    from p2_whisper8000_then_event_feature import pid_capacity, verify_objective
    from listen_cluster import launch
    out = out.resolve(); q = out / 'queue'; q.mkdir(exist_ok=True)
    with (q / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from p2_speaker_whisper_after_annotation import process_record
        atomic_json(q / 'controller.json', process_record())
        c = read(out / 'config.json'); cluster = read(out / 'cluster.json')
        def state(phase, **kw):
            row = dict(phase=phase, at=time.time(), **kw)
            atomic_json(q / 'state.json', row); print(json.dumps(row), flush=True)
        try:
            for attempt in range(1, 6):
                state('waiting_selected_24_gpus', attempt=attempt)
                while True:
                    if (q / 'STOP').exists(): state('stopped_before_launch'); return
                    checks = resources(cluster) + pid_capacity(cluster)
                    atomic_json(q / 'resources.json', dict(at=time.time(), nodes=checks))
                    if all(n['free'] for n in checks): break
                    time.sleep(10)
                source = read(out / 'run.json')['source']
                if any(sha(p) != v for p, v in source.items()): raise ValueError('Training source changed')
                marker = sorted(out.glob('step_*.complete.json'), reverse=True)[0]
                resume = marker.with_suffix('').with_suffix('.pt')
                if sha(resume) != read(marker)['sha256']: raise ValueError('Resume checksum differs')
                cmds = commands(cluster, out / 'config.json', resume)
                logs = q / ('attempt_' + str(attempt) + '_' + str(time.time_ns()))
                atomic_json(q / 'commands.json', dict(commands=cmds, resume=str(resume), world=24))
                state('training', resume_step=read(marker)['step'], budget=c['training']['steps'], world=24, logs=str(logs))
                done = threading.Event(); cancel = threading.Event(); began = time.time()
                f = out / 'metrics.jsonl'; old = f.stat().st_mtime if f.exists() else 0
                def watch():
                    try:
                        while not done.wait(10):
                            if (q / 'STOP').exists() and not (out / 'STOP').exists():
                                atomic_json(out / 'STOP', dict(reason='Controller stop requested'))
                            stamp = f.stat().st_mtime if f.exists() else 0
                            if time.time() - (stamp if stamp > old else began) > (900 if stamp > old else 1800):
                                atomic_json(q / 'STALL.json', dict(at=time.time(), logs=str(logs)))
                                cancel.set(); return
                            if stamp > old:
                                with f.open('rb') as stream:
                                    stream.seek(max(0, stream.seek(0, 2) - 16000)); lines = stream.read().splitlines()
                                try: row = json.loads(lines[-1])
                                except (ValueError, IndexError): continue
                                verify_objective(row)
                                if not (out / 'RESUME_VERIFIED.json').exists():
                                    atomic_json(out / 'RESUME_VERIFIED.json', dict(success=True, world=24,
                                        resumed_from=5500, sample=row, at=time.time()))
                    except Exception as exc:
                        atomic_json(q / 'WATCH_ERROR.json', dict(error=repr(exc), at=time.time())); cancel.set()
                t = threading.Thread(target=watch, daemon=True); t.start()
                try: launch(cluster, cmds, logs, cancel_event=cancel)
                except Exception as exc:
                    state('retry_pending', error=repr(exc), attempt=attempt)
                    if attempt == 5: raise
                    time.sleep(15); continue
                finally: done.set(); t.join(timeout=2)
                final = read(out / 'status.json')
                if final['complete'] and final['step'] == c['training']['steps']:
                    state('complete', **final); return
                if (out / 'STOP').exists(): state('stopped', **final); return
                raise ValueError('Incomplete training')
        except Exception as exc: state('needs_attention', error=repr(exc)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'train'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume', type=Path)
    a = parser.parse_args()
    if a.command == 'prepare': prepare(a.output.resolve())
    elif a.command == 'run': run(a.output.resolve())
    else:
        import faulthandler
        import signal
        faulthandler.enable(); faulthandler.register(signal.SIGUSR1, all_threads=True)
        from p2_matrix16.config import load
        from p2_matrix16.training import train
        c = load(a.config, formal=True); backend = install(c)
        print(json.dumps(train(c, resume_path=a.resume, backend=backend)), flush=True)
