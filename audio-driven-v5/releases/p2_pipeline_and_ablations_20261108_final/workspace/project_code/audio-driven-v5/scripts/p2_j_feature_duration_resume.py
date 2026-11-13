"""Authorized operational migration: group remaining clips now, save every 500."""
import copy
import csv
import json
from collections import Counter
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from p2_listen.contracts import read, atomic_json, sha
from p2_j_duration_execution import make_pairs, Schedule, install_backend
from p2_j_duration_benchmark import equal_tensors
from p2_j_s1_queue import commands
from listen_cluster import launch


def main():
    import torch
    from p2_matrix16.checkpoint import checked
    from p2_listen.cache import atomic_torch
    from p2_matrix16.config import load
    torch.set_num_threads(2)
    old_config = ROOT / 'configs/p2_j_guided_av_feature_base025_20261102.json'
    base = load(old_config, formal=True)
    old = Path(base['output'])
    out = old / 'duration_now_save500'
    out.mkdir(exist_ok=False)
    def status(state, **kw):
        atomic_json(out / 'controller_state.json', dict(state=state, time=time.time(), **kw))
        print(json.dumps(dict(state=state, **kw)), flush=True)
    try:
        status('validating_migration')
        ended = read(old / 'queue/continue_epoch/completed.json')
        if ended['returncodes'] != [0, 0, 0, 0] or not (old / 'STOP').exists():
            raise ValueError('All four original workers must have stopped gracefully')
        step = read(old / 'status.json')['step']
        if not 0 < step < base['duration_sampling']['start_step']:
            raise ValueError('This migration requires an ungrouped consumed prefix')
        path = old / f'step_{step:06d}.pt'
        saved = checked(path)
        backend = install_backend()
        if saved['step'] != step or saved['identity']['experiment'] != base or saved['identity']['world'] != 32:
            raise ValueError('Original checkpoint/configuration mismatch')
        if saved['identity']['source'] != backend.source_identity():
            raise ValueError('Training implementation changed')
        sample_list = Path(base['j_cache']) / 'sample_list.csv'
        with sample_list.open() as f:
            rows = [dict(requested_duration_s=float(r['duration_s'])) for r in csv.DictReader(f)]
        pairs = make_pairs(rows, base['training']['steps'], 32, base['training']['seed'], step, 'duration', 512)
        before = Schedule(base).pairs
        if pairs[:step*32] != before[:step*32] or Counter(pairs[step*32:]) != Counter(before[step*32:]):
            raise ValueError('Consumed prefix or remaining sample/ordinal multiset changed')
        plan = out / 'duration.schedule.json'
        atomic_json(plan, dict(sample_list_sha256=sha(sample_list), pairs=pairs), immutable=True)
        c = copy.deepcopy(base)
        c['output'] = str(out)
        c['training']['save_every'] = 500
        c['duration_sampling'].update(start_step=step, manifest=str(plan), sha256=sha(plan))
        config = out / 'config.json'
        atomic_json(config, c, immutable=True)
        load(config, formal=True); Schedule(c)
        reverted = copy.deepcopy(c)
        reverted['output'] = base['output']; reverted['duration_sampling'] = base['duration_sampling']
        reverted['training']['save_every'] = base['training']['save_every']
        if reverted != base or backend.data_identity(c) != saved['identity']['data_identity']:
            raise ValueError('Unapproved experiment or dataset change')
        identity = copy.deepcopy(saved['identity']); identity['experiment'] = c
        target = out / path.name
        atomic_torch(target, dict(saved, identity=identity))
        atomic_json(target.with_suffix('.complete.json'), dict(sha256=sha(target), step=step))
        verify = checked(target)
        for key in ('model', 'optimizer', 'rank_rng'):
            if not equal_tensors(saved[key], verify[key]):
                raise ValueError('Migration changed ' + key)
        atomic_json(out / 'run.json', identity, immutable=True)
        pre = read(old / 'preflight.json')
        pre['reused_full_preflight'] = dict(path=str(old / 'preflight.json'), sha256=sha(old / 'preflight.json'),
            reason='Only remaining sample order, checkpoint interval, and output changed; same data/source identity verified')
        atomic_json(out / 'preflight.json', pre, immutable=True)
        atomic_json(out / 'migration.json', dict(source_checkpoint=str(path), source_sha256=sha(path), step=step,
            first_grouped_step=step+1, save_every=500, consumed_prefix_identical=True,
            remaining_sample_ordinals_identical=True, model_optimizer_rng_bitwise_unchanged=True,
            source_unchanged=True, data_identity_unchanged=True), immutable=True)
        atomic_json(old / 'CURRENT_RUN.json', dict(output=str(out), config=str(config), preserved_steps=step,
            reason='User requested immediate duration grouping and checkpoints every 500 updates'))
        old_state = read(old / 'queue/state.json')
        atomic_json(old / 'queue/state_before_authorized_migration.json', old_state)
        atomic_json(old / 'queue/state.json', dict(state='superseded_after_authorized_stop', step=step, output=str(out)))
        del saved, verify
        cluster = read(ROOT / 'configs/p2_j_av_feature_base025_32gpu_20261102.json')
        cmds = commands(cluster, config, resume=target)
        for cmd in cmds:
            cmd[cmd.index(str(ROOT / 'scripts/p2_clip_events.py'))] = str(ROOT / 'scripts/p2_j_duration_train.py')
        atomic_json(out / 'launch_commands.json', dict(cluster=cluster, commands=cmds))
        status('training', resume_step=step, first_grouped_step=step+1, world=32, save_every=500)
        launch(cluster, cmds, out / 'logs')
        final = read(out / 'status.json')
        status('complete' if final['complete'] else 'stopped', **final)
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
