"""Recover the stalled AV+Feature run using existing 3/4 checkpointing."""
import copy
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from p2_listen.contracts import read, atomic_json, sha
from p2_j_duration_execution import Schedule, install_backend
from p2_j_duration_benchmark import equal_tensors
from p2_j_s1_queue import commands
from listen_cluster import launch


def main():
    import torch
    from p2_matrix16.checkpoint import checked
    from p2_listen.cache import atomic_torch
    from p2_matrix16.config import load
    torch.set_num_threads(2)
    root = ROOT / 'experiments/p2_j_guided_av_feature_base025_20261102'
    old = root / 'duration_now_save500'
    out = root / 'duration_memorysafe_save500'
    out.mkdir(exist_ok=False)
    def status(state, **kw):
        atomic_json(out / 'controller_state.json', dict(state=state, time=time.time(), **kw))
        print(json.dumps(dict(state=state, **kw)), flush=True)
    try:
        status('recovering_checkpoint')
        base = load(old / 'config.json', formal=True)
        checkpoint = old / 'step_000070.pt'
        saved = checked(checkpoint)
        backend = install_backend()
        if saved['identity']['experiment'] != base or saved['identity']['source'] != backend.source_identity():
            raise ValueError('Checkpoint experiment/source mismatch')
        c = copy.deepcopy(base); c['output'] = str(out)
        c['training']['gradient_checkpointing'] = 'three_quarters'
        config = out / 'config.json'; atomic_json(config, c, immutable=True)
        load(config, formal=True)
        if Schedule(c).pairs != Schedule(base).pairs or backend.data_identity(c) != saved['identity']['data_identity']:
            raise ValueError('Recovery changed samples or data')
        before = copy.deepcopy(c); before['output'] = base['output']
        before['training']['gradient_checkpointing'] = base['training']['gradient_checkpointing']
        if before != base:raise ValueError('Recovery changed more than checkpointing/output')
        identity = copy.deepcopy(saved['identity']); identity['experiment'] = c
        target = out / checkpoint.name
        atomic_torch(target, dict(saved, identity=identity))
        atomic_json(target.with_suffix('.complete.json'), dict(sha256=sha(target), step=70))
        verify = checked(target)
        for k in ('model', 'optimizer', 'rank_rng'):
            if not equal_tensors(saved[k], verify[k]):raise ValueError('Recovery changed ' + k)
        atomic_json(out / 'run.json', identity, immutable=True)
        pre = read(old / 'preflight.json')
        pre['reused_full_preflight'] = dict(path=str(old / 'preflight.json'), sha256=sha(old / 'preflight.json'),
            reason='Same source and dataset; only supported gradient checkpointing policy/output changed')
        atomic_json(out / 'preflight.json', pre, immutable=True)
        atomic_json(out / 'migration.json', dict(source=str(checkpoint), source_sha256=sha(checkpoint), step=70,
            model_optimizer_rng_bitwise_unchanged=True, sample_schedule_identical=True, data_identity_identical=True,
            previous_last_logged_step=161, reexecuted_updates=91,
            change='alternating -> three_quarters checkpointing; mathematical objective unchanged',
            diagnostic_wrapper_sha256=sha(ROOT / 'scripts/p2_j_train_observed.py')), immutable=True)
        del saved, verify
        atomic_json(old / 'controller_state.json', dict(state='stalled_recovered_from_checkpoint', last_step=161,
            blocked_step=162, output=str(out), root_cause='unconfirmed; node4 GPU2 stalled near memory limit'))
        atomic_json(root / 'CURRENT_RUN.json', dict(output=str(out), config=str(config), resume_step=70,
            reason='Recover step162 stall with existing three_quarters gradient checkpointing'))
        cluster = read(ROOT / 'configs/p2_j_av_feature_base025_32gpu_20261102.json')
        cluster['master_port'] = 29839
        cmds = commands(cluster, config, resume=target)
        for cmd in cmds:
            cmd[cmd.index(str(ROOT / 'scripts/p2_clip_events.py'))] = str(ROOT / 'scripts/p2_j_train_observed.py')
        atomic_json(out / 'launch_commands.json', dict(cluster=cluster, commands=cmds))
        cancel = threading.Event(); done = threading.Event()
        def watch():
            began = time.time()
            while not done.wait(15):
                f = out / 'metrics.jsonl'
                age = time.time() - (f.stat().st_mtime if f.exists() else began)
                limit = 180 if f.exists() else 600
                if age > limit:
                    atomic_json(out / 'STALL.json', dict(time=time.time(), no_progress_seconds=age,
                        limit=limit, action='cancel only owned workers; preserve last complete checkpoint'))
                    cancel.set(); return
        monitor = threading.Thread(target=watch, daemon=True); monitor.start()
        status('training', resume_step=70, world=32, gradient_checkpointing='three_quarters', save_every=500)
        try:launch(cluster, cmds, out / 'logs', cancel_event=cancel)
        finally:done.set();monitor.join(timeout=2)
        final = read(out / 'status.json')
        status('complete' if final['complete'] else 'stopped', **final)
    except Exception as exc:
        status('failed', error=repr(exc));raise


if __name__ == '__main__':main()
