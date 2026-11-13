"""Authorized handoff: preserve Whisper step8000, then strictly resume the 28-rank run."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from p2_listen.contracts import read, sha, atomic_json
from p2_speaker_whisper_after_annotation import process_record, alive_record


def validate_source(identity):
    bad = [name for name, expected in identity['source'].items()
           if not Path(name).is_file() or sha(name) != expected]
    if bad:
        raise ValueError('Resume source changed: ' + repr(bad))


def marker_ready(path, step):
    marker = Path(path).with_suffix('.complete.json')
    return Path(path).is_file() and marker.is_file() and read(marker).get('step') == step


def verify_objective(row):
    keys = ('av', 'event', 'feature_video', 'feature_audio', 'weighted_total')
    if not all(math.isfinite(row[k]) for k in keys):
        raise ValueError('Nonfinite resumed losses')
    expected = row['av'] + .05 * row['event'] + .025 * (row['feature_video'] + row['feature_audio'])
    if not math.isclose(expected, row['weighted_total'], rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError('Resumed objective differs')
    if not all(math.isfinite(row['grad_norm'][k]) and row['grad_norm'][k] > 0
               for k in ('connector', 'events', 'features', 'text_lora')):
        raise ValueError('Resumed gradient branch missing')


def enough_pid_capacity(current, maximum, minimum=4096):
    return maximum == 'max' or int(maximum) - int(current) >= minimum


def pid_capacity(cluster):
    """NCCL/torch/SSH need thread slots even when GPU memory is completely free."""
    from concurrent.futures import ThreadPoolExecutor
    from listen_cluster import transport
    code = ("from pathlib import Path; import json; "
            "p=Path('/sys/fs/cgroup'); "
            "print(json.dumps(dict(current=int((p/'pids.current').read_text()),"
            "maximum=(p/'pids.max').read_text().strip())))")
    def one(node):
        try:
            p = subprocess.run(transport(cluster, node, [cluster['python'], '-c', code]),
                               capture_output=True, text=True, timeout=25)
            if p.returncode:
                return dict(node=node['name'], check='pid_capacity', free=False, error=p.stderr[-500:])
            row = json.loads(p.stdout.strip().splitlines()[-1])
            return dict(node=node['name'], check='pid_capacity',
                        free=enough_pid_capacity(row['current'], row['maximum']),
                        minimum_headroom=4096, **row)
        except Exception as exc:
            return dict(node=node['name'], check='pid_capacity', free=False, error=repr(exc))
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, cluster['nodes']))


def run(work):
    work = Path(work).resolve()
    spec = read(work / 'spec.json')
    source = Path(spec['source_output']); target = Path(spec['target_output'])
    sq = source / 'queue'; tq = target / 'queue'
    def state(phase, **kw):
        row = dict(phase=phase, at=time.time(), **kw)
        atomic_json(work / 'state.json', row)
        print(json.dumps(row), flush=True)
    with (work / 'controller.lock').open('a') as lock, (tq / 'controller.lock').open('a') as target_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(target_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(work / 'controller.json', process_record())
        try:
            config = read(spec['target_config']); identity = read(target / 'run.json')
            if (identity['experiment'] != config or identity['world'] != 28 or
                    config['training']['global_batch'] != 28 or config['loss'] != 'av_event_feature'):
                raise ValueError('Strict 28-rank resume contract differs')
            for key in ('source_config', 'target_config', 'target_cluster'):
                if sha(spec[key]) != spec[key + '_sha256']:
                    raise ValueError('Handoff configuration changed: ' + key)
            validate_source(identity)
            initial = Path(spec['target_resume'])
            if not marker_ready(initial, 5500) or sha(initial) != read(initial.with_suffix('.complete.json'))['sha256']:
                raise ValueError('Target step5500 is not complete')
            state('waiting_whisper_step8000', target_resume=str(initial))
            while read(source / 'status.json')['step'] < 8000:
                if (work / 'STOP').exists():
                    state('cancelled_before_handoff'); return
                time.sleep(1)
            # The trainer logs before its scheduled save. Output/STOP is checked
            # only on the NEXT iteration, after that complete save returns.
            atomic_json(source / 'STOP', dict(reason='User requested step8000 handoff', handoff=str(work), at=time.time()))
            state('waiting_whisper_checkpoint_and_graceful_exit')
            cp = source / 'step_008000.pt'
            deadline = time.monotonic() + 1800
            while True:
                phase = read(sq / 'state.json').get('phase')
                if marker_ready(cp, 8000) and phase == 'stopped':
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('Whisper checkpoint/normal shutdown incomplete; no target launched')
                time.sleep(2)
            if sha(cp) != read(cp.with_suffix('.complete.json'))['sha256']:
                raise ValueError('Whisper step8000 SHA mismatch')
            controller = read(sq / 'controller.json')
            while alive_record(controller, ROOT / 'scripts/p2_whisper_train_after_cache.py', 'run'):
                time.sleep(1)
            # Only now may queue/STOP disarm future restarts: it must never
            # cancel the active checkpoint writer or its distributed peers.
            atomic_json(sq / 'STOP', dict(reason='Completed authorized step8000 handoff', handoff=str(work)))
            atomic_json(work / 'SOURCE_SAVED.json', dict(step=8000, checkpoint=str(cp),
                sha256=read(cp.with_suffix('.complete.json'))['sha256'], source_status=read(source / 'status.json')))
            from p2_j_event_feature_launch import commands, resources
            from listen_cluster import launch
            cluster = read(spec['target_cluster'])
            if sum(len(n['gpus']) for n in cluster['nodes']) != 28:
                raise ValueError('Target cluster is not the original 28 ranks')
            # Keep a copy of the unsaved step5501..5514 metrics before the
            # unchanged trainer truncates them to its actual resumed checkpoint.
            if not (work / 'metrics_before_resume.jsonl').exists():
                shutil.copyfile(target / 'metrics.jsonl', work / 'metrics_before_resume.jsonl')
            for attempt in range(1, 6):
                state('waiting_selected_28_gpus', attempt=attempt)
                while True:
                    checks = resources(cluster) + pid_capacity(cluster)
                    atomic_json(work / 'resources.json', dict(at=time.time(), nodes=checks))
                    if (work / 'STOP').exists():
                        state('stopped_before_target_launch'); return
                    if all(n['free'] for n in checks):
                        break
                    time.sleep(10)
                validate_source(identity)
                # A failed startup may have made real updates. Resume only the
                # newest complete checkpoint; the trainer restores its RNG/Adam.
                markers = sorted(target.glob('step_*.complete.json'), reverse=True)
                marker = next(m for m in markers if read(m)['step'] >= 5500)
                resume = marker.with_suffix('').with_suffix('.pt')
                if sha(resume) != read(marker)['sha256']:
                    raise ValueError('Target checkpoint SHA mismatch')
                if (target / 'STOP').exists():
                    raise ValueError('Target has an explicit STOP; not removing it automatically')
                cmds = commands(cluster, Path(spec['target_config']), resume=resume)
                logs = work / ('attempt_' + str(attempt) + '_' + str(time.time_ns()))
                atomic_json(work / 'launch_commands.json', dict(cluster=cluster, commands=cmds, resume=str(resume)))
                state('target_training', resume_step=read(marker)['step'], world=28, logs=str(logs))
                atomic_json(tq / 'state.json', dict(state='training', time=time.time(), world=28,
                    budget=10202, resume_step=read(marker)['step'], handoff=str(work)))
                cancel = threading.Event(); done = threading.Event()
                began = time.time(); old = (target / 'metrics.jsonl').stat().st_mtime
                def watch():
                    try:
                        while not done.wait(10):
                            if (work / 'STOP').exists():
                                # Graceful request, not launch cancellation.
                                if not (target / 'STOP').exists():
                                    atomic_json(target / 'STOP', dict(reason='Handoff controller stop requested'))
                            stamp = (target / 'metrics.jsonl').stat().st_mtime
                            if time.time() - (stamp if stamp > old else began) > (900 if stamp > old else 1800):
                                atomic_json(work / 'STALL.json', dict(at=time.time(), logs=str(logs)))
                                cancel.set(); return
                            if stamp > old:
                                with (target / 'metrics.jsonl').open('rb') as stream:
                                    stream.seek(max(0, stream.seek(0, 2) - 16000))
                                    lines = stream.read().splitlines()
                                try: row = json.loads(lines[-1])
                                except (ValueError, IndexError): continue
                                if row['step'] > read(marker)['step']:
                                    verify_objective(row)
                                    if not (work / 'RESUME_VERIFIED.json').exists():
                                        atomic_json(work / 'RESUME_VERIFIED.json', dict(at=time.time(),
                                            resume_step=read(marker)['step'], row=row, world=28,
                                            source_unchanged=True, config_unchanged=True))
                    except Exception as exc:
                        atomic_json(work / 'WATCH_FAILED.json', dict(at=time.time(), error=repr(exc)))
                        cancel.set()
                thread = threading.Thread(target=watch, daemon=True); thread.start()
                try:
                    launch(cluster, cmds, logs, cancel_event=cancel)
                except Exception as exc:
                    state('target_retry_pending', attempt=attempt, error=repr(exc))
                    if attempt == 5: raise
                    time.sleep(15)
                    continue
                finally:
                    done.set(); thread.join(timeout=2)
                final = read(target / 'status.json')
                phase = 'complete' if final.get('complete') and final['step'] == 10202 else 'stopped'
                state(phase, **final)
                atomic_json(tq / 'state.json', dict(state=phase, time=time.time(), **final, handoff=str(work)))
                if phase != 'complete' and not (target / 'STOP').exists():
                    raise ValueError('Target ended without full budget or requested stop')
                return
        except Exception as exc:
            state('needs_attention', error=repr(exc)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    run(args.work)
