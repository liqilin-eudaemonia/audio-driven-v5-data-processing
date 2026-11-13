"""Four-node, CPU-only J queue; bounded, training-aware pilot controller."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from listen_pair_j_queue import Queue, Guard, locked, lock, read, write, sha, metrics, slow

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
CONFIG = ROOT / 'configs/listen_pair_j_cpu_cluster.json'
PROD = ROOT / 'data_corpora/seamless_listen_local_full'
PREV = ROOT / 'data_corpora/seamless_listen_v8_audio_final_full'
RELEASE = ROOT / 'releases/listen_pair_segmentation_j_v1_20261029'
RUNTIME = ['scripts/listen_pair_j_queue.py', 'scripts/listen_pair_j_runtime.py',
           'scripts/run_listen_pair_j_cluster.py', 'configs/listen_pair_j_cpu_cluster.json']
REVISION = 'j-cpu-sharded-checkpoint-1'


def verify(work):
    m = read(Path(work) / 'manifest.json')
    identity = dict(m); identity.pop('contract')
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if digest != m['contract']:
        raise ValueError('Manifest contract mismatch')
    for rel, expected in m['code_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('Frozen code changed: ' + rel)
    for rel, expected in m['input_sha256'].items():
        if sha(Path(work) / rel) != expected:
            raise ValueError('Frozen run input changed: ' + rel)
    return m


def initialize(args):
    work = args.work.resolve()
    if work == PROD or PROD in work.parents or work == PREV or PREV in work.parents:
        raise ValueError('A separate output work directory is required')
    work.mkdir(parents=True, exist_ok=True)
    with locked(work / 'init.lock'):
        if (work / 'manifest.json').exists():
            verify(work); print('Already initialized:', work); return
        frozen = read(RELEASE / 'manifest.json')
        for rel, h in frozen['code_sha256'].items():
            if sha(ROOT / rel) != h:
                raise ValueError('J v1 source changed: ' + rel)
        cfg = read(args.config)
        sources = read(PROD / 'catalog.json')['rows']
        lookup = {r['pair']: r for r in sources}
        snapshots = {}
        for folder in ['pair_head_seven_20261029', 'pair_semantic_random5_20261029']:
            directory = ROOT / 'experiments' / folder
            # Only known snapshot filenames; no traversal into dataset directories.
            for path in directory.glob('*.json'):
                if path.stem in lookup:
                    snapshots[path.stem] = str(path)
        if args.pairs:
            names = read(args.pairs)
            if isinstance(names, dict):
                names = names['pairs']
            sources = [lookup[p] for p in names]
        else:
            random.Random(args.seed).shuffle(sources)
            sources = [r for r in sources if r['pair'] not in snapshots]
        items = []; inputs = {}; skipped = []
        for item in sources:
            pair = item['pair']
            paths = [PROD / 'results' / (pair + '.json'), PREV / 'results' / (pair + '.json'),
                     PROD / 'normalized_pairs' / (pair + '.json')]
            if not all(p.exists() for p in paths):
                skipped.append(dict(pair=pair, reason='missing_baseline_or_normalized_descriptor')); continue
            inputs[pair] = dict(norm=str(paths[2]), cache_results=[str(paths[1]), str(paths[0])],
                                files={str(p): sha(p) for p in paths})
            if pair in snapshots:
                inputs[pair].update(snapshot=snapshots[pair], snapshot_sha256=sha(snapshots[pair]))
            items.append(item)
            if not args.all and len(items) >= args.count and not args.pairs:
                break
        if not items:
            raise ValueError('No eligible sources')
        if args.pairs and len(items) != len(sources):
            raise ValueError('Explicit regression source is unavailable')
        input_hashes = {}
        for name in ['feedback.json', 'qc4_review_feedback.json', 'human_rejections.json', 'human_retained.json']:
            dest = work / 'inputs' / name; write(dest, read(PROD / name))
            input_hashes[str(dest.relative_to(work))] = sha(dest)
        dest = work / 'inputs/listener_noise_rejections.json'
        write(dest, read(ROOT / 'data_reviews/listener_noise_rejections.json'))
        input_hashes[str(dest.relative_to(work))] = sha(dest)
        code = dict(frozen['code_sha256']); code.update({rel: sha(ROOT / rel) for rel in RUNTIME})
        extra = [str(ROOT / 'experiments' / folder / 'results') for folder in
                 ['pair_semantic_trial2_20261029', 'pair_semantic_remaining5_20261029', 'pair_semantic_random5_20261029']]
        m = dict(revision=REVISION, policy=frozen['version'], policy_revision=frozen['revision'],
                 nodes=cfg['nodes'], config=cfg, items=items, source_inputs=inputs,
                 code_sha256=code, input_sha256=input_hashes, extra_cache_dirs=extra,
                 seed=args.seed, skipped=skipped, max_attempts=3, device='cpu', render_media=False,
                 scope='explicit_all' if args.all else 'bounded_pilot', training_admitted=False)
        m['contract'] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
        write(work / 'manifest.json', m)
        Queue(work).initialize()
        write(work / 'control.json', dict(session=None, enabled=[], stop=True, heartbeat=0, deadline=0))
    print(json.dumps(dict(work=str(work), sources=len(items), contract=m['contract'])))


def remote(node, cfg, code, payload):
    python = cfg['python']
    if socket.gethostname() == node['hostname']:
        args = [python, '-c', code]
    else:
        args = ['ssh', '-i', node.get('ssh_key', cfg['ssh_key']), '-p', str(node['ssh_port']),
                '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=8',
                '-o', 'StrictHostKeyChecking=yes', 'root@' + node['host'],
                shlex.join([python, '-c', code])]
    r = subprocess.run(args, input=json.dumps(payload), text=True, capture_output=True, timeout=20)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])
    return json.loads(r.stdout.splitlines()[-1])


def launch(work, m, node, session):
    code = """import json,subprocess,sys,os,time
from pathlib import Path
p=json.load(sys.stdin);log=Path(p['log']);log.parent.mkdir(parents=True,exist_ok=True)
with log.open('a') as out:
 child=subprocess.Popen(p['args'],cwd=p['cwd'],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
print(json.dumps(dict(pid=child.pid,start_ticks=ticks)))
"""
    args = [m['config']['python'], '-u', str(SCRIPT), 'worker', '--work', str(work),
            '--node', node['name'], '--session', session]
    value = remote(node, m['config'], code, dict(args=args, cwd=str(ROOT),
                   log=str(work / 'logs' / (node['name'] + '.' + session + '.log'))))
    value.update(node=node['name'], session=session, launched=time.time())
    write(work / 'nodes' / (node['name'] + '.launch.json'), value)
    return value


def terminate(work, m, node, session, force=False):
    path = work / 'nodes' / (node['name'] + '.launch.json')
    if not path.exists():
        return dict(absent=True)
    value = read(path)
    if value['session'] != session:
        return dict(other_session=True)
    code = """import json,os,signal,sys
from pathlib import Path
p=json.load(sys.stdin);root=Path('/proc')/str(p['pid'])
if not root.exists():print(json.dumps(dict(exited=True)));sys.exit()
ticks=(root/'stat').read_text().rsplit(')',1)[1].split()[19]
args=(root/'cmdline').read_bytes().decode().split('\\0')
if ticks!=p['start_ticks'] or p['work'] not in args or p['session'] not in args or p['script'] not in args:
 print(json.dumps(dict(identity_mismatch=True)));sys.exit()
os.killpg(p['pid'],signal.SIGKILL if p['force'] else signal.SIGTERM)
print(json.dumps(dict(signalled=True,force=p['force'])))
"""
    return remote(node, m['config'], code, dict(value, work=str(work), script=str(SCRIPT), force=force))


def worker(args):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1')
    if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
        raise RuntimeError('Could not disable per-process THP')
    allowed = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, allowed[-2:])
    os.nice(10)
    work = args.work.resolve(); m = verify(work)
    node = next(n for n in m['nodes'] if n['name'] == args.node)
    if socket.gethostname() != node['hostname']:
        raise ValueError('Worker launched on the wrong host')
    slot = lock(work / 'nodes' / (args.node + '.slot.lock'), False)
    if slot is None:
        raise RuntimeError('Node already has a worker')
    stopped = threading.Event(); finished = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    state = dict(node=args.node, session=args.session, pid=os.getpid(), hostname=socket.gethostname(),
                 affinity=sorted(os.sched_getaffinity(0)), torch_threads=2, blas_threads=1,
                 cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], stage='starting',
                 pair=None, completed_pairs=0, completed_clips=0, failures=0, resumed_pairs=0)
    mutex = threading.Lock()
    def progress(**kw):
        with mutex:
            state.update(kw, heartbeat=time.time(), cpu_seconds=time.process_time())
            write(work / 'nodes' / (args.node + '.status.json'), state)
    def pulse():
        while not finished.wait(5):
            progress()
    thread = threading.Thread(target=pulse, daemon=True); thread.start(); progress()
    from listen_pair_j_runtime import Engine, Paused
    def check():
        ctrl = read(work / 'control.json'); now = time.time()
        request = work / 'stop.request.json'
        operator_stop = request.exists() and read(request)['at'] >= ctrl.get('started', 0)
        if (operator_stop or stopped.is_set() or ctrl.get('session') != args.session or ctrl.get('stop') or
                args.node not in ctrl.get('enabled', []) or now > ctrl.get('deadline', 0) or
                now - ctrl.get('heartbeat', 0) > m['config']['lease_seconds']):
            raise Paused('Disabled, stopped, expired, or controller lease lost')
    queue = Queue(work); handle = None; pair = None
    try:
        check(); engine = Engine(work, check, progress)
        consecutive = 0
        while True:
            check(); claimed = queue.claim(args.node)
            if claimed is None:
                progress(stage='idle', pair=None); time.sleep(2); continue
            pair, handle = claimed
            try:
                result = engine.process(queue.items[pair]); queue.finish(pair)
                progress(stage='between_pairs', pair=None,
                         completed_pairs=state['completed_pairs'] + 1,
                         completed_clips=state['completed_clips'] + len(result['clips']),
                         resumed_pairs=state['resumed_pairs'] + int(result['checkpoint_resumes'] > 0))
                consecutive = 0
            except Paused:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                write(work / 'errors' / (pair + '.json'), dict(error=repr(e), node=args.node, at=time.time()))
                queue.finish(pair, repr(e)); consecutive += 1
                progress(stage='pair_failed', pair=None, failures=state['failures'] + 1)
                if consecutive >= 3:
                    raise RuntimeError('Three consecutive source failures') from e
            finally:
                handle.close(); handle = None
    except Paused as e:
        progress(stage='paused', reason=str(e), pair=pair)
    except BaseException as e:
        progress(stage='error', reason=repr(e)); raise
    finally:
        finished.set(); thread.join(timeout=6)
        if handle is not None:
            handle.close()
        slot.close()


class TrainingLog:
    def __init__(self, path):
        self.path = Path(path); self.identity = None; self.last_step = None; self.last_size = 0
    def fetch(self):
        stat = self.path.stat(); identity = (stat.st_dev, stat.st_ino)
        if self.identity is not None and self.identity != identity:
            raise ValueError('Training log replaced')
        self.identity = identity
        if stat.st_size < self.last_size:
            raise ValueError('Training log truncated')
        self.last_size = stat.st_size
        with self.path.open('rb') as f:
            f.seek(max(0, stat.st_size - 800000)); tail = f.read().decode(errors='replace')
        rows = []; previous_end = 0
        for match in re.finditer(r'TRAIN_PROGRESS (\{[^\r\n]*\})', tail):
            row = json.loads(match.group(1))
            phase = tail[previous_end:match.start()]
            previous_end = match.end()
            row['excluded_phase'] = bool(re.search(r'Validation:|VALIDATION_|Saving checkpoint|CHECKPOINT_', phase))
            if not row.get('warmup'):
                rows.append(row)
        if not rows:
            return [], stat.st_mtime
        if self.last_step is not None and rows[-1]['step'] < self.last_step:
            raise ValueError('Training step reset')
        new = [r for r in rows if not r['excluded_phase'] and (self.last_step is None or r['step'] > self.last_step)]
        self.last_step = rows[-1]['step']
        return new, stat.st_mtime


def statuses(work, names, session):
    values = {}
    for name in names:
        p = work / 'nodes' / (name + '.status.json')
        if p.exists():
            d = read(p)
            if d.get('session') == session:
                values[name] = d
    return values


def activity(samples, names, start):
    relevant = [s for s in samples if s['at'] >= start]
    fractions = {}
    for name in names:
        flags = []
        for s in relevant:
            d = s['workers'].get(name, {})
            flags.append(d.get('stage') in ('preparing', 'screening', 'loading_models') and
                         bool(d.get('pair')) and s['at'] - d.get('heartbeat', 0) < 20)
        fractions[name] = sum(flags) / len(flags) if flags else 0.
    return fractions


def pilot(args):
    work = args.work.resolve(); m = verify(work); cfg = m['config']
    controller_lock = lock(work / 'controller.lock', False)
    if controller_lock is None:
        raise RuntimeError('Another controller owns this work directory')
    session = uuid.uuid4().hex[:12]; names = []; launched = []; report = None
    end_time = time.time() + 2 * cfg['stage_max_seconds'] + 120
    report = dict(session=session, started=time.time(), stages=[], events=[], complete=False)
    def control(stop=False):
        request = work / 'stop.request.json'
        if not stop and request.exists() and read(request)['at'] >= report['started']:
            raise InterruptedError('Operator stop requested')
        write(work / 'control.json', dict(session=session, enabled=names if not stop else [],
              stop=stop, heartbeat=time.time(), deadline=end_time, started=report['started']))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        log = TrainingLog(cfg['training_log']); rows, mtime = log.fetch()
        if len(rows) < 100 or time.time() - mtime > cfg['training_stale_seconds']:
            raise ValueError('No fresh 100-step training baseline')
        baseline = metrics(rows[-100:]); report['baseline'] = baseline
        report['training_log'] = cfg['training_log']
        write(work / 'pilot_report.json', report)
        for target in [cfg['initial_nodes'], cfg['expanded_nodes']]:
            names = list(target); control()
            for name in names:
                if name not in launched:
                    node = next(n for n in m['nodes'] if n['name'] == name)
                    launch(work, m, node, session); launched.append(name)
            # Consume launch-period log rows; each stage starts a new comparison.
            log.fetch(); stage_start = time.time(); latest_training = stage_start
            history = []; samples = []; guard = Guard(baseline, cfg['sustained_window_steps'])
            stage = dict(nodes=list(names), started=stage_start, status='observing')
            report['stages'].append(stage)
            while time.time() < end_time:
                control(); now = time.time(); fresh, mtime = log.fetch()
                if fresh:
                    latest_training = now
                    for r in fresh:
                        r['observed_at'] = now
                    history.extend(fresh)
                current = statuses(work, names, session)
                sample = dict(at=now, workers=current, new_steps=len(fresh))
                samples.append(sample)
                stage.update(elapsed=now - stage_start, steps=len(history),
                             activity=activity(samples, names, stage_start), windows=guard.windows)
                write(work / 'telemetry' / (session + '.json'), dict(samples=samples, stage=stage))
                if now - latest_training > cfg['training_stale_seconds']:
                    stage['status'] = 'training_progress_stale'; break
                if any(d.get('stage') == 'error' for d in current.values()):
                    stage['status'] = 'worker_error'; break
                if now - stage_start > 60 and any(n not in current for n in names):
                    stage['status'] = 'worker_not_started'; break
                if guard.add(fresh):
                    stage['status'] = 'sustained_slowdown'
                    stage['last_metrics'] = metrics(history[-80:])
                    # A bounded measurement stops its own load immediately on
                    # sustained regression; queued work remains resumable.
                    reduced = []
                    report['events'].append(dict(at=now, reason='two_bad_nonoverlapping_windows_pause_cutting',
                                                  before=list(names), after=list(reduced)))
                    names = list(reduced); control()
                    break
                if len(history) >= cfg['stable_window_steps']:
                    recent = history[-cfg['stable_window_steps']:]
                    stage['last_metrics'] = metrics(recent)
                    fractions = activity(samples, names, recent[0]['observed_at'])
                    stage['comparison_activity'] = fractions
                    stable = not slow(stage['last_metrics'], baseline, cfg['slowdown_ratio'])
                    active = all(v >= cfg['minimum_active_fraction'] for v in fractions.values())
                    progressed = all(d.get('completed_pairs', 0) > 0 or d.get('attempted', 0) > 0
                                     for d in current.values()) and len(current) == len(names)
                    if now - stage_start >= cfg['stage_min_seconds'] and stable and active and progressed:
                        stage['status'] = 'stable_under_load'; break
                if now - stage_start >= cfg['stage_max_seconds']:
                    stage['status'] = 'inconclusive_at_time_limit'; break
                write(work / 'pilot_report.json', report)
                time.sleep(15)
            if stage['status'] == 'observing':
                stage['status'] = 'deadline'
            write(work / 'pilot_report.json', report)
            if stage['status'] != 'stable_under_load':
                break
        report['complete'] = len(report['stages']) == 2 and all(
            s['status'] == 'stable_under_load' for s in report['stages'])
    except BaseException as e:
        report['error'] = repr(e)
        raise
    finally:
        names = []; control(stop=True)
        stop_results = {}
        for name in launched:
            try:
                node = next(n for n in m['nodes'] if n['name'] == name)
                stop_results[name] = terminate(work, m, node, session)
            except Exception as e:
                stop_results[name] = dict(error=repr(e))
        # Workers checkpoint at segment boundaries; lease/deadline also protects
        # against controller crashes. Escalation only targets identity-verified PIDs.
        for _ in range(12):
            current = statuses(work, launched, session)
            if all(current.get(n, {}).get('stage') in ('paused', 'error') for n in launched):
                break
            time.sleep(5)
        for name in launched:
            d = statuses(work, [name], session).get(name, {})
            if d.get('stage') not in ('paused', 'error'):
                try:
                    node = next(n for n in m['nodes'] if n['name'] == name)
                    stop_results[name] = terminate(work, m, node, session, force=True)
                except Exception as e:
                    stop_results[name] = dict(error=repr(e))
        report.update(ended=time.time(), stop_results=stop_results,
                      queue=read(work / 'queue.json'), final_workers=statuses(work, launched, session))
        write(work / 'pilot_report.json', report)
        controller_lock.close()
    print(json.dumps(dict(complete=report['complete'], stages=[s['status'] for s in report['stages']],
                          completed_pairs=report['queue']['completed'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'worker', 'pilot', 'status', 'verify', 'stop'])
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--node'); parser.add_argument('--session')
    args = parser.parse_args()
    if args.command == 'init':
        if args.count < 1:parser.error('--count must be positive')
        initialize(args)
    elif args.command == 'worker':
        if not args.node or not args.session:parser.error('worker requires --node and --session')
        worker(args)
    elif args.command == 'pilot':pilot(args)
    elif args.command == 'verify':print(verify(args.work)['contract'])
    elif args.command == 'status':
        print(json.dumps(dict(queue=read(args.work / 'queue.json'), control=read(args.work / 'control.json')), indent=2))
    elif args.command == 'stop':
        # The controller sees an external stop request before its next heartbeat.
        write(args.work / 'stop.request.json', dict(at=time.time(), reason='operator_stop'))

if __name__ == '__main__':
    main()
