"""Deadline-aware, persistent four-node CPU queue for the full Seamless inventory."""
import argparse
import collections
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

import run_listen_pair_j_cluster as bounded
from listen_pair_j_queue import Queue, Guard, lock, locked, read, write, sha, metrics, slow

ROOT = bounded.ROOT
SCRIPT = Path(__file__).resolve()
CONFIG = ROOT / 'configs/listen_pair_j_full_cpu_cluster.json'
EXTRA_CODE = ['scripts/run_listen_pair_j_full_cluster.py', 'scripts/listen_pair_j_full_runtime.py',
              'configs/listen_pair_j_full_cpu_cluster.json']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def initialize(args):
    work = args.work.resolve()
    if any(work == p or p in work.parents for p in (bounded.PROD, bounded.PREV)):
        raise ValueError('Independent full-run output required')
    work.mkdir(parents=True, exist_ok=True)
    with locked(work / 'init.lock'):
        if (work / 'manifest.json').exists():
            bounded.verify(work); return
        frozen = read(bounded.RELEASE / 'manifest.json')
        for p, h in frozen['code_sha256'].items():
            if sha(ROOT / p) != h: raise ValueError('J source changed: ' + p)
        cfg = read(args.config)
        catalog_path = bounded.PROD / 'catalog.json'
        raw_path = ROOT / 'data_corpora/seamless_listen_v7_full/source_pair_index.json'
        catalog = read(catalog_path); official = {r['pair']: r for r in catalog['rows']}
        if sha(raw_path) != catalog['source_index_sha256']:
            raise ValueError('Raw inventory fingerprint changed')
        items = []
        for raw in read(raw_path):
            pair = raw['label'] + '_' + raw['interaction']
            if pair in official:
                row = dict(official[pair], source_catalog_status='existing_catalog')
            else:
                row = dict(raw, pair=pair, group_id='excluded_' + pair,
                           source_catalog_status='previously_excluded_identity_overlap',
                           training_admitted=False)
            items.append(row)
        random.Random(args.seed).shuffle(items)
        if args.pairs:
            names = set(read(args.pairs)); items = [r for r in items if r['pair'] in names]
            if len(items) != len(names): raise ValueError('Unknown explicit source')
        assert len({r['pair'] for r in items}) == len(items)
        inputs = {}
        for name in ['feedback.json', 'qc4_review_feedback.json', 'human_rejections.json', 'human_retained.json']:
            dest = work / 'inputs' / name; write(dest, read(bounded.PROD / name))
            inputs[str(dest.relative_to(work))] = sha(dest)
        dest = work / 'inputs/listener_noise_rejections.json'
        write(dest, read(ROOT / 'data_reviews/listener_noise_rejections.json'))
        inputs[str(dest.relative_to(work))] = sha(dest)
        known = {}
        for name in ['pair_head_seven_20261029', 'pair_semantic_random5_20261029']:
            for p in (ROOT / 'experiments' / name).glob('*.json'):
                if p.stem in official:
                    known[p.stem] = dict(snapshot=str(p), snapshot_sha256=sha(p))
        cached = {}
        pilot = ROOT / 'experiments/pair_j_cluster_pilot_20261029/run'
        if (pilot / 'manifest.json').exists():
            pm = bounded.verify(pilot)
            if pm['input_sha256'] != inputs:
                raise ValueError('Pilot feedback differs from full-run feedback')
            for item in pm['items']:
                folder = pilot / 'results' / item['pair']; receipt = folder / 'complete.json'
                if receipt.exists():
                    r = read(receipt); path = folder / 'result.json'
                    if r['contract'] != pm['contract'] or sha(path) != r['result_sha256']:
                        raise ValueError('Pilot completion corrupted')
                    cached[item['pair']] = dict(result=str(path), result_sha256=r['result_sha256'],
                                               initial=str(folder / 'initial.json'), contract=pm['contract'])
        code = dict(frozen['code_sha256'])
        code.update({p: sha(ROOT / p) for p in bounded.RUNTIME + EXTRA_CODE})
        requested = args.requested_at or time.time()
        manifest = dict(revision='j-full-cpu-deadline-1', policy=frozen['version'],
                        policy_revision=frozen['revision'], nodes=cfg['nodes'], config=cfg,
                        items=items, code_sha256=code, input_sha256=inputs,
                        source_inputs={}, known_snapshots=known, completed_j_cache=cached,
                        baseline_roots=[str(bounded.PROD), str(bounded.PREV)],
                        extra_cache_dirs=[str(ROOT / 'experiments' / x / 'results') for x in
                            ['pair_semantic_trial2_20261029', 'pair_semantic_remaining5_20261029',
                             'pair_semantic_random5_20261029', 'pair_j_cluster_pilot_20261029/run']],
                        source_index=str(raw_path), source_index_sha256=sha(raw_path),
                        old_catalog_sha256=sha(catalog_path), max_attempts=3,
                        seed=args.seed, requested_at=requested, deadline=requested + cfg['deadline_hours'] * 3600,
                        scope='full_raw_inventory' if not args.pairs else 'explicit_regression',
                        training_admitted=False, device='cpu', render_media=False)
        manifest['contract'] = digest(manifest); write(work / 'manifest.json', manifest)
        Queue(work).initialize()
        write(work / 'control.json', dict(stop=True, enabled={}, session=None, heartbeat=0))
    print(json.dumps(dict(work=str(work), sources=len(items), deadline=manifest['deadline'],
                          contract=manifest['contract'])), flush=True)


def key(node, slot):
    return f'{node}.w{slot:02d}'


def start_slots(work, m, node, slots, session):
    if not slots: return {}
    payload = dict(work=str(work), script=str(SCRIPT), python=m['config']['python'],
                   cwd=str(ROOT), node=node['name'], slots=slots, session=session)
    code = """import json,subprocess,sys,os,time
from pathlib import Path
p=json.load(sys.stdin);out={};work=Path(p['work']);(work/'logs').mkdir(exist_ok=True)
for slot in p['slots']:
 k=p['node']+'.w'+str(slot).zfill(2)
 with (work/'logs'/(k+'.'+p['session']+'.log')).open('a') as f:
  args=[p['python'],'-u',p['script'],'worker','--work',p['work'],'--node',p['node'],'--slot',str(slot),'--session',p['session']]
  child=subprocess.Popen(args,cwd=p['cwd'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
  out[k]=dict(pid=child.pid,start_ticks=ticks,slot=slot,node=p['node'],session=p['session'],launched=time.time())
print(json.dumps(out))"""
    values = bounded.remote(node, m['config'], code, payload)
    for k, value in values.items(): write(work / 'workers' / (k + '.launch.json'), value)
    return values


def stop_slots(work, m, node, launches, force=False):
    if not launches: return {}
    code = """import json,os,sys,signal
from pathlib import Path
p=json.load(sys.stdin);out={}
for k,v in p['launches'].items():
 r=Path('/proc')/str(v['pid'])
 try:
  fields=(r/'stat').read_text().rsplit(')',1)[1].split()
  if fields[0]=='Z':out[k]='exited';continue
  args=(r/'cmdline').read_bytes().decode().split('\\0')
  if fields[19]!=v['start_ticks'] or p['work'] not in args or v['session'] not in args or p['script'] not in args:
   out[k]='identity_mismatch';continue
  os.killpg(v['pid'],signal.SIGKILL if p['force'] else signal.SIGTERM);out[k]='signalled'
 except FileNotFoundError:out[k]='exited'
print(json.dumps(out))"""
    return bounded.remote(node, m['config'], code, dict(launches=launches, work=str(work),
                                                        script=str(SCRIPT), force=force))


def worker(args):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1')
    if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0): raise RuntimeError('THP disable failed')
    allowed = sorted(os.sched_getaffinity(0)); upper = len(allowed) - 2 * args.slot
    affinity = allowed[upper - 2:upper]
    if len(affinity) != 2: raise ValueError('Insufficient CPU affinity slots')
    os.sched_setaffinity(0, affinity); os.nice(10)
    work = args.work.resolve(); m = bounded.verify(work)
    node = next(n for n in m['nodes'] if n['name'] == args.node)
    if socket.gethostname() != node['hostname']: raise ValueError('Wrong node')
    k = key(args.node, args.slot); slot_lock = lock(work / 'workers' / (k + '.slot.lock'), False)
    if slot_lock is None: raise RuntimeError('Worker slot already held')
    stopped = threading.Event(); finished = threading.Event(); mutex = threading.Lock()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    state = dict(node=args.node, slot=args.slot, session=args.session, pid=os.getpid(),
                 affinity=affinity, torch_threads=2, blas_threads=1, device='cpu', stage='starting',
                 pair=None, completed_pairs=0, completed_clips=0, reused=0, fresh=0, failures=0)
    def progress(**kw):
        with mutex:
            state.update(kw, heartbeat=time.time(), cpu_seconds=time.process_time())
            write(work / 'workers' / (k + '.status.json'), state)
    def check():
        c = read(work / 'control.json'); now = time.time()
        stop_file = work / 'stop.request.json'
        operator_stop = stop_file.exists() and read(stop_file)['at'] >= c.get('started', 0)
        if (stopped.is_set() or operator_stop or c.get('stop') or c.get('session') != args.session or
                args.slot >= c.get('enabled', {}).get(args.node, 0) or
                now - c.get('heartbeat', 0) > m['config']['lease_seconds']):
            from listen_pair_j_runtime import Paused
            raise Paused('Stopped, disabled, or controller lease lost')
    def pulse():
        while not finished.wait(10): progress()
    threading.Thread(target=pulse, daemon=True).start(); progress()
    from listen_pair_j_full_runtime import FullEngine
    from listen_pair_j_runtime import Paused
    handle = None
    try:
        check(); engine = FullEngine(work, check, progress); queue = Queue(work)
        while True:
            check(); claim = queue.claim(args.node)
            if claim is None:
                progress(stage='idle', pair=None); time.sleep(5); continue
            pair, handle = claim
            try:
                result = engine.process(queue.items[pair]); queue.finish(pair)
                counts = result.get('counts', {})
                progress(stage='between_pairs', pair=None, completed_pairs=state['completed_pairs'] + 1,
                         completed_clips=state['completed_clips'] + len(result['clips']),
                         reused=state['reused'] + counts.get('reused', 0) + counts.get('reused_completed_result_attempts', 0),
                         fresh=state['fresh'] + counts.get('fresh', 0))
            except Paused: raise
            except Exception as e:
                import traceback
                traceback.print_exc(); queue.finish(pair, repr(e))
                write(work / 'errors' / (pair + '.json'), dict(error=repr(e), node=args.node, slot=args.slot, at=time.time()))
                progress(stage='pair_failed', pair=None, failures=state['failures'] + 1)
            finally:
                handle.close(); handle = None
    except Paused as e: progress(stage='paused', reason=str(e))
    except BaseException as e:
        progress(stage='error', error=repr(e)); raise
    finally:
        finished.set()
        if handle: handle.close()
        slot_lock.close()


def required_workers(remaining, per_worker_rate, seconds_left, margin=.85):
    if remaining <= 0: return 0
    if per_worker_rate <= 0 or seconds_left <= 0: return None
    return math.ceil(remaining / (per_worker_rate * seconds_left * margin))


def controller(args):
    work = args.work.resolve(); m = bounded.verify(work); cfg = m['config']
    owner = lock(work / 'controller.lock', False)
    if owner is None: raise RuntimeError('Controller already running')
    session = uuid.uuid4().hex[:12]; began = time.time(); names = [n['name'] for n in m['nodes']]
    enabled = {n: cfg['initial_workers_per_node'] for n in names}
    launches = {}; events = []; points = []; samples = []; history = []
    status = dict(session=session, started=began, deadline=m['deadline'], total=len(m['items']), events=events)
    baseline_path = work / 'training_baseline.json'
    log = bounded.TrainingLog(cfg['training_log']); rows, _ = log.fetch()
    if baseline_path.exists(): baseline = read(baseline_path)
    else:
        if len(rows) < 100: raise ValueError('Missing 100-step training baseline')
        baseline = metrics(rows[-100:]); write(baseline_path, baseline)
    guard = Guard(baseline); status['baseline'] = baseline
    last_training = began; last_change = began; last_launch_check = 0; ramp_index = 0
    def control(stop=False):
        write(work / 'control.json', dict(session=session, started=began, heartbeat=time.time(),
                                         stop=stop, enabled={} if stop else enabled))
    def record(reason, **kw):
        events.append(dict(at=time.time(), reason=reason, **kw)); print(json.dumps(events[-1]), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        # Drain any still-live workers from an interrupted controller session first.
        old_control = read(work / 'control.json'); control(stop=True)
        old_launches = {}
        for node in m['nodes']:
            values = {}
            for slot in range(cfg['maximum_workers_per_node']):
                p = work / 'workers' / (key(node['name'], slot) + '.launch.json')
                if p.exists(): values[key(node['name'], slot)] = read(p)
            if values:
                stop_slots(work, m, node, values); old_launches[node['name']] = values
        if old_launches:
            time.sleep(15)
            for node in m['nodes']:
                if node['name'] in old_launches: stop_slots(work, m, node, old_launches[node['name']], force=True)
        control()
        for node in m['nodes']:
            launches.update(start_slots(work, m, node, list(range(enabled[node['name']])), session))
        record('started', enabled=dict(enabled))
        q = read(work / 'queue.json'); start_completed = q['completed']
        points.append((time.time(), q['completed'], sum(enabled.values())))
        while True:
            now = time.time(); request = work / 'stop.request.json'
            if request.exists() and read(request)['at'] >= began:
                status['state'] = 'operator_stopped'; break
            control(); q = read(work / 'queue.json'); fresh, _ = log.fetch()
            if fresh: last_training = now; history.extend(fresh)
            current = {}
            for k in launches:
                p = work / 'workers' / (k + '.status.json')
                if p.exists():
                    d = read(p)
                    if d.get('session') == session: current[k] = d
            active = {n: sum(d['node'] == n and d.get('stage') not in ('idle', 'paused', 'error') and
                             now - d.get('heartbeat', 0) < 40 for d in current.values()) for n in names}
            points.append((now, q['completed'], sum(active.values())))
            points = [p for p in points if p[0] >= now - 600]
            elapsed = now - points[0][0]; completed_window = q['completed'] - points[0][1]
            worker_seconds = sum((b[0] - a[0]) * a[2] for a, b in zip(points, points[1:]))
            per_worker_rate = completed_window / worker_seconds if worker_seconds > 0 else 0
            aggregate_rate = completed_window / elapsed if elapsed > 0 else 0
            remaining = len(m['items']) - q['completed']; seconds_left = m['deadline'] - now
            eta = now + remaining / aggregate_rate if aggregate_rate > 0 else None
            required = required_workers(remaining, per_worker_rate, seconds_left, cfg['target_utilization_fraction'])
            current_metrics = metrics(history[-80:]) if len(history) >= 80 else None
            bad = guard.add(fresh)
            if now - last_training > cfg['training_stale_seconds']:
                if any(enabled.values()):
                    enabled = {n: 0 for n in names}; last_change = now
                    record('training_progress_stale_pause', enabled=dict(enabled)); control()
            elif bad and now - last_change > 120:
                enabled = {n: max(1, v // 2) for n, v in enabled.items()}; last_change = now
                ramp_index = len(cfg['ramp_workers_per_node']) - 1
                record('sustained_training_slowdown_reduce', enabled=dict(enabled), metrics=current_metrics)
                guard = Guard(baseline); control()
            elif not any(enabled.values()) and fresh:
                enabled = {n: cfg['initial_workers_per_node'] for n in names}; last_change = now; ramp_index = 0
                guard = Guard(baseline); record('training_resumed_restart', enabled=dict(enabled)); control()
            elif now - last_change >= cfg['ramp_interval_seconds'] and not bad:
                if ramp_index + 1 < len(cfg['ramp_workers_per_node']):
                    ramp_index += 1; target = cfg['ramp_workers_per_node'][ramp_index]
                elif elapsed >= 180 and required is not None and (not current_metrics or not slow(current_metrics, baseline)):
                    target = min(cfg['maximum_workers_per_node'], max(enabled.values()) + 4,
                                 max(cfg['workers_per_node'], math.ceil(required / len(names))))
                else: target = max(enabled.values())
                if target > max(enabled.values()):
                    enabled = {n: target for n in names}; last_change = now
                    record('increase_for_deadline', enabled=dict(enabled), required_workers=required, eta=eta); control()
            # Enabled slots are reconciled in small per-node launch batches. Dead
            # workers are identity-checked before replacement; pair locks protect resume.
            if now - last_launch_check >= 30:
                for node in m['nodes']:
                    need = []; retiring = {}
                    for slot in range(cfg['maximum_workers_per_node']):
                        k = key(node['name'], slot); d = current.get(k, {}); launch = launches.get(k)
                        if slot >= enabled[node['name']]:
                            if launch and d.get('stage') not in ('paused', 'error'):
                                retiring[k] = launch
                            continue
                        if not launch: need.append(slot)
                        elif d.get('stage') in ('paused', 'error') or now - d.get('heartbeat', launch['launched']) > 90:
                            stop_slots(work, m, node, {k: launch}, force=True); need.append(slot)
                    if retiring: stop_slots(work, m, node, retiring)
                    if need:
                        try: launches.update(start_slots(work, m, node, need, session))
                        except Exception as e: record('launch_retry_later', node=node['name'], error=repr(e))
                last_launch_check = now
            status.update(state='running', at=now, completed=q['completed'], failed=len(q['failed']),
                          remaining=remaining, inflight=len(q['inflight']), enabled=enabled, active=active,
                          sources_per_second=aggregate_rate, per_worker_sources_per_second=per_worker_rate,
                          eta=eta, seconds_left=seconds_left, required_workers=required,
                          deadline_at_risk=bool(eta and eta > m['deadline']), training=current_metrics,
                          training_progress_age=now - last_training,
                          completed_this_session=q['completed'] - start_completed)
            write(work / 'status.json', status)
            with (work / 'telemetry.jsonl').open('a') as f:
                f.write(json.dumps({k: v for k, v in status.items() if k not in ('events', 'baseline')}) + '\n')
            if q['completed'] + len(q['failed']) == len(m['items']) and not q['inflight']:
                status['state'] = 'complete' if not q['failed'] else 'finished_with_errors'; break
            if seconds_left <= 0 and not status.get('deadline_exceeded_recorded'):
                record('deadline_exceeded_continue_to_finish', remaining=remaining)
                status['deadline_exceeded_recorded'] = True
            time.sleep(cfg['controller_interval_seconds'])
    except BaseException as e:
        status.update(state='controller_error', error=repr(e)); raise
    finally:
        control(stop=True)
        for node in m['nodes']:
            try: stop_slots(work, m, node, {k: v for k, v in launches.items() if v['node'] == node['name']})
            except Exception as e: record('stop_error', node=node['name'], error=repr(e))
        status.update(ended=time.time()); write(work / 'status.json', status); owner.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['init', 'run', 'worker', 'status', 'stop', 'verify'])
    p.add_argument('--work', type=Path, required=True); p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--seed', type=int, default=20261103); p.add_argument('--pairs', type=Path)
    p.add_argument('--requested-at', type=float); p.add_argument('--node'); p.add_argument('--slot', type=int)
    p.add_argument('--session'); args = p.parse_args()
    if args.command == 'init': initialize(args)
    elif args.command == 'run': controller(args)
    elif args.command == 'worker':
        if args.node is None or args.slot is None or args.session is None: p.error('Worker identity required')
        worker(args)
    elif args.command == 'verify': print(bounded.verify(args.work)['contract'])
    elif args.command == 'status': print(json.dumps(read(args.work / 'status.json'), indent=2))
    elif args.command == 'stop': write(args.work / 'stop.request.json', dict(at=time.time()))


if __name__ == '__main__': main()
