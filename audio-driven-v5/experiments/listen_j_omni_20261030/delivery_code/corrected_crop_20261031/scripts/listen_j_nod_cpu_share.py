#!/usr/bin/env python3
"""Borrow host-local CPU for an existing nod queue, preserving its output contract.

The original worker retains source claims, quality admission and result receipts.
Only its decode resource directory and collision-free worker slot are replaced.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from listen_benchmark.j_annotation import acquire, read, write
from listen_j_annotation_supervisor import alive, identity, spawn


def settings(work, node):
    cfg = read(work/'nod_cpu_share/config.json')
    host = read(work/'config.json')['nodes'][node]
    if socket.gethostname() != host['hostname']:
        raise ValueError('Wrong execution host')
    nc = cfg['nodes'][node]
    if not 0 <= nc['workers'] <= nc['max_workers'] <= 80:
        raise ValueError('Invalid bounded CPU worker count')
    if nc['slot_base'] < 10000:
        raise ValueError('Borrowed slots must not overlap original workers')
    return cfg, nc, host


def borrowed_worker(work, node, slot):
    cfg, nc, host = settings(work, node)
    if not nc['slot_base'] <= slot < nc['slot_base'] + nc['max_workers']:
        raise ValueError('Wrong worker slot')
    import listen_j_annotation_nod as original
    original_annotate = original.annotate
    resource_work = Path(host['resource_work'])
    # Use exactly the same decoder/model/thresholds. Resource ownership follows
    # the compute host; result ownership continues to follow the source queue.
    def annotate(row, face, nod, parameters, model, intervals, source_work, fast_crop=False):
        return original_annotate(row, face, nod, parameters, model, intervals,
                                 resource_work, fast_crop=fast_crop)
    original.annotate = annotate
    sys.argv = [str(ROOT/'scripts/listen_j_annotation_nod.py'),
                '--work', cfg['source_work'], '--slot', str(slot)]
    original.main()


def supervise(work, node):
    cfg, nc, host = settings(work, node)
    local = work/'nod_cpu_share'/node
    local.mkdir(parents=True, exist_ok=True)
    lock = acquire(local/'supervisor.lock', False)
    if lock is None:
        raise RuntimeError('CPU sharing supervisor already running')
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    state = read(local/'supervisor.json') if (local/'supervisor.json').exists() else {}
    state.update(pid=os.getpid(), start_ticks=identity(os.getpid()), hostname=socket.gethostname(),
                 started_at=time.time(), stage='running')
    children = state.setdefault('children', {})
    restarts = state.setdefault('restarts', {})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', GLOG_minloglevel='2')
    try:
        while True:
            cfg, nc, host = settings(work, node)
            stopping = stop[0] or (local/'STOP').exists() or (work/'nod_cpu_share/STOP').exists()
            desired = 0 if stopping else nc['workers']
            for i in range(nc['max_workers']):
                key = str(nc['slot_base']+i)
                child = children.get(key)
                if i >= desired:
                    if alive(child) and not child.get('drain_requested_at'):
                        os.kill(child['pid'], signal.SIGTERM)
                        child['drain_requested_at'] = time.time()
                    continue
                if alive(child):
                    continue
                recent = [t for t in restarts.get(key, []) if time.time()-t < 3600]
                if len(recent) >= 3:
                    state.setdefault('restart_blocked', {})[key] = recent
                    continue
                children[key] = spawn(local, 'nod-'+key,
                    [cfg['python'], str(Path(__file__).resolve()), 'worker',
                     '--work', str(work), '--node', node, '--slot', key], env)
                restarts[key] = recent+[time.time()]
            state.update(heartbeat=time.time(), desired_workers=desired,
                         live_workers=sum(alive(c) for c in children.values()),
                         stage='draining' if stopping else 'running')
            write(local/'supervisor.json', state)
            if stopping and not state['live_workers']:
                break
            time.sleep(5)
    finally:
        state.update(stage='stopped', heartbeat=time.time())
        write(local/'supervisor.json', state)
        lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['worker', 'supervise'])
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--node', required=True)
    p.add_argument('--slot', type=int)
    a = p.parse_args()
    if a.command == 'worker':
        if a.slot is None: p.error('--slot required')
        borrowed_worker(a.work.resolve(), a.node, a.slot)
    else:
        supervise(a.work.resolve(), a.node)
