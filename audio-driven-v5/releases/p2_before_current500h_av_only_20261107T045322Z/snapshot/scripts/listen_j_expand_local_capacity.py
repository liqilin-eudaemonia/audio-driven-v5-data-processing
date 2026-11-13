#!/usr/bin/env python3
"""Use released local GPUs and assist existing quality queues with CPU workers."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.j_annotation import acquire, read, write
from listen_benchmark.j_quality_pipeline import live, identity, spawn, QUALITY_PYTHON, ENTRY

BASE = ROOT / 'experiments/listen_j_annotation_three_nodes_20261030'
WORK = ROOT / 'experiments/listen_j_annotation_node12_20261030'
QUALITY = ROOT / 'data_corpora/seamless_listen_j_visual_quality_v1_20261030'
STATE = BASE / 'expansion_24'


def free_devices():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    occupied = {line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
    return {int(i): uuid.strip() not in occupied and float(memory) < 1500
            for i, uuid, memory in (line.split(',') for line in raw.splitlines())}


def activate(gpus):
    config = read(WORK / 'config.json')
    reference = read(QUALITY / 'node8/original_config.json')
    ids = sorted(set(config['gpu_ids']) | set(gpus))
    # Configure CPU feed and per-card execution before the supervisor sees new GPUs.
    tuning = read(WORK / 'resource_tuning.json')
    tuning['decode_slots'] = 14 * len(ids)
    write(WORK / 'resource_tuning.json', tuning)
    config['rolling_canary_gpus'] = ids
    for gpu in gpus:
        config['gpu_execution_overrides'][str(gpu)] = reference['gpu_execution_overrides'][str(gpu)]
    config['nod_workers'] = 10 * len(ids)
    config['gpu_ids'] = ids
    write(WORK / 'config.json', config)
    for path in (BASE / 'nodes.json', QUALITY / 'annotation_nodes.json'):
        scope = read(path)
        scope['nodes']['node12']['gpu_ids'] = ids
        scope['total_gpus'] = sum(len(n['gpu_ids']) for n in scope['nodes'].values())
        write(path, scope)
    return ids


def main():
    STATE.mkdir(exist_ok=True)
    owner = acquire(STATE / 'capacity.lock', False)
    if owner is None:
        return
    if socket.gethostname() != read(WORK / 'config.json')['expected_hostname']:
        raise RuntimeError('This capacity controller belongs to node12')
    if not read(STATE / 'cross_host_lock_validation.json')['passed']:
        raise RuntimeError('Cross-host queue locking must be verified')
    from listen_benchmark.j_quality_pipeline import runtime_contract
    for node in ('node17', 'node8'):
        if runtime_contract(QUALITY / node) != read(QUALITY / node / 'contract.json'):
            raise RuntimeError('Local quality implementation differs from frozen contract')
    path = STATE / 'capacity.json'
    state = read(path) if path.exists() else {'children': {}, 'events': []}
    state.update(pid=os.getpid(), start_ticks=identity(os.getpid()), started_at=time.time())
    confirmations = {}; stop = False

    def stopping(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    try:
        while not stop:
            state['heartbeat'] = time.time()
            try:
                current = read(WORK / 'config.json')['gpu_ids']
                free = free_devices()
                pending = [gpu for gpu in range(8) if gpu not in current]
                for gpu in pending:
                    confirmations[gpu] = confirmations.get(gpu, 0) + 1 if free.get(gpu) else 0
                released = [gpu for gpu in pending if confirmations[gpu] >= 2]
                if released:
                    # Requery immediately before enabling; never kill another workload.
                    latest = free_devices()
                    released = [gpu for gpu in released if latest.get(gpu)]
                    if released:
                        current = activate(released)
                        state['events'].append({'at': time.time(), 'activated': released})
                state['gpu_ids'] = current
                state['waiting_gpus'] = [gpu for gpu in range(8) if gpu not in current]
                for node in ('node17', 'node8'):
                    work = QUALITY / node
                    queue = read(work / 'queues/quality/state.json')
                    cfg = read(work / 'pipeline_config.json')
                    remaining = cfg['scope_sources'] - queue['completed'] - len(queue['failed'])
                    for offset in range(16):
                        slot = 1000 + offset
                        name = f'{node}-{slot}'
                        if remaining <= 0 or (work / 'quality_complete.json').exists():
                            continue
                        if not live(state['children'].get(name, {})):
                            child = spawn(STATE, name, [QUALITY_PYTHON, str(ENTRY), 'worker',
                                '--work', str(work), '--slot', str(slot)])
                            os.setpriority(os.PRIO_PROCESS, child['pid'], 10)
                            state['children'][name] = child
                state['quality_workers_live'] = sum(live(c) for c in state['children'].values())
                state.pop('error', None)
            except Exception as error:
                state['error'] = repr(error)
            write(path, state)
            time.sleep(15)
    finally:
        state['ended_at'] = time.time()
        write(path, state)
        owner.close()


def watchdog():
    """Restart only this capacity controller, preserving its live CPU children."""
    STATE.mkdir(exist_ok=True)
    owner = acquire(STATE / 'capacity_watchdog.lock', False)
    if owner is None:
        return
    state = {'pid': os.getpid(), 'start_ticks': identity(os.getpid()), 'started_at': time.time()}
    stop = False
    def stopping(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    try:
        while not stop and not (STATE / 'STOP_CAPACITY_WATCHDOG').exists():
            path = STATE / 'capacity.json'
            controller = read(path) if path.exists() else {}
            if not live(controller) and not live(state.get('last_launch', {})):
                state['last_launch'] = spawn(STATE, 'capacity-controller',
                    [QUALITY_PYTHON, str(Path(__file__).resolve())])
            state['heartbeat'] = time.time()
            write(STATE / 'capacity_watchdog.json', state)
            time.sleep(15)
    finally:
        state['ended_at'] = time.time()
        write(STATE / 'capacity_watchdog.json', state)
        owner.close()


if __name__ == '__main__':
    if sys.argv[1:] == ['--watchdog']:
        watchdog()
    elif sys.argv[1:]:
        raise SystemExit('Usage: listen_j_expand_local_capacity.py [--watchdog]')
    else:
        main()
