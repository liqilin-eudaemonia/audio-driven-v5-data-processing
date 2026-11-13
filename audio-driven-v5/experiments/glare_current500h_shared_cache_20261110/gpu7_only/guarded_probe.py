"""Pause only the owned GLARE worker for a bounded probe; resume on all exits."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ROOT = WORK.parents[1]
spec = importlib.util.spec_from_file_location('probe_guards', WORK/'optimization_execution_v2/controller.py')
guards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guards)


def resume(record):
    current = guards.signature(record['pid'])
    if guards.owned(current, 'visual', 7) and current['start_ticks'] == record['start_ticks']:
        os.kill(record['pid'], signal.SIGCONT)


def watchdog(out, timeout):
    record = guards.read(out/'PAUSE.json')['record']
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (out/'RESUMED.json').exists():
            return
        if (WORK/'STOP').exists():
            break
        time.sleep(1)
    resume(record)
    guards.save(out/'WATCHDOG_RESUMED.json', dict(at=time.time(), worker_pid=record['pid']))


def run(script, out, timeout):
    assert script.parent == HERE and script.name.startswith('benchmark_')
    assert out.parent == HERE and not (out/'PAUSE.json').exists()
    assert not (WORK/'STOP').exists()
    state = guards.read(WORK/'status_all.json')
    record = guards.read(HERE/'registry.json')['visual']
    current = guards.signature(record['pid'])
    assert guards.owned(current, 'visual', 7) and current['start_ticks'] == record['start_ticks']
    out.mkdir(exist_ok=True)
    guards.save(out/'PAUSE.json', dict(at=time.time(), record=record, before=state, other_jobs_kept_running=True))
    with (out/'watchdog.log').open('a') as log:
        watch = subprocess.Popen(['python', str(Path(__file__)), 'watchdog', '--out', str(out),
                                  '--timeout', str(timeout+60)], cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    os.kill(record['pid'], signal.SIGSTOP)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='7', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=':'.join(str(p) for p in (
                   ROOT/'experiments/glare_current500h_nonprosody_cache_20261110/runtime_deps',
                   ROOT/'compare/.envs/glare-deps', ROOT/'src', ROOT/'scripts')))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    child = None
    try:
        with (out/'benchmark.log').open('a') as log:
            child = subprocess.Popen([str(guards.pipeline.GPU_PY), str(script)], cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        guards.save(out/'LAUNCHED.json', dict(pid=child.pid, at=time.time(), watchdog_pid=watch.pid))
        rc = child.wait(timeout=timeout)
        print(json.dumps(dict(benchmark_exit=rc, pid=child.pid)), flush=True)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
        resume(record)
        guards.save(out/'RESUMED.json', dict(at=time.time(), worker_pid=record['pid'], resumed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'watchdog'])
    parser.add_argument('--script', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=480)
    args = parser.parse_args()
    assert 30 <= args.timeout <= 660
    if args.action == 'watchdog':
        watchdog(args.out.resolve(), args.timeout)
    else:
        run(args.script.resolve(), args.out.resolve(), args.timeout)
