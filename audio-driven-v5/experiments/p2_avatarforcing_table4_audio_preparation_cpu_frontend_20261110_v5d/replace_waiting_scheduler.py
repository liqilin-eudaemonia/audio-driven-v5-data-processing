"""Replace only our verified idle scheduler; preserve all model protocols."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import parallel as source

WORK = Path(__file__).resolve().parent


def process(pid):
    root = Path('/proc') / str(pid)
    if not root.exists():
        return None
    return dict(pid=pid, stat=(root/'stat').read_text(),
        argv=(root/'cmdline').read_bytes().replace(b'\0', b' ').decode(),
        children=(root/'task'/str(pid)/'children').read_text().split())


def main():
    assert not (WORK/'RESOURCE_SCHEDULER_LAUNCH.json').exists(), 'Already launched; inspect the recorded PID'
    status = source.read(WORK/'PARALLEL_STATUS.json')
    runner = source.read(WORK/'PARALLEL_RUNNER_STATUS.json')
    assert status['pid'] == 3852802 and runner['pid'] == 3849509
    assert status['completed'] == 0 and status['live_pids'] == []
    manager_pid, runner_pid = status['pid'], runner['pid']
    assert str(WORK/'parallel.py') in process(manager_pid)['argv']
    assert str(WORK/'parallel_runner.py') in process(runner_pid)['argv']
    os.kill(manager_pid, signal.SIGSTOP)
    os.kill(runner_pid, signal.SIGSTOP)
    try:
        time.sleep(.2)
        manager = process(manager_pid)
        parent = process(runner_pid)
        assert not manager['children'], manager
        assert parent['children'] == [str(manager_pid)], parent
        latest = source.read(WORK/'PARALLEL_STATUS.json')
        assert latest['completed'] == 0 and latest['live_pids'] == []
        source.save(WORK/'IDLE_SCHEDULER_REPLACEMENT.json', dict(
            implementation_sha256=source.sha(__file__), old_status=latest, old_runner_status=runner,
            stopped_processes=[manager, parent], reason='Remove redundant32GiB floor; retain5%allocator+1GiBcontext+16GiBreserve',
            stopped_unix=time.time()))
        for pid in [runner_pid, manager_pid]:
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
        for _ in range(100):
            states = [process(p) for p in [runner_pid, manager_pid]]
            if all(s is None or s['stat'].rsplit(')', 1)[1].strip().startswith('Z') for s in states):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Old schedulers have not exited; do not start another')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='7', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        native = source.ROOT/'compare/.envs/avatarforcing_native201/bin/python'
        with (WORK/'resource_manager.log').open('a') as log:
            child = subprocess.Popen([str(native), '-u', str(WORK/'resource_manager.py')], cwd=source.ROOT,
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        source.save(WORK/'RESOURCE_SCHEDULER_LAUNCH.json', dict(pid=child.pid, launched_unix=time.time(),
            scheduler_sha256=source.sha(WORK/'resource_manager.py'), physical_gpu=7, verified_idle_predecessors_terminal=True))
        source.save(WORK/'PARALLEL_RUNNER_STATUS.json', dict(stage='running_budget_scheduler', pid=child.pid,
            previous_runner_pid=runner_pid, updated_unix=time.time()))
        print(child.pid, flush=True)
    except BaseException:
        for pid in [manager_pid, runner_pid]:
            if process(pid) is not None:
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
        raise


if __name__ == '__main__':
    main()
