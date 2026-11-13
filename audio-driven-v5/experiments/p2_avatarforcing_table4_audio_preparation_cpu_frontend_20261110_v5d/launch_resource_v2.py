"""Launch once after verifying that our previous idle/resource jobs exited."""
import fcntl
import os
from pathlib import Path
import subprocess
import time

import parallel as source

WORK = Path(__file__).resolve().parent


def main():
    dest = WORK/'RESOURCE_SCHEDULER_V2_LAUNCH.json'
    assert not dest.exists(), 'Already launched: inspect recorded process, never duplicate'
    states = []
    for pid in [3849509, 3852802, 3863937, 3863967]:
        p = Path('/proc')/str(pid)/'stat'
        value = p.read_text() if p.exists() else None
        assert value is None or value.rsplit(')',1)[1].strip().startswith('Z'), (pid, value)
        states.append(dict(pid=pid, terminal_stat=value))
    with (WORK/'parallel_manager.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='7', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    native = source.ROOT/'compare/.envs/avatarforcing_native201/bin/python'
    with (WORK/'resource_manager_v2.log').open('a') as log:
        child = subprocess.Popen([str(native), '-u', str(WORK/'resource_manager_v2.py')],
            cwd=source.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=child.pid, launched_unix=time.time(), predecessor_processes=states,
        implementation_sha256=source.sha(WORK/'resource_manager_v2.py'),
        executor_sha256=source.sha(WORK/'resource_execute.py'), physical_gpu=7,
        stop_after='86 separation streams and independent artifact audit; no AF generation or scoring')
    source.save(dest, receipt)
    source.save(WORK/'PARALLEL_RUNNER_STATUS.json', dict(stage='running_budget_scheduler_v2',
        pid=child.pid, updated_unix=time.time(), no_downstream_AF_launched=True))
    print(child.pid, flush=True)


if __name__ == '__main__':
    main()
