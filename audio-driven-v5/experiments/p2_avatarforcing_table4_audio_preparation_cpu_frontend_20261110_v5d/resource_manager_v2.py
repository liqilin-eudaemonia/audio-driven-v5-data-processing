"""Run the already verified IIANet shards within explicit GPU memory budgets.

The original manager added an unexplained 32GiB floor on top of its actual
allocator/context/reserve calculation. This scheduler retains that calculation
and all four-process model/parity protocols, without the redundant floor.
"""
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import parallel as original

WORK = Path(__file__).resolve().parent


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    plan = original.prepare()
    assert original.read(WORK / 'PARALLEL_PARITY.json')['passed']
    protocol = dict(schema='same-IIANet-budget-scheduler-v2',
        implementation_sha256=original.sha(__file__),
        resource_executor_sha256=original.sha(WORK / 'resource_execute.py'),
        mathematical_executor_sha256=original.sha(WORK / 'parallel.py'),
        parent_protocol_sha256=original.sha(WORK / 'PARALLEL_PROTOCOL.json'),
        parity_sha256=original.sha(WORK / 'PARALLEL_PARITY.json'),
        allocator_fraction=plan['allocator_fraction'],
        context_budget_mib=plan['context_budget_mib'], reserve_mib=plan['reserve_mib'],
        required='reserve + unallocated budgets of live workers + one whole worker budget',
        removed_redundant_floor_mib=32768,
        no_model_or_data_or_numerical_changes=True)
    dest = WORK / 'RESOURCE_SCHEDULER_V2_PROTOCOL.json'
    if dest.exists():
        assert original.read(dest) == protocol
    else:
        original.save(dest, protocol)
    children = []
    launched = set()
    with (WORK / 'parallel_manager.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                for child in children:
                    if child.poll() not in [None, 0]:
                        raise RuntimeError(f'Shard {child.pid} exited {child.returncode}; inspect preserved logs')
                done = sum((WORK/'execution'/f'worker_{i}'/'SEPARATION_COMPLETE.json').exists() for i in range(4))
                live = [c for c in children if c.poll() is None]
                if done == 4 and not live:
                    original.collect()
                    original.save(WORK/'PARALLEL_STATUS.json', dict(stage='complete', pairs=43, streams=86,
                        pid=os.getpid(), scheduler_protocol_sha256=original.sha(dest), updated_unix=time.time()))
                    return
                free, total, used = original.gpu()
                budget = math.ceil(total * plan['allocator_fraction']) + plan['context_budget_mib']
                required = plan['reserve_mib'] + sum(max(0, budget-used.get(c.pid, 0)) for c in live) + budget
                pending = [i for i in range(4) if i not in launched and not (WORK/'execution'/f'worker_{i}'/'SEPARATION_COMPLETE.json').exists()]
                if pending and free >= required:
                    i = pending[0]
                    with (WORK/'execution'/f'worker_{i}'/'run.log').open('a') as log:
                        child = subprocess.Popen([sys.executable, '-u', str(WORK/'resource_execute.py'), f'worker_{i}'],
                            cwd=original.ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
                    children.append(child)
                    launched.add(i)
                    live.append(child)
                completed = len(list((WORK/'execution').glob('worker_*/separated/*/*/complete.json')))
                original.save(WORK/'PARALLEL_STATUS.json', dict(stage='separating_parallel', pid=os.getpid(),
                    completed=completed, total=86, live_pids=[c.pid for c in live], free_mib=free,
                    required_to_add_mib=required, worker_budget_mib=budget,
                    scheduler_protocol_sha256=original.sha(dest), updated_unix=time.time()))
                time.sleep(5)
        except BaseException as error:
            original.save(WORK/'RESOURCE_SCHEDULER_V2_FAILURE.json', dict(error=repr(error), pid=os.getpid(), updated_unix=time.time()))
            for child in children:
                if child.poll() is None:
                    child.terminate()
            raise


if __name__ == '__main__':
    main()
