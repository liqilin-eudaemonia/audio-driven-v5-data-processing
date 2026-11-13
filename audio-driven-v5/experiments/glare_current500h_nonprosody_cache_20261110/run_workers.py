"""Detached node11 GPU4-7 controller. Never stops unrelated GPU processes."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'scripts')]
from prepare_glare_nonprosody_cache import GPUS, load_protocol, read, save


def command(gpu):
    return [str(ROOT / 'compare/.envs/avatarforcing/bin/python'),
            str(ROOT / 'scripts/prepare_glare_nonprosody_cache.py'),
            'worker', '--gpu', str(gpu)]


def main():
    lock = (WORK / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = load_protocol(WORK)
    assert read(WORK / 'CANARY_EQUIVALENCE.json')['passed']
    assert read(WORK / 'CANARY_RESUME.json')['passed']
    assert read(WORK / 'CPU_TESTS.json')['passed']
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    children, attempts, accepted, logs = {}, {}, {}, []
    def launch(gpu):
        attempts[gpu] = attempts.get(gpu, 0) + 1
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                   PYTHONPATH=':'.join(str(x) for x in [WORK / 'runtime_deps',
                       ROOT / 'compare/.envs/glare-deps', ROOT / 'src', ROOT / 'scripts']))
        stream = (WORK / 'workers' / ('gpu' + str(gpu) + '.log')).open('a')
        logs.append(stream)
        children[gpu] = subprocess.Popen(command(gpu), env=env, cwd=ROOT,
                                         stdout=stream, stderr=subprocess.STDOUT)
    (WORK / 'workers').mkdir(exist_ok=True)
    for gpu in GPUS:
        launch(gpu)
    started = time.time()
    while True:
        states = {}
        for gpu in GPUS:
            child = children[gpu]
            code = child.poll()
            path = WORK / 'workers' / ('gpu' + str(gpu) + '.json')
            state = read(path) if path.exists() else {}
            if state.get('pid') != child.pid:
                state = {'stage': 'initializing'}
            states[str(gpu)] = dict(state, process_pid=child.pid, process_returncode=code,
                                    attempts=attempts[gpu])
            if code is not None and gpu not in accepted:
                good = code == 0 and state.get('stage') == 'finished' and state.get('failed') == 0
                if good:
                    accepted[gpu] = True
                elif attempts[gpu] < 3 and not stopping[0] and not (WORK / 'STOP').exists():
                    launch(gpu)
                else:
                    accepted[gpu] = False
        snapshot = dict(stage='running', controller_pid=os.getpid(), host=os.uname().nodename,
                        samples=p['samples'], gpus=list(GPUS), workers=states,
                        completed=sum(s.get('completed', 0) + s.get('reused', 0) for s in states.values()),
                        worker_failed=sum(s.get('failed', 0) for s in states.values()),
                        started=started, heartbeat=time.time(),
                        prosody='deferred_by_user', full_glare_training_admitted=False)
        if stopping[0] or (WORK / 'STOP').exists():
            for child in children.values():
                if child.poll() is None:
                    child.terminate()
            for child in children.values():
                child.wait()
            snapshot['stage'] = 'stopped'
            save(WORK / 'status.json', snapshot)
            return
        if len(accepted) == 4:
            passed = all(accepted.values()) and snapshot['completed'] == p['samples']
            snapshot['stage'] = 'lia_complete' if passed else 'needs_error_review'
            save(WORK / 'status.json', snapshot)
            save(WORK / 'LIA_STAGE_RESULT.json', dict(passed=passed, status=snapshot,
                 full_glare_training_admitted=False,
                 remaining=['prosody', 'six_channel_target_assembly', 'training_admission']))
            return
        save(WORK / 'status.json', snapshot)
        time.sleep(10)


if __name__ == '__main__':
    main()
