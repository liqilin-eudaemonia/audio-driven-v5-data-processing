"""Capacity-aware continuation of the already authorized GPU7 evaluation."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from p2_avatarforcing_table4_generate import read, save, sha


def status(phase, **extra):
    save(WORK / 'NATIVE_PIPELINE_STATUS.json', dict(phase=phase, pid=os.getpid(),
        updated_unix=time.time(), script_sha256=sha(__file__), **extra))


def wait_marker(name, phase):
    while not (WORK / name).exists():
        current = read(WORK / 'PREPROCESS_STATUS.json') if (WORK / 'PREPROCESS_STATUS.json').exists() else {}
        status(phase, preprocessing=current)
        time.sleep(20)
    assert read(WORK / name)['passed'], name


def run_script(name, action, logname, marker=None):
    if marker and (WORK / marker).exists():
        assert read(WORK / marker)['passed'], marker
        return
    if name not in ['native_score.py', 'official_score.py']:
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', '7', '--query-gpu=memory.free',
                       '--format=csv,noheader,nounits'], text=True).strip())
            minimum = 37950 if name == 'native_generate.py' else 32768
            if free >= minimum:
                break
            status('waiting_gpu_capacity', next_script=name, free_mib=free, required_mib=minimum)
            time.sleep(20)
    status('running', script=name, action=action)
    runtime = WORK / 'NATIVE_GENERATION_RUNTIME.json'
    def archive_runtime():
        if runtime.exists():
            history = WORK / 'runtime_history'
            history.mkdir(exist_ok=True)
            dest = history / (sha(runtime) + '.json')
            if not dest.exists():
                dest.write_bytes(runtime.read_bytes())
            assert sha(dest) == sha(runtime)
    if name == 'native_generate.py':
        archive_runtime()
    try:
        with (WORK / logname).open('a') as log:
            subprocess.run([sys.executable, '-u', str(WORK / name), action], cwd=ROOT,
                           env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        if name == 'native_generate.py':
            archive_runtime()
    if marker:
        assert read(WORK / marker)['passed'], marker
    if name == 'native_generate.py':
        receipts = [read(p) for p in (WORK / 'native_generated').glob('*/complete.json')]
        assert len({r['initial_cpu_rng'] for r in receipts}) == 1
        assert len({r['initial_cuda_rng'] for r in receipts}) == 1
        for receipt in receipts:
            saved_runtime = WORK / 'runtime_history' / (receipt['runtime_sha256'] + '.json')
            assert sha(saved_runtime) == receipt['runtime_sha256']


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    with (WORK / 'native_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            # Do not launch the full native run before both preprocessing and
            # the actual six-stream feature-extraction probe have succeeded.
            wait_marker('OFFICIAL_SPECTRE_PROBE.json', 'waiting_official_spectre_probe')
            wait_marker('PREPROCESS_COMPLETE.json', 'waiting_preprocessing')
            assert read(WORK / 'PREPROCESS_COMPLETE.json')['roles'] == 86
            run_script('native_generate.py', 'all', 'native_generate_all.log', 'NATIVE_GENERATION_COMPLETE.json')
            if (WORK / 'OFFICIAL_PARALLEL_PLAN.json').exists():
                wait_marker('OFFICIAL_EXTRACTION_COMPLETE.json', 'waiting_parallel_official_spectre')
            else:
                run_script('official_spectre.py', 'all', 'official_spectre_all.log', 'OFFICIAL_EXTRACTION_COMPLETE.json')
            run_script('official_score.py', 'score', 'official_score.log')
            assert read(WORK / 'OFFICIAL_EXECUTION_COMPLETE.json')['passed_execution']
            status('execution_complete', table4_reproduction_requires_review=True,
                   matched_cells=read(WORK / 'OFFICIAL_EXECUTION_COMPLETE.json')['matched_cells'])
        except Exception as error:
            status('failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
