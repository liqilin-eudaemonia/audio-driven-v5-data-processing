"""Wait on the existing live trial, then run one bounded parity experiment."""
from pathlib import Path
import json
import os
import subprocess
import time

W = Path(__file__).resolve().parent


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    status = W / 'monitor_snapshots/cuda_extended_waiter.json'
    observer = json.loads((W / 'monitor_snapshots/groups4_observer.json').read_text())
    ticks = identity(observer['pid'])
    if ticks is None:
        raise RuntimeError('Verify completed observer before starting a new waiter')
    command = Path(f'/proc/{observer["pid"]}/cmdline').read_bytes()
    assert b'observe_groups4_trial.py' in command
    record = {'pid': os.getpid(), 'start_ticks': identity(os.getpid()), 'started_at': time.time(),
              'observer_pid': observer['pid'], 'observer_start_ticks': ticks}
    for _ in range(80):
        ready = (W / 'monitor_snapshots/groups4_trial_complete.json').exists()
        failed = (W / 'monitor_snapshots/groups4_trial_error.json').exists()
        record.update(at=time.time(), stage='waiting_for_groups4_measurement')
        status.write_text(json.dumps(record, indent=2))
        if failed:
            raise RuntimeError('Four-group trial failed; inspect before adding probe load')
        if ready:
            result = json.loads((W / 'monitor_snapshots/groups4_gpu7_stable_window.json').read_text())
            assert result['complete'] and result['summary']['valid_for_throughput_comparison']
            free = subprocess.check_output(['nvidia-smi', '--id=6', '--query-gpu=memory.free',
                                            '--format=csv,noheader,nounits'], text=True, timeout=20)
            if int(free.strip()) < 4096:
                raise RuntimeError('Less than 4 GiB free on probe GPU6; preserve production')
            record.update(stage='running_extended_parity_probe', launched_at=time.time())
            status.write_text(json.dumps(record, indent=2))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6')
            subprocess.run(['/private/software/conda/envs/vllm/bin/python',
                            str(W / 'probe_cuda_decode_extended.py')], env=env, check=True)
            record.update(stage='complete', finished_at=time.time())
            status.write_text(json.dumps(record, indent=2))
            return
        if identity(observer['pid']) != ticks:
            raise RuntimeError('Observer exited without completion marker; inspect its result')
        print(json.dumps({'at': time.time(), 'waiting_on_live_observer': observer['pid']}), flush=True)
        time.sleep(30)
    raise TimeoutError('Observation window elapsed; no process restart requested')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        (W / 'monitor_snapshots/cuda_extended_waiter_error.json').write_text(
            json.dumps({'at': time.time(), 'error': repr(exc)}, indent=2))
        raise
