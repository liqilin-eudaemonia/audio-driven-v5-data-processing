"""Observe newly added nod workers, then measure unchanged GPU production."""
from pathlib import Path
import json, os, subprocess, time

W = Path(__file__).resolve().parent

def read(name):
    return json.loads((W / name).read_text())

def save(name, value):
    p = W / name
    tmp = p.with_suffix('.nodtrial-tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, p)

def live(record):
    try:
        fields = Path('/proc/{}/stat'.format(record['pid'])).read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == str(record['start_ticks'])
    except (FileNotFoundError, KeyError):
        return False

def main():
    started = time.time()
    save('monitor_snapshots/nod80_observer.json', {'at': started, 'pid': os.getpid()})
    before = read('monitor_snapshots/nod80_before.json')
    while time.time() - started < 900:
        s = read('supervisor.json')
        if not live(s) or not live(read('watchdog.json')):
            raise RuntimeError('Supervision process missing')
        assert read('config.json')['nod_workers'] == 80
        for name, previous in before['gpu_workers'].items():
            current = s['children'][name]
            if (current['pid'], current['start_ticks']) != (previous['pid'], previous['start_ticks']):
                raise RuntimeError('GPU identity changed during nod capacity trial')
        ready = []
        for i in range(64, 80):
            name = f'nod-{i}'
            child = s['children'].get(name)
            path = W / 'workers' / (name + '.json')
            if not child or not live(child) or not path.exists():
                continue
            h = json.loads(path.read_text())
            if h.get('pid') != child['pid']:
                continue
            if h.get('errors', 0):
                raise RuntimeError(f'New worker {name} has an input error')
            if h.get('revision') == 3 and h.get('completed_clips', 0) >= 3:
                ready.append(name)
        save('monitor_snapshots/nod80_adoption.json', {'at': time.time(), 'ready': ready})
        print(json.dumps({'at': time.time(), 'ready_new_nod_workers': len(ready)}), flush=True)
        if len(ready) == 16 and not s.get('rolling_restart') and not s.get('warming_gpus'):
            subprocess.run(['python3', str(W / 'measure_runtime_window.py'), '--tag',
                            'process_all_nod80_stable_window', '--seconds', '600', '--revision', '11'], check=True)
            report = read('monitor_snapshots/process_all_nod80_stable_window.json')
            if not report['summary']['valid_for_throughput_comparison']:
                raise RuntimeError('Trial window was not stable')
            save('monitor_snapshots/nod80_trial_complete.json', {
                'at': time.time(), 'measurement': 'process_all_nod80_stable_window.json'})
            return
        time.sleep(15)
    raise TimeoutError('New nod workers did not all produce initial output')

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('monitor_snapshots/nod80_trial_error.json', {'at': time.time(), 'error': repr(exc)})
        raise
