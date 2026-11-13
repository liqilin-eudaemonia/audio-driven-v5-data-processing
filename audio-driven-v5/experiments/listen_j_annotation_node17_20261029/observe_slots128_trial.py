"""Wait for live CPU-slot adoption, then measure unchanged model workers."""
from pathlib import Path
import json, os, subprocess, time
from measure_runtime_window import snapshot

W = Path(__file__).resolve().parent

def read(name):
    return json.loads((W / name).read_text())

def save(name, data):
    p = W / name
    tmp = p.with_suffix('.slottrial-tmp')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)

def main():
    started = time.time()
    baseline = read('monitor_snapshots/slots128_before.json')['snapshot']
    save('monitor_snapshots/slots128_observer.json', {'at': started, 'pid': os.getpid()})
    adopted = {}
    while time.time() - started < 1200:
        now = snapshot(W)
        if not now['guards_live'] or now['rolling'] or now['warming']:
            raise RuntimeError('Supervision or rollout state changed')
        if now['tuning'] != {'decode_workers': 16, 'decode_slots': 128} or now['nod_decode_slots'] != 128:
            raise RuntimeError('Unexpected runtime tuning')
        if set(now['workers']) != set(baseline['workers']):
            raise RuntimeError('Worker membership changed')
        ready = []
        for name, old in baseline['workers'].items():
            current = now['workers'][name]
            if (current['pid'], current['start_ticks']) != (old['pid'], old['start_ticks']) or not current['live'] or not current['heartbeat_matches']:
                raise RuntimeError(f'Worker restarted or disappeared: {name}')
            if current['execution'] != old['execution']:
                raise RuntimeError('Model execution changed')
            if name.startswith('gpu-') and current['decode_slots'] == 128 and current['decode_workers'] == 16:
                adopted.setdefault(name, current['completed'])
                # At most two old groups can remain; wait through a new group.
                if current['completed'] >= adopted[name] + 150:
                    ready.append(name)
        save('monitor_snapshots/slots128_adoption.json', {
            'at': time.time(), 'adopted_completed': adopted, 'ready': ready})
        print(json.dumps({'at': time.time(), 'adopted': len(adopted), 'ready': len(ready)}), flush=True)
        if len(ready) == 8:
            subprocess.run(['python3', str(W / 'measure_runtime_window.py'), '--tag',
                            'decode_slots128_stable_window', '--seconds', '600', '--revision', '11'], check=True)
            report = read('monitor_snapshots/decode_slots128_stable_window.json')
            if not report['summary']['valid_for_throughput_comparison']:
                raise RuntimeError('CPU-slot window was not stable')
            save('monitor_snapshots/slots128_trial_complete.json', {
                'at': time.time(), 'measurement': 'decode_slots128_stable_window.json'})
            return
        time.sleep(15)
    raise TimeoutError('Not all workers drained their old preparation groups')

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('monitor_snapshots/slots128_trial_error.json', {'at': time.time(), 'error': repr(exc)})
        raise
