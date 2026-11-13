"""Observe one supervised graph-size trial; leave production running."""
from pathlib import Path
import json, os, subprocess, time

W = Path(__file__).resolve().parent
SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def read(name):
    return json.loads((W / name).read_text())

def save(name, data):
    p = W / name
    tmp = p.with_suffix('.graphtrial-tmp')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)

def live(record):
    try:
        fields = Path('/proc/{}/stat'.format(record['pid'])).read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == str(record['start_ticks'])
    except (FileNotFoundError, KeyError):
        return False

def main():
    started = time.time()
    save('monitor_snapshots/graph256_observer.json', {'at': started, 'pid': os.getpid()})
    before = read('monitor_snapshots/graph256_before.json')
    captured = None
    while time.time() - started < 1200:
        s = read('supervisor.json')
        if not live(s) or not live(read('watchdog.json')):
            raise RuntimeError('Supervision process missing')
        for name, old in before['other_gpu_workers'].items():
            current = s['children'][name]
            if not live(current) or (old['pid'], old['start_ticks']) != (current['pid'], current['start_ticks']):
                raise RuntimeError(f'Unchanged GPU worker changed identity: {name}')
        c = s['children']['gpu-7']
        h = read('workers/gpu-7.json')
        ready = (live(c) and h.get('pid') == c['pid'] and h.get('revision') == 11
                 and h.get('execution', {}).get('decode_graph_capture_sizes') == SIZES)
        if ready:
            if captured is None:
                q = read('queues/vlm/state.json')
                captured = {'at': time.time(), 'worker': c, 'initial_completed': h['completed_clips'],
                            'snapshot': {'vlm_queue_range': {'offset': q['offset'], 'inflight': list(q['inflight'])}}}
                save('monitor_snapshots/graph256_gpu7_before_output.json', captured)
            if (c['pid'], c['start_ticks']) != (captured['worker']['pid'], captured['worker']['start_ticks']):
                raise RuntimeError('Candidate worker restarted')
            if h.get('failed_clips', 0):
                raise RuntimeError('Candidate worker reported input failures')
            if h['completed_clips'] >= captured['initial_completed'] + 100 and not s.get('rolling_restart') and not s.get('warming_gpus'):
                subprocess.run(['python3', str(W / 'audit_rolling_canary.py'), '--worker', 'gpu-7',
                                '--decode-backend', 'process', '--count', '100', '--tag', 'graph256_gpu7_first100_audit',
                                '--queue-snapshot', str(W / 'monitor_snapshots/graph256_gpu7_before_output.json')], check=True)
                subprocess.run(['python3', str(W / 'measure_runtime_window.py'), '--tag',
                                'graph256_gpu7_stable_window', '--seconds', '600', '--revision', '11'], check=True)
                report = read('monitor_snapshots/graph256_gpu7_stable_window.json')
                if not report['summary']['valid_for_throughput_comparison']:
                    raise RuntimeError('Candidate window was not stable')
                save('monitor_snapshots/graph256_trial_complete.json', {
                    'at': time.time(), 'measurement': 'graph256_gpu7_stable_window.json'})
                return
        save('monitor_snapshots/graph256_trial_status.json', {
            'at': time.time(), 'candidate_ready': ready, 'stage': h.get('stage'),
            'completed': h.get('completed_clips'), 'rolling': s.get('rolling_restart'), 'warming': s.get('warming_gpus')})
        print(json.dumps({'at': time.time(), 'candidate_ready': ready,
                          'stage': h.get('stage'), 'completed': h.get('completed_clips')}), flush=True)
        time.sleep(15)
    raise TimeoutError('Graph trial did not reach initial output; inspect before continuing')

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('monitor_snapshots/graph256_trial_error.json', {'at': time.time(), 'error': repr(exc)})
        raise
