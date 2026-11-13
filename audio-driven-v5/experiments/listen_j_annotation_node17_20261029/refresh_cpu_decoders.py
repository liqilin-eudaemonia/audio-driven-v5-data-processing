"""One-off bounded CPU-pool refresh; never restarts model workers."""
from pathlib import Path
import json, os, time

W = Path(__file__).resolve().parent

def read(name):
    return json.loads((W / name).read_text())

def save(name, data):
    path = W / name
    tmp = path.with_suffix('.refresh-tmp')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)

def set_workers(count):
    for name in ('selected.json', 'resource_tuning.json'):
        data = read(name)
        data['decode_workers'] = count
        data['decoder_refresh_at'] = time.time()
        save(name, data)

def main():
    targets = {}
    for i in (0, 1, 2, 7):
        h = read(f'workers/gpu-{i}.json')
        assert h['revision'] == 11
        targets[f'gpu-{i}'] = {'pid': h['pid'], 'observed_completed': None}
    assert read('resource_tuning.json')['decode_workers'] == 16
    assert read('selected.json')['decode_workers'] == 16
    start = time.time()
    save('monitor_snapshots/cpu_decoder_refresh_before.json', {
        'at': start, 'targets': targets, 'tuning': read('resource_tuning.json'),
        'purpose': 'Reload terminal-frame-support-v2 in existing CPU children only'})
    passed = False
    try:
        set_workers(17)
        while time.time() - start < 1200:
            done = []
            for name, target in targets.items():
                h = read(f'workers/{name}.json')
                if h['pid'] != target['pid']:
                    raise RuntimeError(f'{name} changed model process during CPU refresh')
                if h.get('decode_workers_active') == 17:
                    if target['observed_completed'] is None:
                        target['observed_completed'] = h['completed_clips']
                    if h['completed_clips'] >= target['observed_completed'] + 100:
                        done.append(name)
            save('monitor_snapshots/cpu_decoder_refresh_status.json', {
                'at': time.time(), 'targets': targets, 'done': done})
            print(json.dumps({'at': time.time(), 'done': done}), flush=True)
            if len(done) == len(targets):
                passed = True
                break
            time.sleep(15)
    finally:
        # Restore the tested capacity even when observation fails.
        if read('resource_tuning.json')['decode_workers'] == 17:
            set_workers(16)
        save('monitor_snapshots/cpu_decoder_refresh_complete.json', {
            'at': time.time(), 'passed': passed, 'restored_workers': 16,
            'targets': targets})
    if not passed:
        raise RuntimeError('CPU refresh observation did not complete; capacity restored')

if __name__ == '__main__':
    main()
