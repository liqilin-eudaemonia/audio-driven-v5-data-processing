"""Hot CPU pool refresh plus GPU6 RGB canary; GPU model processes stay live."""
from pathlib import Path
import json
import os
import subprocess
import time
from measure_runtime_window import snapshot

W = Path(__file__).resolve().parent


def read(name):
    return json.loads((W / name).read_text())


def save(name, data):
    p = W / name
    tmp = p.with_suffix('.rgbtrial-tmp')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)


def restore_workers():
    for name in ['selected.json', 'resource_tuning.json']:
        data = read(name)
        if data['decode_workers'] == 17:
            data.update(decode_workers=16, rgb_refresh_restored_at=time.time())
            save(name, data)
        elif data['decode_workers'] != 16:
            raise RuntimeError('Runtime capacity changed externally; inspect before adjusting')


def main():
    started = time.time()
    baseline = read('monitor_snapshots/rgb_reuse_before.json')['snapshot']
    save('monitor_snapshots/rgb_reuse_observer.json', {'at': started, 'pid': os.getpid()})
    phase = 'refresh17'
    adopted = {}
    while time.time() - started < 1800:
        now = snapshot(W)
        if not now['guards_live'] or now['rolling'] or now['warming']:
            raise RuntimeError('Supervision/rollout changed; retain model processes and inspect')
        if now['rgb_reuse_gpus'] != [6] or now['nod_decode_slots'] != 112:
            raise RuntimeError('Unexpected RGB selection or nod slot count')
        if set(now['workers']) != set(baseline['workers']):
            raise RuntimeError('Worker membership changed')
        for name, old in baseline['workers'].items():
            current = now['workers'][name]
            if (current['pid'], current['start_ticks']) != (old['pid'], old['start_ticks']):
                raise RuntimeError(f'Model or nod process restarted: {name}')
            if not current['live'] or not current['heartbeat_matches'] or current['failed'] > old['failed']:
                raise RuntimeError(f'Worker liveness or input failure changed: {name}')
            if current['execution'] != old['execution']:
                raise RuntimeError('Model execution changed')
        gpus = {n: v for n, v in now['workers'].items() if n.startswith('gpu-')}
        ready = []
        if phase == 'refresh17':
            if now['tuning'] != {'decode_workers': 17, 'decode_slots': 112}:
                raise RuntimeError('Unexpected refresh tuning')
            if all(v['decode_workers'] == 17 for v in gpus.values()):
                restore_workers()
                phase = 'drain16'
        else:
            if now['tuning'] != {'decode_workers': 16, 'decode_slots': 112}:
                raise RuntimeError('Unexpected steady-state tuning')
            for name, worker in gpus.items():
                if worker['decode_workers'] == 16:
                    adopted.setdefault(name, worker['completed'])
                    capacity = worker['execution'].get('rolling_annotation_groups') or 2
                    if worker['completed'] >= adopted[name] + (capacity + 1) * 50:
                        ready.append(name)
            if len(ready) == 8:
                subprocess.run(['python3', str(W / 'audit_rolling_canary.py'), '--worker', 'gpu-6',
                                '--decode-backend', 'process', '--annotation-groups', '2', '--rgb-reuse',
                                '--count', '100', '--tag', 'rgb_reuse_gpu6_first100_audit',
                                '--queue-snapshot', str(W / 'monitor_snapshots/rgb_reuse_before.json')], check=True)
                subprocess.run(['python3', str(W / 'measure_runtime_window.py'), '--tag',
                                'rgb_reuse_gpu6_stable_window', '--seconds', '600', '--revision', '11',
                                '--worker-revision', 'gpu-7=12'], check=True)
                report = read('monitor_snapshots/rgb_reuse_gpu6_stable_window.json')
                if not report['summary']['valid_for_throughput_comparison']:
                    raise RuntimeError('RGB reuse window was not stable')
                save('monitor_snapshots/rgb_reuse_trial_complete.json', {'at': time.time(),
                     'measurement': 'rgb_reuse_gpu6_stable_window.json', 'all_model_processes_preserved': True})
                return
        status = {'at': time.time(), 'phase': phase, 'ready': ready, 'adopted_completed': adopted,
                  'gpu_decode_workers': {n: v['decode_workers'] for n, v in gpus.items()}}
        save('monitor_snapshots/rgb_reuse_trial_status.json', status)
        print(json.dumps(status), flush=True)
        time.sleep(15)
    raise TimeoutError('CPU refresh/drain observation timed out; inspect existing processes')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('monitor_snapshots/rgb_reuse_trial_error.json', {'at': time.time(), 'error': repr(exc)})
        try:
            data = read('resource_tuning.json')
            if data.get('rgb_reuse_gpus') == [6]:
                data['rgb_reuse_gpus'] = []
                save('resource_tuning.json', data)
            restore_workers()
        except Exception as rollback:
            save('monitor_snapshots/rgb_reuse_rollback_error.json', {'at': time.time(), 'error': repr(rollback)})
        raise
