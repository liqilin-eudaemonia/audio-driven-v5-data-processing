"""Read-only hourly training patrol; the existing controller owns recovery.

Kept outside checkpointed training sources. Never signals ranks, changes data,
or allocates GPUs. SSH failure is observation failure, not proof of worker death.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from listen_cluster import transport


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def controller_alive():
    try:
        r = read(WORK / 'queue/controller.json')
        proc = Path('/proc') / str(r['pid'])
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (proc / 'cmdline').read_bytes().split(b'\0')
        return (stat[0] != 'Z' and stat[19] == str(r['start_ticks'])
                and str(ROOT / 'scripts/p2_whisper_event_feature_train31.py').encode() in argv
                and b'run' in argv)
    except (OSError, ValueError, KeyError):
        return False


def metric_rows():
    rows = []
    lines = (WORK / 'metrics.jsonl').read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            if index != len(lines) - 1:
                raise
    return rows


def performance(rows, config):
    recent = rows[-128:]
    baseline = [r for r in rows if 100 <= r['step'] <= 1000 and r['step'] < recent[0]['step']]
    result = dict(recent_steps=[recent[0]['step'], recent[-1]['step']],
                  mean_step_s=statistics.mean(r['seconds'] for r in recent),
                  mean_input_wait_s=statistics.mean(r['input_wait_seconds'] for r in recent))
    # Match clip-duration bins; a longer sample mix alone must not trigger a restart.
    plan = read(config['duration_sampling']['manifest'])['pairs']
    with (Path(config['j_cache']) / 'sample_list.csv').open() as f:
        durations = [float(r['duration_s']) for r in csv.DictReader(f)]
    batch = config['training']['global_batch']
    def bucket(row):
        pairs = plan[(row['step'] - 1) * batch:row['step'] * batch]
        return round(statistics.mean(durations[i] for i, _ in pairs) * 2)
    groups = {}
    for row in baseline:
        groups.setdefault(bucket(row), []).append(row['seconds'])
    ratios = []
    for row in recent:
        reference = groups.get(bucket(row), [])
        if len(reference) >= 5:
            ratios.append(row['seconds'] / statistics.median(reference))
    ratio = statistics.median(ratios) if ratios else None
    result.update(matched_steps=len(ratios), duration_matched_ratio=ratio,
                  slowdown_candidate=len(ratios) >= 64 and ratio > 1.25,
                  remaining_compute_hours=(config['training']['steps'] - rows[-1]['step']) * result['mean_step_s'] / 3600)
    return result


def node_status(cluster, node, rank_dir):
    record = read(rank_dir / (node['name'] + '_workers.json'))
    code = '''import json,subprocess,socket
from pathlib import Path
pids=REPLACE_PIDS
workers=[]
for pid in pids:
 try:
  p=Path('/proc')/str(pid)
  argv=(p/'cmdline').read_bytes().split(b'\\0')
  state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
  workers.append(dict(pid=pid,alive=state!='Z' and b'train' in argv and any(x.endswith(b'/p2_whisper_event_feature_train31.py') for x in argv)))
 except OSError: workers.append(dict(pid=pid,alive=False))
r=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20,check=True)
gpus=[]
for line in r.stdout.splitlines():
 i,u,m,t=line.split(',');gpus.append(dict(gpu=int(i),util=float(u),used_mib=float(m),temperature=float(t)))
print(json.dumps(dict(hostname=socket.gethostname(),workers=workers,gpus=gpus)))
'''.replace('REPLACE_PIDS', repr(record['pids']))
    failures = []
    for attempt in range(3):
        try:
            p = subprocess.run(transport(cluster, node, [cluster['python'], '-c', code]),
                               capture_output=True, text=True, timeout=40)
            if p.returncode == 0:
                value = json.loads(p.stdout)
                return dict(value, node=node['name'], checked=True, selected_gpus=node['gpus'], retries=failures)
            failures.append(p.stderr[-1500:])
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            failures.append(repr(exc))
        if attempt < 2:
            time.sleep(2)
    return dict(node=node['name'], checked=False, observation_errors=failures)


def inspect():
    config = read(WORK / 'config.json')
    cluster = read(WORK / 'cluster.json')
    state = read(WORK / 'queue/state.json')
    rows = metric_rows()
    latest = rows[-1]
    alerts = []
    for row in rows[-128:]:
        expected = row['av'] + .05 * row['event'] + .025 * (row['feature_video'] + row['feature_audio'])
        if not math.isfinite(expected) or abs(expected - row['weighted_total']) > 1e-8:
            alerts.append('nonfinite_or_incorrect_loss'); break
        if any(not math.isfinite(row['grad_norm'][k]) or row['grad_norm'][k] <= 0
               for k in ['connector', 'events', 'features', 'text_lora', 'whisper']):
            alerts.append('missing_or_nonfinite_gradient'); break
    alive = controller_alive()
    status = read(WORK / 'status.json')
    age = time.time() - (WORK / 'metrics.jsonl').stat().st_mtime
    if not alive and not status['complete']:
        alerts.append('controller_not_live')
    if age > 900 and not status['complete']:
        alerts.append('no_progress_15min')
    perf = performance(rows, config)
    if perf['slowdown_candidate']:
        alerts.append('duration_matched_slowdown_requires_diagnosis')
    rank_dir = (Path(state['logs']) / 'ranks' if state.get('logs') else sorted((WORK / 'queue').glob('attempt_*/ranks'))[-1])
    with ThreadPoolExecutor(max_workers=4) as pool:
        nodes = list(pool.map(lambda n: node_status(cluster, n, rank_dir), cluster['nodes']))
    if not status['complete']:
        for node in nodes:
            if not node['checked']:
                alerts.append(node['node'] + '_observation_unavailable')
            elif not all(p['alive'] for p in node['workers']):
                alerts.append(node['node'] + '_rank_not_live')
    markers = sorted(WORK.glob('step_*.complete.json'))
    marker = read(markers[-1]) if markers else None
    if marker:
        model = markers[-1].with_suffix('').with_suffix('.pt')
        marker = dict(marker, path=str(model), bytes=model.stat().st_size)
    if not marker or latest['step'] - marker['step'] > 500:
        alerts.append('checkpoint_missing_or_overdue')
    terminal = False
    if status['complete'] and marker and marker['step'] == config['training']['steps']:
        h = hashlib.sha256()
        with Path(marker['path']).open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        terminal = h.hexdigest() == marker['sha256']
        if not terminal:
            alerts.append('final_checkpoint_hash_mismatch')
    return dict(at=time.time(), utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                status=status, controller_live=alive, progress_age_s=age, performance=perf,
                nodes=nodes, checkpoint=marker, alerts=alerts, terminal_verified=terminal,
                reserved_gpu='node8:7; may be occupied by the user; excluded from training',
                recovery_owner='existing train31 controller; this patrol never kills/restarts ranks')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    out = WORK / 'hourly_monitor'
    out.mkdir(exist_ok=True)
    with (out / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        write(out / 'process.json', dict(pid=os.getpid(), start_ticks=stat[19], script=str(Path(__file__).resolve())))
        while not (out / 'STOP').exists():
            began = time.time()
            try:
                report = inspect()
                write(out / ('report-' + str(int(began)) + '.json'), report)
                write(out / 'latest.json', report)
                print(json.dumps(report, ensure_ascii=False), flush=True)
                if report['terminal_verified'] or args.once:
                    return
            except Exception as exc:
                write(out / 'observation_error.json', dict(at=time.time(), error=repr(exc)))
                print(repr(exc), flush=True)
                if args.once:
                    raise
            due = began + 3600
            write(out / 'schedule.json', dict(last_started_at=began, next_due_at=due, interval_s=3600))
            while time.time() < due and not (out / 'STOP').exists():
                time.sleep(min(30, max(0, due-time.time())))


if __name__ == '__main__':
    main()
