"""One-shot, fail-closed five-minute idle gate for the next unfinished candidate."""
from pathlib import Path
import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
WATCH = WORK / 'idle_resume'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from p2_listen.contracts import read, sha, atomic_json
from p2_loss_search28.config import CANDIDATES, source_identity, validate, validate_cluster
from listen_cluster import transport

INTERVAL = 30
HOLD = 300
EXPECTED = {n: list(range(8)) for n in ('node12', 'node8', 'node4')}
EXPECTED['node11'] = list(range(4))


class IdleWindow:
    def __init__(self):
        self.since = self.last = None

    def update(self, idle, now):
        if not idle or (self.last is not None and not 0 <= now - self.last <= 90):
            self.since = None
        self.last = now
        if idle and self.since is None:
            self.since = now
        elapsed = 0 if self.since is None else now - self.since
        return elapsed > HOLD, elapsed


def stop_identity(path):
    if not path.exists():
        return None
    return dict(sha256=sha(path), mtime_ns=path.stat().st_mtime_ns)


def next_candidate():
    for name in CANDIDATES:
        out = WORK / 'runs' / name
        receipt = out / 'STAGE_500.json'
        if not receipt.exists():
            return name
        r = read(receipt)
        marker = read(out / 'step_000500.complete.json')
        if r.get('success') is not True or r.get('step') != 500 or r.get('sha256') != marker['sha256']:
            raise ValueError('Invalid completed stage receipt: ' + name)
    raise ValueError('All seven 500-step candidates already complete; no implicit promotion')


def valid_node(node, value):
    gpus = value['gpus']
    if len(gpus) != len(node['gpus']) or {r['index'] for r in gpus} != set(node['gpus']):
        raise ValueError('Incomplete/duplicate GPU inventory for ' + node['name'])
    if len({r['uuid'] for r in gpus}) != len(gpus) or any(not r['uuid'].startswith('GPU-') for r in gpus):
        raise ValueError('Invalid GPU identities')
    for r in gpus:
        if not 0 <= r['utilization'] <= 100 or not 0 <= r['used_mib'] <= r['total_mib']:
            raise ValueError('Unavailable GPU utilization/memory')
    return not value['compute_jobs'] and all(r['utilization'] <= 5 and r['used_mib'] <= 1024 for r in gpus)


PROBE = '''import subprocess,json
def query(args):
 return subprocess.check_output(['nvidia-smi',*args,'--format=csv,noheader,nounits'],text=True)
rows=[]
for line in query(['--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total']).splitlines():
 i,u,b,m,t=[v.strip() for v in line.split(',')]
 if int(i) in MASK: rows.append(dict(index=int(i),uuid=u,utilization=int(b),used_mib=int(m),total_mib=int(t)))
uuids={r['uuid'] for r in rows}
apps=[]
for line in query(['--query-compute-apps=pid,gpu_uuid']).splitlines():
 pid,u=[v.strip() for v in line.split(',')]
 if u in uuids: apps.append(dict(pid=int(pid),uuid=u))
print(json.dumps(dict(gpus=rows,compute_jobs=apps)))
'''


def observe(cluster):
    if {n['name']: n['gpus'] for n in cluster['nodes']} != EXPECTED or len(cluster['nodes']) != 4:
        raise ValueError('Only the original 28 GPU allocation is allowed')
    def one(node):
        r = subprocess.run(transport(cluster, node, [cluster['python'], '-c', PROBE.replace('MASK', repr(node['gpus']))]),
                           capture_output=True, text=True, check=True, timeout=20)
        value = json.loads(r.stdout)
        return dict(node=node['name'], **value, idle=valid_node(node, value))
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        values = list(pool.map(one, cluster['nodes']))
    return dict(idle=all(n['idle'] for n in values), nodes=values, at=time.time())


def validate_request(request):
    if (WATCH / 'STOP').exists():
        raise RuntimeError('Idle continuation cancelled by its STOP')
    if request['source'] != source_identity() or request['watcher_sha256'] != sha(__file__):
        raise ValueError('Approved code changed')
    if next_candidate() != request['candidate']:
        raise ValueError('Next candidate changed; refusing a different experiment')
    if request.get('resume_checkpoint'):
        checkpoint = Path(request['resume_checkpoint'])
        marker = checkpoint.with_suffix('.complete.json')
        markers = sorted(checkpoint.parent.glob('step_*.complete.json'))
        if not checkpoint.is_file() or not markers or markers[-1] != marker or read(marker) != request['resume_checkpoint_marker']:
            raise ValueError('Pinned resume checkpoint/receipt changed; never restart from scratch')
    for filename, identity in request['stops'].items():
        if stop_identity(Path(filename)) != identity:
            raise RuntimeError('STOP changed after this request; newer stop takes precedence')


def archive_stops(request):
    validate_request(request)
    archive = WATCH / 'archived_stops' / str(time.time_ns())
    archive.mkdir(parents=True)
    for index, (filename, identity) in enumerate(request['stops'].items()):
        if identity is not None:
            Path(filename).rename(archive / (str(index) + '_STOP'))
    atomic_json(archive / 'REQUEST.json', request)


def train_one(request, cluster):
    from p2_loss_search28 import queue, resources
    original_idle = resources.idle
    def restricted_stages():
        # queue.run already owns the original queue.lock here. Recheck before
        # archiving only STOP files explicitly superseded by this user request.
        validate_request(request)
        snap = observe(cluster)
        if not snap['idle']:
            raise RuntimeError('Allocation became busy before launch; no overlapping training')
        atomic_json(WATCH / 'TRIGGERED.json', dict(at=time.time(), request=request, resources=snap))
        archive_stops(request)
        yield 500, [request['candidate']]
    def strict_idle(cl):
        snapshot = observe(cl)
        if not snapshot['idle']:
            raise RuntimeError('Selected allocation became busy at the native launch gate')
        native = original_idle(cl)
        if not native['idle']:
            raise RuntimeError('Native launch gate found competing compute jobs')
        return snapshot
    # Keep the frozen native training entry/checkpoint/source verification.
    # Limit scheduling to exactly one candidate, never later groups/promotions.
    with patch.object(queue, 'stages', restricted_stages), patch.object(resources, 'idle', strict_idle):
        queue.run()
    out = WORK / 'runs' / request['candidate']
    if not (out / 'STAGE_500.json').exists():
        raise RuntimeError('Training stopped before complete step500; inspect native checkpoint')
    atomic_json(WATCH / 'COMPLETE.json', dict(success=True, at=time.time(), candidate=request['candidate'],
                target=500, stage_receipt=read(out / 'STAGE_500.json'), additional_candidates_started=False))


def main(once=False):
    WATCH.mkdir(exist_ok=True)
    with (WATCH / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = read(WATCH / 'REQUEST.json')
        validate_request(request)
        config = validate(read(WORK / 'runs' / request['candidate'] / 'config.json'))
        cluster = validate_cluster(config, read(Path(config['output']) / 'cluster.json'))
        if once:
            result = observe(cluster)
            atomic_json(WATCH / 'PREFLIGHT_RESOURCES.json', result)
            print(json.dumps(result))
            return
        ticks = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
        atomic_json(WATCH / 'CONTROLLER.json', dict(pid=os.getpid(), start_ticks=ticks, at=time.time(),
                    interval_seconds=INTERVAL, required_idle_seconds=HOLD, candidate=request['candidate']))
        window = IdleWindow()
        while not (WATCH / 'STOP').exists():
            validate_request(request)
            try:
                snapshot = observe(cluster)
                ready, elapsed = window.update(snapshot['idle'], time.monotonic())
                status = dict(stage='waiting_for_28_idle_gpus', at=time.time(), idle_seconds=elapsed,
                              candidate=request['candidate'], target=500, resources=snapshot)
            except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as exc:
                ready, _ = window.update(False, time.monotonic())
                status = dict(stage='observation_error_timer_reset', at=time.time(), error=repr(exc), idle_seconds=0)
            atomic_json(WATCH / 'STATUS.json', status)
            if ready:
                atomic_json(WATCH / 'STATUS.json', dict(stage='starting_next_candidate', at=time.time(),
                            candidate=request['candidate'], target=500, observed_idle_seconds=elapsed))
                train_one(request, cluster)
                atomic_json(WATCH / 'STATUS.json', dict(stage='complete_one_candidate', at=time.time(),
                            candidate=request['candidate'], target=500))
                return
            deadline = time.monotonic() + INTERVAL
            while time.monotonic() < deadline:
                if (WATCH / 'STOP').exists():
                    break
                time.sleep(min(1, max(0, deadline-time.monotonic())))
        atomic_json(WATCH / 'STATUS.json', dict(stage='cancelled', at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true', help='read-only resource preflight; never launches')
    args = parser.parse_args()
    try:
        main(args.once)
    except Exception as exc:
        atomic_json(WATCH / 'FAILURE.json', dict(at=time.time(), error=repr(exc), traceback=traceback.format_exc()))
        raise
